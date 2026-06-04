//! A concrete [`planner::catalog::Catalog`] implementation backed by Parquet
//! directories on disk.
//!
//! The catalog stores tables in a `RwLock<HashMap>` keyed by name, so multiple
//! threads can resolve and create tables concurrently — many readers (lookups)
//! coexist with infrequent writers (`CREATE TABLE`). The only kind of table
//! supported today is a directory of Parquet files: a `CREATE TABLE` statement
//! must carry a `WITH (path = '...')` option, and the catalog validates that
//! the path points to a directory it can open as a [`ParquetTable`].
//!
//! Each `Catalog::table` lookup hands back a fresh [`Box<dyn Table>`] cloned
//! from the master entry, so per-binding filter pushdown can mutate the
//! row-group set in place without affecting the master or other queries.
//! [`Table::pushdown_filter`] keeps row groups whose min/max statistics could
//! still satisfy the predicate and drops the rest. Pushdown always reports
//! `false` because per-row evaluation is still required on the survivors.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::Path;
use std::sync::{Arc, RwLock};

use arrow_array::{ArrayRef, BooleanArray, Datum, Scalar};
use dispatch::{
    DataFlowDispatcher, ParquetTable, ParquetTableError, Projection, RecordBatchOperatorSpec,
    RowGroupMetadata, ScanEqualityPredicate, table_input_with_eq_predicates,
};
use planner::catalog::{
    Catalog, Column, CreateTableRequest, Error as CatalogError, Result as CatalogResult, Table,
};
use planner::expression::{Compare, CompareType, Expression, Ref, TableFilter};
use thiserror::Error;

const PATH_OPTION: &str = "path";

#[derive(Debug, Error)]
pub enum Error {
    #[error("CREATE TABLE missing required option `{PATH_OPTION}`")]
    MissingPath,
    #[error("path `{0}` does not exist")]
    PathNotFound(String),
    #[error("path `{0}` is not a directory")]
    PathNotDirectory(String),
    #[error("`IF NOT EXISTS` is not supported")]
    IfNotExistsUnsupported,
    #[error(transparent)]
    ParquetTable(#[from] ParquetTableError),
    #[error("table `{0}` already exists")]
    TableExists(String),
    #[error(transparent)]
    Arrow(#[from] arrow_schema::ArrowError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<Error> for CatalogError {
    fn from(value: Error) -> Self {
        CatalogError::Other(Box::new(value))
    }
}

/// Concurrent catalog of [`ParquetCatalogTable`]s, keyed by table name.
#[derive(Debug, Default)]
pub struct ParquetCatalog {
    tables: RwLock<HashMap<String, ParquetCatalogTable>>,
}

impl ParquetCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve `name` to a fresh [`ParquetCatalogTable`] (same semantics as
    /// [`Catalog::table`], but typed). Useful for callers (and tests) that
    /// need state the [`Table`] trait does not expose.
    ///
    /// NOTE: Performance wise, cloning all the row groups for every new query resolve
    /// might be expensive for very large dataset (10/100tb+). If this ever becomes
    /// a bottleneck, we can consider using another mechanism.
    pub fn parquet_table(&self, name: &str) -> Option<ParquetCatalogTable> {
        self.tables.read().unwrap().get(name).cloned()
    }

    fn create_parquet_table(&self, request: CreateTableRequest) -> Result<()> {
        if request.if_not_exists {
            return Err(Error::IfNotExistsUnsupported);
        }
        // Validate inputs (path option, filesystem) and open the parquet
        // directory before taking the write lock.
        let path = request.options.get(PATH_OPTION).ok_or(Error::MissingPath)?;
        let path_buf = Path::new(path);
        if !path_buf.exists() {
            return Err(Error::PathNotFound(path.clone()));
        }
        if !path_buf.is_dir() {
            return Err(Error::PathNotDirectory(path.clone()));
        }
        let parquet = ParquetTable::from_directory(path_buf)?;

        let table = ParquetCatalogTable {
            columns: request.columns,
            parquet,
            eq_predicates: Vec::new(),
        };

        match self.tables.write().unwrap().entry(request.name) {
            Entry::Occupied(entry) => Err(Error::TableExists(entry.key().clone())),
            Entry::Vacant(entry) => {
                entry.insert(table);
                Ok(())
            }
        }
    }
}

impl Catalog for ParquetCatalog {
    /// Resolve `name` to a fresh, independently-mutable [`ParquetCatalogTable`].
    /// Each binding gets its own clone so per-query filter pushdown can prune
    /// row groups without affecting the master entry or other concurrent
    /// queries.
    fn table(&self, name: &str) -> Option<Box<dyn Table>> {
        self.parquet_table(name).map(|t| Box::new(t) as _)
    }

    fn create_table(&self, request: CreateTableRequest) -> CatalogResult<()> {
        Ok(self.create_parquet_table(request)?)
    }
}

/// A catalog table backed by a directory of Parquet files. Cloned per-binding
/// so each query can mutate its own copy via [`Table::pushdown_filter`].
#[derive(Clone, Debug)]
pub struct ParquetCatalogTable {
    pub columns: Vec<Column>,
    pub parquet: ParquetTable,
    /// Equality predicates pushed down by DuckDB. The scan uses them to prune
    /// row groups by dictionary contents; the upstream `Filter` still runs, so
    /// pruning is a pure optimization.
    pub eq_predicates: Vec<ScanEqualityPredicate>,
}

impl Table for ParquetCatalogTable {
    fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
    ) -> RecordBatchOperatorSpec {
        let parquet = Arc::new(self.parquet.clone());
        table_input_with_eq_predicates(
            dispatcher,
            &parquet,
            projection,
            false,
            Arc::new(self.eq_predicates.clone()),
        )
    }

    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    fn pushdown_filter(&mut self, filter: TableFilter) -> CatalogResult<bool> {
        let TableFilter::Expression(expr) = filter else {
            return Ok(false);
        };
        let Expression::Compare(compare) = expr.as_ref() else {
            return Ok(false);
        };
        let (reference, constant) = match (compare.left.as_ref(), compare.right.as_ref()) {
            (Expression::Ref(r), Expression::Constant(k))
            | (Expression::Constant(k), Expression::Ref(r)) => (r, k),
            _ => return Ok(false),
        };

        let mut retain_err = Ok(());
        self.parquet.row_groups_mut().retain(|rg| {
            match should_filter_row_group(rg, compare, reference, constant) {
                Ok(b) => !b,
                Err(e) => {
                    retain_err = Err(e);
                    true
                }
            }
        });
        retain_err?;

        // Record equality predicates so the scan can prune row groups whose
        // dictionary for this column excludes the constant. The upstream
        // `Filter` is kept (we return `Ok(false)`), so this is purely an
        // optimization and never affects correctness.
        if matches!(compare.compare_type, CompareType::Equal) {
            self.eq_predicates.push(ScanEqualityPredicate {
                column_idx: reference.column_idx,
                value: constant.clone(),
            });
        }

        Ok(false)
    }
}

/// Decide whether a row group can be skipped (filtered out) for the given
/// predicate, based on the column's min/max statistics. Returns `Ok(true)`
/// to drop the row group, `Ok(false)` to keep it.
///
/// When stats are missing we return `Ok(false)` — a missing bound is not
/// proof of absence, so the row group must still be scanned.
fn should_filter_row_group(
    row_group: &RowGroupMetadata,
    compare: &Compare,
    column: &Ref,
    constant: &Scalar<ArrayRef>,
) -> Result<bool> {
    let Some(stats) = row_group
        .columns
        .get(column.column_idx)
        .and_then(|c| c.statistics.as_ref())
    else {
        return Ok(false);
    };
    let (Some(min), Some(max)) = (stats.min.as_ref(), stats.max.as_ref()) else {
        return Ok(false);
    };

    // The min/max statistics come back typed as the physical parquet column
    // (e.g. a DATE column stored as UInt16), while the constant carries the
    // logical SQL type (e.g. Date32). The comparison kernels below require
    // matching types, so when they differ we can't prune from stats — keep the
    // row group (always safe; just no pruning).
    let (min_arr, _) = Datum::get(min);
    let (const_arr, _) = Datum::get(constant);
    if min_arr.data_type() != const_arr.data_type() {
        return Ok(false);
    }

    Ok(match compare.compare_type {
        // `col <> k` is true on every row unless every row in this group
        // equals `k` — provable only when min == max == k.
        CompareType::NotEqual => {
            bool_kernel(min, constant, arrow_ord::cmp::eq)?
                && bool_kernel(max, constant, arrow_ord::cmp::eq)?
        }
        // `col = k` can never match when k is strictly outside [min, max].
        CompareType::Equal => {
            bool_kernel(constant, min, arrow_ord::cmp::lt)?
                || bool_kernel(constant, max, arrow_ord::cmp::gt)?
        }
        // `col < k` matches nothing when every value is >= k, i.e. min >= k.
        CompareType::Less => bool_kernel(min, constant, arrow_ord::cmp::gt_eq)?,
        // `col > k` matches nothing when every value is <= k, i.e. max <= k.
        CompareType::Greater => bool_kernel(max, constant, arrow_ord::cmp::lt_eq)?,
        // `col <= k` matches nothing when every value is > k, i.e. min > k.
        CompareType::LessEqual => bool_kernel(min, constant, arrow_ord::cmp::gt)?,
        // `col >= k` matches nothing when every value is < k, i.e. max < k.
        CompareType::GreaterEqual => bool_kernel(max, constant, arrow_ord::cmp::lt)?,
    })
}

fn bool_kernel(
    a: &Scalar<ArrayRef>,
    b: &Scalar<ArrayRef>,
    kernel: fn(&dyn Datum, &dyn Datum) -> Result<BooleanArray, arrow_schema::ArrowError>,
) -> Result<bool> {
    let result = kernel(a as &dyn Datum, b as &dyn Datum)?;
    Ok(result.len() == 1 && result.value(0))
}
