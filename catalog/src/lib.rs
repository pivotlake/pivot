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

use arrow_array::{Array, ArrayRef, Datum, Int64Array, Scalar};
use dispatch::{
    ParquetTable, ParquetTableError, Projection, RecordBatchOperatorSpec, RowGroupFilter,
    RowGroupMetadata, table_input,
};
use planner::catalog::{
    Catalog, Column, CreateTableRequest, Error as CatalogError, Result as CatalogResult, Table,
};
use planner::expression::{Compare, Expression, Ref, TableFilter};
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
        let mut parquet = ParquetTable::from_directory(path_buf)?;
        sort_row_groups_by_event_time_desc(&mut parquet, &request.columns);

        let table = ParquetCatalogTable {
            columns: request.columns,
            parquet,
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
}

impl Table for ParquetCatalogTable {
    fn compile(
        &self,
        projection: Projection,
        row_group_filter: Option<RowGroupFilter>,
    ) -> RecordBatchOperatorSpec {
        let parquet = Arc::new(self.parquet.clone());
        table_input(&parquet, projection, false, row_group_filter)
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

        Ok(false)
    }
}

/// Reorder a table's row groups by their `EventTime` column's min statistic,
/// highest first. If there is no `EventTime` column, or its stats aren't a
/// BIGINT min that we can read, the table is left untouched.
///
/// This is a deliberate ClickBench-specific knob: with `ORDER BY EventTime`
/// queries we want to control the work-stealing order to study how the
/// dynamic-filter pruning interacts with it. Once we want this for other
/// columns / other tables we'll lift the column name out into a catalog
/// option.
fn sort_row_groups_by_event_time_desc(parquet: &mut ParquetTable, columns: &[Column]) {
    let Some(event_time_idx) = columns.iter().position(|c| c.name == "EventTime") else {
        return;
    };
    parquet.row_groups_mut().sort_by(|a, b| {
        let a_min = min_event_time(a, event_time_idx);
        let b_min = min_event_time(b, event_time_idx);
        // `None` (missing stats) sorts to the end of the queue — we
        // can't position it intelligently, so don't let it crowd out
        // groups with usable stats.
        match (b_min, a_min) {
            (Some(a), Some(b)) => b.cmp(&a),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        }
    });
}

fn min_event_time(rg: &RowGroupMetadata, col_idx: usize) -> Option<i64> {
    let stats = rg.columns.get(col_idx)?.statistics.as_ref()?;
    let min = stats.min.as_ref()?;
    let (arr, _) = min.get();
    let arr = arr.as_any().downcast_ref::<Int64Array>()?;
    if arr.is_empty() || arr.is_null(0) {
        return None;
    }
    Some(arr.value(0))
}

/// Decide whether a row group can be skipped (filtered out) for the given
/// predicate, based on the column's min/max statistics. Returns `Ok(true)`
/// to drop the row group, `Ok(false)` to keep it.
fn should_filter_row_group(
    row_group: &RowGroupMetadata,
    compare: &Compare,
    column: &Ref,
    constant: &Scalar<ArrayRef>,
) -> Result<bool> {
    Ok(planner::row_group_stats::row_group_eliminated(
        row_group,
        column.column_idx,
        compare.compare_type,
        constant,
    )?)
}
