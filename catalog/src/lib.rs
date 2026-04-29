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
    ParquetTable, ParquetTableError, Projection, RecordBatchOperatorSpec, RowGroupMetadata,
    table_input,
};
use planner::catalog::{
    Catalog, Column, CreateTableRequest, Error as CatalogError, Result as CatalogResult, Table,
};
use planner::expression::{CompareType, ConstantComparison, Expression, Ref, TableFilter};
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
    fn compile(&self, projection: Projection) -> RecordBatchOperatorSpec {
        let parquet = Arc::new(self.parquet.clone());
        table_input(&parquet, projection, false)
    }

    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    fn pushdown_filter(&mut self, filter: TableFilter) -> bool {
        let TableFilter::ConstantComparison(cmp) = filter else {
            return false;
        };

        let Expression::Ref(reference) = cmp.column_ref.as_ref() else {
            return false;
        };

        self.parquet
            .row_groups_mut()
            .retain(|rg| should_retain_row_group(rg, reference, &cmp));

        false
    }
}

/// Decide whether a row group could contain rows satisfying the predicate,
/// based on the column's min/max statistics. Returns `true` to keep the row
/// group (it might match and must be scanned) and `false` to prune it.
fn should_retain_row_group(
    row_group: &RowGroupMetadata,
    column: &Ref,
    compare: &ConstantComparison,
) -> bool {
    let Some(stats) = row_group
        .columns
        .get(column.column_idx)
        .and_then(|c| c.statistics.as_ref())
    else {
        return true;
    };
    let (Some(min), Some(max)) = (stats.min.as_ref(), stats.max.as_ref()) else {
        return true;
    };

    match compare.compare_type {
        // `col <> k` is true on every row unless every row in this group
        // equals `k`
        CompareType::NotEqual => {
            !(scalars_equal(min, &compare.constant) && scalars_equal(max, &compare.constant))
        }
    }
}

fn scalars_equal(a: &Scalar<ArrayRef>, b: &Scalar<ArrayRef>) -> bool {
    let result: BooleanArray = match arrow_ord::cmp::eq(a as &dyn Datum, b as &dyn Datum) {
        Ok(arr) => arr,
        Err(_) => return false,
    };
    result.len() == 1 && result.value(0)
}
