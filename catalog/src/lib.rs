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

// Internal engine crate: the Parquet pipeline's public factories document their
// behaviour by linking to the private operators they build (e.g.
// `DecompressorFactory` → `Decompressor`). That's intentional here — we're not a
// published API — so allow public docs to reference private items.
#![allow(rustdoc::private_intra_doc_links)]

pub mod parquet;

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::Path;
use std::sync::{Arc, RwLock};

use crate::parquet::{
    ParquetTable, ParquetTableError, ScanEqualityPredicate, materialize, row_group_eliminated,
    row_group_filter_from, scan_order_from, table_input_with_filter_and_eq_predicates,
};
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use planner::catalog::{
    Catalog, Column, CreateTableRequest, DynamicScanPredicate, Error as CatalogError,
    Result as CatalogResult, Table,
};
use planner::expression::{CompareType, Expression, TableFilter};
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
        dynamic_filters: Vec<DynamicScanPredicate>,
        emit_row_group_metadata: bool,
    ) -> RecordBatchOperatorSpec {
        let parquet = Arc::new(self.parquet.clone());
        // Order the scan by the Top-N's key so its boundary tightens after the
        // first row group and the rest get pruned, instead of racing file order.
        let scan_order = scan_order_from(&dynamic_filters);
        table_input_with_filter_and_eq_predicates(
            dispatcher,
            &parquet,
            projection,
            emit_row_group_metadata,
            row_group_filter_from(dynamic_filters),
            scan_order,
            Arc::new(self.eq_predicates.clone()),
        )
    }

    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    fn clone_box(&self) -> Box<dyn Table> {
        Box::new(self.clone())
    }

    fn materialize(
        &self,
        input: RecordBatchOperatorSpec,
        projection: Projection,
    ) -> RecordBatchOperatorSpec {
        materialize(input, Arc::new(self.parquet.clone()), projection)
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

        // Statically prune row groups the predicate proves can't match, using
        // the shared stats logic (also used for dynamic filters at scan time).
        let mut retain_err: CatalogResult<()> = Ok(());
        self.parquet.row_groups_mut().retain(|rg| {
            match row_group_eliminated(
                rg.as_ref(),
                reference.column_idx,
                compare.compare_type,
                constant,
            ) {
                Ok(eliminated) => !eliminated,
                Err(e) => {
                    retain_err = Err(CatalogError::Other(Box::new(e)));
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
