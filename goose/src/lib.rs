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

pub mod lake;
pub mod metadata;
pub mod parquet;
pub mod store;

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::Path;
use std::sync::{Arc, RwLock};

use crate::parquet::{
    ParquetTable, ParquetTableError, ScanEqualityPredicate, row_group_eliminated,
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
const URL_OPTION: &str = "url";

#[derive(Debug, Error)]
pub enum Error {
    #[error("CREATE TABLE needs a `{PATH_OPTION}` (local directory) or `{URL_OPTION}` (goose catalog) option")]
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
    #[error(transparent)]
    Store(#[from] store::StoreError),
    #[error("data file `{0}` is on remote object storage; remote data reads are not wired up yet")]
    RemoteDataNotSupported(String),
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
        self.insert(request.name, table)
    }

    /// Create a table from a goose catalog snapshot (`WITH (url = '…')`).
    ///
    /// The URL names one catalog root (one table). If the latest snapshot
    /// already defines a table, we attach to it — building a [`ParquetTable`]
    /// over its recorded data files. If the catalog is empty, we CAS-commit a
    /// new snapshot defining this table (declared columns, no data files yet);
    /// ingest fills in data files via its own commits.
    ///
    /// Remote (`s3://`/`gs://`) *data* files are not yet readable, so attaching
    /// a catalog whose files live on object storage errors clearly; the catalog
    /// metadata itself may still live on object storage.
    ///
    /// Runs on a dispatch worker (it calls [`ParquetTable::from_files`]).
    fn create_lake_table(&self, request: CreateTableRequest) -> Result<()> {
        if request.if_not_exists {
            return Err(Error::IfNotExistsUnsupported);
        }
        let url = request.options.get(URL_OPTION).ok_or(Error::MissingPath)?;
        let store = store::open_store(url)?;
        let snapshot = store::latest_snapshot(store.as_ref())?;

        // The URL identifies a single table, so take the sole table the snapshot
        // describes (if any).
        let existing = snapshot.schemas.iter().flat_map(|s| &s.tables).next();

        let parquet = match existing {
            Some(table) => {
                let mut paths = Vec::with_capacity(table.files.len());
                for file in &table.files {
                    match lake::local_path(url, &file.location) {
                        Some(p) => paths.push(p),
                        None => {
                            return Err(Error::RemoteDataNotSupported(file.location.clone()));
                        }
                    }
                }
                ParquetTable::from_files(&paths)?
            }
            None => {
                // New catalog: commit an initial snapshot defining this table.
                let columns: Vec<metadata::Column> = request
                    .columns
                    .iter()
                    .map(|c| metadata::Column {
                        name: c.name.clone(),
                        type_sql: lake::pivot_type_to_sql(&c.col_type).to_string(),
                    })
                    .collect();
                let name = request.name.clone();
                store::commit(store.as_ref(), |snap| {
                    let schema = ensure_main_schema(snap);
                    schema.tables.push(metadata::Table {
                        name: name.clone(),
                        columns: columns.clone(),
                        files: Vec::new(),
                    });
                    Ok(())
                })?;
                ParquetTable::new(Vec::new())
            }
        };

        let table = ParquetCatalogTable {
            columns: request.columns,
            parquet,
            eq_predicates: Vec::new(),
        };
        self.insert(request.name, table)
    }

    /// Insert a built table, failing if the name is already taken.
    fn insert(&self, name: String, table: ParquetCatalogTable) -> Result<()> {
        match self.tables.write().unwrap().entry(name) {
            Entry::Occupied(entry) => Err(Error::TableExists(entry.key().clone())),
            Entry::Vacant(entry) => {
                entry.insert(table);
                Ok(())
            }
        }
    }

    /// Route a `CREATE TABLE` to the local-directory or goose-catalog backend
    /// based on which option it carries.
    fn create_any(&self, request: CreateTableRequest) -> Result<()> {
        if request.options.contains_key(URL_OPTION) {
            self.create_lake_table(request)
        } else {
            self.create_parquet_table(request)
        }
    }
}

/// Get the `main` schema, creating it if a snapshot somehow lacks one.
fn ensure_main_schema(snap: &mut metadata::CatalogSnapshot) -> &mut metadata::Schema {
    if !snap.schemas.iter().any(|s| s.name == "main") {
        snap.schemas.push(metadata::Schema {
            name: "main".to_string(),
            tables: Vec::new(),
        });
    }
    snap.schemas.iter_mut().find(|s| s.name == "main").unwrap()
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
        Ok(self.create_any(request)?)
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
    ) -> RecordBatchOperatorSpec {
        let parquet = Arc::new(self.parquet.clone());
        // Order the scan by the Top-N's key so its boundary tightens after the
        // first row group and the rest get pruned, instead of racing file order.
        let scan_order = scan_order_from(&dynamic_filters);
        table_input_with_filter_and_eq_predicates(
            dispatcher,
            &parquet,
            projection,
            false,
            row_group_filter_from(dynamic_filters),
            scan_order,
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
