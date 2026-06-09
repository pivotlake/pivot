//! A concrete [`planner::catalog::Catalog`] implementation backed by Parquet.
//!
//! The catalog stores tables in a `RwLock<HashMap>` keyed by name, so multiple
//! threads can resolve and create tables concurrently — many readers (lookups)
//! coexist with infrequent writers (`CREATE TABLE`). A `CREATE TABLE` statement
//! names its backing store with one of two `WITH` options:
//! * `WITH (path = '<dir>')` — a local directory of Parquet files.
//! * `WITH (url = '<s3://…/gs://…/local>')` — a goose object-store catalog: the
//!   latest `_goose_log/` snapshot is resolved into a [`ParquetTable`] over its
//!   data files (read locally or, for remote files, over the io_uring ring via
//!   presigned URLs), or a new snapshot is CAS-committed if the catalog is empty.
//!
//! Each `Catalog::table` lookup hands back a fresh [`Box<dyn Table>`] cloned
//! from the master entry, so per-binding filter pushdown accumulates on that
//! binding alone. [`Table::pushdown_filter`] just *records* each single-column
//! predicate; the pruning (dropping row groups whose min/max can't match, and
//! dictionary-pruning for equality) happens in [`Table::compile`], once the
//! row-group metadata is in hand. Pushdown always reports `false` because
//! per-row evaluation is still required on the survivors.

// Internal engine crate: the Parquet pipeline's public factories document their
// behaviour by linking to the private operators they build (e.g.
// `DecompressorFactory` → `Decompressor`). That's intentional here — we're not a
// published API — so allow public docs to reference private items.
#![allow(rustdoc::private_intra_doc_links)]

pub mod lake;
pub mod manifest;
pub mod metadata;
pub mod parquet;
pub mod store;
pub mod table_store;

pub use manifest::{InMemoryTableManifest, ManifestEntry, ObjectStoreManifest, TableManifest};
pub use table_store::TableObjectStore;

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::parquet::{
    ParquetTable, ParquetTableError, ScanEqualityPredicate, row_group_eliminated,
    row_group_filter_from, scan_order_from, table_input_with_filter_and_eq_predicates,
};
use arrow_array::{ArrayRef, RecordBatch, Scalar};
use dispatch::{
    DataFlowDispatcher, Nullary, NullaryFactory, NullaryResult, Projection,
    RecordBatchOperatorSpec, Sender, WorkStatus,
};
use planner::catalog::{
    Catalog, Column, CreateTableRequest, DynamicScanPredicate, Error as CatalogError,
    Result as CatalogResult, Table,
};
use planner::expression::{CompareType, Expression, TableFilter};
use std::sync::atomic::{AtomicBool, Ordering};
use thiserror::Error;

const PATH_OPTION: &str = "path";

#[derive(Debug, Error)]
pub enum Error {
    #[error(
        "CREATE TABLE needs a `{PATH_OPTION}` option, or a server started with a database directory"
    )]
    MissingPath,
    #[error("path `{0}` does not exist")]
    PathNotFound(String),
    #[error("path `{0}` is not a directory")]
    PathNotDirectory(String),
    #[error(
        "table path `{0}` must be a local path: a local database cannot hold an object-store table"
    )]
    RemoteTablePath(String),
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
    #[error(transparent)]
    Manifest(#[from] manifest::ManifestError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<Error> for CatalogError {
    fn from(value: Error) -> Self {
        CatalogError::Other(Box::new(value))
    }
}

/// Concurrent catalog of [`ParquetCatalogTable`]s, keyed by table name.
///
/// `CREATE TABLE`/`ATTACH` reads every data file's footer **once**, when the
/// table is defined (in parallel over the dispatch worker pool), building the
/// [`ParquetTable`] (row groups + open file handles) that every later query
/// reuses. The table map is shared (`Arc`) with the write operator the
/// `CREATE TABLE` plan compiles to, so that plan can commit the built table.
#[derive(Debug)]
pub struct ParquetCatalog {
    tables: Arc<RwLock<HashMap<String, ParquetCatalogTable>>>,
    /// Durable record of which tables exist. In-memory by default (ephemeral);
    /// for a database opened with [`open`](Self::open) it persists to the
    /// database's object store and is reloaded into `tables` at startup.
    manifest: Arc<dyn TableManifest>,
    /// The database root directory (a persisted local database). A `CREATE TABLE`
    /// with no explicit `path` puts the new table under `<root>/<name>`; `None`
    /// for an in-memory database, where every table must name its own `path`.
    root: Option<PathBuf>,
}

impl Default for ParquetCatalog {
    fn default() -> Self {
        Self {
            tables: Arc::new(RwLock::new(HashMap::new())),
            manifest: Arc::new(InMemoryTableManifest),
            root: None,
        }
    }
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

    /// Open a persisted database rooted at `uri` (a local directory, or — with an
    /// object-store manifest — `s3://…`/`gs://…`), reloading every table the
    /// manifest records. Each is re-materialized from its data directory, so a
    /// restart restores the same catalog.
    pub fn open(uri: &str, dispatcher: &DataFlowDispatcher) -> Result<Self> {
        let manifest: Arc<dyn TableManifest> =
            Arc::new(ObjectStoreManifest::new(store::open_store(uri)?));
        let catalog = Self {
            tables: Arc::new(RwLock::new(HashMap::new())),
            manifest,
            root: local_root(uri),
        };
        for entry in catalog.manifest.load()? {
            let location = catalog.resolve(&entry.location);
            let parquet = Arc::new(materialize(dispatcher, &location)?);
            catalog.tables.write().unwrap().insert(
                entry.name,
                ParquetCatalogTable {
                    columns: entry.columns,
                    location,
                    parquet,
                    predicates: Vec::new(),
                },
            );
        }
        Ok(catalog)
    }

    /// Create a table and register it in the manifest.
    ///
    /// The data lives in a directory: an explicit `WITH (path = '…')` — a local
    /// path, inside or outside the database root, never an object-store URL — or
    /// `<database-root>/<name>` when none is given. `CREATE TABLE x (…)` with no
    /// path on an in-memory database has nowhere to put data and is rejected.
    /// Every footer under the directory is read (in parallel) into the reusable
    /// row groups; an empty directory yields an empty table.
    fn create(
        &self,
        request: CreateTableRequest,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec> {
        if request.if_not_exists {
            return Err(Error::IfNotExistsUnsupported);
        }
        // Reject a duplicate before persisting it; the write operator re-checks
        // under the lock as a race backstop.
        if self.tables.read().unwrap().contains_key(&request.name) {
            return Err(Error::TableExists(request.name));
        }

        let stored = self.locate(&request)?;
        let location = self.resolve(&stored);
        let parquet = Arc::new(materialize(dispatcher, &location)?);

        self.manifest.insert(&ManifestEntry {
            name: request.name.clone(),
            columns: request.columns.clone(),
            location: stored,
        })?;

        let table = ParquetCatalogTable {
            columns: request.columns,
            location,
            parquet,
            predicates: Vec::new(),
        };
        Ok(self.write_table_spec(dispatcher, request.name, table))
    }

    /// The data directory for a new table, in the form stored in the manifest: an
    /// explicit local `path` (kept as given), or the table name relative to the
    /// database root. For the latter it creates `<root>/<name>` so an empty table
    /// has somewhere to hold data.
    fn locate(&self, request: &CreateTableRequest) -> Result<PathBuf> {
        if let Some(path) = request.options.get(PATH_OPTION) {
            if has_object_store_scheme(path) {
                return Err(Error::RemoteTablePath(path.clone()));
            }
            let dir = PathBuf::from(path);
            if !dir.exists() {
                return Err(Error::PathNotFound(path.clone()));
            }
            if !dir.is_dir() {
                return Err(Error::PathNotDirectory(path.clone()));
            }
            Ok(dir)
        } else if let Some(root) = &self.root {
            std::fs::create_dir_all(root.join(&request.name))?;
            Ok(PathBuf::from(&request.name))
        } else {
            Err(Error::MissingPath)
        }
    }

    /// Resolve a manifest-stored location to an absolute path: a relative one is
    /// taken against the database root, an absolute one used as-is.
    fn resolve(&self, location: &Path) -> PathBuf {
        match &self.root {
            Some(root) if location.is_relative() => root.join(location),
            _ => location.to_path_buf(),
        }
    }

    /// Build the dataflow that commits an already-materialized table into the
    /// catalog: one worker inserts it (failing if the name is taken), the rest
    /// are no-ops. Sharing the table map (`Arc`) lets the executed plan — not
    /// the coordinator — perform the write.
    fn write_table_spec(
        &self,
        dispatcher: &DataFlowDispatcher,
        name: String,
        table: ParquetCatalogTable,
    ) -> RecordBatchOperatorSpec {
        let already_written = Arc::new(AtomicBool::new(false));
        RecordBatchOperatorSpec::from_nullary(
            dispatcher,
            (0..dispatcher.worker_count()).map(|_| CatalogTableWriterFactory {
                tables: self.tables.clone(),
                name: name.clone(),
                table: table.clone(),
                already_written: already_written.clone(),
            }),
        )
    }
}

/// Read every Parquet footer under `dir` into a table's row groups. A missing or
/// empty directory yields an empty table — a freshly-created, data-less table.
fn materialize(dispatcher: &DataFlowDispatcher, dir: &Path) -> Result<ParquetTable> {
    if !dir.exists() {
        return Ok(ParquetTable::new(Vec::new()));
    }
    Ok(ParquetTable::from_directory(dispatcher, dir)?)
}

/// The local root directory for a database `uri`, or `None` if it names a remote
/// object store (whose tables aren't local paths). `file://` is local.
fn local_root(uri: &str) -> Option<PathBuf> {
    if is_remote_uri(uri) {
        None
    } else {
        Some(PathBuf::from(uri.strip_prefix("file://").unwrap_or(uri)))
    }
}

/// Whether a database URI names a remote object store (matching `open_store`).
fn is_remote_uri(uri: &str) -> bool {
    uri.starts_with("s3://") || uri.starts_with("s3a://") || uri.starts_with("gs://")
}

/// Whether a table `path` carries any object-store / URL scheme, which a table
/// location never may — it must be a bare local path.
fn has_object_store_scheme(path: &str) -> bool {
    is_remote_uri(path) || path.starts_with("gcs://") || path.starts_with("file://")
}

impl Catalog for ParquetCatalog {
    /// Resolve `name` to a fresh, independently-mutable [`ParquetCatalogTable`].
    /// Each binding gets its own clone so per-query filter pushdown can prune
    /// row groups without affecting the master entry or other concurrent
    /// queries.
    fn table(&self, name: &str) -> Option<Box<dyn Table>> {
        self.parquet_table(name).map(|t| Box::new(t) as _)
    }

    fn create_table(
        &self,
        request: CreateTableRequest,
        dispatcher: &DataFlowDispatcher,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        Ok(self.create(request, dispatcher)?)
    }
}

/// Per-worker factory for [`CatalogTableWriter`].
struct CatalogTableWriterFactory {
    tables: Arc<RwLock<HashMap<String, ParquetCatalogTable>>>,
    name: String,
    table: ParquetCatalogTable,
    already_written: Arc<AtomicBool>,
}

impl NullaryFactory<RecordBatch> for CatalogTableWriterFactory {
    type Nullary = CatalogTableWriter;

    fn build_nullary(self) -> CatalogTableWriter {
        CatalogTableWriter {
            tables: self.tables,
            name: self.name,
            table: Some(self.table),
            already_written: self.already_written,
            ran: false,
        }
    }
}

/// The dataflow `CREATE TABLE` compiles to: the first worker to claim the shared
/// flag commits the already-materialized [`ParquetCatalogTable`] into the
/// catalog map (failing if the name is taken); the rest are no-ops. Emits no
/// rows. The footers were already read in parallel when this plan was compiled.
struct CatalogTableWriter {
    tables: Arc<RwLock<HashMap<String, ParquetCatalogTable>>>,
    name: String,
    /// Taken by the worker that wins the write; left `None` on the no-op workers.
    table: Option<ParquetCatalogTable>,
    already_written: Arc<AtomicBool>,
    ran: bool,
}

impl Nullary<RecordBatch> for CatalogTableWriter {
    fn run<S: Sender<RecordBatch>>(&mut self, _sender: &mut S) -> NullaryResult<WorkStatus> {
        if self.ran {
            return Ok(WorkStatus::Pending);
        }
        self.ran = true;
        if !self.already_written.swap(true, Ordering::SeqCst) {
            let table = self.table.take().expect("winning worker holds the table");
            match self.tables.write().unwrap().entry(self.name.clone()) {
                Entry::Occupied(entry) => {
                    let err: Box<dyn std::error::Error + Send + Sync> =
                        Box::new(Error::TableExists(entry.key().clone()));
                    return Err(err.into());
                }
                Entry::Vacant(entry) => {
                    entry.insert(table);
                }
            }
        }
        Ok(WorkStatus::Ran)
    }

    fn finish<S: Sender<RecordBatch>>(&mut self, _sender: &mut S) -> NullaryResult<bool> {
        Ok(self.ran)
    }
}

/// A single-column constant comparison (`col <cmp> const`) pushed down by
/// DuckDB during binding. Recorded as-is; applied at [`compile`](Table::compile)
/// time after the row-group metadata exists — min/max stats prune row groups,
/// and equality additionally prunes by dictionary contents in the decoder. The
/// upstream `Filter` always runs, so this is a pure optimization.
#[derive(Clone, Debug)]
struct PushedPredicate {
    column_idx: usize,
    compare_type: CompareType,
    value: Scalar<ArrayRef>,
}

/// A catalog table backed by a directory of Parquet files. Cloned per-binding
/// so each query accumulates its own pushed-down predicates via
/// [`Table::pushdown_filter`] without affecting others.
#[derive(Clone, Debug)]
pub struct ParquetCatalogTable {
    pub columns: Vec<Column>,
    /// The absolute directory holding this table's Parquet data. Kept so a future
    /// `refresh()` can re-read the latest files without consulting the manifest.
    pub location: PathBuf,
    /// The table's row-group metadata, read once when the table was defined and
    /// shared across every binding/query (cheap `Arc` clone). Pruning a binding's
    /// predicates clones the row-group `Vec` and filters it — no footer re-read.
    pub parquet: Arc<ParquetTable>,
    /// Single-column predicates pushed down for this binding (recorded here
    /// because the `Table` trait gives no channel from `pushdown_filter` to
    /// `compile`); applied as a filter when the scan is compiled.
    predicates: Vec<PushedPredicate>,
}

impl Table for ParquetCatalogTable {
    fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        dynamic_filters: Vec<DynamicScanPredicate>,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // Equality predicates additionally let the decoder skip row groups whose
        // dictionary for that column excludes the constant.
        let eq_predicates: Vec<ScanEqualityPredicate> = self
            .predicates
            .iter()
            .filter(|p| matches!(p.compare_type, CompareType::Equal))
            .map(|p| ScanEqualityPredicate {
                column_idx: p.column_idx,
                value: p.value.clone(),
            })
            .collect();

        // Prune the (already-materialized) row groups by the pushed-down
        // predicates' stats. No footer re-read — the row groups were built once
        // when the table was defined.
        let parquet = Arc::new(self.pruned_parquet());
        // Order the scan by the Top-N's key so its boundary tightens after the
        // first row group and the rest get pruned, instead of racing file order.
        let scan_order = scan_order_from(&dynamic_filters);
        Ok(table_input_with_filter_and_eq_predicates(
            dispatcher,
            &parquet,
            projection,
            false,
            row_group_filter_from(dynamic_filters),
            scan_order,
            Arc::new(eq_predicates),
        ))
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

        // Just record it. The actual pruning (min/max row-group elimination and
        // equality/dictionary pruning) happens in `compile`, once the row-group
        // metadata exists. The upstream `Filter` is kept (we return `Ok(false)`),
        // so this is purely an optimization and never affects correctness.
        self.predicates.push(PushedPredicate {
            column_idx: reference.column_idx,
            compare_type: compare.compare_type,
            value: constant.clone(),
        });

        Ok(false)
    }
}

impl ParquetCatalogTable {
    /// Clone this table's row groups and keep only those that survive this
    /// binding's pushed-down predicates — i.e. what [`Table::compile`] actually
    /// scans. A min/max stat that proves no row in a group can match drops it; a
    /// stats-comparison error means "can't prune" (kept) — never wrong, just
    /// unoptimized. No footer I/O: the row groups were materialized once when the
    /// table was defined. Exposed so pruning can be asserted directly.
    pub fn pruned_parquet(&self) -> ParquetTable {
        let mut parquet = (*self.parquet).clone();
        parquet.row_groups_mut().retain(|rg| {
            !self.predicates.iter().any(|p| {
                row_group_eliminated(rg.as_ref(), p.column_idx, p.compare_type, &p.value)
                    .unwrap_or(false)
            })
        });
        parquet
    }
}
