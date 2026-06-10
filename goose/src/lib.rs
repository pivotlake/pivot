//! A concrete [`planner::catalog::Catalog`] implementation backed by Parquet.
//!
//! The catalog stores tables in a `RwLock<HashMap>` keyed by name, so multiple
//! threads can resolve and create tables concurrently — many readers (lookups)
//! coexist with infrequent writers (`CREATE TABLE`). Backing the map is the
//! database's [`ObjectStore`]: an in-memory store by
//! default ([`ParquetCatalog::new`], ephemeral), or — for a database opened with
//! [`ParquetCatalog::open`] on a local directory *or* a remote `s3://`/`gs://`
//! root — a persisted one. It holds the table [`manifest`], so a restart reloads
//! every table.
//!
//! Each table's data is a directory/prefix of Parquet files: `<name>` under the
//! database root by default, or an explicit `WITH (path = '<dir>')` — a bare
//! local path (absolute or relative), never an object-store URL. `CREATE TABLE`
//! reads every footer under that location (in parallel over the dispatch pool)
//! into the reusable [`ParquetTable`] row groups — local files directly, remote
//! ones presigned and range-read over the ring — then records the table in the
//! manifest.
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

pub mod manifest;
pub mod parquet;
mod sql_type;
pub mod store;

pub use manifest::ManifestEntry;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::parquet::{
    ParquetTable, ParquetTableError, ScanEqualityPredicate, materialize, row_group_eliminated,
    row_group_filter_from, scan_order_from, table_input_with_filter_and_eq_predicates,
};
use crate::store::{DataFileLocation, MemoryStore, ObjectStore};
use arrow_array::{ArrayRef, Scalar};
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
    #[error("path `{0}` does not exist")]
    PathNotFound(String),
    #[error("path `{0}` is not a directory")]
    PathNotDirectory(String),
    #[error(
        "table path `{0}` must be a plain path, not a URL — a table's storage is the database's, so the path carries no scheme"
    )]
    TablePathWithScheme(String),
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
/// `CREATE TABLE` compiles to a single dataflow that reads every data file's
/// footer **once** (in parallel over the worker pool) and, at its terminal stage,
/// commits the assembled [`ParquetTable`] (row groups + open file handles) into
/// this shared map — which every later query then reuses.
#[derive(Debug)]
pub struct ParquetCatalog {
    tables: Arc<RwLock<HashMap<String, ParquetCatalogTable>>>,
    /// The database's object store: an in-memory [`MemoryStore`] by default
    /// (ephemeral), or a local directory / S3 / GCS for one opened with
    /// [`open`](Self::open). It holds the table [`manifest`] and the data of
    /// tables that live under the database root; `WITH (path = …)` tables read
    /// their own local directory directly.
    store: Arc<dyn ObjectStore>,
}

impl Default for ParquetCatalog {
    fn default() -> Self {
        Self {
            tables: Arc::new(RwLock::new(HashMap::new())),
            store: Arc::new(MemoryStore::new()),
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

    /// Open a persisted database rooted at `uri` — a local directory (or
    /// `file://…`), or a remote `s3://…`/`gs://…` object store — reloading every
    /// table the manifest records. Each is re-materialized from its data
    /// location, so a restart restores the same catalog.
    pub fn open(uri: &str, dispatcher: &DataFlowDispatcher) -> Result<Self> {
        let catalog = Self {
            tables: Arc::new(RwLock::new(HashMap::new())),
            store: Arc::from(store::open_store(uri)?),
        };
        for entry in manifest::load(&*catalog.store)? {
            // Reload on the coordinator (open runs there, before serving): take
            // the value path and register it directly — the table is already in
            // the manifest, so there's nothing to commit, just to materialize.
            let files = catalog.data_files(&entry.location)?;
            let parquet = Arc::new(ParquetTable::from_locations(dispatcher, files)?);
            let (name, table) = registered_table(entry, parquet);
            catalog.tables.write().unwrap().insert(name, table);
        }
        Ok(catalog)
    }

    /// Compile a `CREATE TABLE` to the dataflow that runs it: read every Parquet
    /// footer under the table's location in parallel and, at the final stage,
    /// record the assembled table in the manifest and the catalog map. Fetch and
    /// write are one spec — the caller executes it; nothing happens here but the
    /// (cheap, read-only) directory listing.
    ///
    /// The data lives at a directory/prefix: an explicit `WITH (path = '…')` —
    /// always a plain path, never an object-store URL — or `<name>` under the
    /// database root when none is given. An empty/absent location yields an
    /// empty table (registered with no row groups).
    fn create(
        &self,
        request: CreateTableRequest,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec> {
        if request.if_not_exists {
            return Err(Error::IfNotExistsUnsupported);
        }
        // Reject a duplicate up front; the commit re-checks under the lock as a
        // race backstop.
        if self.tables.read().unwrap().contains_key(&request.name) {
            return Err(Error::TableExists(request.name));
        }

        let location = self.locate(&request)?;
        let files = self.data_files(&location)?;
        let entry = ManifestEntry {
            name: request.name,
            columns: request.columns,
            location,
        };

        // The commit runs on the dataflow's last worker once the table is
        // assembled: record it durably in the manifest, then in the in-memory map
        // (re-checking the name under the lock as a race backstop).
        let tables = self.tables.clone();
        let store = self.store.clone();
        Ok(crate::parquet::create_load_and_commit_spec(
            dispatcher,
            &files,
            move |parquet| {
                let mut map = tables.write().unwrap();
                if map.contains_key(&entry.name) {
                    return Err(Box::new(Error::TableExists(entry.name))
                        as Box<dyn std::error::Error + Send + Sync>);
                }
                manifest::insert(&*store, &entry)?;
                let (name, table) = registered_table(entry, parquet);
                map.insert(name, table);
                Ok(())
            },
        ))
    }

    /// Register one newly-written local Parquet data file with table `name`,
    /// making its rows visible to every query that binds the table from now on
    /// (in-flight bindings keep the snapshot they cloned). This is how ingest
    /// publishes each file it flushes without waiting for a re-`CREATE`.
    ///
    /// The file's footer is read over the dispatch worker pool *before* the
    /// catalog lock is taken (so it must run on a coordinator thread, never
    /// inside `run_on_worker`); only the in-memory append happens under the
    /// write lock. Registering a path the table already holds is a no-op, so a
    /// replayed notification can't double-count rows. The manifest is not
    /// touched: it records the table's *location*, and a reload re-lists that
    /// directory, so the new file's durability comes from the directory itself.
    ///
    /// The file must live in the table's data directory — and only a table over
    /// an absolute local directory can accept one (a store-relative or remote
    /// location is not where local files land); anything else is reported as
    /// [`RegisterOutcome::LocationMismatch`] and the catalog is untouched.
    pub fn register_data_file(
        &self,
        dispatcher: &DataFlowDispatcher,
        name: &str,
        path: &Path,
    ) -> Result<RegisterOutcome> {
        let loaded = ParquetTable::from_files(dispatcher, &[path])?;

        let mut map = self.tables.write().unwrap();
        let Some(table) = map.get_mut(name) else {
            return Ok(RegisterOutcome::NoSuchTable);
        };
        if !file_in_table_dir(path, &table.location) {
            return Ok(RegisterOutcome::LocationMismatch);
        }
        if table
            .parquet
            .row_groups()
            .iter()
            .any(|rg| rg.source.local_path() == Some(path))
        {
            return Ok(RegisterOutcome::AlreadyRegistered);
        }
        let mut combined = (*table.parquet).clone();
        let offset = combined.row_groups().len();
        combined
            .row_groups_mut()
            .extend(loaded.row_groups().iter().enumerate().map(|(i, rg)| {
                let mut rg = (**rg).clone();
                rg.global_row_group_idx = offset + i;
                Arc::new(rg)
            }));
        table.parquet = Arc::new(combined);
        Ok(RegisterOutcome::Registered)
    }

    /// Swap a set of table `name`'s local data files for another, atomically as
    /// far as the catalog is concerned: drop the row groups of every file in
    /// `removed`, append those of the (freshly-read) `added` files, and
    /// renumber the global row-group indices. This is compaction's commit —
    /// several small files become one merged file in a single step, so no
    /// binding ever sees the rows doubled or missing. Queries already bound
    /// keep their old snapshot; their open fds keep even an unlinked file's
    /// bytes readable until they finish.
    ///
    /// Footers of `added` are read over the dispatch pool before the lock (same
    /// coordinator requirement as
    /// [`register_data_file`](Self::register_data_file)). Returns `false` when
    /// no table named `name` exists.
    pub fn replace_data_files(
        &self,
        dispatcher: &DataFlowDispatcher,
        name: &str,
        removed: &[PathBuf],
        added: &[PathBuf],
    ) -> Result<bool> {
        let loaded = ParquetTable::from_files(dispatcher, added)?;
        let removed: std::collections::HashSet<&Path> =
            removed.iter().map(PathBuf::as_path).collect();

        let mut map = self.tables.write().unwrap();
        let Some(table) = map.get_mut(name) else {
            return Ok(false);
        };
        let row_groups = table
            .parquet
            .row_groups()
            .iter()
            .filter(|rg| !rg.source.local_path().is_some_and(|p| removed.contains(p)))
            .cloned()
            .chain(loaded.row_groups().iter().cloned())
            .enumerate()
            .map(|(i, rg)| {
                if rg.global_row_group_idx == i {
                    rg
                } else {
                    let mut renumbered = (*rg).clone();
                    renumbered.global_row_group_idx = i;
                    Arc::new(renumbered)
                }
            })
            .collect();
        table.parquet = Arc::new(ParquetTable::new(row_groups));
        Ok(true)
    }

    /// The location stored for a new table: an explicit `path` (kept as given),
    /// or the table name under the database root.
    ///
    /// A table path is always a plain path, never a URL (no scheme) — where it
    /// physically lives is the database's storage, not the path's. An *absolute*
    /// path names an external directory on the server's local filesystem
    /// (whatever the database's store) and must already exist; a relative one
    /// lives in the store under the database root.
    fn locate(&self, request: &CreateTableRequest) -> Result<String> {
        let Some(path) = request.options.get(PATH_OPTION) else {
            return Ok(request.name.clone());
        };
        if is_url(path) {
            return Err(Error::TablePathWithScheme(path.clone()));
        }
        let dir = Path::new(path);
        if dir.is_absolute() {
            if !dir.exists() {
                return Err(Error::PathNotFound(path.clone()));
            }
            if !dir.is_dir() {
                return Err(Error::PathNotDirectory(path.clone()));
            }
        }
        Ok(path.clone())
    }

    /// The Parquet data files at `location`, ready for the metadata fetch. An
    /// absolute path is an external local directory, read from the filesystem; a
    /// relative location lives under the database root, in the store (which yields
    /// a local path or presigned remote per object). An empty/absent location
    /// yields no files.
    fn data_files(&self, location: &str) -> Result<Vec<DataFileLocation>> {
        if Path::new(location).is_absolute() {
            return Ok(store::local_parquet_files(Path::new(location))?);
        }
        Ok(self
            .store
            .list(location)?
            .into_iter()
            .filter(|object| object.key.ends_with(".parquet"))
            .map(|object| self.store.data_file(&object.key, object.size))
            .collect::<crate::store::Result<Vec<_>>>()?)
    }
}

/// What [`ParquetCatalog::register_data_file`] did with the file. The non-
/// `Registered` outcomes are not errors — a file can legitimately land before
/// its table is created — but the caller (ingest) wants to log them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegisterOutcome {
    /// Appended to the table; new binds see its rows.
    Registered,
    /// The table already holds this path (a replayed notification); no change.
    AlreadyRegistered,
    /// No table by that name exists yet; the file is picked up by a later
    /// `CREATE TABLE` or restart instead.
    NoSuchTable,
    /// The table exists but its data does not live in this file's directory
    /// (or is not an absolute local directory at all); refused.
    LocationMismatch,
}

/// Whether `file` sits directly in the table data directory `location`. Only an
/// absolute local location qualifies — a relative one lives inside the
/// database's object store, which is not where a locally-written file is.
/// Both sides are canonicalized so spelling differences don't refuse a
/// legitimate registration.
fn file_in_table_dir(file: &Path, location: &str) -> bool {
    let dir = Path::new(location);
    if !dir.is_absolute() {
        return false;
    }
    match (std::fs::canonicalize(dir), file.parent()) {
        (Ok(dir), Some(parent)) => std::fs::canonicalize(parent).is_ok_and(|p| p == dir),
        _ => false,
    }
}

/// The map entry for a registered table: its name, paired with the catalog table
/// built from its manifest entry and freshly-materialized row groups.
fn registered_table(
    entry: ManifestEntry,
    parquet: Arc<ParquetTable>,
) -> (String, ParquetCatalogTable) {
    (
        entry.name,
        ParquetCatalogTable {
            columns: entry.columns,
            location: entry.location,
            parquet,
            predicates: Vec::new(),
        },
    )
}

/// Whether `s` carries a URL scheme (`s3://`, `gs://`, `file://`, …). A table
/// `path` never may — it must be a bare path within the database.
fn is_url(s: &str) -> bool {
    s.contains("://")
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
    /// Where this table's Parquet data lives (a directory/key prefix in the
    /// database store). Kept so a future `refresh()` can re-read the latest files.
    pub location: String,
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
        emit_row_group_metadata: bool,
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
            emit_row_group_metadata,
            row_group_filter_from(dynamic_filters),
            scan_order,
            Arc::new(eq_predicates),
        ))
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
        // Late materialization re-reads rows by their *global* row-group index,
        // so it uses the full table, not the pruned scan view.
        materialize(input, self.parquet.clone(), projection)
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
