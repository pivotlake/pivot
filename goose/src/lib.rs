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
pub mod metadata;
pub mod parquet;
pub mod store;

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::Path;
use std::sync::{Arc, RwLock};

use crate::parquet::{
    ParquetSource, ParquetTable, ParquetTableError, ScanEqualityPredicate, row_group_eliminated,
    row_group_filter_from, scan_order_from, table_input_with_filter_and_eq_predicates,
};
use arrow_array::{ArrayRef, Scalar};
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
    #[error(
        "CREATE TABLE needs a `{PATH_OPTION}` (local directory) or `{URL_OPTION}` (goose catalog) option"
    )]
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
    #[error("table mixes local and remote data files, which is not supported")]
    MixedDataFiles,
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
        let source = ParquetSource::from_directory(path_buf)?;

        let table = ParquetCatalogTable {
            columns: request.columns,
            source,
            predicates: Vec::new(),
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
    /// Data files are resolved by location: local ones are read from disk;
    /// remote (`s3://`/`gs://`) ones are presigned and range-read over the ring.
    /// A table's files share the catalog root's store, so they're all local or
    /// all remote — a mix is rejected.
    ///
    /// Runs on a dispatch worker (it calls [`ParquetTable::from_files`] /
    /// [`ParquetTable::from_remote_files`]).
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

        let source = match existing {
            Some(table) => {
                // Resolve each data file to a local path or a presigned remote
                // URL. A table's files share the root's store, so they're either
                // all local or all remote.
                let mut paths = Vec::new();
                let mut urls = Vec::new();
                for file in &table.files {
                    match lake::resolve_data_file(url, &file.location) {
                        lake::Resolved::Local(p) => paths.push(p),
                        lake::Resolved::Remote(uri) => urls.push(store::presign_get(&uri)?),
                    }
                }
                match (paths.is_empty(), urls.is_empty()) {
                    (_, true) => ParquetSource::from_files(&paths),
                    (true, false) => ParquetSource::from_remote_files(&urls),
                    (false, false) => {
                        return Err(Error::MixedDataFiles);
                    }
                }
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
                ParquetSource::from_files::<&Path>(&[])
            }
        };

        let table = ParquetCatalogTable {
            columns: request.columns,
            source,
            predicates: Vec::new(),
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
    /// The table's data files, by location. The row-group metadata is fetched
    /// (in parallel) when the scan is compiled, not held here.
    pub source: ParquetSource,
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

        // Fetch this table's row-group metadata (in parallel across workers),
        // then prune by the pushed-down predicates' stats.
        let parquet = Arc::new(self.pruned_parquet(dispatcher)?);
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
    /// Materialize this table's row-group metadata (reading every footer in
    /// parallel) and keep only the row groups that survive this binding's
    /// pushed-down predicates — i.e. what [`Table::compile`] actually scans. A
    /// min/max stat that proves no row in a group can match drops it; a
    /// stats-comparison error means "can't prune" (kept) — never wrong, just
    /// unoptimized. Exposed so resolution + pruning can be asserted directly.
    pub fn pruned_parquet(&self, dispatcher: &DataFlowDispatcher) -> Result<ParquetTable> {
        let mut parquet = self.source.materialize(dispatcher)?;
        parquet.row_groups_mut().retain(|rg| {
            !self.predicates.iter().any(|p| {
                row_group_eliminated(rg.as_ref(), p.column_idx, p.compare_type, &p.value)
                    .unwrap_or(false)
            })
        });
        Ok(parquet)
    }
}
