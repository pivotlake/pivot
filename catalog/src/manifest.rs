//! The catalog's in-memory table state and the database index.
//!
//! One durable JSON document lives here: the **database manifest**
//! ([`CatalogManifest`], at [`MANIFEST_KEY`]): the index of which tables
//! exist and where each one's data lives. Written with a plain `put` (the
//! catalog control plane is single-writer).
//!
//! Everything else is the **in-memory** shape of one table's committed state
//! ([`TableState`] and its [`ManifestEntry`]s), extracted from the table's
//! durable record, its Delta Lake transaction log (see [`crate::delta`]),
//! and compared against by partition pruning.

use std::sync::Arc;

use planner::catalog::Column;
use serde::{Deserialize, Serialize};

use crate::FileRef;
use crate::parquet::RowGroupMetadata;
use crate::store::{ObjectPath, ObjectStore};

/// Key of the [`CatalogManifest`] document within the database's object store.
const MANIFEST_KEY: &str = "_pivot_manifest.json";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] crate::store::StoreError),
    #[error("manifest json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// One data file of a table: its identity ([`FileRef`]) paired with its
/// materialized row groups (file-local order).
#[derive(Clone)]
pub struct TableFile {
    pub(crate) file: FileRef,
    pub(crate) row_groups: Vec<Arc<RowGroupMetadata>>,
}

impl TableFile {
    /// Pair a file's identity with the row groups read from its footer. Built by
    /// the metadata fetcher (one per file) and the catalog's table assembly.
    pub(crate) fn new(file: FileRef, row_groups: Vec<Arc<RowGroupMetadata>>) -> Self {
        Self { file, row_groups }
    }

    /// This file's row groups, in file-local order.
    pub(crate) fn row_groups(&self) -> &[Arc<RowGroupMetadata>] {
        &self.row_groups
    }
}

/// The sort key's range within one file: the `sort_by` columns at the file's
/// first and last row (the file is sorted, so these bound every row). Each is a
/// one-row arrow-json object, e.g. `{"Timestamp": 100}`. Lets a reader prune a
/// file on a sort-key range predicate without fetching its footer. Persisted as
/// the `minValues`/`maxValues` of the file's delta stats.
#[derive(Clone, Serialize, Deserialize)]
pub struct SortBounds {
    pub min: serde_json::Value,
    pub max: serde_json::Value,
}

/// One committed data file: its store identity ([`FileRef`]) plus the optional
/// partition tuple and sort-key bounds a partitioning/sorting writer stamps on
/// it. Both are `None` for files written without that metadata (an
/// unpartitioned/unsorted table, or files discovered at CREATE TABLE).
/// Persisted as one `add` action in the table's delta log.
#[derive(Clone)]
pub struct ManifestEntry {
    pub file: FileRef,
    pub partition: Option<serde_json::Value>,
    pub sort_bounds: Option<SortBounds>,
}

impl ManifestEntry {
    /// An entry with no partition/sort metadata (unpartitioned + unsorted table,
    /// or a writer that doesn't record it).
    pub fn new(file: FileRef) -> Self {
        Self {
            file,
            partition: None,
            sort_bounds: None,
        }
    }

    /// Whether this file *can* hold a row matching every partition filter — a
    /// soft test, so it never wrongly drops a file. A filter on a non-partition
    /// column, an entry with no recorded tuple, or a tuple missing the column all
    /// keep the entry (absence of a value is not proof of a mismatch). Only a
    /// recorded partition value that differs from the filter's excludes it. Both
    /// values are the JSON arrow-json produced (the sink for the tuple, the query
    /// for the constant), so they compare directly.
    pub fn maybe_matches_partition(
        &self,
        partition_by: &[String],
        filters: &[PartitionEqFilter],
    ) -> bool {
        let Some(tuple) = self.partition.as_ref().and_then(|v| v.as_object()) else {
            return true;
        };
        filters.iter().all(|filter| {
            !partition_by.iter().any(|c| c == &filter.column)
                || match tuple.get(&filter.column) {
                    Some(value) => *value == filter.value,
                    None => true,
                }
        })
    }
}

/// A `partition column = constant` predicate the query pushed down, with the
/// constant already encoded to the JSON shape a partition tuple records (via
/// arrow-json, the same encoder the sink uses). [`ManifestEntry::maybe_matches_partition`]
/// uses it to skip a file whose recorded partition value can't match *before*
/// its footer is fetched — the HTTP a stats prune can't save, since stats live
/// in the footer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionEqFilter {
    pub column: String,
    pub value: serde_json::Value,
}

/// One table's committed state at one delta log version, as a plain
/// **in-memory value**: its declared schema (`columns`), its partition and
/// sort specs (column names; either may be empty), and its committed file
/// list (`entries`) at `version`. Extracted whole from the table's delta log
/// by [`crate::delta`]; a change swaps the whole value.
///
/// This exists instead of holding delta-rs's `DeltaTable` directly, for three
/// reasons:
///
/// - `DeltaTable` is a live async handle (an object-store client plus mutable
///   snapshot state), not a value. The catalog needs a cheaply-`Clone`,
///   immutable value: every transaction freezes the table set by cloning the
///   map, and every binding/pruning read happens lock-free on any thread with
///   zero I/O.
/// - It is the decode-once boundary. Queries need pivot-shaped data (exact
///   pivot [`Column`] types, partition tuples in the arrow-json shape pruning
///   compares against, sort bounds as JSON) while the log speaks delta shapes
///   (string field metadata, raw `partitionValues`, stats JSON). Decoding and
///   validating happen once per log version, not on every access.
/// - It keeps delta-rs contained in [`crate::delta`]: everything downstream
///   (snapshots, bindings, pruning, `metadata()`) is format-agnostic.
///
/// Mirrors the live [`CatalogTable`] state the catalog keeps in memory.
///
/// [`CatalogTable`]: crate::catalog::CatalogTable
#[derive(Clone)]
pub struct TableState {
    pub version: u64,
    pub columns: Vec<Column>,
    /// Partition columns, in order (identity partitioning); empty = unpartitioned.
    pub partition_by: Vec<String>,
    /// Sort columns, in order; empty = unsorted.
    pub sort_by: Vec<String>,
    pub entries: Vec<ManifestEntry>,
    /// Each committed entry's footer, materialized: the per-file row groups a
    /// scan reads. Derived data, not recorded in the log: [`crate::delta`]
    /// extracts a state with this empty, and the catalog attaches the footers
    /// (fetching only the ones a previous state didn't already hold) before
    /// the state is installed, so state and footers always change together.
    pub files: Vec<TableFile>,
}

/// One table's entry in the [`CatalogManifest`]: its name and the
/// [location](ObjectPath) its Parquet data lives at.
#[derive(Clone, Serialize, Deserialize)]
pub struct CatalogManifestTableEntry {
    pub(crate) name: String,
    pub(crate) location: ObjectPath,
}

impl CatalogManifestTableEntry {
    pub fn new(name: String, location: ObjectPath) -> Self {
        Self { name, location }
    }
}

/// The database manifest: the index of which tables exist and where their data
/// lives. A monotonically increasing `version` records how many times it has
/// changed.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct CatalogManifest {
    pub(crate) version: u64,
    pub(crate) tables: Vec<CatalogManifestTableEntry>,
}

impl CatalogManifest {
    /// Load the database manifest, or an empty one if the database is brand new.
    pub fn load(store: &dyn ObjectStore) -> Result<Self> {
        match store.get(&ObjectPath::new(MANIFEST_KEY))? {
            Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
            None => Ok(Self::default()),
        }
    }

    /// Write the database manifest, overwriting the previous version.
    pub fn store(&self, store: &dyn ObjectStore) -> Result<()> {
        store.put(&ObjectPath::new(MANIFEST_KEY), &serde_json::to_vec(self)?)?;
        Ok(())
    }

    /// Add (or replace) a table entry and bump the manifest version.
    pub(crate) fn upsert(&mut self, entry: CatalogManifestTableEntry) {
        self.tables.retain(|t| t.name != entry.name);
        self.tables.push(entry);
        self.version += 1;
    }
}
