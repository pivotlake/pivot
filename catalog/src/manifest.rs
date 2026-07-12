//! The catalog's in-memory table state and the database index.
//!
//! One durable JSON document lives here: the **database manifest**
//! ([`CatalogManifest`], at [`MANIFEST_KEY`]): the index of which tables
//! exist and where each one's data lives. Written with a plain `put` (the
//! catalog control plane is single-writer).
//!
//! Everything else is the **in-memory** shape of one table's committed state
//! ([`TableManifest`] and its [`ManifestEntry`]s), extracted from the table's
//! durable record, its Delta Lake transaction log (see [`crate::delta`]),
//! and compared against by partition pruning.

use planner::catalog::Column;
use serde::{Deserialize, Serialize};

use crate::FileRef;
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

/// One table's committed state at one delta log version: its declared schema
/// (`columns`), its partition and sort specs (column names; either may be
/// empty), and its committed file list (`entries`) at `version`. Extracted
/// whole from the table's delta log; a change swaps the whole value. Mirrors
/// the live [`CatalogTable`] state the catalog keeps in memory.
///
/// [`CatalogTable`]: crate::catalog::CatalogTable
#[derive(Clone)]
pub struct TableManifest {
    pub version: u64,
    pub columns: Vec<Column>,
    /// Partition columns, in order (identity partitioning); empty = unpartitioned.
    pub partition_by: Vec<String>,
    /// Sort columns, in order; empty = unsorted.
    pub sort_by: Vec<String>,
    pub entries: Vec<ManifestEntry>,
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
