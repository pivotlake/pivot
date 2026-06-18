//! The catalog's durable metadata, as JSON in the database's [`ObjectStore`].
//!
//! Two documents, both folded out of what used to be a separate "table log":
//!
//! - the **database manifest** ([`CatalogManifest`], at [`MANIFEST_KEY`]) — the
//!   index of which tables exist and where each one's Parquet data lives. Written
//!   with a plain `put` (the catalog control plane is single-writer).
//! - a per-table **table manifest** ([`TableManifest`], under
//!   [`TABLE_MANIFEST_DIR`]) — the table's declared schema plus its committed
//!   file list. It is **versioned**: each commit writes a new
//!   `_pivot_tables/<name>/<version>.json` with `put_if_absent`, so a commit is a
//!   compare-and-swap (it fails if that version already exists) and the latest
//!   version a `list` finds is the table's current state. That is what lets two
//!   processes register/replace files concurrently — and what a stale in-memory
//!   copy reconciles against via `refresh`.

use planner::catalog::Column;
use serde::{Deserialize, Serialize};

use crate::FileRef;
use crate::store::{ObjectPath, ObjectStore};

/// Key of the [`CatalogManifest`] document within the database's object store.
const MANIFEST_KEY: &str = "_pivot_manifest.json";
/// Directory the per-table [`TableManifest`] versions live under
/// (`_pivot_tables/<name>/<version>.json`). Kept under the database root so a
/// table over an external data directory never has catalog metadata written
/// into it.
const TABLE_MANIFEST_DIR: &str = "_pivot_tables";
/// Version numbers are zero-padded to this width so a `list` returns them in
/// numeric order and the lexicographic max is the latest.
const VERSION_DIGITS: usize = 20;
/// The version a table's first commit gets.
pub const FIRST_VERSION: u64 = 1;
/// How many of the most-recent manifest versions to keep when pruning. A query
/// resolves a snapshot (one version) and reads its files for the whole query;
/// the snapshot's files are only physically deleted once *its* version is pruned
/// (deferred deletions, see [`PendingDeletions`]). So the retained tail must
/// outlast a query: while it runs, compaction must not advance far enough to
/// prune the version it's reading. During catch-up the compacter commits a few
/// versions per second, so a handful of retained versions (the old value) let a
/// 1 s query's snapshot be pruned out from under it — a `404` mid-scan. This
/// tail keeps a snapshot alive for ~a minute at that rate, well past any query,
/// while staying cheap: a few hundred tiny JSON manifests, and a slightly longer
/// `list` per bind.
const VERSIONS_RETAINED: u64 = 128;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] crate::store::StoreError),
    #[error("manifest json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("table `{0}` is in the catalog manifest but its table manifest is missing")]
    MissingTableManifest(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The sort key's range within one file: the `sort_by` columns at the file's
/// first and last row (the file is sorted, so these bound every row). Each is a
/// one-row arrow-json object, e.g. `{"Timestamp": 100}`. Lets a reader prune a
/// file on a sort-key range predicate without fetching its footer.
#[derive(Clone, Serialize, Deserialize)]
pub struct SortBounds {
    pub min: serde_json::Value,
    pub max: serde_json::Value,
}

/// One file in a table manifest: its store identity ([`FileRef`]) plus the
/// optional partition tuple and sort-key bounds the partitioning/sorting ingest
/// sink stamps on it. Both are `None` for files written without that metadata
/// (an unpartitioned/unsorted table, or compaction output today).
#[derive(Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    #[serde(flatten)]
    pub file: FileRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partition: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
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
}

/// The data files a compaction swap removed at a given table-manifest version,
/// recorded so the objects are deleted *lazily* — only when that version is
/// pruned, by which point the retention tail guarantees no live reader still
/// references them. This is how lakehouses avoid a reader hitting a file deleted
/// out from under it (Iceberg snapshot expiry, Delta `VACUUM`): a commit only
/// stops referencing the old files; a later GC removes them.
#[derive(Default, Serialize, Deserialize)]
pub struct PendingDeletions {
    /// Already-`location`-resolved store keys of the files to delete.
    pub paths: Vec<ObjectPath>,
}

/// One table's durable record: its declared schema (`columns`), its partition and
/// sort specs (column names; either may be empty), and its committed file list
/// (`entries`) at `version`. Mirrors the live [`CatalogTable`] state the catalog
/// keeps in memory.
///
/// [`CatalogTable`]: crate::catalog::CatalogTable
#[derive(Clone, Serialize, Deserialize)]
pub struct TableManifest {
    pub version: u64,
    pub columns: Vec<Column>,
    /// Partition columns, in order (identity partitioning); empty = unpartitioned.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub partition_by: Vec<String>,
    /// Sort columns, in order; empty = unsorted.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sort_by: Vec<String>,
    pub entries: Vec<ManifestEntry>,
}

impl TableManifest {
    /// The directory holding table `name`'s manifest versions.
    fn dir(name: &str) -> ObjectPath {
        ObjectPath::new(format!("{TABLE_MANIFEST_DIR}/{name}"))
    }

    /// The store key of one `version` of table `name`'s manifest.
    fn key(name: &str, version: u64) -> ObjectPath {
        Self::dir(name).join(&format!("{version:0VERSION_DIGITS$}.json"))
    }

    /// The store key of the deferred-deletions record for `version` of table
    /// `name`. Named so it never parses as a version number (the `load`/`prune`
    /// listings ignore it).
    fn deletions_key(name: &str, version: u64) -> ObjectPath {
        Self::dir(name).join(&format!("deletions-{version:0VERSION_DIGITS$}.json"))
    }

    /// Record the data files a swap at `version` removed, for deletion when that
    /// version is later pruned. A no-op for an empty list.
    pub fn record_deletions(
        store: &dyn ObjectStore,
        name: &str,
        version: u64,
        paths: &[ObjectPath],
    ) -> Result<()> {
        if paths.is_empty() {
            return Ok(());
        }
        let doc = PendingDeletions {
            paths: paths.to_vec(),
        };
        store.put(&Self::deletions_key(name, version), &serde_json::to_vec(&doc)?)?;
        Ok(())
    }

    /// Load table `name`'s latest committed manifest — one `list` of its version
    /// directory for the highest version, then a `get` of it. The database
    /// manifest is the index of which tables exist, so a name listed there with
    /// no manifest is a corrupt catalog — an error, not an absent table.
    pub fn load(store: &dyn ObjectStore, name: &str) -> Result<Self> {
        let version = store
            .list(&Self::dir(name))?
            .iter()
            .filter_map(|file| {
                file.path
                    .name()
                    .strip_suffix(".json")
                    .and_then(|v| v.parse::<u64>().ok())
            })
            .max()
            .ok_or_else(|| Error::MissingTableManifest(name.to_string()))?;
        let bytes = store
            .get(&Self::key(name, version))?
            .ok_or_else(|| Error::MissingTableManifest(name.to_string()))?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Commit this manifest at its `version` via compare-and-swap: `Ok(true)` if
    /// this writer created the version, `Ok(false)` if that version already
    /// exists (a concurrent writer won — the caller should reload and retry on
    /// top of the new latest).
    pub fn commit(&self, store: &dyn ObjectStore, name: &str) -> Result<bool> {
        Ok(store.put_if_absent(&Self::key(name, self.version), &serde_json::to_vec(self)?)?)
    }

    /// GC of superseded versions: delete every committed version of table
    /// `name` older than the [`VERSIONS_RETAINED`] most recent. Driven by the
    /// compacter's background sweep, not by commit — and safe to run
    /// concurrently with commits and other sweeps, since a double delete of the
    /// same key is harmless.
    pub fn prune_old_versions(store: &dyn ObjectStore, name: &str, latest: u64) -> Result<()> {
        let cutoff = latest.saturating_sub(VERSIONS_RETAINED);
        if cutoff == 0 {
            return Ok(());
        }
        let stale: Vec<u64> = store
            .list(&Self::dir(name))?
            .iter()
            .filter_map(|f| {
                f.path
                    .name()
                    .strip_suffix(".json")
                    .and_then(|v| v.parse::<u64>().ok())
            })
            .filter(|&v| v < cutoff)
            .collect();
        for v in stale {
            // NOTE: we intentionally do NOT physically delete this version's
            // swapped-out data files here. Under heavy compaction the version
            // *listing* (eventually-consistent) can leave a query's resolved
            // snapshot lagging far behind the prune cutoff; deleting its files
            // 404s the in-flight scan. Leaving the files as orphans lets that scan
            // read a valid (if slightly stale) snapshot instead — correct snapshot
            // isolation. Orphans are reclaimed by a separate consistency-aware GC
            // (compare live objects against every retained manifest), not here.
            // We still drop the deletions record and the version manifest so the
            // metadata directory stays small and `list` stays cheap.
            store.delete(&Self::deletions_key(name, v))?;
            store.delete(&Self::key(name, v))?;
        }
        Ok(())
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::LocalStore;
    use tempfile::TempDir;

    fn manifest(version: u64) -> TableManifest {
        TableManifest {
            version,
            columns: vec![],
            partition_by: vec![],
            sort_by: vec![],
            entries: vec![],
        }
    }

    fn live_versions(store: &dyn ObjectStore, name: &str) -> Vec<u64> {
        store
            .list(&TableManifest::dir(name))
            .unwrap()
            .iter()
            .filter_map(|f| {
                f.path
                    .name()
                    .strip_suffix(".json")
                    .and_then(|v| v.parse().ok())
            })
            .collect()
    }

    /// Pruning keeps only the most-recent [`VERSIONS_RETAINED`] versions (plus
    /// the latest itself), deletes everything older, and leaves the table still
    /// loadable at its latest version.
    #[test]
    fn prune_keeps_only_the_recent_tail() {
        let dir = TempDir::new().unwrap();
        let store = LocalStore::new(dir.path());
        let latest = VERSIONS_RETAINED + 22;
        for v in FIRST_VERSION..=latest {
            assert!(manifest(v).commit(&store, "t").unwrap());
        }

        TableManifest::prune_old_versions(&store, "t", latest).unwrap();

        let remaining = live_versions(&store, "t");
        let cutoff = latest - VERSIONS_RETAINED;
        assert!(
            remaining.iter().all(|&v| v >= cutoff),
            "no version below the cutoff survives"
        );
        assert_eq!(remaining.len() as u64, VERSIONS_RETAINED + 1);
        assert_eq!(TableManifest::load(&store, "t").unwrap().version, latest);
    }

    /// With fewer versions than the retained tail, pruning deletes nothing.
    #[test]
    fn prune_below_retention_is_a_noop() {
        let dir = TempDir::new().unwrap();
        let store = LocalStore::new(dir.path());
        for v in FIRST_VERSION..=3 {
            manifest(v).commit(&store, "t").unwrap();
        }

        TableManifest::prune_old_versions(&store, "t", 3).unwrap();

        assert_eq!(live_versions(&store, "t").len(), 3);
    }

    /// Pruning the same range twice (concurrent compacters, or a retry) is
    /// harmless — a delete of an already-gone version is not an error.
    #[test]
    fn prune_is_idempotent() {
        let dir = TempDir::new().unwrap();
        let store = LocalStore::new(dir.path());
        let latest = VERSIONS_RETAINED + 12;
        for v in FIRST_VERSION..=latest {
            manifest(v).commit(&store, "t").unwrap();
        }

        TableManifest::prune_old_versions(&store, "t", latest).unwrap();
        TableManifest::prune_old_versions(&store, "t", latest).unwrap();

        assert_eq!(
            live_versions(&store, "t").len() as u64,
            VERSIONS_RETAINED + 1
        );
    }

    /// Pruning a version drops its manifest and its deletions record, but does
    /// NOT physically delete the swapped-out data files — they're left as orphans
    /// so a query snapshot lagging behind the cutoff reads valid files instead of
    /// a 404. (A separate consistency-aware GC reclaims true orphans.)
    #[test]
    fn pruning_drops_metadata_but_keeps_data_files() {
        let dir = TempDir::new().unwrap();
        let store = LocalStore::new(dir.path());
        let latest = VERSIONS_RETAINED + 22;
        for v in FIRST_VERSION..=latest {
            assert!(manifest(v).commit(&store, "t").unwrap());
        }
        let data = ObjectPath::new("data/old.parquet");
        store.put(&data, b"rows").unwrap();
        TableManifest::record_deletions(&store, "t", 5, std::slice::from_ref(&data)).unwrap();

        TableManifest::prune_old_versions(&store, "t", latest).unwrap();

        assert!(
            store.get(&data).unwrap().is_some(),
            "the swapped-out data file is kept (orphaned), not deleted"
        );
        assert!(
            store
                .get(&TableManifest::deletions_key("t", 5))
                .unwrap()
                .is_none(),
            "the deletions record is pruned"
        );
        assert!(
            store.get(&TableManifest::key("t", 5)).unwrap().is_none(),
            "the version manifest is pruned"
        );
    }
}
