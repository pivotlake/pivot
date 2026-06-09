//! The catalog **control plane** over a generic [`ObjectStore`]: one table's
//! goose catalog root, with its `_goose_log/` snapshot history and the CAS
//! commit loop.
//!
//! A [`TableObjectStore`] wraps a [`Box<dyn ObjectStore>`] (local, S3, or GCS)
//! and adds the catalog operations the [`ParquetCatalog`](crate::ParquetCatalog)
//! needs: read the latest snapshot, atomically commit a new one, and resolve a
//! table's recorded data files into [`DataFileLocation`]s the metadata fetch can
//! materialize. Holding one of these per table is what lets a future `refresh()`
//! re-read the latest snapshot and rebuild the [`ParquetTable`](crate::parquet::ParquetTable).

use crate::lake;
use crate::metadata::{CatalogSnapshot, FORMAT_VERSION, Table};
use crate::parquet::DataFileLocation;
use crate::store::{ObjectStore, PutOutcome, Result, StoreError, open_store, presign_get};

/// Directory (key prefix) under the catalog root holding the versioned snapshot
/// files: `_goose_log/<zero-padded-version>.json`.
const LOG_DIR: &str = "_goose_log";
/// Zero-pad versions to a fixed width so lexical key order matches numeric order.
const VERSION_WIDTH: usize = 20;
/// Optimistic-retry ceiling for a contended commit.
const MAX_COMMIT_RETRIES: u32 = 10_000;

/// One table's goose catalog: a generic object store plus the snapshot history
/// and CAS commit loop rooted at `root`.
#[derive(Debug)]
pub struct TableObjectStore {
    store: Box<dyn ObjectStore>,
    /// The catalog root URI; used to resolve relative data-file locations.
    root: String,
}

impl TableObjectStore {
    /// Open the catalog at `uri` (`s3://…`, `gs://…`, or a local path).
    pub fn open(uri: &str) -> Result<Self> {
        Ok(Self {
            store: open_store(uri)?,
            root: uri.to_string(),
        })
    }

    /// The catalog root URI.
    pub fn root(&self) -> &str {
        &self.root
    }

    /// Load the latest committed snapshot: LIST `_goose_log/`, take the highest
    /// version, GET and parse it. Returns the empty snapshot if the log has no
    /// commits yet.
    pub fn latest_snapshot(&self) -> Result<CatalogSnapshot> {
        let keys = self.store.list(LOG_DIR)?;
        let latest = keys
            .iter()
            .filter_map(|k| parse_version(k).map(|v| (v, k)))
            .max_by_key(|(v, _)| *v)
            .map(|(_, k)| k.clone());

        let Some(key) = latest else {
            return Ok(CatalogSnapshot::empty());
        };
        let bytes = self.store.get(&key)?.ok_or_else(|| {
            StoreError::Http(format!("snapshot {key} vanished between list and get"))
        })?;
        Ok(CatalogSnapshot::from_slice(&bytes)?)
    }

    /// Compare-and-swap commit: read the latest snapshot, apply `mutate` to a
    /// copy, then atomically create `_goose_log/<version+1>.json`. On a lost race
    /// (another writer committed that version first) it reloads and retries. This
    /// single create-if-absent is the only synchronization the catalog needs.
    ///
    /// `mutate` may return an error to abort the commit (no retry); a storage
    /// error on the create also aborts. Returns the committed snapshot.
    pub fn commit<F>(&self, mut mutate: F) -> Result<CatalogSnapshot>
    where
        F: FnMut(&mut CatalogSnapshot) -> Result<()>,
    {
        for _ in 0..MAX_COMMIT_RETRIES {
            let mut snap = self.latest_snapshot()?;
            let base = snap.version;
            mutate(&mut snap)?;
            snap.version = base + 1;
            snap.format_version = FORMAT_VERSION;
            let key = format!(
                "{LOG_DIR}/{:0width$}.json",
                snap.version,
                width = VERSION_WIDTH
            );
            match self.store.put_if_absent(&key, &snap.to_vec())? {
                PutOutcome::Created => return Ok(snap),
                PutOutcome::AlreadyExists => continue,
            }
        }
        Err(StoreError::TooMuchContention(MAX_COMMIT_RETRIES))
    }

    /// Resolve a snapshot table's recorded data files into the locations the
    /// metadata fetch reads: local files by path, remote ones by a presigned GET
    /// URL. The snapshot-recorded size travels with each, so locating the footer
    /// needs no `stat` (local) or HEAD/probe (remote). The list may mix local and
    /// remote entries; [`ParquetTable::from_locations`](crate::parquet::ParquetTable::from_locations)
    /// handles both.
    pub fn resolve_data_files(&self, table: &Table) -> Result<Vec<DataFileLocation>> {
        table
            .files
            .iter()
            .map(|file| {
                Ok(match lake::resolve_data_file(&self.root, &file.location) {
                    lake::Resolved::Local(path) => DataFileLocation::Local {
                        path,
                        size: file.size,
                    },
                    lake::Resolved::Remote(uri) => DataFileLocation::Remote {
                        url: presign_get(&uri)?,
                        size: file.size,
                    },
                })
            })
            .collect()
    }
}

/// Parse a snapshot version from a `_goose_log/<version>.json` key.
fn parse_version(key: &str) -> Option<i64> {
    key.rsplit('/')
        .next()?
        .strip_suffix(".json")?
        .parse::<i64>()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{Column, DataFile, Schema};

    fn snapshot(version: i64, table: &str) -> CatalogSnapshot {
        CatalogSnapshot {
            format_version: FORMAT_VERSION,
            version,
            schemas: vec![Schema {
                name: "main".into(),
                tables: vec![Table {
                    name: table.into(),
                    columns: vec![Column {
                        name: "id".into(),
                        type_sql: "INTEGER".into(),
                    }],
                    files: vec![DataFile {
                        location: "_goose_data/a.parquet".into(),
                        size: 123,
                        row_count: 1,
                    }],
                }],
            }],
        }
    }

    /// Write a snapshot at an explicit version, bypassing `commit`'s auto-numbering
    /// (so a test can commit out of order).
    fn write_snapshot(root: &str, version: i64, snap: &CatalogSnapshot) {
        let store = open_store(root).unwrap();
        let key = format!("{LOG_DIR}/{:0width$}.json", version, width = VERSION_WIDTH);
        assert_eq!(
            store.put_if_absent(&key, &snap.to_vec()).unwrap(),
            PutOutcome::Created
        );
    }

    #[test]
    fn latest_snapshot_picks_highest_version() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_str().unwrap();
        // Write out of order to prove sorting is by parsed version, not write order.
        write_snapshot(root, 2, &snapshot(2, "events"));
        write_snapshot(root, 10, &snapshot(10, "latest"));
        write_snapshot(root, 1, &snapshot(1, "first"));

        let snap = TableObjectStore::open(root)
            .unwrap()
            .latest_snapshot()
            .unwrap();
        assert_eq!(snap.version, 10);
        assert!(snap.table("main", "latest").is_some());
        assert!(snap.table("main", "events").is_none());
    }

    #[test]
    fn latest_snapshot_of_empty_log_is_version_zero() {
        let dir = tempfile::tempdir().unwrap();
        let store = TableObjectStore::open(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(store.latest_snapshot().unwrap().version, 0);
    }

    #[test]
    fn commit_increments_over_the_latest_version() {
        let dir = tempfile::tempdir().unwrap();
        let store = TableObjectStore::open(dir.path().to_str().unwrap()).unwrap();

        let first = store
            .commit(|snap| {
                snap.schemas.push(Schema {
                    name: "main".into(),
                    tables: Vec::new(),
                });
                Ok(())
            })
            .unwrap();
        assert_eq!(first.version, 1);
        assert_eq!(store.latest_snapshot().unwrap().version, 1);

        let second = store.commit(|_| Ok(())).unwrap();
        assert_eq!(second.version, 2);
    }
}
