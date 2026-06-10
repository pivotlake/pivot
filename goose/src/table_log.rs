//! The versioned **table log**: each table's durable, append-only record of
//! which Parquet data files it consists of.
//!
//! Layout in the database's [`ObjectStore`]: one directory per table,
//! `_goose_logs/<table>/`, holding one immutable JSON document per version,
//! named by a zero-padded version number (`…/00000000000000000007.json`). A
//! version is the table's **complete** file list at that point — name and size
//! of every Parquet file, relative to the table's data location — and the
//! highest version is the current one.
//!
//! Versions are immutable once written: a change (a flushed file, a
//! compaction's swap) is a **new** version, committed with the store's
//! compare-and-swap ([`ObjectStore::put_if_absent`]). Two writers racing to
//! commit version `n+1` resolve cleanly — exactly one wins; the loser re-reads
//! the new latest and retries on top of it. That makes the log the multi-writer
//! source of truth: a file is part of the table iff the current version lists
//! it, so a half-written data file (or a compaction's leftover input) is simply
//! invisible.
//!
//! The log holds file *lists*; table definitions (name, columns, location)
//! stay in the [`manifest`](crate::manifest).

use crate::store::{FileRef, ObjectStore, StoreError};
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum TableLogError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("malformed table log document `{key}`: {reason}")]
    Parse { key: String, reason: String },
    #[error("table log version {version} of `{table}` listed but unreadable")]
    Missing { table: String, version: u64 },
}

pub type Result<T> = std::result::Result<T, TableLogError>;

/// Root directory of all table logs within the database store.
const LOG_DIR: &str = "_goose_logs";
/// Version numbers are zero-padded to this width so lexicographic key order is
/// numeric order.
const VERSION_DIGITS: usize = 20;
/// The version a table's very first commit gets.
pub const FIRST_VERSION: u64 = 1;

/// A table's complete file list at one version. The highest committed version
/// is the table's current state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableVersion {
    pub version: u64,
    pub files: Vec<FileRef>,
}

impl TableVersion {
    /// The version a change on top of this state must commit as.
    pub fn next(&self) -> u64 {
        self.version + 1
    }
}

/// One table's log: every operation against `_goose_logs/<table>/` in one
/// store, bound once instead of threading `(store, table)` through each call.
pub struct TableLog<'a> {
    store: &'a dyn ObjectStore,
    table: &'a str,
}

impl<'a> TableLog<'a> {
    pub fn new(store: &'a dyn ObjectStore, table: &'a str) -> Self {
        Self { store, table }
    }

    /// The highest committed version *number*, from one LIST — the cheap
    /// staleness probe. `None` for a table with no log (yet). For the
    /// version's contents, [`read`](Self::read) it (or use
    /// [`read_latest`](Self::read_latest)).
    pub fn latest_version(&self) -> Result<Option<u64>> {
        let objects = self.store.list(&format!("{LOG_DIR}/{}", self.table))?;
        Ok(objects
            .iter()
            .filter_map(|object| parse_version(&object.name))
            .max())
    }

    /// Read one committed version's document.
    pub fn read(&self, version: u64) -> Result<TableVersion> {
        let key = self.version_key(version);
        let Some(bytes) = self.store.get(&key)? else {
            return Err(TableLogError::Missing {
                table: self.table.to_string(),
                version,
            });
        };
        let doc: VersionDoc = serde_json::from_slice(&bytes).map_err(|e| TableLogError::Parse {
            key,
            reason: e.to_string(),
        })?;
        Ok(TableVersion {
            version,
            files: doc.files,
        })
    }

    /// Read the current (highest) version's document, or `None` if the table
    /// has no log. One LIST plus one GET.
    pub fn read_latest(&self) -> Result<Option<TableVersion>> {
        match self.latest_version()? {
            Some(version) => Ok(Some(self.read(version)?)),
            None => Ok(None),
        }
    }

    /// Atomically commit `files` as version `version`. `Ok(true)` means this
    /// writer won; `Ok(false)` means someone else committed that version
    /// first — re-read the latest and retry the change on top of it.
    pub fn commit(&self, version: u64, files: &[FileRef]) -> Result<bool> {
        let doc = VersionDoc {
            version,
            files: files.to_vec(),
        };
        let bytes = serde_json::to_vec_pretty(&doc).map_err(|e| TableLogError::Parse {
            key: self.version_key(version),
            reason: format!("serializing: {e}"),
        })?;
        Ok(self
            .store
            .put_if_absent(&self.version_key(version), &bytes)?)
    }

    /// Key of one version document.
    fn version_key(&self, version: u64) -> String {
        format!(
            "{LOG_DIR}/{}/{version:0width$}.json",
            self.table,
            width = VERSION_DIGITS
        )
    }
}

/// Parse a listed key back into its version number. Foreign objects (temp
/// files, unrelated names) yield `None` and are skipped — the log only trusts
/// names it would itself write.
fn parse_version(key: &str) -> Option<u64> {
    let name = key.rsplit('/').next()?;
    let digits = name.strip_suffix(".json")?;
    if digits.len() != VERSION_DIGITS {
        return None;
    }
    digits.parse().ok()
}

/// Serialized form of one version document. Carries its version number too, so
/// the document is self-describing independent of its key.
#[derive(Serialize, Deserialize)]
struct VersionDoc {
    version: u64,
    files: Vec<FileRef>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::MemoryStore;

    fn file(name: &str, size: u64) -> FileRef {
        FileRef {
            name: name.into(),
            size,
        }
    }

    #[test]
    fn versions_round_trip_and_latest_wins() {
        let store = MemoryStore::new();
        let log = TableLog::new(&store, "t");
        assert_eq!(log.read_latest().unwrap(), None);

        assert!(log.commit(1, &[file("a.parquet", 10)]).unwrap());
        assert!(
            log.commit(2, &[file("a.parquet", 10), file("b.parquet", 20)])
                .unwrap()
        );

        let current = log.read_latest().unwrap().unwrap();
        assert_eq!(current.version, 2);
        assert_eq!(
            current.files,
            vec![file("a.parquet", 10), file("b.parquet", 20)]
        );
        assert_eq!(current.next(), 3);
        // Older versions stay readable (immutable history).
        assert_eq!(log.read(1).unwrap().files, vec![file("a.parquet", 10)]);
    }

    #[test]
    fn conflicting_commit_loses_and_state_is_the_winners() {
        let store = MemoryStore::new();
        let log = TableLog::new(&store, "t");
        assert!(log.commit(1, &[file("a.parquet", 1)]).unwrap());
        // A racing writer targeting the same version loses cleanly.
        assert!(!log.commit(1, &[file("b.parquet", 2)]).unwrap());
        assert_eq!(
            log.read_latest().unwrap().unwrap().files,
            vec![file("a.parquet", 1)]
        );
    }

    #[test]
    fn logs_are_per_table_and_foreign_keys_are_ignored() {
        let store = MemoryStore::new();
        assert!(
            TableLog::new(&store, "a")
                .commit(1, &[file("x.parquet", 1)])
                .unwrap()
        );
        assert_eq!(TableLog::new(&store, "b").read_latest().unwrap(), None);

        // A stray object in the log directory is not a version.
        store.put("_goose_logs/a/garbage.tmp", b"junk").unwrap();
        assert_eq!(
            TableLog::new(&store, "a").latest_version().unwrap(),
            Some(1)
        );
    }
}
