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

use crate::store::{ObjectStore, StoreError};
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

/// One data file in a version: its name within the table's data location, and
/// its size in bytes — carried so the reader can locate the Parquet footer
/// without a `stat`/HEAD per file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoggedFile {
    pub name: String,
    pub size: u64,
}

/// A table's complete file list at one version. The highest committed version
/// is the table's current state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableVersion {
    pub version: u64,
    pub files: Vec<LoggedFile>,
}

impl TableVersion {
    /// The version a change on top of this state must commit as. On `None`
    /// (no log yet) start at 1.
    pub fn next_after(current: Option<&TableVersion>) -> u64 {
        current.map_or(1, |v| v.version + 1)
    }
}

/// Key of one version document.
fn version_key(table: &str, version: u64) -> String {
    format!(
        "{LOG_DIR}/{table}/{version:0width$}.json",
        width = VERSION_DIGITS
    )
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

/// The highest committed version number of `table`'s log, from one LIST.
/// `None` for a table with no log (yet).
pub fn latest_version(store: &dyn ObjectStore, table: &str) -> Result<Option<u64>> {
    let dir = format!("{LOG_DIR}/{table}");
    let objects = store.list(&dir)?;
    Ok(objects
        .iter()
        .filter_map(|object| parse_version(&object.key))
        .max())
}

/// Read one committed version of `table`'s log.
pub fn read(store: &dyn ObjectStore, table: &str, version: u64) -> Result<TableVersion> {
    let key = version_key(table, version);
    let Some(bytes) = store.get(&key)? else {
        return Err(TableLogError::Missing {
            table: table.to_string(),
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

/// The current (highest) version of `table`'s log, or `None` if the table has
/// no log. One LIST plus one GET.
pub fn latest(store: &dyn ObjectStore, table: &str) -> Result<Option<TableVersion>> {
    match latest_version(store, table)? {
        Some(version) => Ok(Some(read(store, table, version)?)),
        None => Ok(None),
    }
}

/// Atomically commit `files` as version `version` of `table`. `Ok(true)` means
/// this writer won; `Ok(false)` means someone else committed that version first
/// — re-read [`latest`] and retry the change on top of it.
pub fn commit(
    store: &dyn ObjectStore,
    table: &str,
    version: u64,
    files: &[LoggedFile],
) -> Result<bool> {
    let doc = VersionDoc {
        version,
        files: files.to_vec(),
    };
    let bytes = serde_json::to_vec_pretty(&doc).map_err(|e| TableLogError::Parse {
        key: version_key(table, version),
        reason: format!("serializing: {e}"),
    })?;
    Ok(store.put_if_absent(&version_key(table, version), &bytes)?)
}

/// Serialized form of one version document. Carries its version number too, so
/// the document is self-describing independent of its key.
#[derive(Serialize, Deserialize)]
struct VersionDoc {
    version: u64,
    files: Vec<LoggedFile>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::MemoryStore;

    fn file(name: &str, size: u64) -> LoggedFile {
        LoggedFile {
            name: name.into(),
            size,
        }
    }

    #[test]
    fn versions_round_trip_and_latest_wins() {
        let store = MemoryStore::new();
        assert_eq!(latest(&store, "t").unwrap(), None);

        assert!(commit(&store, "t", 1, &[file("a.parquet", 10)]).unwrap());
        assert!(
            commit(
                &store,
                "t",
                2,
                &[file("a.parquet", 10), file("b.parquet", 20)]
            )
            .unwrap()
        );

        let current = latest(&store, "t").unwrap().unwrap();
        assert_eq!(current.version, 2);
        assert_eq!(
            current.files,
            vec![file("a.parquet", 10), file("b.parquet", 20)]
        );
        // Older versions stay readable (immutable history).
        assert_eq!(
            read(&store, "t", 1).unwrap().files,
            vec![file("a.parquet", 10)]
        );
    }

    #[test]
    fn conflicting_commit_loses_and_state_is_the_winners() {
        let store = MemoryStore::new();
        assert!(commit(&store, "t", 1, &[file("a.parquet", 1)]).unwrap());
        // A racing writer targeting the same version loses cleanly.
        assert!(!commit(&store, "t", 1, &[file("b.parquet", 2)]).unwrap());
        assert_eq!(
            latest(&store, "t").unwrap().unwrap().files,
            vec![file("a.parquet", 1)]
        );
    }

    #[test]
    fn logs_are_per_table_and_foreign_keys_are_ignored() {
        let store = MemoryStore::new();
        assert!(commit(&store, "a", 1, &[file("x.parquet", 1)]).unwrap());
        assert_eq!(latest(&store, "b").unwrap(), None);

        // A stray object in the log directory is not a version.
        store.put("_goose_logs/a/garbage.tmp", b"junk").unwrap();
        assert_eq!(latest_version(&store, "a").unwrap(), Some(1));
    }

    #[test]
    fn next_after_starts_at_one() {
        assert_eq!(TableVersion::next_after(None), 1);
        let v = TableVersion {
            version: 7,
            files: vec![],
        };
        assert_eq!(TableVersion::next_after(Some(&v)), 8);
    }
}
