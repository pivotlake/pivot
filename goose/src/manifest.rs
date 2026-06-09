//! The **table manifest**: a database's record of which tables exist and where
//! each table's Parquet data lives.
//!
//! A database has one manifest. The default ([`InMemoryTableManifest`]) keeps
//! its tables only in memory, so they vanish when the server restarts — fine for
//! ad-hoc / test use. A persisted manifest ([`ObjectStoreManifest`]) writes a
//! small JSON document to an [`ObjectStore`] — a local directory *or* an S3/GCS
//! bucket — so restarting the server reloads the same tables.
//!
//! The manifest only tracks table *definitions* (name, declared schema, and the
//! directory holding the data). The Parquet row-group metadata itself is read
//! from those directories when the catalog loads — the manifest is the small,
//! durable index that points at them. The declared columns are kept here (not
//! re-derived from the Parquet on load) because they are the authoritative
//! logical schema — possibly a deliberate reinterpretation of the physical types
//! — and because an empty table has no Parquet to derive them from.
//!
//! A database has a single storage class fixed at startup: a *local* database's
//! table locations are always local filesystem paths (absolute, or relative to
//! the database root) — never an `s3://`/`gs://`/`file://` URL — and a remote
//! database's always live in its object store. The two never mix.

use crate::lake::{pivot_type_to_sql, sql_type_to_pivot};
use crate::store::ObjectStore;
use planner::catalog::Column;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error(transparent)]
    Store(#[from] crate::store::StoreError),
    #[error("malformed table manifest: {0}")]
    Parse(String),
}

pub type Result<T> = std::result::Result<T, ManifestError>;

/// One table's entry in the manifest: its identity, declared schema, and the
/// directory its Parquet data lives in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestEntry {
    pub name: String,
    pub columns: Vec<Column>,
    /// The directory holding this table's Parquet files. Absolute, or relative
    /// to the database root. Always a local filesystem path for a local database.
    pub location: PathBuf,
}

/// A database's durable record of its tables.
///
/// Implementations back the same registry over different media: nothing
/// ([`InMemoryTableManifest`]) or an [`ObjectStore`] ([`ObjectStoreManifest`],
/// local or remote). The catalog calls [`load`](Self::load) once at startup to
/// recover existing tables, and [`insert`](Self::insert) on every `CREATE TABLE`.
pub trait TableManifest: Debug + Send + Sync {
    /// Every table the database has, recovered from durable storage. Called once
    /// when the catalog starts; empty for a fresh or in-memory database.
    fn load(&self) -> Result<Vec<ManifestEntry>>;

    /// Durably record a newly-created table.
    fn insert(&self, entry: &ManifestEntry) -> Result<()>;
}

/// The default manifest: tables live only in the catalog's in-memory map and are
/// gone on restart. Records nothing.
#[derive(Debug, Default)]
pub struct InMemoryTableManifest;

impl TableManifest for InMemoryTableManifest {
    fn load(&self) -> Result<Vec<ManifestEntry>> {
        Ok(Vec::new())
    }

    fn insert(&self, _entry: &ManifestEntry) -> Result<()> {
        Ok(())
    }
}

/// Key of the manifest document within the database's object store.
const MANIFEST_KEY: &str = "_pivot_manifest.json";
/// On-disk format version of the manifest document.
const MANIFEST_VERSION: u32 = 1;

/// A manifest persisted as a single JSON document in an [`ObjectStore`] — a local
/// directory (a `--path` database) or an S3/GCS bucket. The same code serves
/// both; only the backing store differs.
#[derive(Debug)]
pub struct ObjectStoreManifest {
    store: Box<dyn ObjectStore>,
}

impl ObjectStoreManifest {
    /// Wrap a store as a persisted manifest. The store is rooted at the database
    /// directory; the manifest document lives at its `_pivot_manifest.json` key.
    pub fn new(store: Box<dyn ObjectStore>) -> Self {
        Self { store }
    }

    fn read_doc(&self) -> Result<ManifestDoc> {
        match self.store.get(MANIFEST_KEY)? {
            None => Ok(ManifestDoc::default()),
            Some(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| ManifestError::Parse(format!("{MANIFEST_KEY}: {e}"))),
        }
    }
}

impl TableManifest for ObjectStoreManifest {
    fn load(&self) -> Result<Vec<ManifestEntry>> {
        self.read_doc()?
            .tables
            .into_iter()
            .map(TableRecord::into_entry)
            .collect()
    }

    fn insert(&self, entry: &ManifestEntry) -> Result<()> {
        let mut doc = self.read_doc()?;
        doc.tables.push(TableRecord::from_entry(entry));
        let bytes = serde_json::to_vec_pretty(&doc)
            .map_err(|e| ManifestError::Parse(format!("serializing manifest: {e}")))?;
        self.store.put(MANIFEST_KEY, &bytes)?;
        Ok(())
    }
}

/// Serialized form of the whole manifest document.
#[derive(Serialize, Deserialize)]
struct ManifestDoc {
    version: u32,
    tables: Vec<TableRecord>,
}

impl Default for ManifestDoc {
    fn default() -> Self {
        Self {
            version: MANIFEST_VERSION,
            tables: Vec::new(),
        }
    }
}

/// Serialized form of one table entry.
#[derive(Serialize, Deserialize)]
struct TableRecord {
    name: String,
    columns: Vec<ColumnRecord>,
    location: String,
}

/// Serialized form of one column: name plus its SQL type spelling (the same
/// spelling the planner declared), so the type round-trips through the manifest.
#[derive(Serialize, Deserialize)]
struct ColumnRecord {
    name: String,
    #[serde(rename = "type")]
    type_sql: String,
}

impl TableRecord {
    fn from_entry(entry: &ManifestEntry) -> Self {
        Self {
            name: entry.name.clone(),
            columns: entry
                .columns
                .iter()
                .map(|c| ColumnRecord {
                    name: c.name.clone(),
                    type_sql: pivot_type_to_sql(&c.col_type).to_string(),
                })
                .collect(),
            location: entry.location.to_string_lossy().into_owned(),
        }
    }

    fn into_entry(self) -> Result<ManifestEntry> {
        let columns = self
            .columns
            .into_iter()
            .map(|c| {
                let col_type = sql_type_to_pivot(&c.type_sql).ok_or_else(|| {
                    ManifestError::Parse(format!(
                        "unknown type `{}` for column `{}` of table `{}`",
                        c.type_sql, c.name, self.name
                    ))
                })?;
                Ok(Column {
                    name: c.name,
                    col_type,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ManifestEntry {
            name: self.name,
            columns,
            location: PathBuf::from(self.location),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::LocalStore;
    use planner::types::Type;

    fn entry(name: &str) -> ManifestEntry {
        ManifestEntry {
            name: name.into(),
            columns: vec![Column {
                name: "a".into(),
                col_type: Type::Int32,
            }],
            location: PathBuf::from(name),
        }
    }

    #[test]
    fn in_memory_manifest_persists_nothing() {
        let manifest = InMemoryTableManifest;
        manifest.insert(&entry("t")).unwrap();
        assert!(manifest.load().unwrap().is_empty());
    }

    #[test]
    fn object_store_manifest_round_trips_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let open = || ObjectStoreManifest::new(Box::new(LocalStore::new(dir.path())));

        assert!(open().load().unwrap().is_empty());

        open().insert(&entry("first")).unwrap();
        open().insert(&entry("second")).unwrap();

        // A freshly-opened manifest sees both — as a restart would.
        let loaded = open().load().unwrap();
        assert_eq!(loaded, vec![entry("first"), entry("second")]);
    }
}
