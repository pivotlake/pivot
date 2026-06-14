//! The catalog's durable metadata, as JSON in the database's [`ObjectStore`].
//!
//! Two documents, both folded out of what used to be a separate "table log":
//!
//! - the **database manifest** ([`CatalogManifest`], at [`MANIFEST_KEY`]) — the
//!   index of which tables exist and where each one's Parquet data lives.
//! - a per-table **table manifest** ([`TableManifest`], under
//!   [`TABLE_MANIFEST_DIR`]) — the table's declared schema plus its committed
//!   file list, at one version.
//!
//! Both serialize as JSON and round-trip through plain `get`/`put`; the catalog
//! holds the live copies in memory and rewrites them on a change.

use planner::catalog::Column;
use serde::{Deserialize, Serialize};

use crate::FileRef;
use crate::store::ObjectStore;

/// Key of the [`CatalogManifest`] document within the database's object store.
const MANIFEST_KEY: &str = "_pivot_manifest.json";
/// Directory the per-table [`TableManifest`] documents live under, keyed by
/// table name (`_goose_tables/<name>.json`). Kept under the database root so a
/// table over an external data directory never has catalog metadata written
/// into it.
const TABLE_MANIFEST_DIR: &str = "_goose_tables";
/// The version a table's first commit gets.
pub const FIRST_VERSION: u64 = 1;

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

/// One table's durable record: its declared schema (`columns`) and its committed
/// file list (`entries`) at `version`. Mirrors the live [`CatalogTable`] state
/// the catalog keeps in memory.
///
/// [`CatalogTable`]: crate::catalog::CatalogTable
#[derive(Clone, Serialize, Deserialize)]
pub struct TableManifest {
    pub version: u64,
    pub columns: Vec<Column>,
    pub entries: Vec<FileRef>,
}

impl TableManifest {
    /// The store key of table `name`'s manifest document.
    fn key(name: &str) -> String {
        format!("{TABLE_MANIFEST_DIR}/{name}.json")
    }

    /// Load table `name`'s manifest. The database manifest is the index of which
    /// tables exist, so a name listed there with no manifest is a corrupt
    /// catalog — an error, not an absent table.
    pub fn load(store: &dyn ObjectStore, name: &str) -> Result<Self> {
        let bytes = store
            .get(&Self::key(name))?
            .ok_or_else(|| Error::MissingTableManifest(name.to_string()))?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Write table `name`'s manifest, overwriting any previous version.
    pub fn store(&self, store: &dyn ObjectStore, name: &str) -> Result<()> {
        store.put(&Self::key(name), &serde_json::to_vec(self)?)?;
        Ok(())
    }
}

/// One table's entry in the [`CatalogManifest`]: its name and the location its
/// Parquet data lives at (relative to the database root, or an absolute store
/// path).
#[derive(Clone, Serialize, Deserialize)]
pub struct CatalogManifestTableEntry {
    pub(crate) name: String,
    pub(crate) location: String,
}

impl CatalogManifestTableEntry {
    pub fn new(name: String, location: String) -> Self {
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
        match store.get(MANIFEST_KEY)? {
            Some(bytes) => Ok(serde_json::from_slice(&bytes)?),
            None => Ok(Self::default()),
        }
    }

    /// Write the database manifest, overwriting the previous version.
    pub fn store(&self, store: &dyn ObjectStore) -> Result<()> {
        store.put(MANIFEST_KEY, &serde_json::to_vec(self)?)?;
        Ok(())
    }

    /// Add (or replace) a table entry and bump the manifest version.
    pub(crate) fn upsert(&mut self, entry: CatalogManifestTableEntry) {
        self.tables.retain(|t| t.name != entry.name);
        self.tables.push(entry);
        self.version += 1;
    }
}
