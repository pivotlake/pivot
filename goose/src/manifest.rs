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
//!   `_goose_tables/<name>/<version>.json` with `put_if_absent`, so a commit is a
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
/// (`_goose_tables/<name>/<version>.json`). Kept under the database root so a
/// table over an external data directory never has catalog metadata written
/// into it.
const TABLE_MANIFEST_DIR: &str = "_goose_tables";
/// Version numbers are zero-padded to this width so a `list` returns them in
/// numeric order and the lexicographic max is the latest.
const VERSION_DIGITS: usize = 20;
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
    /// The directory holding table `name`'s manifest versions.
    fn dir(name: &str) -> ObjectPath {
        ObjectPath::new(format!("{TABLE_MANIFEST_DIR}/{name}"))
    }

    /// The store key of one `version` of table `name`'s manifest.
    fn key(name: &str, version: u64) -> ObjectPath {
        Self::dir(name).join(&format!("{version:0VERSION_DIGITS$}.json"))
    }

    /// Load table `name`'s latest committed manifest — one `list` of its version
    /// directory for the highest version, then a `get` of it. The database
    /// manifest is the index of which tables exist, so a name listed there with
    /// no manifest is a corrupt catalog — an error, not an absent table.
    pub fn load(store: &dyn ObjectStore, name: &str) -> Result<Self> {
        let version = store
            .list(&Self::dir(name))?
            .iter()
            .filter_map(|file| file.name.strip_suffix(".json").and_then(|v| v.parse::<u64>().ok()))
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
