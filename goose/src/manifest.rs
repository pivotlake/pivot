//! The **table manifest**: a database's record of which tables exist and where
//! each table's Parquet data lives.
//!
//! A database has one manifest. The default ([`InMemoryTableManifest`]) keeps
//! its tables only in memory, so they vanish when the server restarts — fine for
//! ad-hoc / test use. A *persisted* manifest (e.g. a `DiskTableManifest` rooted
//! at the database directory) durably records every `CREATE TABLE`, so restarting
//! the server reloads the same tables.
//!
//! The manifest only tracks table *definitions* (name, declared schema, and the
//! directory holding the data). The Parquet row-group metadata itself is read
//! from those directories when the catalog loads — the manifest is the small,
//! durable index that points at them.
//!
//! A database has a single storage class fixed at startup: a *local* database's
//! table locations are always local filesystem paths (absolute, or relative to
//! the database root) — never an `s3://`/`gs://`/`file://` URL — and a remote
//! database's are always in its object store. The two never mix.

use planner::catalog::Column;
use std::fmt::Debug;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("io error on manifest `{path}`: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing manifest `{path}`: {message}")]
    Parse { path: String, message: String },
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
/// ([`InMemoryTableManifest`]), a local directory, or an object store. The
/// catalog calls [`load`](Self::load) once at startup to recover existing tables,
/// and [`insert`](Self::insert) on every `CREATE TABLE`.
pub trait TableManifest: Debug + Send + Sync {
    /// Every table the database has, recovered from durable storage. Called once
    /// when the catalog starts; empty for a fresh or in-memory database.
    fn load(&self) -> Result<Vec<ManifestEntry>>;

    /// Durably record a newly-created table.
    fn insert(&self, entry: &ManifestEntry) -> Result<()>;

    /// The database root directory, if this manifest is persisted to one. A
    /// `CREATE TABLE` with no explicit path puts the new table under here
    /// (`<root>/<name>`). `None` for the in-memory manifest, which has no root.
    fn root(&self) -> Option<&Path>;
}

/// The default manifest: tables live only in the catalog's in-memory map and are
/// gone on restart. Records nothing and has no root.
#[derive(Debug, Default)]
pub struct InMemoryTableManifest;

impl TableManifest for InMemoryTableManifest {
    fn load(&self) -> Result<Vec<ManifestEntry>> {
        Ok(Vec::new())
    }

    fn insert(&self, _entry: &ManifestEntry) -> Result<()> {
        Ok(())
    }

    fn root(&self) -> Option<&Path> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_manifest_persists_nothing() {
        let manifest = InMemoryTableManifest;
        manifest
            .insert(&ManifestEntry {
                name: "t".into(),
                columns: Vec::new(),
                location: PathBuf::from("t"),
            })
            .unwrap();
        // Nothing was retained, and there's no root to place tables under.
        assert!(manifest.load().unwrap().is_empty());
        assert!(manifest.root().is_none());
    }
}
