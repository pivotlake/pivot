use std::path::PathBuf;
use std::sync::Arc;
use planner::catalog::Column;
use crate::FileRef;
use crate::store::ObjectStore;


/// Root directory of all table logs within the database store.
const LOG_DIR: &str = "_goose_logs";
/// Version numbers are zero-padded to this width so lexicographic key order is
/// numeric order.
const VERSION_DIGITS: usize = 20;
/// The version a table's very first commit gets.
pub const FIRST_VERSION: u64 = 1;


/// Key of the CatalogManifest document within the database's object store.
const MANIFEST_KEY: &str = "_pivot_manifest.json";

// TODO: these should all be serialize/deserialize

pub struct TableManifest {
    pub columns: Vec<Column>,
    pub entries: Vec<FileRef>
}


pub struct CatalogManifestTableEntry {
    pub(crate) name: String,
    path: PathBuf,
}

impl CatalogManifestTableEntry {
    pub fn new(name: String) -> Self {
        Self {
            name,
            path: Default::default(),
        }
    }
}

pub struct CatalogManifest {
    tables: Vec<CatalogManifestTableEntry>,
    version: usize,
}
