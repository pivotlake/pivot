//! The catalog's master record of one table: its definition plus its current
//! content — the files at one [`table_log`](crate::table_log) version.

use std::sync::Arc;

use crate::catalog::TableBinding;
use crate::manifest::ManifestEntry;
use crate::parquet::{ParquetTable, RowGroupMetadata};
use crate::store::FileRef;
use planner::catalog::Column;

/// One data file of a table: its identity ([`FileRef`]) paired with its
/// materialized row groups (file-local order).
#[derive(Clone)]
pub(super) struct TableFile {
    pub(super) file: FileRef,
    pub(super) row_groups: Vec<Arc<RowGroupMetadata>>,
}

/// The catalog's master record of one table: its definition (name, declared
/// columns, data location) and its current content — the per-file row groups at
/// one log version. A change (a registered file, a compaction swap) replaces
/// `version` + `files` whole via [`set_version`](Self::set_version); the scan
/// view a query sees is derived from `files` on demand, so there is no cached
/// copy to keep in sync.
pub(super) struct CatalogTable {
    pub(super) name: String,
    pub(super) columns: Vec<Column>,
    pub(super) location: String,
    pub(super) version: u64,
    pub(super) files: Vec<TableFile>,
}

impl CatalogTable {
    pub(super) fn new(manifest: ManifestEntry, version: u64, files: Vec<TableFile>) -> Self {
        Self {
            name: manifest.name,
            columns: manifest.columns,
            location: manifest.location,
            version,
            files,
        }
    }

    /// Swap in a newer log version's content.
    pub(super) fn set_version(&mut self, version: u64, files: Vec<TableFile>) {
        self.version = version;
        self.files = files;
    }

    /// A fresh per-query [`TableBinding`] over a flat scan view of the current
    /// files: every file's row groups concatenated in log order, where a row
    /// group's global index is simply its position. Derived here (cheap `Arc`
    /// clones) rather than cached, so the view can never disagree with `files`.
    pub(super) fn binding(&self) -> TableBinding {
        let row_groups = self
            .files
            .iter()
            .flat_map(|f| f.row_groups.iter().cloned())
            .collect();
        TableBinding::new(self.columns.clone(), Arc::new(ParquetTable::new(row_groups)))
    }

    /// The current file list as [`FileRef`]s — what a commit on top of this
    /// state builds on, and what a compacter scans.
    pub(super) fn logged_files(&self) -> Vec<FileRef> {
        self.files.iter().map(|f| f.file.clone()).collect()
    }
}
