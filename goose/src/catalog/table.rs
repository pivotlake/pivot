//! The catalog's master record of one table: its definition plus its current
//! content — the files at one manifest version.

use std::path::Path;
use std::sync::Arc;

use crate::catalog::TableBinding;
use crate::manifest::{TableManifest};
use crate::parquet::{ParquetTable, RowGroupMetadata};
use crate::store::{FileRef, ObjectStore};
use planner::catalog::Column;
use crate::RegisterOutcome;

/// One data file of a table: its identity ([`FileRef`]) paired with its
/// materialized row groups (file-local order).
#[derive(Clone)]
pub struct TableFile {
    pub(super) file: FileRef,
    pub(super) row_groups: Vec<Arc<RowGroupMetadata>>,
}

impl TableFile {
    /// Pair a file's identity with the row groups read from its footer. Built by
    /// the metadata fetcher (one per file) and the catalog's table assembly.
    pub(crate) fn new(file: FileRef, row_groups: Vec<Arc<RowGroupMetadata>>) -> Self {
        Self { file, row_groups }
    }

    /// This file's row groups, in file-local order.
    pub(crate) fn row_groups(&self) -> &[Arc<RowGroupMetadata>] {
        &self.row_groups
    }
}

/// The catalog's master record of one table: its definition (name, declared
/// columns, data location) and its current content — the per-file row groups at
/// one log version. A change (a registered file, a compaction swap) replaces
/// `version` + `files` whole via [`set_version`](Self::set_version); the scan
/// view a query sees is derived from `files` on demand, so there is no cached
/// copy to keep in sync.
pub struct CatalogTable {
    pub(super) manifest: TableManifest,
    pub(super) version: u64,
    pub(super) files: Vec<TableFile>,
    // todo: remove this comment, but this allows us to refresh in place
    store: Arc<dyn ObjectStore>
}

impl CatalogTable {
    /// Assemble the master record from a table's definition (`manifest`), the
    /// manifest `version` those files were read at, and the per-file row groups
    /// (`files`). `store` is retained so the entry can reload itself in place.
    pub(super) fn new(
        manifest: TableManifest,
        version: u64,
        files: Vec<TableFile>,
        store: Arc<dyn ObjectStore>,
    ) -> Self {
        Self {
            manifest,
            version,
            files,
            store,
        }
    }


    /// Register one newly-written local Parquet data file with table `name`:
    /// read its footer (over the dispatch pool), commit a new log version
    /// whose file list is `latest + this file` — retrying past CAS conflicts
    /// with other writers — and swap the new version into the in-memory entry.
    /// Registering a name the log already holds is a no-op, so a replayed
    /// notification can't double-count rows.
    ///
    /// The file must live in the table's data directory — and only a table
    /// over an absolute local directory can accept a *path* (a store-relative
    /// location is not where local files land); anything else is reported as
    /// [`RegisterOutcome::LocationMismatch`] and nothing is committed.
    pub fn register_data_file(&self, path: &Path) -> crate::Result<RegisterOutcome> {
        // TODO
        todo!()
    }
    
    pub fn files(&self) -> &[TableFile] {
        &self.files
    }

    pub fn refresh() {
        todo!()
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
        TableBinding::new(self.manifest.columns.clone(), Arc::new(ParquetTable::new(row_groups)))
    }

    /// The current file list as [`FileRef`]s — what a commit on top of this
    /// state builds on, and what a compacter scans.
    pub(super) fn logged_files(&self) -> Vec<FileRef> {
        self.files.iter().map(|f| f.file.clone()).collect()
    }
}
