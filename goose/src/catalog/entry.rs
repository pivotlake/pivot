//! The catalog's master record of one table: the binding template every query
//! clones, plus the versioned per-file row-group state it is derived from.

use std::sync::Arc;

use crate::catalog::ParquetCatalogTable;
use crate::manifest::ManifestEntry;
use crate::parquet::{ParquetTable, RowGroupMetadata};
use crate::table_log::LoggedFile;

/// One data file of a table entry: its logged identity paired with its
/// materialized row groups (file-local order; global indices are assigned
/// when the entry's files are flattened into the scan view).
#[derive(Clone)]
pub(super) struct TableFileEntry {
    pub(super) file: LoggedFile,
    pub(super) row_groups: Vec<Arc<RowGroupMetadata>>,
}

/// The master record of one table.
#[derive(Clone)]
pub(super) struct TableEntry {
    /// The table's name — the catalog map's key.
    pub(super) name: String,
    /// What [`Catalog::table`](planner::catalog::Catalog::table) clones for
    /// each binding.
    pub(super) table: ParquetCatalogTable,
    /// The [`table_log`](crate::table_log) version this entry reflects (`0` =
    /// a log-less legacy table, loaded by directory listing; its first commit
    /// seeds the log).
    pub(super) version: u64,
    /// The version's files with their row groups, in log order.
    pub(super) files: Vec<TableFileEntry>,
}

impl TableEntry {
    pub(super) fn new(manifest: ManifestEntry, version: u64, files: Vec<TableFileEntry>) -> Self {
        let mut entry = Self {
            name: manifest.name,
            table: ParquetCatalogTable::empty(manifest.columns, manifest.location),
            version,
            files,
        };
        entry.reflatten();
        entry
    }

    /// Rebuild the binding template's scan view from the per-file state:
    /// concatenate every file's row groups in log order and (re)assign global
    /// indices. Row groups whose index already matches are shared, not cloned.
    pub(super) fn reflatten(&mut self) {
        let rows = self
            .files
            .iter()
            .flat_map(|f| f.row_groups.iter())
            .enumerate()
            .map(|(global_idx, rg)| {
                if rg.global_row_group_idx == global_idx {
                    rg.clone()
                } else {
                    let mut renumbered = (**rg).clone();
                    renumbered.global_row_group_idx = global_idx;
                    Arc::new(renumbered)
                }
            })
            .collect();
        self.table.parquet = Arc::new(ParquetTable::new(rows));
    }

    /// The entry's file list in logged form — the base a commit builds on when
    /// the table has no log yet (version 0).
    pub(super) fn logged_files(&self) -> Vec<LoggedFile> {
        self.files.iter().map(|f| f.file.clone()).collect()
    }
}
