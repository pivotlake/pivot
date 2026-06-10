//! The catalog's master record of one table: its definition plus its
//! [`TableState`] — the content at one log version.

use std::sync::Arc;

use crate::catalog::TableBinding;
use crate::manifest::ManifestEntry;
use crate::parquet::{ParquetTable, RowGroupMetadata};
use crate::table_log::LoggedFile;
use planner::catalog::Column;

/// One data file of a table state: its logged identity paired with its
/// materialized row groups (file-local order).
#[derive(Clone)]
pub(super) struct TableFile {
    pub(super) logged: LoggedFile,
    pub(super) row_groups: Vec<Arc<RowGroupMetadata>>,
}

/// A table's content at one log version: the per-file row groups, and the
/// flattened scan view **derived from them at construction** — the only way to
/// get a `TableState` is [`new`](Self::new), so the view can never disagree
/// with the files. A change (a registered file, a compaction swap) is a new
/// `TableState`, swapped into the entry whole.
pub(super) struct TableState {
    version: u64,
    files: Vec<TableFile>,
    view: Arc<ParquetTable>,
}

impl TableState {
    /// Build the state for `version` from its files: concatenate every file's
    /// row groups in log order and assign global indices, yielding the scan
    /// view. Row groups whose index already matches are shared, not cloned.
    pub(super) fn new(version: u64, files: Vec<TableFile>) -> Self {
        let rows = files
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
        Self {
            version,
            files,
            view: Arc::new(ParquetTable::new(rows)),
        }
    }

    /// The [`table_log`](crate::table_log) version this state reflects (`0` =
    /// a log-less legacy table, loaded by directory listing; its first commit
    /// seeds the log).
    pub(super) fn version(&self) -> u64 {
        self.version
    }

    /// The state's files with their row groups, in log order.
    pub(super) fn files(&self) -> &[TableFile] {
        &self.files
    }

    /// The file list in logged form — what a commit on top of this state
    /// builds on, and what a compacter scans.
    pub(super) fn logged_files(&self) -> Vec<LoggedFile> {
        self.files.iter().map(|f| f.logged.clone()).collect()
    }
}

/// The master record of one table: its definition (name, declared columns,
/// data location) and its current [`TableState`].
pub(super) struct TableEntry {
    /// The table's name — the catalog map's key.
    pub(super) name: String,
    pub(super) columns: Vec<Column>,
    pub(super) location: String,
    pub(super) state: TableState,
}

impl TableEntry {
    pub(super) fn new(manifest: ManifestEntry, state: TableState) -> Self {
        Self {
            name: manifest.name,
            columns: manifest.columns,
            location: manifest.location,
            state,
        }
    }

    /// A fresh per-query [`TableBinding`] over the current state's scan view.
    pub(super) fn binding(&self) -> TableBinding {
        TableBinding::new(
            self.columns.clone(),
            self.location.clone(),
            self.state.view.clone(),
        )
    }
}
