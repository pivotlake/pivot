//! Transaction-owned bookkeeping for INSERT's Parquet writes.
//!
//! Dispatch performs every byte transfer on the io_uring ring (see
//! [`insert_sink`](super::insert_sink)); this object never touches an object
//! store on the data path. It only stages the manifest entry each successful
//! write produces — recorded straight from the dispatch worker that saw the
//! completion — and, once the query's dataflow has fully drained, either commits
//! those entries as one Delta version per table or deletes the files on rollback.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::manifest::ManifestEntry;

use super::{CatalogSnapshot, CatalogTable, Error, Result};

/// One table's staged files, pinned to the snapshot copy that commit publishes.
struct PendingTable {
    table: CatalogTable,
    entries: Vec<ManifestEntry>,
}

/// Shared transaction state INSERT sink operators record into.
///
/// The sink operators run on dispatch workers and record a [`ManifestEntry`] as
/// each of their writes lands; commit and rollback run afterwards on the server's
/// blocking pool, once the dataflow has drained (so every write has completed and
/// every entry is recorded). A single mutex guards the staged set — recording is
/// rare relative to the byte transfer, so the lock is never contended on a hot
/// path.
pub(super) struct TransactionWriter {
    snapshot: Arc<CatalogSnapshot>,
    pending: Mutex<HashMap<String, PendingTable>>,
}

impl std::fmt::Debug for TransactionWriter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TransactionWriter")
            .finish_non_exhaustive()
    }
}

impl TransactionWriter {
    pub(super) fn new(snapshot: Arc<CatalogSnapshot>) -> Self {
        Self {
            snapshot,
            pending: Mutex::new(HashMap::new()),
        }
    }

    /// Record one uploaded file for `table`. Called from a dispatch worker as the
    /// write completes; the entry is published on [`commit`](Self::commit) or its
    /// file deleted on [`rollback`](Self::rollback).
    pub(in crate::catalog) fn record(&self, table: &str, entry: ManifestEntry) -> Result<()> {
        let snapshot_table = self
            .snapshot
            .tables
            .get(table)
            .ok_or_else(|| Error::InsertTableMissing(table.to_string()))?;
        let mut pending = self.pending.lock().unwrap();
        pending
            .entry(table.to_string())
            .or_insert_with(|| PendingTable {
                table: snapshot_table.clone(),
                entries: Vec::new(),
            })
            .entries
            .push(entry);
        Ok(())
    }

    /// Commit each table's staged files in one Delta version, returning the
    /// updated table copies for immediate catalog publication.
    pub(super) fn commit(&self) -> Result<Vec<CatalogTable>> {
        let mut pending = self.pending.lock().unwrap();
        let mut committed = Vec::with_capacity(pending.len());
        for entry in pending.values_mut() {
            entry
                .table
                .commit_added_files(std::mem::take(&mut entry.entries))?;
            committed.push(entry.table.clone());
        }
        Ok(committed)
    }

    /// Delete every uncommitted Parquet file this transaction successfully wrote.
    /// Best-effort: attempts every deletion and returns the first failure only
    /// after visiting them all.
    pub(super) fn rollback(&self) -> Result<()> {
        let pending = self.pending.lock().unwrap();
        let mut first_error = None;
        for table in pending.values() {
            for entry in &table.entries {
                if let Err(error) = table.table.delete_data_file(&entry.file.path)
                    && first_error.is_none()
                {
                    first_error = Some(error);
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}
