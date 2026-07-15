//! Transaction-owned Parquet writes for INSERT.
//!
//! INSERT's dispatch workers encode files and upload them over their io_uring
//! (the same ring that fetches Parquet), then push each uploaded file's
//! [`ManifestEntry`] into the shared [`InsertCollector`]. Because the uploads
//! are part of the dataflow, they have all completed by the time the server's
//! `collect()` returns — so [`commit`](TransactionWriter::commit) simply drains
//! the collector and publishes one Delta version per table, and
//! [`rollback`](TransactionWriter::rollback) deletes what was uploaded. No
//! dedicated writer thread and no blocking object-store calls off the ring.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::manifest::ManifestEntry;

use super::{CatalogSnapshot, CatalogTable, Error, Result};

/// Gathers the manifest entries for files INSERT's workers have uploaded. The
/// upload sink on each worker records an entry as its file lands (a brief lock
/// held only to push); commit drains them grouped by table, rollback deletes
/// them.
pub(in crate::catalog) struct InsertCollector {
    entries: Mutex<HashMap<String, Vec<ManifestEntry>>>,
}

impl InsertCollector {
    fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Record a successfully uploaded file for `table`. Called from a dispatch
    /// worker when one of its uploads completes.
    pub(in crate::catalog) fn record(&self, table: String, entry: ManifestEntry) {
        self.entries
            .lock()
            .unwrap()
            .entry(table)
            .or_default()
            .push(entry);
    }

    /// Take every collected entry, leaving the collector empty.
    fn take(&self) -> HashMap<String, Vec<ManifestEntry>> {
        std::mem::take(&mut self.entries.lock().unwrap())
    }

    /// Put entries back after a partial commit: those whose Delta commit failed
    /// return here so rollback/`Drop` can still delete their uploaded files.
    fn restore(&self, entries: impl IntoIterator<Item = (String, Vec<ManifestEntry>)>) {
        let mut guard = self.entries.lock().unwrap();
        for (table, mut files) in entries {
            guard.entry(table).or_default().append(&mut files);
        }
    }
}

/// Shared transaction state captured by INSERT operators.
///
/// INSERT setup obtains the [`InsertCollector`] while holding the lifecycle
/// mutex. Dispatch workers then record uploaded files into it directly. Commit
/// and rollback run from the server's blocking pool after the dataflow (uploads
/// included) has fully drained.
pub(super) struct TransactionWriter {
    snapshot: Arc<CatalogSnapshot>,
    collector: Arc<InsertCollector>,
    /// Set once commit or rollback has consumed the collected entries, so
    /// `Drop` knows there is nothing left to clean up.
    finalized: AtomicBool,
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
            collector: Arc::new(InsertCollector::new()),
            finalized: AtomicBool::new(false),
        }
    }

    /// The collector INSERT's upload sinks record into. A read-only transaction
    /// never records anything, so this costs nothing until the first upload.
    pub(super) fn collector(&self) -> Arc<InsertCollector> {
        self.collector.clone()
    }

    /// Commit every uploaded file: one Delta version per table, returning the
    /// updated table copies for immediate catalog publication.
    ///
    /// If a table's Delta commit fails, its (and every still-uncommitted table's)
    /// entries are put back into the collector before returning the error, so the
    /// caller's rollback — or this writer's `Drop` — still deletes those uploaded
    /// files rather than orphaning them. `finalized` is set only once the whole
    /// commit succeeds.
    pub(super) fn commit(&self) -> Result<Vec<CatalogTable>> {
        let mut pending: Vec<(String, Vec<ManifestEntry>)> =
            self.collector.take().into_iter().collect();

        let mut committed = Vec::with_capacity(pending.len());
        while let Some((table, added)) = pending.pop() {
            let result = self
                .snapshot
                .tables
                .get(&table)
                .cloned()
                .ok_or_else(|| Error::InsertTableMissing(table.clone()))
                .and_then(|mut copy| copy.commit_added_files(added.clone()).map(|()| copy));
            match result {
                Ok(copy) => committed.push(copy),
                Err(error) => {
                    // This table isn't committed; re-stash it and every table
                    // still pending so cleanup deletes their uploads. Tables
                    // already committed are durable in Delta and left as is.
                    self.collector
                        .restore(std::iter::once((table, added)).chain(pending));
                    return Err(error);
                }
            }
        }
        self.finalized.store(true, Ordering::Relaxed);
        Ok(committed)
    }

    /// Delete every uploaded file, so a failed or rolled-back INSERT leaves no
    /// orphans. Best-effort: it attempts every deletion and returns the first
    /// failure only after visiting them all.
    pub(super) fn rollback(&self) -> Result<()> {
        let entries = self.collector.take();
        self.finalized.store(true, Ordering::Relaxed);
        delete_uploaded_files(&self.snapshot, entries)
    }
}

impl Drop for TransactionWriter {
    fn drop(&mut self) {
        // A transaction dropped without an explicit commit/rollback (e.g. the
        // connection went away mid-INSERT) still leaves uploaded orphans; clean
        // them up off the reactor.
        if self.finalized.load(Ordering::Relaxed) {
            return;
        }
        let entries = self.collector.take();
        if entries.is_empty() {
            return;
        }
        let snapshot = self.snapshot.clone();
        let _ = std::thread::Builder::new()
            .name("pivot-insert-cleanup".to_string())
            .spawn(move || {
                let _ = delete_uploaded_files(&snapshot, entries);
            });
    }
}

/// Delete every uploaded file in `entries`, attempting all of them and returning
/// the first failure (if any) only after visiting them all.
fn delete_uploaded_files(
    snapshot: &CatalogSnapshot,
    entries: HashMap<String, Vec<ManifestEntry>>,
) -> Result<()> {
    let mut first_error = None;
    for (table, files) in entries {
        let Some(copy) = snapshot.tables.get(&table) else {
            continue;
        };
        for entry in files {
            if let Err(error) = copy.delete_data_file(&entry.file.path)
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
