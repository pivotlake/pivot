//! Transaction-owned metadata for Parquet files staged by INSERT.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::manifest::ManifestEntry;

use super::{CatalogSnapshot, CatalogTable, Error, Result};

/// A worker-local handle that registers one table's files before their async
/// uploads start.
#[derive(Clone)]
pub(in crate::catalog) struct WriteSender {
    writer: Arc<TransactionWriter>,
    table: String,
}

impl WriteSender {
    /// Retain a file's final identity so commit can publish it and rollback can
    /// delete it, including when its upload later fails or is cancelled.
    pub(in crate::catalog) fn submit(&self, entry: ManifestEntry) -> Result<()> {
        self.writer.stage(&self.table, entry)
    }
}

struct PendingTable {
    table: CatalogTable,
    entries: Vec<ManifestEntry>,
}

#[derive(Default)]
struct WriterResult {
    tables: HashMap<String, PendingTable>,
}

enum WriterState {
    Open(WriterResult),
    Finalized,
}

/// Shared transaction state captured by INSERT operators.
///
/// Dispatch performs every Parquet upload. This state only records each path
/// before launch, then commits the successfully drained dataflow's entries in
/// one Delta version or removes them on rollback.
pub(super) struct TransactionWriter {
    snapshot: Arc<CatalogSnapshot>,
    state: Mutex<WriterState>,
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
            state: Mutex::new(WriterState::Open(WriterResult::default())),
        }
    }

    /// Obtain a table-scoped registration handle for an INSERT pipeline.
    pub(super) fn write_sender(self: &Arc<Self>, table: String) -> Result<WriteSender> {
        if !self.snapshot.tables.contains_key(&table) {
            return Err(Error::InsertTableMissing(table));
        }
        if !matches!(*self.state.lock().unwrap(), WriterState::Open(_)) {
            return Err(Error::InsertWriterUnavailable);
        }
        Ok(WriteSender {
            writer: self.clone(),
            table,
        })
    }

    fn stage(&self, table: &str, entry: ManifestEntry) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        let WriterState::Open(result) = &mut *state else {
            return Err(Error::InsertWriterUnavailable);
        };
        let snapshot_table = self
            .snapshot
            .tables
            .get(table)
            .ok_or_else(|| Error::InsertTableMissing(table.to_string()))?;
        result
            .tables
            .entry(table.to_string())
            .or_insert_with(|| PendingTable {
                table: snapshot_table.clone(),
                entries: Vec::new(),
            })
            .entries
            .push(entry);
        Ok(())
    }

    /// Commit each table's staged files in one Delta version and return the
    /// updated copies for immediate catalog publication.
    pub(super) fn commit(&self) -> Result<Vec<CatalogTable>> {
        let mut state = self.state.lock().unwrap();
        let WriterState::Open(result) = &mut *state else {
            return Err(Error::InsertWriterUnavailable);
        };

        let mut committed = Vec::with_capacity(result.tables.len());
        for pending in result.tables.values_mut() {
            pending.table.commit_added_files(pending.entries.clone())?;
            pending.entries.clear();
            committed.push(pending.table.clone());
        }
        *state = WriterState::Finalized;
        Ok(committed)
    }

    /// Delete every Parquet path registered by this transaction.
    pub(super) fn rollback(&self) -> Result<()> {
        let result = {
            let mut state = self.state.lock().unwrap();
            match std::mem::replace(&mut *state, WriterState::Finalized) {
                WriterState::Open(result) => result,
                WriterState::Finalized => return Err(Error::InsertWriterUnavailable),
            }
        };
        delete_pending_files(result)
    }
}

impl Drop for TransactionWriter {
    fn drop(&mut self) {
        let state = std::mem::replace(self.state.get_mut().unwrap(), WriterState::Finalized);
        let WriterState::Open(result) = state else {
            return;
        };
        let _ = std::thread::Builder::new()
            .name("pivot-insert-cleanup".to_string())
            .spawn(move || {
                let _ = delete_pending_files(result);
            });
    }
}

/// Best-effort cleanup that attempts every deletion and returns the first
/// failure only after all pending files have been visited.
fn delete_pending_files(result: WriterResult) -> Result<()> {
    let mut first_error = None;
    for pending in result.tables.into_values() {
        for entry in pending.entries {
            if let Err(error) = pending.table.delete_data_file(&entry.file.path)
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
