//! Transaction-owned Parquet writes for INSERT.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;

use crate::manifest::ManifestEntry;
use crate::parquet_writing::EncodedFile;
use crate::store::ObjectPath;

use super::{CatalogSnapshot, CatalogTable, Error, Result};

struct WriteCommand {
    table: String,
    encoded: EncodedFile,
}

struct PendingTable {
    table: CatalogTable,
    entries: Vec<ManifestEntry>,
}

#[derive(Default)]
struct WriterResult {
    tables: HashMap<String, PendingTable>,
    error: Option<Error>,
}

struct WriterThread {
    sender: mpsc::Sender<WriteCommand>,
    join: JoinHandle<WriterResult>,
}

enum WriterState {
    Idle,
    Running(WriterThread),
    Finishing,
    Completed(WriterResult),
    Finalized,
}

/// Shared transaction state captured by INSERT operators.
///
/// Dispatch workers only call [`submit`](Self::submit), which sends into an
/// unbounded in-memory channel. The dedicated writer performs all blocking
/// object-store calls. Commit and rollback join it from the server's blocking
/// pool after the dataflow has drained.
pub(super) struct TransactionWriter {
    snapshot: Arc<CatalogSnapshot>,
    state: Mutex<Option<WriterState>>,
}

impl std::fmt::Debug for TransactionWriter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TransactionWriter")
            .finish_non_exhaustive()
    }
}

impl TransactionWriter {
    /// Create a lazy writer. No thread is started until the first encoded file
    /// is submitted, so read-only transactions pay no thread cost.
    pub(super) fn new(snapshot: Arc<CatalogSnapshot>) -> Self {
        Self {
            snapshot,
            state: Mutex::new(Some(WriterState::Idle)),
        }
    }

    /// Queue one encoded file. This never waits for file or network I/O.
    pub(super) fn submit(&self, table: String, encoded: EncodedFile) -> Result<()> {
        let mut locked = self.state.lock().unwrap();
        let state = locked.take().ok_or(Error::InsertWriterUnavailable)?;
        let writer = match state {
            WriterState::Idle => self.start_writer()?,
            WriterState::Running(writer) => writer,
            other => {
                *locked = Some(other);
                return Err(Error::InsertWriterUnavailable);
            }
        };
        let send_result = writer.sender.send(WriteCommand { table, encoded });
        *locked = Some(WriterState::Running(writer));
        send_result.map_err(|_| Error::InsertWriterStopped)
    }

    /// Start the transaction's single blocking-I/O thread and its unbounded
    /// in-memory command channel.
    fn start_writer(&self) -> Result<WriterThread> {
        let (sender, receiver) = mpsc::channel();
        let snapshot = self.snapshot.clone();
        let join = std::thread::Builder::new()
            .name("pivot-insert-writer".to_string())
            .spawn(move || write_files(snapshot, receiver))?;
        Ok(WriterThread { sender, join })
    }

    /// Close the command channel and join the writer exactly once.
    ///
    /// The server calls commit and rollback from Tokio's blocking pool, so this
    /// join never occupies a dispatch worker or an async reactor thread.
    fn finish_writer(&self) -> Result<()> {
        let running = {
            let mut locked = self.state.lock().unwrap();
            match locked.take().ok_or(Error::InsertWriterUnavailable)? {
                WriterState::Idle => {
                    *locked = Some(WriterState::Completed(WriterResult::default()));
                    return Ok(());
                }
                WriterState::Running(writer) => {
                    *locked = Some(WriterState::Finishing);
                    writer
                }
                WriterState::Completed(result) => {
                    *locked = Some(WriterState::Completed(result));
                    return Ok(());
                }
                other => {
                    *locked = Some(other);
                    return Err(Error::InsertWriterUnavailable);
                }
            }
        };

        drop(running.sender);
        let result = running
            .join
            .join()
            .map_err(|_| Error::InsertWriterPanicked)?;
        self.state
            .lock()
            .unwrap()
            .replace(WriterState::Completed(result));
        Ok(())
    }

    /// Drain writes, commit each table's staged files in one Delta version, and
    /// return the updated table copies for immediate catalog publication.
    pub(super) fn commit(&self) -> Result<Vec<CatalogTable>> {
        self.finish_writer()?;
        let mut locked = self.state.lock().unwrap();
        let state = locked.as_mut().ok_or(Error::InsertWriterUnavailable)?;
        let WriterState::Completed(result) = state else {
            return Err(Error::InsertWriterUnavailable);
        };
        if let Some(error) = result.error.take() {
            return Err(error);
        }

        let mut committed = Vec::with_capacity(result.tables.len());
        for pending in result.tables.values_mut() {
            pending.table.commit_added_files(pending.entries.clone())?;
            pending.entries.clear();
            committed.push(pending.table.clone());
        }
        *state = WriterState::Finalized;
        Ok(committed)
    }

    /// Drain writes and delete every uncommitted Parquet file successfully
    /// created by this transaction.
    pub(super) fn rollback(&self) -> Result<()> {
        self.finish_writer()?;
        let mut locked = self.state.lock().unwrap();
        let state = locked.take().ok_or(Error::InsertWriterUnavailable)?;
        let WriterState::Completed(result) = state else {
            *locked = Some(state);
            return Err(Error::InsertWriterUnavailable);
        };
        let cleanup = delete_pending_files(result);
        *locked = Some(WriterState::Finalized);
        cleanup
    }
}

impl Drop for TransactionWriter {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap().take();
        let Some(state) = state else {
            return;
        };
        match state {
            WriterState::Idle | WriterState::Finalized => {}
            WriterState::Completed(result) => {
                let _ = std::thread::Builder::new()
                    .name("pivot-insert-cleanup".to_string())
                    .spawn(move || {
                        let _ = delete_pending_files(result);
                    });
            }
            WriterState::Running(writer) => {
                let _ = std::thread::Builder::new()
                    .name("pivot-insert-cleanup".to_string())
                    .spawn(move || {
                        drop(writer.sender);
                        if let Ok(result) = writer.join.join() {
                            let _ = delete_pending_files(result);
                        }
                    });
            }
            WriterState::Finishing => {}
        }
    }
}

/// Run on `pivot-insert-writer`: persist encoded bytes and retain the exact
/// file entries that commit will publish or rollback will delete.
fn write_files(
    snapshot: Arc<CatalogSnapshot>,
    receiver: mpsc::Receiver<WriteCommand>,
) -> WriterResult {
    let mut result = WriterResult::default();
    for command in receiver {
        if result.error.is_some() {
            continue;
        }
        let Some(snapshot_table) = snapshot.tables.get(&command.table) else {
            result.error = Some(Error::InsertTableMissing(command.table));
            continue;
        };
        let pending = result
            .tables
            .entry(command.table)
            .or_insert_with(|| PendingTable {
                table: snapshot_table.clone(),
                entries: Vec::new(),
            });
        let path = ObjectPath::new(format!("pivot-{}.parquet", uuid::Uuid::new_v4()));
        match pending.table.write_data_file(path, &command.encoded.bytes) {
            Ok(file) => pending.entries.push(ManifestEntry {
                file,
                partition: command.encoded.partition,
                sort_bounds: command.encoded.sort_bounds,
            }),
            Err(error) => result.error = Some(error),
        }
    }
    result
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
