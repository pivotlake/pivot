//! Transaction-owned Parquet writes for INSERT.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{JoinHandle, Thread};

use crate::manifest::ManifestEntry;
use crate::parquet_writing::EncodedFile;
use crate::store::ObjectPath;
use crossbeam_deque::{Injector, Steal};

use super::{CatalogSnapshot, CatalogTable, Error, Result};

struct WriteCommand {
    table: String,
    encoded: EncodedFile,
}

struct WriteQueue {
    commands: Injector<WriteCommand>,
    producers: AtomicUsize,
}

impl WriteQueue {
    fn new() -> Self {
        Self {
            commands: Injector::new(),
            producers: AtomicUsize::new(1),
        }
    }
}

/// A worker-local producer for the transaction's Parquet writer queue.
///
/// Submitting a file does not acquire the transaction lifecycle mutex. The
/// final producer to drop wakes the writer so it can drain and exit.
pub(in crate::catalog) struct WriteSender {
    queue: Arc<WriteQueue>,
    writer: Thread,
}

impl Clone for WriteSender {
    fn clone(&self) -> Self {
        self.queue.producers.fetch_add(1, Ordering::Relaxed);
        Self {
            queue: self.queue.clone(),
            writer: self.writer.clone(),
        }
    }
}

impl WriteSender {
    /// Enqueue an encoded file and wake the blocking writer thread.
    pub(in crate::catalog) fn submit(&self, table: String, encoded: EncodedFile) {
        self.queue.commands.push(WriteCommand { table, encoded });
        self.writer.unpark();
    }
}

impl Drop for WriteSender {
    fn drop(&mut self) {
        if self.queue.producers.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.writer.unpark();
        }
    }
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
    sender: WriteSender,
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
/// INSERT setup obtains a [`WriteSender`] while holding the lifecycle mutex.
/// Dispatch workers then clone that producer and submit directly to a shared
/// lock-free queue. The dedicated writer performs all blocking object-store
/// calls. Commit and rollback join it from the server's blocking pool after the
/// dataflow has drained.
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

    /// Obtain a producer for an INSERT pipeline.
    ///
    /// This is the only lifecycle-mutex acquisition needed by that pipeline.
    /// Its dispatch workers clone the returned producer and enqueue files
    /// without returning to the transaction state.
    pub(super) fn write_sender(&self) -> Result<WriteSender> {
        let mut locked = self.state.lock().unwrap();
        let state = locked.take().ok_or(Error::InsertWriterUnavailable)?;
        let writer = match state {
            WriterState::Idle => match self.start_writer() {
                Ok(writer) => writer,
                Err(error) => {
                    *locked = Some(WriterState::Idle);
                    return Err(error);
                }
            },
            WriterState::Running(writer) => writer,
            other => {
                *locked = Some(other);
                return Err(Error::InsertWriterUnavailable);
            }
        };
        let sender = writer.sender.clone();
        *locked = Some(WriterState::Running(writer));
        Ok(sender)
    }

    /// Start the transaction's single blocking-I/O thread and shared queue.
    fn start_writer(&self) -> Result<WriterThread> {
        let queue = Arc::new(WriteQueue::new());
        let snapshot = self.snapshot.clone();
        let writer_queue = queue.clone();
        let join = std::thread::Builder::new()
            .name("pivot-insert-writer".to_string())
            .spawn(move || write_files(snapshot, writer_queue))?;
        let sender = WriteSender {
            queue,
            writer: join.thread().clone(),
        };
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
fn write_files(snapshot: Arc<CatalogSnapshot>, queue: Arc<WriteQueue>) -> WriterResult {
    let mut result = WriterResult::default();
    while let Some(command) = next_write(&queue) {
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

/// Wait for the next queued file, or finish after the final producer drops and
/// all previously submitted files have been drained.
fn next_write(queue: &WriteQueue) -> Option<WriteCommand> {
    loop {
        match queue.commands.steal() {
            Steal::Success(command) => return Some(command),
            Steal::Retry => continue,
            Steal::Empty if queue.producers.load(Ordering::Acquire) == 0 => {
                // A final producer may have pushed between the first steal and
                // our producer-count load. With no producers left, a second
                // empty result is stable and means the queue is fully drained.
                match queue.commands.steal() {
                    Steal::Success(command) => return Some(command),
                    Steal::Retry => continue,
                    Steal::Empty => return None,
                }
            }
            Steal::Empty => std::thread::park(),
        }
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
