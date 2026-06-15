//! [`ParquetSink<T>`]: buffer raw items for one logical stream and, once enough
//! rows accumulate (or a timer fires), turn them into Parquet files and append
//! each to the sink's **catalog table** (writing into the table's own data
//! location, through catalog's store), making the rows queryable at once.
//!
//! The sink is signal-agnostic: it only knows `T: ToRecordBatch`. The receive
//! path (a gRPC handler) does **no** conversion — it just `append`s a cheap,
//! unconverted item. All the CPU work happens at flush, where it belongs:
//!
//! - **Conversion** (`T` → Arrow `RecordBatch`: id encoding, attribute JSON,
//!   column building) and **encoding** (column encode + Snappy compression) are
//!   both CPU-bound, so both fan out across the **dispatch worker pool** via
//!   work-stealing `values_input` sources — the same thread-per-core pool that
//!   runs queries, not the async runtime.
//! - **Appending the file** (the store write + the manifest commit) is I/O and
//!   runs on a blocking task, never on a worker — blocking a pinned worker on a
//!   round-trip would stall queries.
//!
//! One flush produces one or more Parquet files. The table the sink appends to
//! must already exist (checked when the sink is built); the sink only ever
//! *writes into* a table, it never creates one.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use catalog::ParquetCatalog;
use catalog::store::ObjectPath;
use dispatch::DataFlowDispatcher;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

use crate::parquet_writing::{self, ToRecordBatch};

/// Rows per Parquet row group. A flush's items are cut into row groups of about
/// this many rows as they stream through the write pipeline.
pub(crate) const ROW_GROUP_ROWS: usize = 128 * 1024;
/// Row groups per Parquet file. Once this many accumulate, a file is emitted and
/// written; the flush's remainder lands in a final, smaller file.
pub(crate) const ROW_GROUPS_PER_FILE: usize = 8;

/// Object-safe view of a sink for the lifecycle paths (flush timer, shutdown
/// drain), which don't care about the concrete item type. `tonic::async_trait`
/// is reused only to get an async method on a `dyn` trait.
#[tonic::async_trait]
pub(crate) trait Flushable: Send + Sync {
    async fn flush(&self);
}

/// Buffers items for one stream and flushes them to Parquet files. Cheap to
/// share behind an `Arc`; all mutable state lives behind a `Mutex` so the gRPC
/// handlers and the flush timer can both append.
pub struct ParquetSink<T> {
    /// The **catalog table** this sink appends to: its name is also the
    /// file-name prefix and the log identity (e.g. `otel_logs`). Each flush
    /// writes into this table's data location and commits the file.
    name: String,
    /// Flush once the buffer holds at least this many rows.
    flush_rows: usize,
    dispatcher: DataFlowDispatcher,
    /// The catalog the table lives in. The sink resolves [`name`](Self::name) to
    /// it on each flush to write into the table's location and commit the file;
    /// it never creates the table (existence is checked when the sink is built).
    catalog: Arc<ParquetCatalog>,
    buffer: Mutex<Buffer<T>>,
    /// Monotonic file sequence, so two flushes in the same millisecond don't
    /// collide on a filename.
    seq: AtomicU64,
}

struct Buffer<T> {
    items: Vec<T>,
    rows: usize,
}

// Manual impl: the derived one would add a spurious `T: Default` bound.
impl<T> Default for Buffer<T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            rows: 0,
        }
    }
}

impl<T> ParquetSink<T> {
    /// Create a sink that appends to the catalog table named `name`. The table
    /// must already exist in `catalog` — the caller checks that before building
    /// the sink (the sink only writes into a table, never creates one).
    pub fn new(
        name: impl Into<String>,
        flush_rows: usize,
        dispatcher: DataFlowDispatcher,
        catalog: Arc<ParquetCatalog>,
    ) -> Self {
        Self {
            name: name.into(),
            flush_rows: flush_rows.max(1),
            dispatcher,
            catalog,
            buffer: Mutex::new(Buffer::default()),
            seq: AtomicU64::new(0),
        }
    }
}

#[tonic::async_trait]
impl<T: ToRecordBatch> Flushable for ParquetSink<T> {
    async fn flush(&self) {
        self.flush_now().await;
    }
}

impl<T: ToRecordBatch> ParquetSink<T> {
    /// Append an item, flushing eagerly if the row threshold is crossed.
    /// Row-less items are ignored.
    pub async fn append(&self, item: T) {
        let rows = item.num_rows();
        if rows == 0 {
            return;
        }
        let drained = {
            let mut buf = self.buffer.lock().await;
            buf.rows += rows;
            buf.items.push(item);
            if buf.rows >= self.flush_rows {
                Some(std::mem::take(&mut *buf))
            } else {
                None
            }
        };
        if let Some(buf) = drained {
            self.write(buf.items).await;
        }
    }

    /// Flush whatever is currently buffered (timer tick / shutdown drain).
    pub async fn flush_now(&self) {
        let drained = {
            let mut buf = self.buffer.lock().await;
            std::mem::take(&mut *buf)
        };
        if !drained.items.is_empty() {
            self.write(drained.items).await;
        }
    }

    /// Run `items` through the write pipeline and write each Parquet file it
    /// produces.
    ///
    /// The pipeline (see [`parquet_writing`]) flattens items, cuts row groups,
    /// and encodes pages across the dispatch worker pool, streaming finished
    /// file buffers out as they complete. Iterating that stream blocks, so it
    /// runs on a blocking thread; each finished file is handed to the async
    /// runtime, which writes/uploads it (I/O) while the pipeline keeps working.
    /// A failed pipeline run or write is logged and dropped rather than blocking
    /// — the drop-under-pressure stance OTLP exporters expect.
    async fn write(&self, items: Vec<T>) {
        let dispatcher = self.dispatcher.clone();
        // Small buffer: bounds how many finished files sit in memory ahead of
        // the writer, applying backpressure to the pipeline's blocking drainer.
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(4);

        let pipeline = tokio::task::spawn_blocking(move || -> Result<(), String> {
            let files =
                parquet_writing::run(&dispatcher, items, ROW_GROUP_ROWS, ROW_GROUPS_PER_FILE);
            for file in files {
                let bytes = file.map_err(|e| format!("write pipeline failed: {e}"))?;
                // Receiver gone (shouldn't happen) — stop draining.
                if tx.blocking_send(bytes).is_err() {
                    break;
                }
            }
            Ok(())
        });

        // Write each file as it streams out.
        while let Some(bytes) = rx.recv().await {
            let seq = self.seq.fetch_add(1, Ordering::Relaxed);
            let millis = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);
            let file_name = format!("{}-{}-{:06}.parquet", self.name, millis, seq);
            self.write_file(file_name, bytes).await;
        }

        match pipeline.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => error!(sink = %self.name, error = %e, "parquet write pipeline failed"),
            Err(e) => error!(sink = %self.name, error = %e, "write pipeline task panicked"),
        }
    }

    /// Append one finished file (`file_name`, relative to the table's location)
    /// to the sink's catalog table: write the bytes into the table's location
    /// and commit the file, making its rows queryable at once. Runs on a
    /// blocking thread — the store write and the footer-reading commit dataflow
    /// both block. Failures are logged, never propagated (the drop-under-pressure
    /// stance OTLP exporters expect); a `None` means the table was dropped while
    /// ingest is running.
    async fn write_file(&self, file_name: String, bytes: Vec<u8>) {
        let catalog = self.catalog.clone();
        let table = self.name.clone();
        let path = ObjectPath::new(file_name.clone());
        let appended = tokio::task::spawn_blocking(move || {
            catalog
                .table_handle(&table)
                .map(|mut handle| handle.append_data_file(path, &bytes))
        })
        .await;
        match appended {
            Ok(Some(Ok(()))) => {
                info!(sink = %self.name, file = file_name, "appended parquet to table")
            }
            Ok(Some(Err(e))) => {
                warn!(sink = %self.name, file = file_name, error = %e, "appending parquet to table failed")
            }
            Ok(None) => {
                warn!(sink = %self.name, file = file_name, "table no longer exists; dropping flushed file")
            }
            Err(e) => {
                error!(sink = %self.name, file = file_name, error = %e, "parquet append task panicked")
            }
        }
    }
}
