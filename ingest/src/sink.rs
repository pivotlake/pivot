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
use dispatch::{DataFlowDispatcher, DataFlowError};
use tokio::sync::Mutex;
use tracing::{error, info, warn};

use crate::parquet_writing::{self, EncodedFile, ToRecordBatch};

/// Cumulative, lock-free counters for one sink's flush activity - read by the
/// introspection API to show live ingest throughput per table.
#[derive(Default)]
pub(crate) struct SinkStats {
    rows: AtomicU64,
    bytes: AtomicU64,
    files: AtomicU64,
    flushes: AtomicU64,
    last_flush_unix_ms: AtomicU64,
}

/// A point-in-time read of a sink's `SinkStats`, tagged with its table.
#[derive(Debug, Clone)]
pub struct SinkStatsSnapshot {
    pub table: String,
    pub rows: u64,
    pub bytes: u64,
    pub files: u64,
    pub flushes: u64,
    /// Unix-epoch milliseconds of the last committed file, or 0 if none yet.
    pub last_flush_unix_ms: u64,
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Rows per Parquet row group. A flush's items are cut into row groups of about
/// this many rows as they stream through the write pipeline.
pub(crate) const ROW_GROUP_ROWS: usize = 128 * 1024;
/// Row groups per Parquet file. Once this many accumulate, a file is emitted and
/// written; the flush's remainder lands in a final, smaller file.
pub(crate) const ROW_GROUPS_PER_FILE: usize = 8;
/// Finished files buffered between the encode pipeline and the writer: small, so
/// it bounds in-flight files and backpressures the pipeline onto the writer.
const IN_FLIGHT_FILES: usize = 4;

/// Object-safe view of a sink for the lifecycle paths (flush timer, shutdown
/// drain), which don't care about the concrete item type. `tonic::async_trait`
/// is reused only to get an async method on a `dyn` trait.
#[tonic::async_trait]
pub(crate) trait Flushable: Send + Sync {
    async fn flush(&self);

    /// A snapshot of the sink's cumulative flush counters, for introspection.
    fn stats(&self) -> SinkStatsSnapshot;
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
    stats: SinkStats,
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
            stats: SinkStats::default(),
        }
    }

    /// A snapshot of this sink's cumulative flush counters.
    pub(crate) fn stats_snapshot(&self) -> SinkStatsSnapshot {
        SinkStatsSnapshot {
            table: self.name.clone(),
            rows: self.stats.rows.load(Ordering::Relaxed),
            bytes: self.stats.bytes.load(Ordering::Relaxed),
            files: self.stats.files.load(Ordering::Relaxed),
            flushes: self.stats.flushes.load(Ordering::Relaxed),
            last_flush_unix_ms: self.stats.last_flush_unix_ms.load(Ordering::Relaxed),
        }
    }
}

#[tonic::async_trait]
impl<T: ToRecordBatch> Flushable for ParquetSink<T> {
    async fn flush(&self) {
        self.flush_now().await;
    }

    fn stats(&self) -> SinkStatsSnapshot {
        self.stats_snapshot()
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
        // Count rows as they arrive (not at flush), so the ingest rate the
        // dashboard derives is continuous rather than a step at each flush.
        self.stats.rows.fetch_add(rows as u64, Ordering::Relaxed);
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
    /// The pipeline (see [`parquet_writing`]) flattens items into Arrow batches,
    /// then partitions, encodes, and assembles them into Parquet files across the
    /// dispatch worker pool, streaming finished files out as they complete.
    /// Iterating that stream blocks, so it runs on a blocking thread; each
    /// finished file is handed to the async runtime, which writes/uploads it (I/O)
    /// while the pipeline keeps working.
    /// A failed pipeline run or write is logged and dropped rather than blocking
    /// — the drop-under-pressure stance OTLP exporters expect.
    async fn write(&self, items: Vec<T>) {
        // Rows are counted on arrival (see `append`); here we only tally the
        // flush itself. Bytes/files are tallied per file as each one commits.
        self.stats.flushes.fetch_add(1, Ordering::Relaxed);

        // The table carries the partition/sort spec; read it per flush (the table
        // may have been dropped out from under us).
        let (partition_by, sort_by) = match self.catalog.table_handle(&self.name) {
            Some(table) => (table.partition_by().to_vec(), table.sort_by().to_vec()),
            None => {
                warn!(sink = %self.name, "table no longer exists; dropping flush");
                return;
            }
        };

        let dispatcher = self.dispatcher.clone();
        // Each finished file streams out as an `EncodedFile` (bytes + the metadata
        // to record).
        let (tx, mut rx) = tokio::sync::mpsc::channel::<EncodedFile>(IN_FLIGHT_FILES);

        // One pipeline, parameterised by the table's spec: it splits by partition,
        // files per partition, and tags each `EncodedFile` with its tuple +
        // sort_bounds (both `None` when the spec is empty).
        let pipeline = tokio::task::spawn_blocking(move || -> Result<(), DataFlowError> {
            for file in parquet_writing::encode_items(
                &dispatcher,
                items,
                partition_by.into(),
                sort_by.into(),
                ROW_GROUP_ROWS,
                ROW_GROUPS_PER_FILE,
            ) {
                if tx.blocking_send(file?).is_err() {
                    return Ok(());
                }
            }
            Ok(())
        });

        // Write each file as it streams out, under an opaque uuid name so
        // concurrent ingestors never collide.
        while let Some(encoded) = rx.recv().await {
            let file_name = format!("pivot-{}.parquet", uuid::Uuid::new_v4());
            self.write_file(file_name, encoded).await;
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
    async fn write_file(&self, file_name: String, encoded: EncodedFile) {
        let byte_len = encoded.bytes.len() as u64;
        let catalog = self.catalog.clone();
        let table = self.name.clone();
        let path = ObjectPath::new(file_name.clone());
        let appended = tokio::task::spawn_blocking(move || {
            catalog.table_handle(&table).map(|mut handle| {
                handle
                    .append_data_file(path, &encoded.bytes, encoded.partition, encoded.sort_bounds)
                    .map(|()| {
                        // Publish the committed copy so the very next query's
                        // snapshot sees the rows, without waiting for the
                        // background catalog refresh.
                        catalog.publish_table(handle);
                    })
            })
        })
        .await;
        match appended {
            Ok(Some(Ok(()))) => {
                self.stats.bytes.fetch_add(byte_len, Ordering::Relaxed);
                self.stats.files.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .last_flush_unix_ms
                    .store(now_unix_ms(), Ordering::Relaxed);
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
