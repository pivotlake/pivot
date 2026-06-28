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

use catalog::ParquetCatalog;
use dispatch::DataFlowDispatcher;
use tokio::sync::Mutex;
use tracing::warn;

use crate::parquet_writing::ToRecordBatch;
use crate::write::encode_and_append;

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

    /// Run `items` through the shared write path
    /// ([`encode_and_append`](crate::write::encode_and_append)) and write each
    /// Parquet file it produces into the sink's catalog table.
    ///
    /// A failed flush is logged and dropped rather than blocking — the
    /// drop-under-pressure stance OTLP exporters expect. (The Kafka source uses
    /// the same write path but acts on the error, gating its offset commit.)
    async fn write(&self, items: Vec<T>) {
        if let Err(e) = encode_and_append(&self.catalog, &self.name, &self.dispatcher, items).await {
            warn!(sink = %self.name, error = %e, "flush failed; dropping");
        }
    }
}
