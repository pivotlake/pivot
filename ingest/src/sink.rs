//! [`ParquetSink<T>`]: buffer raw items for one logical stream and, once enough
//! rows accumulate (or a timer fires), turn them into a single Parquet file and
//! write it to the configured [`SinkDestination`].
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
//! - **Writing the bytes** is I/O — a local file write or a network upload to
//!   object storage — so it happens on the async runtime, never on a worker.
//!   Blocking a pinned worker on a network round-trip would stall queries.
//!
//! One flush produces one Parquet file.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use dispatch::DataFlowDispatcher;
use goose::ParquetCatalog;
use goose::store::{ObjectPath, ObjectStore, open_store};
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::parquet_writing::{self, ToRecordBatch};

/// Rows per Parquet row group. A flush's items are cut into row groups of about
/// this many rows as they stream through the write pipeline.
pub(crate) const ROW_GROUP_ROWS: usize = 128 * 1024;
/// Row groups per Parquet file. Once this many accumulate, a file is emitted and
/// written; the flush's remainder lands in a final, smaller file.
pub(crate) const ROW_GROUPS_PER_FILE: usize = 8;

/// Where a sink writes its Parquet files.
#[derive(Debug, Clone)]
pub enum SinkDestination {
    /// A local directory. Files land flat in it, so the stream is queryable
    /// with `CREATE TABLE ... WITH (path = '<dir>')`.
    Local(PathBuf),
    /// An object-store URL (`gs://bucket/prefix` or `s3://bucket/prefix`).
    ///
    /// **Write-only export**: the files are written through goose's store but
    /// not registered with the catalog (the catalog's own database store, not
    /// this sink's, is what reads a table — they only align for a local sink).
    /// Credentials come from the environment (`GOOGLE_APPLICATION_CREDENTIALS`,
    /// `AWS_*`, workload identity, …).
    Remote(String),
}

impl SinkDestination {
    /// Parse a destination string: `gs://…` / `s3://…` select object storage,
    /// anything else (optionally `file://`-prefixed) is a local directory.
    pub fn parse(s: &str) -> Self {
        if s.starts_with("gs://") || s.starts_with("s3://") || s.starts_with("s3a://") {
            SinkDestination::Remote(s.to_string())
        } else {
            let local = s.strip_prefix("file://").unwrap_or(s);
            SinkDestination::Local(PathBuf::from(local))
        }
    }
}

/// The resolved write target for a sink: a goose [`ObjectStore`] rooted at the
/// destination (a local dir, S3, or GCS — built once so misconfiguration
/// surfaces at startup) plus a human-readable form for logs. The same store
/// stack the catalog reads through, so a flushed file is written and registered
/// through one mechanism. `registers` is set for a **local** destination (whose
/// files the catalog can read back and so register for immediate visibility); a
/// remote destination is a write-only export.
struct Backend {
    store: Arc<dyn ObjectStore>,
    registers: bool,
    /// The destination, for log lines.
    display: String,
}

impl Backend {
    /// Resolve a [`SinkDestination`] into a goose store, creating a local
    /// directory or building/validating the remote client up front so
    /// misconfiguration is reported at startup.
    fn build(dest: &SinkDestination) -> std::io::Result<Self> {
        let (uri, registers, display) = match dest {
            SinkDestination::Local(dir) => {
                // Create the directory now so a bad path is reported at startup.
                std::fs::create_dir_all(dir)?;
                let uri = dir.to_str().ok_or_else(|| {
                    std::io::Error::other(format!("non-UTF-8 sink path {}", dir.display()))
                })?;
                (uri.to_string(), true, dir.display().to_string())
            }
            SinkDestination::Remote(url) => (url.clone(), false, url.clone()),
        };
        let store: Arc<dyn ObjectStore> = open_store(&uri).map_err(std::io::Error::other)?.into();
        Ok(Self {
            store,
            registers,
            display,
        })
    }

    /// Write `bytes` as `file_name` directly under the destination. goose's
    /// store does the atomic write (a sibling temp object renamed into place),
    /// so a concurrent reader never sees a half-written `.parquet`. The store is
    /// synchronous, so the write runs on a blocking thread.
    async fn put(&self, file_name: &str, bytes: Vec<u8>) -> Result<(), String> {
        let store = self.store.clone();
        let path = ObjectPath::new(file_name);
        tokio::task::spawn_blocking(move || store.put(&path, &bytes))
            .await
            .map_err(|e| format!("blocking write task: {e}"))?
            .map_err(|e| format!("object store put: {e}"))
    }
}

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
    /// Used as the file-name prefix, in log lines, **and as the catalog table
    /// name** each flushed file is registered under (e.g. `otel_logs`).
    name: String,
    backend: Backend,
    /// Flush once the buffer holds at least this many rows.
    flush_rows: usize,
    dispatcher: DataFlowDispatcher,
    /// Every locally-written file is registered with the table named
    /// [`name`](Self::name), so its rows are queryable immediately — no
    /// re-`CREATE TABLE` needed. (Remote destinations are write-only exports
    /// and skip registration; a missing table just defers visibility to its
    /// `CREATE TABLE`.)
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
    /// Create a sink writing to `destination` (local dirs are created up front;
    /// object-store clients are built and validated here too).
    pub fn new(
        name: impl Into<String>,
        destination: &SinkDestination,
        flush_rows: usize,
        dispatcher: DataFlowDispatcher,
        catalog: Arc<ParquetCatalog>,
    ) -> std::io::Result<Self> {
        Ok(Self {
            name: name.into(),
            backend: Backend::build(destination)?,
            flush_rows: flush_rows.max(1),
            dispatcher,
            catalog,
            buffer: Mutex::new(Buffer::default()),
            seq: AtomicU64::new(0),
        })
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
            match self.backend.put(&file_name, bytes).await {
                Ok(()) => {
                    info!(sink = %self.name, dest = %self.backend.display, file = file_name, "flushed parquet");
                    if self.backend.registers {
                        self.register(&file_name).await;
                    }
                }
                Err(e) => {
                    error!(sink = %self.name, file = file_name, error = %e, "failed writing parquet")
                }
            }
        }

        match pipeline.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => error!(sink = %self.name, error = %e, "parquet write pipeline failed"),
            Err(e) => error!(sink = %self.name, error = %e, "write pipeline task panicked"),
        }
    }

    /// Register a freshly-written file (`file_name`, relative to the table's
    /// data location) with the catalog table named after this sink, making its
    /// rows queryable at once. Runs on a blocking thread: the registration reads
    /// the file's footer by driving a dataflow (the decode itself lands on the
    /// dispatch workers), which blocks the calling thread. A failure only delays
    /// visibility — the file is in the table's directory, so the next
    /// `CREATE TABLE`/restart still finds it — so it is logged, never propagated.
    async fn register(&self, file_name: &str) {
        let catalog = self.catalog.clone();
        let table = self.name.clone();
        let path = ObjectPath::new(file_name);
        let registered = tokio::task::spawn_blocking(move || {
            // A writer evolves a cloned-out table handle; `None` (no such table
            // yet) just defers visibility to the table's `CREATE TABLE`.
            catalog
                .table_handle(&table)
                .map(|mut handle| handle.register_data_file(path))
        })
        .await;
        let result = match registered {
            Ok(result) => result,
            Err(e) => {
                error!(sink = %self.name, error = %e, "catalog registration task panicked");
                return;
            }
        };
        match result {
            Some(Ok(())) => {
                info!(sink = %self.name, file = file_name, "registered parquet in catalog")
            }
            // Non-fatal: the file is on disk, so the next `CREATE TABLE`/restart
            // still finds it (e.g. a location mismatch, or a transient store error).
            Some(Err(e)) => {
                warn!(sink = %self.name, file = file_name, error = %e, "catalog registration failed")
            }
            None => {
                debug!(
                    sink = %self.name, file = file_name,
                    "no catalog table for sink; file becomes visible at CREATE TABLE"
                )
            }
        }
    }
}
