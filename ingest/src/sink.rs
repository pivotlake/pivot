//! [`ParquetSink`]: buffer Arrow [`RecordBatch`]es for one logical stream and,
//! once enough rows accumulate (or a timer fires), encode them into a single
//! Parquet file and write it to the configured [`SinkDestination`].
//!
//! Work is split by where it belongs:
//!
//! - **Encoding** (column encoding + Snappy compression) is CPU-bound, so it
//!   runs on a **dispatch worker** via [`DataFlowDispatcher::run_on_worker`].
//!   This is the whole point of running ingest inside the server: the heavy
//!   lifting lands on the same thread-per-core pool that runs queries.
//! - **Writing the bytes** is I/O — a local file write or a network upload to
//!   object storage — so it happens on the async runtime, never on a worker.
//!   Blocking a pinned worker on a network round-trip would stall queries.
//!
//! The sink is signal-agnostic: it only ever sees `RecordBatch`es. Each OTLP
//! service flattens its payload into a batch and hands it over. One flush
//! produces one Parquet file.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use arrow_array::RecordBatch;
use dispatch::{DataFlowDispatcher, values_input};
use object_store::aws::AmazonS3Builder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::path::Path as StorePath;
use object_store::{ObjectStore, PutPayload};
use tokio::sync::Mutex;
use tracing::{error, info};

use crate::parquet_writer::{EncodedPage, assemble_parquet, build_page_jobs, encode_page};

/// Subdirectory (inside a local sink directory) where files are written before
/// being atomically renamed into place. It is a directory, so
/// `ParquetTable::from_directory` — which only looks at *files* in the top
/// level — never tries to parse a half-written file.
const INFLIGHT_DIR: &str = ".inflight";

/// Where a sink writes its Parquet files.
#[derive(Debug, Clone)]
pub enum SinkDestination {
    /// A local directory. Files land flat in it, so the stream is queryable
    /// with `CREATE TABLE ... WITH (path = '<dir>')`.
    Local(PathBuf),
    /// An object-store URL (`gs://bucket/prefix` or `s3://bucket/prefix`).
    ///
    /// **Write-only**: pivot's reader only opens local directories, so object
    /// storage is an export target — read it elsewhere (e.g. DuckDB
    /// `read_parquet`). Credentials come from the environment
    /// (`GOOGLE_APPLICATION_CREDENTIALS`, `AWS_*`, workload identity, …).
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

/// The resolved write target for a sink.
enum Backend {
    Local {
        dir: PathBuf,
        inflight: PathBuf,
    },
    Remote {
        store: Arc<dyn ObjectStore>,
        prefix: StorePath,
        /// Original URL, for human-readable log lines.
        url: String,
    },
}

impl Backend {
    /// Resolve a [`SinkDestination`] into a usable backend, creating local
    /// directories or building the object-store client up front so
    /// misconfiguration is reported at startup.
    fn build(dest: &SinkDestination) -> std::io::Result<Self> {
        match dest {
            SinkDestination::Local(dir) => {
                let inflight = dir.join(INFLIGHT_DIR);
                std::fs::create_dir_all(&inflight)?;
                Ok(Backend::Local {
                    dir: dir.clone(),
                    inflight,
                })
            }
            SinkDestination::Remote(url) => {
                let (store, prefix) = build_object_store(url).map_err(std::io::Error::other)?;
                Ok(Backend::Remote {
                    store,
                    prefix,
                    url: url.clone(),
                })
            }
        }
    }

    /// Write `bytes` as `file_name`, returning a printable location for logs.
    /// Local writes go to the `.inflight` dir then atomically rename into place
    /// (so a concurrent reader never sees a partial file); remote writes upload
    /// directly.
    async fn put(&self, file_name: &str, bytes: Vec<u8>) -> Result<String, String> {
        match self {
            Backend::Local { dir, inflight } => {
                let final_path = dir.join(file_name);
                let inflight_path = inflight.join(file_name);
                let dest = final_path.clone();
                tokio::task::spawn_blocking(move || -> Result<(), String> {
                    std::fs::write(&inflight_path, &bytes)
                        .map_err(|e| format!("write {}: {e}", inflight_path.display()))?;
                    std::fs::rename(&inflight_path, &dest)
                        .map_err(|e| format!("rename into {}: {e}", dest.display()))?;
                    Ok(())
                })
                .await
                .map_err(|e| format!("blocking write task: {e}"))??;
                Ok(final_path.display().to_string())
            }
            Backend::Remote { store, prefix, url } => {
                let path = prefix.child(file_name);
                store
                    .put(&path, PutPayload::from(bytes))
                    .await
                    .map_err(|e| format!("object store put: {e}"))?;
                Ok(format!("{}/{}", url.trim_end_matches('/'), file_name))
            }
        }
    }
}

/// Build an [`ObjectStore`] and base prefix from a `gs://` / `s3://` URL.
/// Credentials are read from the environment.
fn build_object_store(url: &str) -> Result<(Arc<dyn ObjectStore>, StorePath), String> {
    if let Some(rest) = url.strip_prefix("gs://") {
        let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
        let store = GoogleCloudStorageBuilder::from_env()
            .with_bucket_name(bucket)
            .build()
            .map_err(|e| format!("building GCS store for `{bucket}`: {e}"))?;
        Ok((Arc::new(store), StorePath::from(prefix)))
    } else if let Some(rest) = url
        .strip_prefix("s3://")
        .or_else(|| url.strip_prefix("s3a://"))
    {
        let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
        let store = AmazonS3Builder::from_env()
            .with_bucket_name(bucket)
            .build()
            .map_err(|e| format!("building S3 store for `{bucket}`: {e}"))?;
        Ok((Arc::new(store), StorePath::from(prefix)))
    } else {
        Err(format!("unsupported object-store url `{url}`"))
    }
}

/// Buffers rows for one stream and flushes them to Parquet files. Cheap to
/// share behind an `Arc`; all mutable state lives behind a `Mutex` so the gRPC
/// handlers and the flush timer can both append.
pub struct ParquetSink {
    /// Used as the file-name prefix and in log lines (e.g. `otel_logs`).
    name: String,
    backend: Backend,
    /// Flush once the buffer holds at least this many rows.
    flush_rows: usize,
    dispatcher: DataFlowDispatcher,
    buffer: Mutex<Buffer>,
    /// Monotonic file sequence, so two flushes in the same millisecond don't
    /// collide on a filename.
    seq: AtomicU64,
}

#[derive(Default)]
struct Buffer {
    batches: Vec<RecordBatch>,
    rows: usize,
}

impl ParquetSink {
    /// Create a sink writing to `destination` (local dirs are created up front;
    /// object-store clients are built and validated here too).
    pub fn new(
        name: impl Into<String>,
        destination: &SinkDestination,
        flush_rows: usize,
        dispatcher: DataFlowDispatcher,
    ) -> std::io::Result<Self> {
        Ok(Self {
            name: name.into(),
            backend: Backend::build(destination)?,
            flush_rows: flush_rows.max(1),
            dispatcher,
            buffer: Mutex::new(Buffer::default()),
            seq: AtomicU64::new(0),
        })
    }

    /// Append a batch, flushing eagerly if the row threshold is crossed. Empty
    /// batches are ignored.
    pub async fn append(&self, batch: RecordBatch) {
        if batch.num_rows() == 0 {
            return;
        }
        let drained = {
            let mut buf = self.buffer.lock().await;
            buf.rows += batch.num_rows();
            buf.batches.push(batch);
            if buf.rows >= self.flush_rows {
                Some(std::mem::take(&mut *buf))
            } else {
                None
            }
        };
        if let Some(buf) = drained {
            self.write(buf.batches).await;
        }
    }

    /// Flush whatever is currently buffered (timer tick / shutdown drain).
    pub async fn flush_now(&self) {
        let drained = {
            let mut buf = self.buffer.lock().await;
            std::mem::take(&mut *buf)
        };
        if !drained.batches.is_empty() {
            self.write(drained.batches).await;
        }
    }

    /// Encode `batches` into one Parquet file, **parallelizing the encode at
    /// the page level across the dispatch worker pool**, then write the file to
    /// the destination.
    ///
    /// Each `(row group, column)` becomes one page job; the jobs fan out via a
    /// work-stealing source ([`values_input`]) and each is PLAIN-encoded +
    /// snappy-compressed on whatever worker steals it (CPU, on the pool). The
    /// encoded pages are stitched into a single file (cheap, serial) and the
    /// bytes written/uploaded here on the async runtime (I/O). A failed
    /// encode/write is logged and dropped rather than blocking the pipeline —
    /// the drop-under-pressure stance OTLP exporters expect.
    async fn write(&self, batches: Vec<RecordBatch>) {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let file_name = format!("{}-{}-{:06}.parquet", self.name, millis, seq);

        let total_rows: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
        let schema = batches[0].schema();

        // Page-level parallel encode on the worker pool, then serial assembly.
        // `collect()` blocks, so park the whole thing on a blocking thread.
        let dispatcher = self.dispatcher.clone();
        let encoded = tokio::task::spawn_blocking(move || -> Result<Vec<u8>, String> {
            // Concatenate each column and cut ~1 MiB pages (one row group).
            let jobs = build_page_jobs(&batches)?;
            drop(batches); // the page jobs hold Arc clones of the columns
            let pages = values_input(&dispatcher, jobs)
                .map_each(encode_page)
                .collect()
                .map_err(|e| format!("encode dataflow failed: {e}"))?;
            let pages: Vec<EncodedPage> = pages.into_iter().collect::<Result<Vec<_>, _>>()?;
            assemble_parquet(&schema, total_rows as i64, pages)
        })
        .await;

        let bytes = match encoded {
            Ok(Ok(bytes)) => bytes,
            Ok(Err(e)) => {
                error!(sink = %self.name, error = %e, "parquet encode failed");
                return;
            }
            Err(e) => {
                error!(sink = %self.name, error = %e, "parquet encode task panicked");
                return;
            }
        };

        // I/O on the async runtime (local write or object-store upload).
        match self.backend.put(&file_name, bytes).await {
            Ok(location) => {
                info!(sink = %self.name, file = %location, rows = total_rows, "flushed parquet")
            }
            Err(e) => {
                error!(sink = %self.name, file = %file_name, error = %e, "failed writing parquet")
            }
        }
    }
}
