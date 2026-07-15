//! INSERT's terminal upload sink.
//!
//! The write pipeline's last stage: it consumes each finished [`EncodedFile`],
//! uploads its bytes over the dispatch worker's io_uring (a local file write or a
//! single HTTP PUT/POST, exactly like the read path but outbound), and records
//! the uploaded file's [`ManifestEntry`] into the transaction's
//! [`InsertCollector`]. Once every upload a worker issued has landed, its sink
//! emits the one-row `count` `RecordBatch` that becomes the INSERT command tag
//! (emitted once across all workers, guarded by a shared flag).
//!
//! This replaces the old dedicated writer thread that did blocking `store.put`s:
//! the uploads now ride the same ring as reads, and there is no object-store I/O
//! off it.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use dispatch::io::{RemoteUpload, UploadDest, UploadMethod, UploadRequest};
use dispatch::{Sender, Unary, UnaryError, UnaryFactory, UnaryResult};

use crate::manifest::ManifestEntry;
use crate::parquet_writing::EncodedFile;
use crate::store::{FileRef, ObjectPath, UploadMethod as StoreUploadMethod, UploadTarget};

use super::CatalogTable;
use super::transaction::InsertCollector;

/// Files one worker's sink buffers and uploads at once. Bounds the encoded bytes
/// held in memory per worker while still overlapping several uploads; the
/// worker's own `PIVOT_UPLOAD_INFLIGHT` caps the ring's depth across all sinks.
const MAX_INFLIGHT: usize = 8;

/// Map any catalog/IO error into the out-of-crate operator error arm.
fn op_err(error: impl std::fmt::Display) -> UnaryError {
    UnaryError::Operator(Box::new(std::io::Error::other(error.to_string())))
}

/// Builds one [`UploadSink`] per worker, each sharing the transaction's
/// collector, the running row tally, and the one-shot count-emitted flag.
pub(super) struct UploadSinkFactory {
    table: CatalogTable,
    table_name: String,
    collector: Arc<InsertCollector>,
    rows_written: Arc<AtomicU64>,
    count_emitted: Arc<AtomicBool>,
}

impl UploadSinkFactory {
    /// One factory per worker for the INSERT into `table_name`. `rows_written` is
    /// the tally the conform stage accumulates; `collector` gathers uploaded
    /// files for commit.
    pub(super) fn factories(
        count: usize,
        table: CatalogTable,
        table_name: String,
        collector: Arc<InsertCollector>,
        rows_written: Arc<AtomicU64>,
    ) -> Vec<UploadSinkFactory> {
        let count_emitted = Arc::new(AtomicBool::new(false));
        (0..count)
            .map(|_| UploadSinkFactory {
                table: table.clone(),
                table_name: table_name.clone(),
                collector: collector.clone(),
                rows_written: rows_written.clone(),
                count_emitted: count_emitted.clone(),
            })
            .collect()
    }
}

impl UnaryFactory<EncodedFile, RecordBatch> for UploadSinkFactory {
    type Unary = UploadSink;

    fn build_unary(self) -> UploadSink {
        UploadSink {
            table: self.table,
            table_name: self.table_name,
            collector: self.collector,
            rows_written: self.rows_written,
            count_emitted: self.count_emitted,
            pending: Vec::new(),
            entries: HashMap::new(),
            in_flight: 0,
            next_token: 0,
        }
    }
}

/// One worker's upload sink. Turns each [`EncodedFile`] into an upload, hands the
/// uploads to the worker's ring, and commits each file's [`ManifestEntry`] to the
/// collector as it lands.
pub(super) struct UploadSink {
    table: CatalogTable,
    table_name: String,
    collector: Arc<InsertCollector>,
    rows_written: Arc<AtomicU64>,
    count_emitted: Arc<AtomicBool>,
    /// Uploads built from consumed files, not yet handed to the worker.
    pending: Vec<UploadRequest>,
    /// The manifest entry for each in-flight upload, keyed by its token, moved to
    /// the collector once the upload lands.
    entries: HashMap<u64, ManifestEntry>,
    /// Uploads handed to the worker but not yet completed.
    in_flight: usize,
    /// Per-sink token, matching an upload to its staged entry on completion.
    next_token: u64,
}

impl UploadSink {
    /// Turn a store upload target into a ring upload destination, opening the
    /// local file (creating parent dirs) or resolving the remote endpoint.
    fn build_dest(target: UploadTarget) -> std::io::Result<UploadDest> {
        match target {
            UploadTarget::Local(path) => {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let file = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(&path)?;
                Ok(UploadDest::Local(Arc::new(file)))
            }
            UploadTarget::Remote {
                url,
                method,
                auth,
                content_type,
            } => {
                let method = match method {
                    StoreUploadMethod::Put => UploadMethod::Put,
                    StoreUploadMethod::Post => UploadMethod::Post,
                };
                let remote = RemoteUpload::open(url, method, auth, content_type)?;
                Ok(UploadDest::Remote(Arc::new(remote)))
            }
        }
    }
}

impl Unary<EncodedFile, RecordBatch> for UploadSink {
    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        encoded: EncodedFile,
        _sender: &mut S,
    ) -> UnaryResult<()> {
        let path = ObjectPath::new(format!("pivot-{}.parquet", uuid::Uuid::new_v4()));
        let target = self.table.upload_target(&path).map_err(op_err)?;
        let dest = Self::build_dest(target).map_err(op_err)?;

        let bytes: Arc<[u8]> = Arc::from(encoded.bytes);
        let size = bytes.len() as u64;
        let token = self.next_token;
        self.next_token += 1;

        self.entries.insert(
            token,
            ManifestEntry {
                file: FileRef { path, size },
                partition: encoded.partition,
                sort_bounds: encoded.sort_bounds,
            },
        );
        self.pending.push(UploadRequest { dest, bytes, token });
        Ok(())
    }

    fn next_upload_requests(&mut self) -> UnaryResult<Vec<UploadRequest>> {
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }
        let requests = std::mem::take(&mut self.pending);
        self.in_flight += requests.len();
        Ok(requests)
    }

    fn process_upload_response<S: Sender<RecordBatch>>(
        &mut self,
        _sender: &mut S,
        request: UploadRequest,
    ) -> UnaryResult<()> {
        let entry = self
            .entries
            .remove(&request.token)
            .expect("upload completed for an unknown token");
        self.collector.record(self.table_name.clone(), entry);
        self.in_flight -= 1;
        Ok(())
    }

    /// Backpressure the encode pipeline once this worker is holding a full batch
    /// of uploads (pending + in flight), so encoded bytes don't pile up unbounded.
    fn ready_for_more_work(&mut self) -> bool {
        self.pending.len() + self.in_flight < MAX_INFLIGHT
    }

    fn finish<S: Sender<RecordBatch>>(&mut self, sender: &mut S) -> UnaryResult<bool> {
        // Not done until every upload this worker issued has landed (and its entry
        // is in the collector); the worker keeps draining the ring meanwhile.
        if !self.pending.is_empty() || self.in_flight > 0 {
            return Ok(false);
        }
        // Emit the row count exactly once across all workers. By now every sibling
        // has drained its input, so `rows_written` is final.
        if !self.count_emitted.swap(true, Ordering::Relaxed) {
            let count = UInt64Array::from(vec![self.rows_written.load(Ordering::Relaxed)]);
            let schema = Schema::new(vec![Field::new("count", DataType::UInt64, false)]);
            let batch =
                RecordBatch::try_new(Arc::new(schema), vec![Arc::new(count)]).map_err(op_err)?;
            sender.send(batch)?;
        }
        Ok(true)
    }
}
