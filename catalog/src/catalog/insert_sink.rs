//! The dispatch sink that uploads INSERT'd Parquet files over the io_uring ring.
//!
//! One [`InsertSink`] runs per worker as the terminal stage of the INSERT encode
//! pipeline. It consumes finished [`EncodedFile`]s, asks the table's store to
//! prepare a ring write for each (a local file to `write`, or a signed S3/GCS
//! request to `PUT`/`POST`), and yields those writes to the worker's I/O ring —
//! no byte transfer happens on the worker thread. As each write lands it records
//! the file's [`ManifestEntry`] against the transaction; once its input has
//! drained and every write has completed, one worker emits the inserted row count
//! (the INSERT's command tag).

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use dispatch::io::{FsWriteRequest, HttpWriteRequest};
use dispatch::{Sender, Unary, UnaryError, UnaryFactory, UnaryResult};

use crate::manifest::ManifestEntry;
use crate::parquet_writing::EncodedFile;
use crate::store::{DataWriteTarget, FileRef, ObjectPath, ObjectStore};

use super::transaction::TransactionWriter;

/// Max Parquet files one sink keeps outstanding (buffered, prepared, or in
/// flight) before it backpressures the encode pipeline, bounding memory.
const MAX_OUTSTANDING: usize = 64;

/// Max writes handed to the worker in one `next_*_write_requests` call, keeping
/// the ring's submission queue comfortably unsaturated.
const SUBMIT_BATCH: usize = 16;

fn to_operator_error(error: impl std::error::Error + Send + Sync + 'static) -> UnaryError {
    UnaryError::Operator(Box::new(error))
}

/// Builds one worker's [`InsertSink`]. Cloned once per worker by
/// [`OperatorSpec::chain`](dispatch::OperatorSpec::chain); the fields are all
/// shared handles, so a clone is cheap.
pub(super) struct InsertSinkFactory {
    pub store: Arc<dyn ObjectStore>,
    pub location: ObjectPath,
    pub table: String,
    pub writer: Arc<TransactionWriter>,
    pub rows_written: Arc<AtomicU64>,
    pub count_emitted: Arc<AtomicBool>,
}

impl UnaryFactory<EncodedFile, RecordBatch> for InsertSinkFactory {
    type Unary = InsertSink;

    fn build_unary(self) -> InsertSink {
        InsertSink {
            store: self.store,
            location: self.location,
            table: self.table,
            writer: self.writer,
            rows_written: self.rows_written,
            count_emitted: self.count_emitted,
            buffered: VecDeque::new(),
            ready_fs: VecDeque::new(),
            ready_http: VecDeque::new(),
            in_flight: HashMap::new(),
            next_tag: 0,
        }
    }
}

pub(super) struct InsertSink {
    store: Arc<dyn ObjectStore>,
    location: ObjectPath,
    table: String,
    writer: Arc<TransactionWriter>,
    /// Total inserted rows (incremented upstream during conform), shared across
    /// workers; the row count the finishing worker reports.
    rows_written: Arc<AtomicU64>,
    /// Latches so exactly one worker emits the count row.
    count_emitted: Arc<AtomicBool>,
    /// Encoded files consumed but not yet prepared into a ring write.
    buffered: VecDeque<EncodedFile>,
    /// Prepared local-file writes awaiting submission to the ring.
    ready_fs: VecDeque<FsWriteRequest>,
    /// Prepared object uploads awaiting submission to the ring.
    ready_http: VecDeque<HttpWriteRequest>,
    /// The manifest entry each in-flight write records on completion, keyed by
    /// the write's tag.
    in_flight: HashMap<u64, ManifestEntry>,
    next_tag: u64,
}

impl InsertSink {
    fn outstanding(&self) -> usize {
        self.buffered.len() + self.ready_fs.len() + self.ready_http.len() + self.in_flight.len()
    }

    /// Turn every buffered [`EncodedFile`] into a prepared ring write: name the
    /// file, ask the store for a write target, and stash the manifest entry the
    /// write records once its bytes land. The file's identity is known up front
    /// (its size is the encoded length), so the entry is complete before the
    /// write is even submitted. A store/URL failure surfaces out of the
    /// `next_*_write_requests` hook and cancels the INSERT.
    fn prepare(&mut self) -> UnaryResult<()> {
        while let Some(encoded) = self.buffered.pop_front() {
            let EncodedFile {
                bytes,
                partition,
                sort_bounds,
            } = encoded;
            let path = ObjectPath::new(format!("pivot-{}.parquet", uuid::Uuid::new_v4()));
            let size = bytes.len() as u64;
            let data = Arc::new(bytes);
            let key = self.location.resolve(&path);
            let target = self
                .store
                .open_data_write(&key, &data)
                .map_err(to_operator_error)?;
            let tag = self.next_tag;
            self.next_tag += 1;
            match target {
                DataWriteTarget::Local(file) => self.ready_fs.push_back(FsWriteRequest {
                    file: Arc::new(file),
                    data,
                    tag,
                }),
                DataWriteTarget::Remote {
                    remote,
                    method,
                    headers,
                } => self.ready_http.push_back(HttpWriteRequest {
                    remote,
                    method,
                    headers,
                    data,
                    tag,
                }),
            }
            self.in_flight.insert(
                tag,
                ManifestEntry {
                    file: FileRef { path, size },
                    partition,
                    sort_bounds,
                },
            );
        }
        Ok(())
    }

    /// Record the file a completed write produced against the transaction.
    fn record_completed(&mut self, tag: u64) -> UnaryResult<()> {
        let entry = self
            .in_flight
            .remove(&tag)
            .expect("write completion for an in-flight file");
        self.writer
            .record(&self.table, entry)
            .map_err(to_operator_error)
    }
}

impl Unary<EncodedFile, RecordBatch> for InsertSink {
    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        encoded: EncodedFile,
        _sender: &mut S,
    ) -> UnaryResult<()> {
        self.buffered.push_back(encoded);
        Ok(())
    }

    /// Backpressure the encode pipeline once too many files are outstanding, so a
    /// large INSERT doesn't buffer every encoded file in memory at once.
    fn ready_for_more_work(&mut self) -> bool {
        self.outstanding() < MAX_OUTSTANDING
    }

    fn next_fs_write_requests(&mut self) -> UnaryResult<Vec<FsWriteRequest>> {
        self.prepare()?;
        let count = self.ready_fs.len().min(SUBMIT_BATCH);
        Ok(self.ready_fs.drain(..count).collect())
    }

    fn next_http_write_requests(&mut self) -> UnaryResult<Vec<HttpWriteRequest>> {
        self.prepare()?;
        let count = self.ready_http.len().min(SUBMIT_BATCH);
        Ok(self.ready_http.drain(..count).collect())
    }

    fn process_fs_write_response<S: Sender<RecordBatch>>(
        &mut self,
        _sender: &mut S,
        request: FsWriteRequest,
    ) -> UnaryResult<()> {
        self.record_completed(request.tag)
    }

    fn process_http_write_response<S: Sender<RecordBatch>>(
        &mut self,
        _sender: &mut S,
        request: HttpWriteRequest,
    ) -> UnaryResult<()> {
        self.record_completed(request.tag)
    }

    fn finish<S: Sender<RecordBatch>>(&mut self, sender: &mut S) -> UnaryResult<bool> {
        // Not done while any file is still buffered, prepared, or in flight: stay
        // alive (report "still working") so the worker keeps draining our writes
        // through the ring and delivering their completions.
        if self.outstanding() != 0 {
            return Ok(false);
        }
        // Every write is durable and recorded; one worker emits the total row
        // count. `finish` runs only after all siblings' inputs have drained, so
        // conform is complete and `rows_written` is final.
        if !self.count_emitted.swap(true, Ordering::Relaxed) {
            let count = UInt64Array::from(vec![self.rows_written.load(Ordering::Relaxed)]);
            let schema = Schema::new(vec![Field::new("count", DataType::UInt64, false)]);
            sender.send(RecordBatch::try_new(
                Arc::new(schema),
                vec![Arc::new(count)],
            )?)?;
        }
        Ok(true)
    }
}
