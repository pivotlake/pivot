//! The Parquet upload sink: the worker-side dataflow tail that writes
//! assembled files into an object store, reports each one for its owner to
//! commit, and emits the written row count.
//!
//! The sink knows nothing about what records the files afterwards: an
//! INSERT's transaction and a compaction both drain what it uploaded and
//! commit it through their own table format.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use crossbeam_deque::{Injector, Steal};
use dispatch::io::{
    FsWriteRequest, HttpUploadRequest, LocalFile, OpenFile, OperatorIO, RemoteFile,
};
use dispatch::{
    BuildContext, Built, BuiltUnary, OperatorFactory, OperatorSpec, RecordBatchOperatorSpec,
    Sender, Unary, UnaryFactory, stealable,
};
use object_storage::{DataFileLocation, FileRef, ObjectPath, ObjectStore};
use planner::catalog::Column;

use super::AssembledFile;
use crate::RowGroupMetadata;

/// Upload assembled Parquet files over the ring and push every completion onto
/// `uploaded_files`. INSERT and compaction prepare files through different front
/// halves of the write pipeline and share this storage-facing tail.
/// `on_start` runs once, when the dataflow is built and before any file can
/// land: an INSERT stages the pending insert its transaction commits there;
/// compaction, which commits its files itself, has nothing to run.
pub fn upload_files_spec<OF>(
    store: Arc<dyn ObjectStore>,
    location: ObjectPath,
    declared_columns: Arc<[Column]>,
    uploaded_files: Arc<Injector<UploadedFile>>,
    on_start: Option<Box<dyn FnOnce() + Send>>,
    encoded: OperatorSpec<AssembledFile, OF>,
) -> RecordBatchOperatorSpec
where
    OF: OperatorFactory<AssembledFile> + 'static,
{
    let workers = encoded.dispatcher().worker_count();
    let topology = encoded.dispatcher().topology();
    // One shared total; every worker's `Upload` adds its completions to it and the
    // last worker into `finish` emits it once all uploads have landed, so no
    // separate fan-in is needed.
    let rows = Arc::new(AtomicUsize::new(0));
    let remaining_workers = Arc::new(AtomicUsize::new(workers));
    // One factory per worker, but only the first runs `on_start`.
    let mut on_start = on_start;
    let uploads = encoded.chain(
        stealable::<AssembledFile>(topology).into_iter().collect(),
        (0..workers)
            .map(|_| {
                UploadFactory::new(
                    store.clone(),
                    location.clone(),
                    declared_columns.clone(),
                    uploaded_files.clone(),
                    on_start.take(),
                    rows.clone(),
                    remaining_workers.clone(),
                )
            })
            .collect(),
    );
    RecordBatchOperatorSpec::from_spec(uploads)
}

/// One finished upload waiting to be committed, drained from `uploaded_files` by
/// whoever owns it (a statement's transaction, or compaction): the file's
/// path under the directory it was written to, with its size, the partition
/// values its rows share (`None` for an unpartitioned write), and the row
/// groups its footer records.
pub struct UploadedFile {
    pub file: FileRef,
    pub partition: Option<crate::PartitionValues>,
    pub row_groups: Vec<Arc<RowGroupMetadata>>,
}

/// The files one write uploaded, waiting for whoever owns the write to
/// record them in its table format or remove them. Shared between the upload
/// sink, which pushes each file as it lands, and the owner, which takes them
/// all at commit or discard.
#[derive(Clone, Default)]
pub struct UploadedFiles {
    queue: Arc<Injector<UploadedFile>>,
}

impl UploadedFiles {
    pub fn new() -> Self {
        Self::default()
    }

    /// The queue the upload sink pushes onto.
    pub fn queue(&self) -> Arc<Injector<UploadedFile>> {
        Arc::clone(&self.queue)
    }

    /// Every file uploaded so far, in the order they landed, leaving the
    /// queue empty.
    pub fn take(&self) -> Vec<UploadedFile> {
        let mut files = Vec::new();
        loop {
            match self.queue.steal() {
                Steal::Success(file) => files.push(file),
                Steal::Retry => continue,
                Steal::Empty => return files,
            }
        }
    }

    /// Delete every file uploaded so far from `store`, under `location`, the
    /// directory the sink wrote them to. For a write that will not be
    /// recorded: nothing names the files, so no reader can be holding one.
    pub fn delete_from(&self, store: &dyn ObjectStore, location: &ObjectPath) {
        delete_uploaded_files(store, location, &self.take());
    }
}

/// Delete `files`, uploaded under `location` and never recorded, from
/// `store`. Best effort: a file that cannot be removed is logged and left
/// behind, unreferenced.
pub fn delete_uploaded_files(
    store: &dyn ObjectStore,
    location: &ObjectPath,
    files: &[UploadedFile],
) {
    for uploaded in files {
        let key = location.resolve(&uploaded.file.path);
        if let Err(error) = store.delete(&key) {
            tracing::warn!(
                file = %key.as_str(),
                %error,
                "could not remove the uploaded file of a write that did not land"
            );
        }
    }
}

pub(super) struct UploadFactory {
    store: Arc<dyn ObjectStore>,
    location: ObjectPath,
    /// The table's declared column types, reconciled with each uploaded file's
    /// footer when its row groups are built.
    declared_columns: Arc<[Column]>,
    uploaded_files: Arc<Injector<UploadedFile>>,
    /// Run once when this factory builds its operator. Held by one worker's
    /// factory only.
    on_start: Option<Box<dyn FnOnce() + Send>>,
    /// Total inserted rows, shared across every worker's `Upload`. Each worker's
    /// completions add to it; the emitting worker reads it once the finish barrier
    /// guarantees all uploads landed.
    rows: Arc<AtomicUsize>,
    /// Counts workers that have not finished uploading yet. The last worker
    /// emits the result after every worker has contributed its row count.
    remaining_workers: Arc<AtomicUsize>,
}

impl UploadFactory {
    pub fn new(
        store: Arc<dyn ObjectStore>,
        location: ObjectPath,
        declared_columns: Arc<[Column]>,
        uploaded_files: Arc<Injector<UploadedFile>>,
        on_start: Option<Box<dyn FnOnce() + Send>>,
        rows: Arc<AtomicUsize>,
        remaining_workers: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            store,
            location,
            declared_columns,
            uploaded_files,
            on_start,
            rows,
            remaining_workers,
        }
    }
}

impl UnaryFactory<AssembledFile, RecordBatch> for UploadFactory {
    type Unary = Upload;

    /// Building the operator is the dataflow starting, so `on_start` runs
    /// here: before any file can land.
    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self::Unary> {
        if let Some(on_start) = self.on_start {
            on_start();
        }
        Built::new(Upload {
            store: self.store,
            location: self.location,
            declared_columns: self.declared_columns,
            uploaded_files: self.uploaded_files,
            rows: self.rows,
            remaining_workers: self.remaining_workers,
            in_flight: HashMap::new(),
        })
    }
}

struct PendingUpload {
    file: object_storage::FileRef,
    key: ObjectPath,
    partition: Option<crate::PartitionValues>,
    /// The footer metadata the writer produced for this file, used to record its
    /// row groups once the upload lands (no re-parsing the file we just wrote).
    metadata: crate::thrift::footer::FileMetaData,
}

pub(super) struct Upload {
    store: Arc<dyn ObjectStore>,
    location: ObjectPath,
    /// The table's declared column types, reconciled with each uploaded file's
    /// footer when its row groups are built (see [`UploadFactory`]).
    declared_columns: Arc<[Column]>,
    /// Shared queue of finished uploads: each is pushed here for its owner (a
    /// statement's transaction, or compaction) to drain and commit.
    uploaded_files: Arc<Injector<UploadedFile>>,
    /// Running total of inserted rows, shared by every worker (see
    /// [`UploadFactory`]). Added to as completions land; read once at `finish`.
    rows: Arc<AtomicUsize>,
    /// Shared worker countdown; the last worker into `finish` emits the result
    /// row (see [`UploadFactory`]).
    remaining_workers: Arc<AtomicUsize>,
    /// Uploads whose write has been issued but not yet completed, keyed by their
    /// encoded-bytes pointer. The write/upload request shares that `Arc<[u8]>`,
    /// so a completion finds its file by the pointer - completions can arrive in
    /// any order. Unbounded: every file the dataflow hands us starts its write
    /// right away and they all run concurrently (some disks and object stores
    /// have enough latency that serializing would leave throughput on the table).
    in_flight: HashMap<usize, PendingUpload>,
}

impl Upload {
    /// Finish the in-flight upload whose encoded bytes are at `id`: its data-file
    /// IO has landed, so build the table file and hand it to the statement's
    /// transaction.
    fn complete(&mut self, id: usize) -> dispatch::UnaryResult<()> {
        let pending = self
            .in_flight
            .remove(&id)
            .expect("completion for an upload not in flight");
        let source = self.store.source(&pending.key).map_err(crate::op_err)?;
        // Row count is taken from the metadata before it's consumed below.
        let num_rows = pending.metadata.num_rows as usize;
        // Build the row groups from the footer the writer already produced; only
        // the location is bound now, since it names the stored file (which exists
        // only once the upload has landed) that future scans read.
        let loaded = crate::file_row_groups_from_metadata(
            pending.file.clone(),
            source,
            pending.metadata,
            &crate::TableColumns::by_name(self.declared_columns.clone()),
        )
        .map_err(crate::op_err)?;
        // Data-file IO is complete, but publication belongs to the statement's
        // transaction. Its commit drains this queue and appends every file to
        // Delta only after the whole dataflow has succeeded.
        self.uploaded_files.push(UploadedFile {
            file: pending.file,
            partition: pending.partition,
            row_groups: loaded.row_groups,
        });
        // Relaxed: the finish barrier (AcqRel on the sibling counter) publishes
        // this add to whichever worker reads the total.
        self.rows.fetch_add(num_rows, Ordering::Relaxed);
        Ok(())
    }
}

impl Unary<AssembledFile, RecordBatch> for Upload {
    fn consume(
        &mut self,
        encoded: AssembledFile,
        _sender: &mut dyn Sender<RecordBatch>,
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let path = ObjectPath::new(format!("pivot-{}.parquet", uuid::Uuid::new_v4()));
        let key = self.location.resolve(&path);
        let data = Arc::new(encoded.bytes);
        // The write/upload request below shares this `Arc`, so its address keys
        // the in-flight entry a later completion resolves against.
        let id = Arc::as_ptr(&data) as usize;
        let file = object_storage::FileRef {
            path,
            size: data.len() as u64,
        };

        let open_file = match self.store.sink(&key).map_err(crate::op_err)? {
            DataFileLocation::Local(path) => {
                let file_handle = OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(path)
                    .map_err(crate::op_err)?;
                OpenFile::Local(LocalFile::new(file_handle).map_err(crate::op_err)?)
            }
            DataFileLocation::Remote { url, auth } => {
                let remote = RemoteFile::open(url, auth, data.len() as u64)
                    .map(Arc::new)
                    .map_err(crate::op_err)?;
                OpenFile::Remote(remote)
            }
        };
        io.write(open_file, data.clone())?;
        self.in_flight.insert(
            id,
            PendingUpload {
                file,
                key,
                partition: encoded.partition,
                metadata: encoded.metadata,
            },
        );
        Ok(())
    }

    fn process_fs_write_response(
        &mut self,
        _sender: &mut dyn Sender<RecordBatch>,
        request: FsWriteRequest,
    ) -> dispatch::UnaryResult<()> {
        self.complete(Arc::as_ptr(&request.data) as usize)
    }

    fn process_http_upload_response(
        &mut self,
        _sender: &mut dyn Sender<RecordBatch>,
        request: HttpUploadRequest,
    ) -> dispatch::UnaryResult<()> {
        self.complete(Arc::as_ptr(&request.data) as usize)
    }

    /// The uploads are async: their writes/uploads land later on the ring, so the
    /// worker isn't done until `in_flight` drains. Reporting this keeps the finish
    /// barrier from firing until every worker's uploads have completed - only then
    /// is the shared row total final.
    fn has_pending_work(&self) -> bool {
        !self.in_flight.is_empty()
    }

    fn finish(&mut self, sender: &mut dyn Sender<RecordBatch>) -> dispatch::UnaryResult<bool> {
        // Each worker reaches here after its own uploads have landed. The last
        // worker to finish observes every worker's row-count contribution and
        // emits the single result row; earlier workers return without emitting.
        if self.remaining_workers.fetch_sub(1, Ordering::AcqRel) != 1 {
            return Ok(true);
        }
        let total = self.rows.load(Ordering::Acquire) as i64;
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "Count",
                DataType::Int64,
                false,
            )])),
            vec![Arc::new(Int64Array::from(vec![total]))],
        )
        .map_err(crate::op_err)?;
        sender.send(batch)?;
        Ok(true)
    }
}
