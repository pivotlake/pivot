//! The Parquet upload sink: the worker-side dataflow tail that writes
//! assembled files into an object store, reports each one for its owner to
//! commit, and emits the written row count.
//!
//! The sink knows nothing about what records the files afterwards: an
//! INSERT's transaction and a compaction both drain what it uploaded and
//! commit it through their own table format.
//!
//! A file that comes with text indexes has each written into the write's
//! text index directory, named after the data file and the column
//! ([`TextIndexFile::path_for`]), and is reported once all of them and the
//! data file have landed.

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

use super::{AssembledFile, TextIndexFile};
use crate::RowGroupMetadata;

/// Upload assembled Parquet files over the ring and push every completion onto
/// `uploaded_files`. INSERT and compaction prepare files through different front
/// halves of the write pipeline and share this storage-facing tail. Files are
/// written into `directory` of `store`, and each is reported by its path
/// relative to that directory, with the store and the directory themselves.
/// `on_start` runs once, when the dataflow is built and before any file can
/// land: an INSERT stages the pending insert its transaction commits there;
/// compaction, which commits its files itself, has nothing to run.
///
/// The text indexes the files come with are written into
/// `text_index_directory` of `store`, which the caller has made sure exists.
/// It is `None` for a write whose files come with none.
pub fn upload_files_spec<OF>(
    uploaded_files: UploadedFiles,
    store: Arc<dyn ObjectStore>,
    directory: ObjectPath,
    text_index_directory: Option<ObjectPath>,
    declared_columns: Arc<[Column]>,
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
    // Every file the write uploads names the one directory through this.
    let directory = Arc::new(directory);
    let text_index_directory = text_index_directory.map(Arc::new);
    // One factory per worker, but only the first runs `on_start`.
    let mut on_start = on_start;
    let uploads = encoded.chain(
        stealable::<AssembledFile>(topology).into_iter().collect(),
        (0..workers)
            .map(|_| UploadFactory {
                declared_columns: declared_columns.clone(),
                uploaded_files: uploaded_files.clone(),
                store: Arc::clone(&store),
                directory: Arc::clone(&directory),
                text_index_directory: text_index_directory.clone(),
                on_start: on_start.take(),
                rows: rows.clone(),
                remaining_workers: remaining_workers.clone(),
            })
            .collect(),
    );
    RecordBatchOperatorSpec::from_spec(uploads)
}

/// One finished upload waiting to be committed, drained from `uploaded_files` by
/// whoever owns it (a statement's transaction, or compaction): the file's
/// path relative to the directory it was written into, with its size, the
/// partition values its rows share (`None` for an unpartitioned write), and
/// the row groups its footer records, and the columns its rows are in order
/// of, as the write pipeline reports them
/// ([`AssembledFile::sorted_by`](super::AssembledFile::sorted_by)). It
/// carries the place it was written, the store and the directory in it, so
/// whoever holds it records it or removes it with nothing else in hand.
pub struct UploadedFile {
    pub file: FileRef,
    pub partition: Option<crate::PartitionValues>,
    pub row_groups: Vec<Arc<RowGroupMetadata>>,
    pub sorted_by: Arc<[String]>,
    /// The file's text indexes, each by its path relative to the write's
    /// text index directory, with its size.
    pub text_indexes: Vec<FileRef>,
    store: Arc<dyn ObjectStore>,
    /// Shared by every file of one write.
    directory: Arc<ObjectPath>,
    /// Where the text indexes of the write's files went, when it has any.
    text_index_directory: Option<Arc<ObjectPath>>,
}

impl UploadedFile {
    /// The store the file was written to.
    pub fn store(&self) -> &Arc<dyn ObjectStore> {
        &self.store
    }

    /// The directory the file was written into, which its path is relative
    /// to.
    pub fn directory(&self) -> &ObjectPath {
        &self.directory
    }

    /// The directory the file's text indexes were written into, which their
    /// paths are relative to. `None` for a write without text indexes.
    pub fn text_index_directory(&self) -> Option<&ObjectPath> {
        self.text_index_directory.as_deref()
    }

    /// Delete the file, never recorded, and its text indexes from where they
    /// were written. Best effort: a file that cannot be removed is logged
    /// and left behind, unreferenced.
    pub fn delete(&self) {
        let data_file = self.directory.resolve(&self.file.path);
        let text_indexes = self.text_indexes.iter().map(|text_index| {
            self.text_index_directory
                .as_ref()
                .expect("a file with text indexes was written with a directory for them")
                .resolve(&text_index.path)
        });
        for key in std::iter::once(data_file).chain(text_indexes) {
            if let Err(error) = self.store.delete(&key) {
                tracing::warn!(
                    file = %key.as_str(),
                    %error,
                    "could not remove the uploaded file of a write that did not land"
                );
            }
        }
    }
}

/// The files one write uploaded, waiting for whoever owns the write to
/// record them in its table format or remove them. Shared between the
/// upload sink, which pushes each file as it lands, and the owner, which
/// takes them all at commit or discard.
#[derive(Clone, Default)]
pub struct UploadedFiles {
    queue: Arc<Injector<UploadedFile>>,
}

impl UploadedFiles {
    /// No files yet.
    pub fn new() -> Self {
        Self::default()
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

    /// Delete every file uploaded so far from where it was written
    /// ([`UploadedFile::delete`]). For a write that will not be recorded:
    /// nothing names the files, so no reader can be holding one.
    pub fn delete_all(&self) {
        for uploaded in self.take() {
            uploaded.delete();
        }
    }
}

pub(super) struct UploadFactory {
    /// The table's declared column types, reconciled with each uploaded file's
    /// footer when its row groups are built.
    declared_columns: Arc<[Column]>,
    uploaded_files: UploadedFiles,
    /// Where the files are written: the store, and the directory in it.
    store: Arc<dyn ObjectStore>,
    directory: Arc<ObjectPath>,
    /// Where the files' text indexes are written, in the same store.
    text_index_directory: Option<Arc<ObjectPath>>,
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

impl UnaryFactory<AssembledFile, RecordBatch> for UploadFactory {
    type Unary = Upload;

    /// Building the operator is the dataflow starting, so `on_start` runs
    /// here: before any file can land.
    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self::Unary> {
        if let Some(on_start) = self.on_start {
            on_start();
        }
        Built::new(Upload {
            declared_columns: self.declared_columns,
            uploaded_files: self.uploaded_files,
            store: self.store,
            directory: self.directory,
            text_index_directory: self.text_index_directory,
            rows: self.rows,
            remaining_workers: self.remaining_workers,
            in_flight: HashMap::new(),
            text_index_writes: HashMap::new(),
        })
    }
}

struct PendingUpload {
    file: object_storage::FileRef,
    key: ObjectPath,
    partition: Option<crate::PartitionValues>,
    sorted_by: Arc<[String]>,
    /// The footer metadata the writer produced for this file, used to record its
    /// row groups once the upload lands (no re-parsing the file we just wrote).
    metadata: crate::thrift::footer::FileMetaData,
    /// The file's text indexes, by their paths in the text index directory.
    text_indexes: Vec<FileRef>,
    /// How many of the writes of the data file and its text indexes have not
    /// landed yet. The upload is reported when none is left.
    writes_in_flight: usize,
}

pub(super) struct Upload {
    /// The table's declared column types, reconciled with each uploaded file's
    /// footer when its row groups are built (see [`UploadFactory`]).
    declared_columns: Arc<[Column]>,
    /// The shared queue of finished uploads: each is pushed there for its
    /// owner (a statement's transaction, or compaction) to drain and commit.
    uploaded_files: UploadedFiles,
    /// Where the files are written, the store and the directory in it,
    /// which each finished upload is stamped with.
    store: Arc<dyn ObjectStore>,
    directory: Arc<ObjectPath>,
    /// Where the files' text indexes are written, in the same store.
    text_index_directory: Option<Arc<ObjectPath>>,
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
    /// The writes of text indexes that have not landed yet: each one's
    /// bytes pointer, and the `in_flight` key of the upload it belongs to.
    text_index_writes: HashMap<usize, usize>,
}

impl Upload {
    /// Open `key` of the store for a write of `len` bytes: a new local file,
    /// or an upload to a remote object.
    fn open_for_write(&self, key: &ObjectPath, len: usize) -> dispatch::UnaryResult<OpenFile> {
        Ok(match self.store.sink(key).map_err(crate::op_err)? {
            DataFileLocation::Local(path) => {
                let file_handle = OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(path)
                    .map_err(crate::op_err)?;
                OpenFile::Local(LocalFile::new(file_handle).map_err(crate::op_err)?)
            }
            DataFileLocation::Remote { url, auth } => {
                let remote = RemoteFile::open(url, auth, len as u64)
                    .map(Arc::new)
                    .map_err(crate::op_err)?;
                OpenFile::Remote(remote)
            }
        })
    }

    /// The write whose bytes are at `written` has landed: it is the data
    /// file of an upload or one of its text indexes. Once every write of the
    /// upload has landed, finish it.
    fn complete_write(&mut self, written: usize) -> dispatch::UnaryResult<()> {
        let id = self.text_index_writes.remove(&written).unwrap_or(written);
        let pending = self
            .in_flight
            .get_mut(&id)
            .expect("completion for an upload not in flight");
        pending.writes_in_flight -= 1;
        if pending.writes_in_flight > 0 {
            return Ok(());
        }
        self.complete(id)
    }

    /// Finish the in-flight upload whose encoded bytes are at `id`: its data
    /// file and its text indexes have landed, so build the table file and hand
    /// it to the statement's transaction.
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
        self.uploaded_files.queue.push(UploadedFile {
            file: pending.file,
            partition: pending.partition,
            row_groups: loaded.row_groups,
            sorted_by: pending.sorted_by,
            text_indexes: pending.text_indexes,
            store: Arc::clone(&self.store),
            directory: Arc::clone(&self.directory),
            text_index_directory: self.text_index_directory.clone(),
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
        let key = self.directory.resolve(&path);
        let data = Arc::new(encoded.bytes);
        // The write/upload request below shares this `Arc`, so its address keys
        // the in-flight entry a later completion resolves against.
        let id = Arc::as_ptr(&data) as usize;
        let file = object_storage::FileRef {
            path,
            size: data.len() as u64,
        };

        io.write(self.open_for_write(&key, data.len())?, data.clone())?;

        let mut text_indexes = Vec::with_capacity(encoded.text_indexes.len());
        for TextIndexFile { column, bytes } in encoded.text_indexes {
            let index_path = TextIndexFile::path_for(&file.path, &column);
            let index_key = self
                .text_index_directory
                .as_ref()
                .expect("a write whose files have text indexes has a directory for them")
                .resolve(&index_path);
            let index_data = Arc::new(bytes);
            self.text_index_writes
                .insert(Arc::as_ptr(&index_data) as usize, id);
            io.write(
                self.open_for_write(&index_key, index_data.len())?,
                index_data.clone(),
            )?;
            text_indexes.push(FileRef {
                path: index_path,
                size: index_data.len() as u64,
            });
        }
        self.in_flight.insert(
            id,
            PendingUpload {
                file,
                key,
                partition: encoded.partition,
                sorted_by: encoded.sorted_by,
                metadata: encoded.metadata,
                writes_in_flight: 1 + text_indexes.len(),
                text_indexes,
            },
        );
        Ok(())
    }

    fn process_fs_write_response(
        &mut self,
        _sender: &mut dyn Sender<RecordBatch>,
        request: FsWriteRequest,
    ) -> dispatch::UnaryResult<()> {
        self.complete_write(Arc::as_ptr(&request.data) as usize)
    }

    fn process_http_upload_response(
        &mut self,
        _sender: &mut dyn Sender<RecordBatch>,
        request: HttpUploadRequest,
    ) -> dispatch::UnaryResult<()> {
        self.complete_write(Arc::as_ptr(&request.data) as usize)
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
