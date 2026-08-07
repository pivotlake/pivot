//! Worker-side data-file upload and row counting for SQL INSERT.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::mem;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::{Int64Array, RecordBatch};

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use crossbeam_deque::Injector;
use dispatch::io::{FsRequest, FsWriteRequest, HttpRequest, HttpUploadRequest, RemoteFile};
use dispatch::{
    DataFlowDispatcher, RecordBatchOperatorSpec, Sender, Unary, UnaryFactory, stealable,
};
use planner::catalog::Column;
use uuid::Uuid;

use crate::catalog::CatalogTable;
use crate::parquet::RowGroupMetadata;
use crate::parquet::writing::{AssembledFile, encode_record_batches_spec, unshred_batch};
use crate::store::{DataFileLocation, FileRef, ObjectPath, ObjectStore};

/// Build the dataflow that writes `input`'s rows into `table` as Parquet and
/// emits the inserted-row count. Each finished file is pushed onto `uploaded_files`
/// for the statement's transaction to commit (by the table's durable id). This is
/// the write mirror of the scan's
/// [`table_input_with_filter_and_eq_predicates`](crate::parquet::table_input_with_filter_and_eq_predicates):
/// it wires the encode pipeline into this module's upload operators. Driven by
/// the binding's `BoundTable::compile_insert` impl (see [`super::binding::TableBinding`]).
pub(super) fn build_insert_spec(
    table: &CatalogTable,
    uploaded_files: Arc<Injector<UploadedFile>>,
    input: RecordBatchOperatorSpec,
    dispatcher: &DataFlowDispatcher,
) -> crate::Result<RecordBatchOperatorSpec> {
    let store = table.store();
    store.prepare_write()?;

    // In `INSERT INTO x VALUES (...)`, VALUES emits batches with unnamed columns
    // because only INSERT knows which table columns receive the rows. DuckDB has
    // already bound and cast each expression to its target column by position, so
    // stamp the table's declared schema onto the batches for Parquet files and
    // partitioning.
    let schema = Arc::new(Schema::new(
        table
            .columns()
            .into_iter()
            .map(|column| {
                let field = Field::new(
                    column.name,
                    planner::types::physical_arrow_type(&column.col_type),
                    false,
                );
                // The Arrow extension tag lives on the field, not on the data
                // type, so a variant built from `physical_arrow_type` alone is
                // indistinguishable from any other struct. Put the tag back, or
                // the shredding stage further down this pipeline does not
                // recognize the column and writes the documents unshredded.
                match column.col_type {
                    planner::types::Type::Variant => {
                        field.with_metadata(crate::parquet::variant_extension_metadata())
                    }
                    _ => field,
                }
            })
            .collect::<Vec<_>>(),
    ));
    let input = input.project({
        let schema = schema.clone();
        move || {
            let schema = schema.clone();
            move |batch| {
                // Selecting a variant column out of a table hands over the
                // physical shape that table shredded it into, while the column
                // declared here is the plain pair. Reassemble the documents, so
                // the rows are shredded for the file they land in rather than
                // carrying the source table's layout across.
                let batch = unshred_batch(batch)
                    .expect("a variant column reassembles into metadata and value");
                RecordBatch::try_new(schema.clone(), batch.columns().to_vec())
                    .expect("INSERT binding matches the target table schema")
            }
        }
    });

    const TARGET_ROWS_PER_GROUP: usize = 128 * 1024;
    Ok(encode_and_upload_spec(
        store,
        table.object_location().clone(),
        table.id(),
        table.columns().into(),
        uploaded_files,
        input,
        schema,
        table.partition_by().to_vec().into(),
        table.sort_by().to_vec().into(),
        TARGET_ROWS_PER_GROUP,
        dispatcher,
    ))
}

/// The shared write stage: encode `input`'s rows into Parquet files and upload
/// each over the ring, pushing every finished file onto `uploaded_files`. Both
/// INSERT (which stamps the durable schema first) and compaction (which feeds a
/// table scan) build on this; whoever owns `uploaded_files` decides how the files
/// commit. The finish emits the inserted-row count as a single batch.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_and_upload_spec(
    store: Arc<dyn ObjectStore>,
    location: ObjectPath,
    table_id: Uuid,
    declared_columns: Arc<[Column]>,
    uploaded_files: Arc<Injector<UploadedFile>>,
    input: RecordBatchOperatorSpec,
    schema: SchemaRef,
    partition_by: Arc<[String]>,
    sort_by: Arc<[String]>,
    target_rows_per_group: usize,
    dispatcher: &DataFlowDispatcher,
) -> RecordBatchOperatorSpec {
    let encoded =
        encode_record_batches_spec(input, schema, partition_by, sort_by, target_rows_per_group);
    let workers = dispatcher.worker_count();
    // One shared total; every worker's `Upload` adds its completions to it and the
    // last worker into `finish` emits it once all uploads have landed, so no
    // separate fan-in is needed.
    let rows = Arc::new(AtomicUsize::new(0));
    let remaining_workers = Arc::new(AtomicUsize::new(workers));
    let uploads = encoded.chain(
        stealable::<AssembledFile>(dispatcher.topology())
            .into_iter()
            .collect(),
        (0..workers)
            .map(|_| {
                UploadFactory::new(
                    store.clone(),
                    location.clone(),
                    table_id,
                    declared_columns.clone(),
                    uploaded_files.clone(),
                    rows.clone(),
                    remaining_workers.clone(),
                )
            })
            .collect(),
    );
    RecordBatchOperatorSpec::from_spec(uploads)
}

/// One finished upload waiting to be committed, drained from `uploaded_files` by
/// whoever owns it (a statement's transaction, or compaction). Its consumer
/// builds the `DeltaFileEntry` (from `file`/`partition`) and
/// `TableFile` (from `file`/`row_groups`) it commits. Carries the table's durable
/// id, not its name, so the commit resolves the live table regardless of a rename.
pub(crate) struct UploadedFile {
    pub table_id: Uuid,
    pub file: FileRef,
    pub partition: Option<crate::PartitionValues>,
    pub row_groups: Vec<Arc<RowGroupMetadata>>,
}

pub(super) struct UploadFactory {
    store: Arc<dyn ObjectStore>,
    location: ObjectPath,
    table_id: Uuid,
    /// The table's declared column types, reconciled with each uploaded file's
    /// footer when its row groups are built.
    declared_columns: Arc<[Column]>,
    uploaded_files: Arc<Injector<UploadedFile>>,
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
        table_id: Uuid,
        declared_columns: Arc<[Column]>,
        uploaded_files: Arc<Injector<UploadedFile>>,
        rows: Arc<AtomicUsize>,
        remaining_workers: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            store,
            location,
            table_id,
            declared_columns,
            uploaded_files,
            rows,
            remaining_workers,
        }
    }
}

impl UnaryFactory<AssembledFile, RecordBatch> for UploadFactory {
    type Unary = Upload;

    fn build_unary(self) -> Self::Unary {
        Upload {
            store: self.store,
            location: self.location,
            table_id: self.table_id,
            declared_columns: self.declared_columns,
            uploaded_files: self.uploaded_files,
            rows: self.rows,
            remaining_workers: self.remaining_workers,
            in_flight: HashMap::new(),
            pending_fs: Vec::new(),
            pending_http: Vec::new(),
        }
    }
}

struct PendingUpload {
    file: crate::store::FileRef,
    key: ObjectPath,
    partition: Option<crate::PartitionValues>,
    /// The footer metadata the writer produced for this file, used to record its
    /// row groups once the upload lands (no re-parsing the file we just wrote).
    metadata: thriftparquet::footer::FileMetaData,
}

pub(super) struct Upload {
    store: Arc<dyn ObjectStore>,
    location: ObjectPath,
    table_id: Uuid,
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
    /// Writes/uploads staged by `consume`, drained by
    /// [`next_fs_requests`](Self::next_fs_requests) / `next_http_requests`.
    pending_fs: Vec<FsRequest>,
    pending_http: Vec<HttpRequest>,
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
        let source = self
            .store
            .source(&pending.key)
            .map_err(crate::parquet::op_err)?;
        // Row count is taken from the metadata before it's consumed below.
        let num_rows = pending.metadata.num_rows as usize;
        // Build the row groups from the footer the writer already produced; only
        // the location is bound now, since it names the stored file (which exists
        // only once the upload has landed) that future scans read.
        let loaded = crate::parquet::file_row_groups_from_metadata(
            pending.file.clone(),
            source,
            pending.metadata,
            &self.declared_columns,
        )
        .map_err(crate::parquet::op_err)?;
        // Data-file IO is complete, but publication belongs to the statement's
        // transaction. Its commit drains this queue and appends every file to
        // Delta only after the whole dataflow has succeeded.
        self.uploaded_files.push(UploadedFile {
            table_id: self.table_id,
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
    ) -> dispatch::UnaryResult<()> {
        let path = ObjectPath::new(format!("pivot-{}.parquet", uuid::Uuid::new_v4()));
        let key = self.location.resolve(&path);
        let data = Arc::new(encoded.bytes);
        // The write/upload request below shares this `Arc`, so its address keys
        // the in-flight entry a later completion resolves against.
        let id = Arc::as_ptr(&data) as usize;
        let file = crate::store::FileRef {
            path,
            size: data.len() as u64,
        };

        match self.store.sink(&key).map_err(crate::parquet::op_err)? {
            DataFileLocation::Local(path) => {
                let file_handle = OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(path)
                    .map_err(crate::parquet::op_err)?;
                self.pending_fs.push(FsRequest::Write(FsWriteRequest {
                    file: Arc::new(file_handle),
                    data: data.clone(),
                }));
            }
            DataFileLocation::Remote { url, auth } => {
                let remote = RemoteFile::open(url, auth, data.len() as u64)
                    .map(Arc::new)
                    .map_err(crate::parquet::op_err)?;
                self.pending_http
                    .push(HttpRequest::Upload(HttpUploadRequest {
                        remote,
                        data: data.clone(),
                    }));
            }
        }
        dispatch::io::note_pending_io();
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

    fn next_fs_requests(&mut self) -> dispatch::UnaryResult<Vec<FsRequest>> {
        Ok(mem::take(&mut self.pending_fs))
    }

    fn next_http_requests(&mut self) -> dispatch::UnaryResult<Vec<HttpRequest>> {
        Ok(mem::take(&mut self.pending_http))
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
        .map_err(crate::parquet::op_err)?;
        sender.send(batch)?;
        Ok(true)
    }
}
