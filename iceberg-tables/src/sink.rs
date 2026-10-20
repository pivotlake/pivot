//! The Parquet upload sink: the worker-side dataflow tail that writes a
//! table's data files into its store and reports each one for its owner to
//! commit, and the Arrow schema those files are written under.
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
use crossbeam_deque::Injector;
use dispatch::io::{
    FsWriteRequest, HttpUploadRequest, LocalFile, OpenFile, OperatorIO, RemoteFile,
};
use dispatch::{
    OperatorFactory, OperatorSpec, RecordBatchOperatorSpec, Sender, Unary, UnaryFactory, stealable,
};
use object_storage::{DataFileLocation, FileRef, ObjectPath, ObjectStore};
use parquet_engine::writing::{AssembledFile, encode_record_batches_spec, unshred_batches_spec};
use parquet_engine::{PartitionValues, RowGroupMetadata};
use planner::catalog::Column;
use uuid::Uuid;

/// The Arrow schema a table's files are written under: its declared columns
/// in order, each tagged the way the writer and the readers need. A column
/// `nullability` marks as not nullable is written REQUIRED, so a NULL in it
/// fails the write rather than landing in a file other readers trust. With
/// `field_ids`, each field also carries the Parquet field id (in `columns`
/// order) a format that resolves columns by id assigned it.
pub fn write_schema(
    columns: &[Column],
    nullability: &[bool],
    field_ids: Option<&[i32]>,
) -> Arc<Schema> {
    let fields = columns
        .iter()
        .enumerate()
        .map(|(position, column)| {
            let field = Field::new(
                column.name.clone(),
                planner::types::physical_arrow_type(&column.col_type),
                nullability[position],
            );
            // The Arrow extension tag lives on the field, not on the data
            // type, so a variant built from `physical_arrow_type` alone is
            // indistinguishable from any other struct. Put the tag back, or
            // the shredding stage further down this pipeline does not
            // recognize the column and writes the documents unshredded.
            let mut metadata = match column.col_type {
                planner::types::Type::Variant => parquet_engine::variant_extension_metadata(),
                _ => HashMap::new(),
            };
            if let Some(field_ids) = field_ids {
                metadata.insert(
                    parquet_engine::PARQUET_FIELD_ID_METADATA_KEY.to_string(),
                    field_ids[position].to_string(),
                );
            }
            field.with_metadata(metadata)
        })
        .collect::<Vec<_>>();
    Arc::new(Schema::new(fields))
}

/// Where an INSERT's files go and what shape they take: everything the sink
/// needs to know about the table it writes into.
pub struct InsertTarget {
    /// The store the table's files live in, with the credentials its writes
    /// need.
    pub store: Arc<dyn ObjectStore>,
    /// The table's directory under the store, which every uploaded file's
    /// path is relative to.
    pub directory: ObjectPath,
    /// Where under the table new files are written: nothing for a table that
    /// keeps its files flat, `data` for one laid out the way other writers
    /// expect. Paths reported for the files carry it.
    pub file_prefix: ObjectPath,
    /// The table's durable identity, stamped on every reported file so the
    /// commit resolves the live table whatever it is named by then.
    pub table_id: Uuid,
    pub columns: Vec<Column>,
    /// Per column, in `columns` order, whether a NULL may be written.
    pub nullability: Vec<bool>,
    /// The field id each column is written under, in `columns` order, for a
    /// table format that resolves columns by id.
    pub field_ids: Option<Vec<i32>>,
    pub partition_by: Vec<String>,
    pub sort_by: Vec<String>,
}

/// Build the dataflow that writes `input`'s rows into `target` as Parquet and
/// emits the inserted-row count. Each finished file is pushed onto
/// `uploaded_files` for the statement's transaction to commit. This is the
/// write mirror of the scan's
/// [`table_input_with_filter_and_eq_predicates`](parquet_engine::table_input_with_filter_and_eq_predicates):
/// it wires the encode pipeline into this module's upload operators.
pub fn build_insert_spec(
    target: &InsertTarget,
    uploaded_files: Arc<Injector<UploadedFile>>,
    input: RecordBatchOperatorSpec,
) -> object_storage::Result<RecordBatchOperatorSpec> {
    target.store.prepare_write()?;

    // In `INSERT INTO x VALUES (...)`, VALUES emits batches with unnamed columns
    // because only INSERT knows which table columns receive the rows. The
    // planner has already bound and cast each expression to its target column
    // by position, so stamp the table's declared schema onto the batches for
    // Parquet files and partitioning.
    let schema = write_schema(
        &target.columns,
        &target.nullability,
        target.field_ids.as_deref(),
    );
    // The batch is first stamped with every column nullable, which checks
    // everything about the columns but their nulls, then restamped with the
    // table's own nullability. A NULL headed for a REQUIRED column is the
    // encoder's to refuse, as a write error, not this projection's to panic
    // over: the checked constructor would.
    let nullable_stamp = write_schema(
        &target.columns,
        &vec![true; target.columns.len()],
        target.field_ids.as_deref(),
    );
    let input = unshred_batches_spec(input).project({
        let schema = schema.clone();
        move || {
            let nullable_stamp = nullable_stamp.clone();
            let schema = schema.clone();
            move |batch| {
                // Selecting a variant column out of a table hands over the
                // physical shape that table shredded it into, while the column
                // declared here is the plain pair. Reassemble the documents, so
                // the rows are shredded for the file they land in rather than
                // carrying the source table's layout across.
                let checked =
                    RecordBatch::try_new(nullable_stamp.clone(), batch.columns().to_vec())
                        .expect("INSERT binding matches the target table schema");
                let row_count = checked.num_rows();
                // SAFETY: `checked` just verified the columns against the same
                // fields in count, data type and row count; `schema` differs
                // from `nullable_stamp` only in which fields are nullable, and
                // a null under a required field is refused by the encoder's
                // leaf builder rather than stored.
                unsafe {
                    RecordBatch::new_unchecked(
                        schema.clone(),
                        checked.columns().to_vec(),
                        row_count,
                    )
                }
            }
        }
    });

    const TARGET_ROWS_PER_GROUP: usize = 128 * 1024;
    // This target measures retained Arrow data, not encoded Parquet bytes. The
    // collector applies it to each pending partition file and cuts only after
    // adding a complete batch. Compression and encoding normally make the
    // resulting files substantially smaller on disk.
    const TARGET_IN_MEMORY_BYTES_PER_FILE: usize = 900 * 1024 * 1024;
    let encoded = encode_record_batches_spec(
        input,
        schema,
        target.partition_by.clone().into(),
        target.sort_by.clone().into(),
        TARGET_ROWS_PER_GROUP,
        TARGET_IN_MEMORY_BYTES_PER_FILE,
    );
    Ok(upload_files_spec(
        target.store.clone(),
        target.directory.clone(),
        target.file_prefix.clone(),
        target.table_id,
        target.columns.clone().into(),
        uploaded_files,
        encoded,
    ))
}

/// Upload assembled Parquet files over the ring and push every completion onto
/// `uploaded_files`. INSERT and compaction prepare files through different front
/// halves of the write pipeline and share this storage-facing tail. Files are
/// written under `file_prefix` below `directory`, and reported by their path
/// relative to `directory`.
#[allow(clippy::too_many_arguments)] // one call per write pipeline; each field is needed
pub fn upload_files_spec<OF>(
    store: Arc<dyn ObjectStore>,
    directory: ObjectPath,
    file_prefix: ObjectPath,
    table_id: Uuid,
    declared_columns: Arc<[Column]>,
    uploaded_files: Arc<Injector<UploadedFile>>,
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
    let uploads = encoded.chain(
        stealable::<AssembledFile>(topology).into_iter().collect(),
        (0..workers)
            .map(|_| UploadFactory {
                store: store.clone(),
                directory: directory.clone(),
                file_prefix: file_prefix.clone(),
                table_id,
                declared_columns: declared_columns.clone(),
                uploaded_files: uploaded_files.clone(),
                rows: rows.clone(),
                remaining_workers: remaining_workers.clone(),
            })
            .collect(),
    );
    RecordBatchOperatorSpec::from_spec(uploads)
}

/// One finished upload waiting to be committed, drained from `uploaded_files` by
/// whoever owns it (a statement's transaction, or compaction). Carries the
/// table's durable id, not its name, so the commit resolves the live table
/// regardless of a rename. `file` is relative to the table's directory.
pub struct UploadedFile {
    pub table_id: Uuid,
    pub file: FileRef,
    pub partition: Option<PartitionValues>,
    pub row_groups: Vec<Arc<RowGroupMetadata>>,
}

struct UploadFactory {
    store: Arc<dyn ObjectStore>,
    directory: ObjectPath,
    file_prefix: ObjectPath,
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

impl UnaryFactory<AssembledFile, RecordBatch> for UploadFactory {
    type Unary = Upload;

    fn build_unary(self) -> Self::Unary {
        Upload {
            store: self.store,
            directory: self.directory,
            file_prefix: self.file_prefix,
            table_id: self.table_id,
            declared_columns: self.declared_columns,
            uploaded_files: self.uploaded_files,
            rows: self.rows,
            remaining_workers: self.remaining_workers,
            in_flight: HashMap::new(),
        }
    }
}

struct PendingUpload {
    file: FileRef,
    key: ObjectPath,
    partition: Option<PartitionValues>,
    /// The footer metadata the writer produced for this file, used to record its
    /// row groups once the upload lands (no re-parsing the file we just wrote).
    metadata: parquet_engine::thrift::footer::FileMetaData,
}

struct Upload {
    store: Arc<dyn ObjectStore>,
    directory: ObjectPath,
    file_prefix: ObjectPath,
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
            .map_err(parquet_engine::op_err)?;
        // Row count is taken from the metadata before it's consumed below.
        let num_rows = pending.metadata.num_rows as usize;
        // Build the row groups from the footer the writer already produced; only
        // the location is bound now, since it names the stored file (which exists
        // only once the upload has landed) that future scans read.
        let loaded = parquet_engine::file_row_groups_from_metadata(
            pending.file.clone(),
            source,
            pending.metadata,
            &parquet_engine::TableColumns::by_name(self.declared_columns.clone()),
        )
        .map_err(parquet_engine::op_err)?;
        // Data-file IO is complete, but publication belongs to the statement's
        // transaction. Its commit drains this queue and records every file
        // only after the whole dataflow has succeeded.
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
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let path = self
            .file_prefix
            .join(&format!("pivot-{}.parquet", Uuid::new_v4()));
        let key = self.directory.resolve(&path);
        let data = Arc::new(encoded.bytes);
        // The write/upload request below shares this `Arc`, so its address keys
        // the in-flight entry a later completion resolves against.
        let id = Arc::as_ptr(&data) as usize;
        let file = FileRef {
            path,
            size: data.len() as u64,
        };

        let open_file = match self.store.sink(&key).map_err(parquet_engine::op_err)? {
            DataFileLocation::Local(path) => {
                let file_handle = OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(path)
                    .map_err(parquet_engine::op_err)?;
                OpenFile::Local(LocalFile::new(file_handle).map_err(parquet_engine::op_err)?)
            }
            DataFileLocation::Remote { url, auth } => {
                let remote = RemoteFile::open(url, auth, data.len() as u64)
                    .map(Arc::new)
                    .map_err(parquet_engine::op_err)?;
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
        .map_err(parquet_engine::op_err)?;
        sender.send(batch)?;
        Ok(true)
    }
}
