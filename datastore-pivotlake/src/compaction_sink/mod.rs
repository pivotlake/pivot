//! Writes a compaction's row groups into data files as they arrive.
//!
//! The compaction hands over encoded row groups one at a time, in order. Each is
//! appended to the current output file, and a file is closed with its footer
//! once it holds its share of the rows, or earlier if the next row group
//! would take it past the size limit. The share comes from how many files
//! the rows not yet in a closed file need at the bytes per row written so
//! far, split evenly between those files, so the last file does not come out
//! much smaller than the rest.
//!
//! A local file is written a row group at a time, so only the row groups
//! whose writes are in flight stay in memory. An S3 object goes up as a
//! multipart upload: its bytes collect until there are
//! [`UPLOAD_PART_BYTES`] of them, which are then uploaded as the next part,
//! so only the part being collected and the parts in flight stay in memory.
//! An object in any other remote store is uploaded in one request, so its
//! row groups collect until the file is closed.
//!
//! Every file is put in the compressed cache as it is written, so the first
//! query to read it finds it there. A row group's bytes are copied into the
//! cache when they are appended, which gives up the buffers they were encoded
//! in, and each write goes out from the cache's copy, which stays in memory
//! until the write lands.
//!
//! The compaction reserves memory for the rows of every row group and reads
//! further rows only while its budget has room. The reservation arrives
//! with the row group, and the writer drops it when the row group's bytes
//! are stored: when its write lands, or for a part, when the part's upload
//! lands. The whole budget can be fewer bytes than a part, so the memory of
//! rows collecting for a part is given back whenever no upload is in
//! flight: nothing else would ever let the compaction go on. Rows that
//! arrive while an upload is in flight keep their memory until it lands,
//! which is what slows the compaction down to a slow upload.
mod multipart;
mod output_file;

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use crossbeam_deque::Injector;
use dispatch::io::{FsWriteRequest, HttpUploadRequest, OperatorIO};
use dispatch::memory::SlabAllocator;
use dispatch::{
    BuildContext, Built, BuiltUnary, OperatorFactory, RecordBatchOperatorSpec, Sender, Unary,
    UnaryFactory, UnaryResult, WorkStatus, to_single_worker_mpsc,
};
use object_storage::{ObjectPath, ObjectStore};
use parquet_engine::writing::{EncodedRowGroup, FileCompaction};
use planner::catalog::Column;

use output_file::{FileWrites, InFlightWrite, OpenOutputFile};
/// Everything a [`RowGroupWriter`] needs that is the same for every file it
/// writes.
#[derive(Clone)]
pub(super) struct CompactionOutput {
    pub store: Arc<dyn ObjectStore>,
    pub location: ObjectPath,
    /// The table's declared column types, reconciled with each written file's
    /// footer when its row groups are built.
    pub declared_columns: Arc<[Column]>,
    /// The schema the row groups were encoded with.
    pub schema: SchemaRef,
    pub partition: Option<crate::PartitionValues>,
    /// How many bytes of row groups a file may hold. A row group larger than
    /// this is a file of its own.
    pub max_file_size: usize,
    /// Every file this compaction has started to put in storage, whether it is
    /// complete or not, for the compaction's owner to delete if the compaction fails.
    pub stored_files: Arc<Injector<ObjectPath>>,
    /// Every multipart upload this compaction has started, whether it is complete
    /// or not, for the compaction's owner to abort if the compaction fails.
    pub started_uploads: Arc<Injector<StartedUpload>>,
    /// Every complete file, for the compaction's owner to commit.
    pub uploaded_files: Arc<Injector<CompactedFile>>,
}

/// A complete output file of a compaction: where it is, the partition
/// values its rows share, and the row groups its footer records.
pub(super) struct CompactedFile {
    pub file: object_storage::FileRef,
    pub partition: Option<crate::PartitionValues>,
    pub row_groups: Vec<Arc<parquet_engine::RowGroupMetadata>>,
}

/// A multipart upload of the object at `key`.
pub(super) struct StartedUpload {
    pub key: ObjectPath,
    pub upload_id: String,
}

/// Append `compaction`'s row groups to data files described by `output`. The
/// returned dataflow emits nothing: the files arrive on
/// `output.uploaded_files`.
pub(super) fn write_row_groups_spec<OF>(
    output: CompactionOutput,
    compaction: FileCompaction<OF>,
) -> RecordBatchOperatorSpec
where
    OF: OperatorFactory<EncodedRowGroup> + 'static,
{
    let worker_count = compaction.row_groups.dispatcher().worker_count();
    let compaction_worker = compaction.worker;
    let row_count = compaction.row_count;
    // The writer runs where the row groups are assembled, so their bytes are
    // released by the worker that owns them.
    let writers = compaction.row_groups.chain(
        to_single_worker_mpsc::<EncodedRowGroup>(worker_count, compaction_worker)
            .into_iter()
            .collect(),
        (0..worker_count)
            .map(|_| RowGroupWriterFactory {
                output: output.clone(),
                row_count,
            })
            .collect(),
    );
    RecordBatchOperatorSpec::from_spec(writers)
}

struct RowGroupWriterFactory {
    output: CompactionOutput,
    row_count: usize,
}

impl UnaryFactory<EncodedRowGroup, RecordBatch> for RowGroupWriterFactory {
    type Unary = RowGroupWriter;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<RowGroupWriter> {
        Built::new(RowGroupWriter {
            output: self.output,
            rows_left: self.row_count,
            allocator: None,
            current: None,
            rows_appended: 0,
            bytes_appended: 0,
            next_file_number: 0,
            files: HashMap::new(),
            in_flight: HashMap::new(),
        })
    }
}
pub(super) struct RowGroupWriter {
    output: CompactionOutput,
    /// How many of the compaction's rows have not arrived. The file holding the
    /// last of them is closed as soon as they do.
    rows_left: usize,
    /// Holds the magic and footer of every file. Initialized on first use so
    /// a writer that receives nothing holds no ring buffer.
    allocator: Option<SlabAllocator>,
    current: Option<OpenOutputFile>,
    /// The rows appended to any file so far and the bytes they took, which
    /// give the bytes a row is expected to take.
    rows_appended: usize,
    bytes_appended: usize,
    next_file_number: usize,
    files: HashMap<usize, FileWrites>,
    /// Issued writes by the address of their bytes, which the completion
    /// carries back.
    in_flight: HashMap<usize, InFlightWrite>,
}

impl RowGroupWriter {
    /// How many rows the file being filled, now holding `file_rows`, should
    /// hold before the next file is started: an even share of the rows not
    /// yet in a closed file, between as many files as they are expected to
    /// need.
    fn rows_for_current_file(&self, file_rows: usize) -> usize {
        let unclosed_rows = file_rows + self.rows_left;
        let expected_bytes =
            unclosed_rows as f64 * self.bytes_appended as f64 / self.rows_appended as f64;
        let files_needed = (expected_bytes / self.output.max_file_size as f64)
            .ceil()
            .max(1.0) as usize;
        unclosed_rows.div_ceil(files_needed)
    }
}

impl Unary<EncodedRowGroup, RecordBatch> for RowGroupWriter {
    fn consume(
        &mut self,
        mut group: EncodedRowGroup,
        _sender: &mut dyn Sender<RecordBatch>,
        io: &mut OperatorIO,
    ) -> UnaryResult<()> {
        let rows = group.row_count();
        let memory = group.take_memory_reservations();
        self.rows_appended += rows;
        self.bytes_appended += group.byte_len();
        let current_is_full = self.current.as_ref().is_some_and(|file| {
            file.builder.row_group_bytes() > 0
                && file.builder.row_group_bytes() + group.byte_len() > self.output.max_file_size
        });
        if current_is_full {
            let full = self.current.take().unwrap();
            self.close_file(io, full)?;
        }
        let mut file = match self.current.take() {
            Some(file) => file,
            None => self.open_file(io)?,
        };
        let allocator = self
            .allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        let bytes = file.builder.append_row_group(group, allocator);
        let bytes = file.cached_file.append(bytes);
        self.append_to_file(io, file.number, &mut file.destination, bytes, memory)?;
        file.rows += rows;

        self.rows_left = self.rows_left.checked_sub(rows).ok_or_else(|| {
            parquet_engine::op_err(std::io::Error::other(
                "a compaction handed its writer more rows than its inputs hold",
            ))
        })?;
        if self.rows_left == 0 || file.rows >= self.rows_for_current_file(file.rows) {
            self.close_file(io, file)?;
        } else {
            self.current = Some(file);
        }
        Ok(())
    }

    fn process_fs_write_response(
        &mut self,
        _sender: &mut dyn Sender<RecordBatch>,
        request: FsWriteRequest,
    ) -> UnaryResult<()> {
        self.record_landed_write(Arc::as_ptr(&request.data) as usize, None, &[])
    }

    fn process_http_upload_response(
        &mut self,
        _sender: &mut dyn Sender<RecordBatch>,
        request: HttpUploadRequest,
    ) -> UnaryResult<()> {
        self.record_landed_write(
            Arc::as_ptr(&request.data) as usize,
            request.etag,
            &request.response_body,
        )
    }

    /// Issue the requests of the files' multipart uploads that an answer
    /// made possible: the parts collected before the store named an upload,
    /// and the request that completes an upload whose last part has landed.
    fn run(
        &mut self,
        _sender: &mut dyn Sender<RecordBatch>,
        io: &mut OperatorIO,
    ) -> UnaryResult<WorkStatus> {
        let file_numbers: Vec<usize> = self.files.keys().copied().collect();
        let mut status = WorkStatus::Pending;
        for file_number in file_numbers {
            if self.issue_next_upload_step(io, file_number)? {
                status = WorkStatus::Ran;
            }
        }
        Ok(status)
    }

    /// The requests are answered later on the ring, and an answer can make
    /// another request due, so the worker is not done until every request is
    /// answered and none is left to issue.
    fn has_pending_work(&self) -> bool {
        !self.in_flight.is_empty()
            || self
                .files
                .values()
                .any(|writes| writes.find_next_upload_step().is_some())
    }

    fn finish(&mut self, _sender: &mut dyn Sender<RecordBatch>) -> UnaryResult<bool> {
        assert!(
            self.current.is_none() && self.files.is_empty(),
            "every file of the compaction was closed and written"
        );
        Ok(true)
    }
}
