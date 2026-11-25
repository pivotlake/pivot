//! Writes a compaction's output into data files as it arrives.
//!
//! The compaction hands over its encoded column chunks a batch at a time, in
//! the order of the output: a row group's first columns arrive, and are
//! written, while its last are still being read. Each batch is appended to
//! the current output file. A file is closed with its footer at the end of
//! a row group, once it holds its share of the rows. The share comes from
//! how many files the rows not yet in a closed file need at the bytes per
//! row written so far, split evenly between those files, so the last file
//! does not come out much smaller than the rest. A row group's size is not
//! known before it is written, so a file can pass the size it is meant to
//! have by part of a row group.
//!
//! A local file is written a batch at a time, so only the batches
//! whose writes are in flight stay in memory. An S3 object goes up as a
//! multipart upload (see [`multipart`]): its bytes collect until there are
//! enough for a part, which is then uploaded, so only the part being
//! collected and the parts in flight stay in memory. An object in any other
//! remote store is uploaded in one request, so its column chunks collect,
//! copied into buffers of their own, until the file is closed.
//!
//! The compaction counts the bytes of every batch in its memory budget
//! and reads further rows only while the budget has room. The count arrives
//! with the batch ([`MemoryReservation`]), and the writer drops it when
//! the batch's bytes are stored: when its write lands, or for a part,
//! when the part's upload lands. That is what slows the compaction down to
//! a slow store.
//!
//! Bytes that only wait for more rows are not counted, since a compaction
//! that waited for them would never go on. The whole budget can be fewer
//! bytes than a part, so the count of the batches collecting for a part
//! is given back whenever no upload is in flight. The bytes that collect
//! for the one request of any other remote object are never counted.

mod multipart;

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use crossbeam_deque::Injector;
use dispatch::io::{
    AuthHeader, FsWriteRequest, HttpUploadRequest, LocalFile, OpenFile, OperatorIO, RemoteFile,
};
use dispatch::memory::{FileBytes, SlabAllocator};
use dispatch::{
    BuildContext, Built, BuiltUnary, OperatorFactory, RecordBatchOperatorSpec, Sender, Unary,
    UnaryFactory, UnaryResult, WorkStatus, to_single_worker_mpsc,
};
use object_storage::{DataFileLocation, FileRef, ObjectPath, ObjectStore, S3Store};
use parquet_engine::thrift::footer::FileMetaData;
use parquet_engine::writing::{
    EncodedColumnChunks, FileCompaction, MemoryReservation, ParquetFileBuilder,
};
use planner::catalog::Column;

use crate::insert_sink::UploadedFile;
use multipart::MultipartUpload;

/// The largest piece encoded column chunks are copied in when they are held for
/// the upload of a whole object, small enough that the pieces fill their ring
/// buffers.
const PACKED_PIECE_BYTES: usize = 256 * 1024;

/// Everything a [`OutputFileWriter`] needs that is the same for every file it
/// writes.
#[derive(Clone)]
pub(super) struct CompactionOutput {
    pub(super) store: Arc<dyn ObjectStore>,
    pub(super) location: ObjectPath,
    /// The table's declared column types, reconciled with each written file's
    /// footer when its row groups are built.
    pub(super) declared_columns: Arc<[Column]>,
    /// The schema the row groups were encoded with.
    pub(super) schema: SchemaRef,
    pub(super) partition: Option<crate::PartitionValues>,
    /// How many bytes of row groups a file is meant to hold.
    pub(super) max_file_size: usize,
    /// Every file this compaction has started to put in storage, whether it
    /// is complete or not, for the compaction's owner to delete if the
    /// compaction fails.
    pub(super) stored_files: Arc<Injector<ObjectPath>>,
    /// Every multipart upload this compaction has started, whether it is
    /// complete or not, for the compaction's owner to abort if the
    /// compaction fails.
    pub(super) started_uploads: Arc<Injector<StartedUpload>>,
    /// Every complete file, for the compaction's owner to commit.
    pub(super) uploaded_files: Arc<Injector<UploadedFile>>,
}

/// A multipart upload of the object at `key`.
pub(super) struct StartedUpload {
    pub(super) key: ObjectPath,
    pub(super) upload_id: String,
}

/// Append `compaction`'s column chunks to data files described by `output`. The
/// returned dataflow emits nothing: the files arrive on
/// `output.uploaded_files`.
pub(super) fn write_output_files_spec<OF>(
    output: CompactionOutput,
    compaction: FileCompaction<OF>,
) -> RecordBatchOperatorSpec
where
    OF: OperatorFactory<EncodedColumnChunks> + 'static,
{
    let worker_count = compaction.column_chunks.dispatcher().worker_count();
    let rows_left = compaction.row_count;
    // The writer runs where the column chunks are put in file order, so a
    // batch is appended to its file without changing workers.
    let writers = compaction.column_chunks.chain(
        to_single_worker_mpsc::<EncodedColumnChunks>(worker_count, compaction.write_worker)
            .into_iter()
            .collect(),
        (0..worker_count)
            .map(|_| OutputFileWriter {
                output: output.clone(),
                rows_left,
                allocator: None,
                open_file: None,
                appended_rows: 0,
                appended_bytes: 0,
                next_file_number: 0,
                unstored_files: HashMap::new(),
                in_flight_writes: HashMap::new(),
            })
            .collect(),
    );
    RecordBatchOperatorSpec::from_spec(writers)
}

/// Where an output file's bytes go.
enum Destination {
    /// A local file, written a piece at a time. `end` is where the next piece
    /// goes.
    Local { file: LocalFile, end: u64 },
    /// An object uploaded whole once the file is closed. `bytes` is what
    /// collected of it.
    Remote {
        url: url::Url,
        auth: Option<AuthHeader>,
        bytes: FileBytes,
    },
    /// An S3 object uploaded a part at a time. The upload is kept with the
    /// file's writes, since its last requests go out after the file is
    /// closed.
    Multipart,
}

/// The file row groups are being appended to.
struct OpenOutputFile {
    number: usize,
    path: ObjectPath,
    key: ObjectPath,
    builder: ParquetFileBuilder,
    destination: Destination,
}

/// A file whose last byte has been issued, waiting for its writes to land.
struct ClosedFile {
    file: FileRef,
    key: ObjectPath,
    metadata: FileMetaData,
}

/// The writes of one output file that have not landed.
#[derive(Default)]
struct UnstoredFile {
    /// How many requests of the file are issued and not answered.
    in_flight_writes: usize,
    closed: Option<ClosedFile>,
    /// The file's multipart upload, when it goes up as one.
    multipart: Option<MultipartUpload>,
}

/// What a request of an output file does.
enum WriteKind {
    /// Puts bytes in a local file, or a whole object in a store.
    Bytes,
    /// Uploads the part of this number of a multipart upload.
    Part(usize),
    /// Starts a multipart upload.
    UploadStart,
    /// Completes a multipart upload.
    UploadCompletion,
}

/// One request that has been issued and has not been answered.
struct InFlightWrite {
    file_number: usize,
    /// What the compaction counts for the row groups this write stores,
    /// given back when it lands. Empty for a request that stores no row
    /// group of its own: a footer, a whole object, or the start or the
    /// completion of an upload.
    memory: Vec<MemoryReservation>,
    kind: WriteKind,
}

pub(super) struct OutputFileWriter {
    output: CompactionOutput,
    /// How many of the compaction's rows have not arrived. The file holding
    /// the last of them is closed as soon as they do.
    rows_left: usize,
    /// Holds the magic and footer of every file, and the copies of the row
    /// groups of a remote object. Initialized on first use so a writer that
    /// receives nothing holds no ring buffer.
    allocator: Option<SlabAllocator>,
    open_file: Option<OpenOutputFile>,
    /// The rows of the row groups appended whole to any file so far, and
    /// the bytes appended, which give the bytes a row is expected to take
    /// when a row group ends.
    appended_rows: usize,
    appended_bytes: usize,
    next_file_number: usize,
    /// The files that are not stored whole yet, by their number.
    unstored_files: HashMap<usize, UnstoredFile>,
    /// Issued writes by the address of their bytes, which the completion
    /// carries back.
    in_flight_writes: HashMap<usize, InFlightWrite>,
}

impl UnaryFactory<EncodedColumnChunks, RecordBatch> for OutputFileWriter {
    type Unary = Self;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self> {
        Built::new(self)
    }
}

impl OutputFileWriter {
    /// How many rows the open file should hold before the next file is
    /// started: an even share of the rows not yet in a closed file, between
    /// as many files as they are expected to need.
    fn count_rows_for_open_file(&self, open_file_rows: usize) -> usize {
        let unclosed_rows = open_file_rows + self.rows_left;
        let expected_bytes =
            unclosed_rows as f64 * self.appended_bytes as f64 / self.appended_rows as f64;
        let files_needed = (expected_bytes / self.output.max_file_size as f64)
            .ceil()
            .max(1.0) as usize;
        unclosed_rows.div_ceil(files_needed)
    }

    /// Start the next output file.
    fn open_next_file(&mut self, io: &mut OperatorIO) -> UnaryResult<OpenOutputFile> {
        let path = ObjectPath::new(format!("pivot-{}.parquet", uuid::Uuid::new_v4()));
        let key = self.output.location.resolve(&path);
        let number = self.next_file_number;
        self.next_file_number += 1;
        self.unstored_files.insert(number, UnstoredFile::default());
        if self.output.store.as_s3().is_some() {
            self.unstored_files
                .get_mut(&number)
                .expect("the file was just started")
                .multipart = Some(MultipartUpload::new(key.clone()));
            self.output.stored_files.push(path.clone());
            self.issue_upload_start(io, number, &key)?;
            return Ok(OpenOutputFile {
                number,
                path,
                key,
                builder: ParquetFileBuilder::new(self.output.schema.clone()),
                destination: Destination::Multipart,
            });
        }
        let sink = self.output.store.sink(&key);
        let destination = match sink.map_err(parquet_engine::op_err)? {
            DataFileLocation::Local(local_path) => {
                let file_handle = OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(local_path)
                    .map_err(parquet_engine::op_err)?;
                self.output.stored_files.push(path.clone());
                Destination::Local {
                    file: LocalFile::new(file_handle).map_err(parquet_engine::op_err)?,
                    end: 0,
                }
            }
            DataFileLocation::Remote { url, auth } => Destination::Remote {
                url,
                auth,
                bytes: FileBytes::new(),
            },
        };
        Ok(OpenOutputFile {
            number,
            path,
            key,
            builder: ParquetFileBuilder::new(self.output.schema.clone()),
            destination,
        })
    }

    /// Put `bytes` at the end of an output file: written now to a local
    /// file, collected for the next part of an S3 object, kept for the
    /// upload of any other remote one. `memory` is what the compaction
    /// counts for the column chunks they hold.
    fn append_to_file(
        &mut self,
        io: &mut OperatorIO,
        file_number: usize,
        destination: &mut Destination,
        bytes: FileBytes,
        memory: Vec<MemoryReservation>,
    ) -> UnaryResult<()> {
        match destination {
            Destination::Local { file, end } => {
                let offset = *end;
                *end += bytes.len() as u64;
                let data = Arc::new(bytes);
                io.write_at(file.clone(), offset, data.clone())?;
                let write = InFlightWrite {
                    file_number,
                    memory,
                    kind: WriteKind::Bytes,
                };
                self.record_issued_write(data, write);
            }
            Destination::Multipart => {
                let upload = self.find_multipart_upload(file_number);
                upload.buffered.append(bytes);
                upload.buffered_memory.extend(memory);
                let has_issued_part = self.issue_next_upload_step(io, file_number)?;
                // Bytes that wait for more to fill a part, with nothing of
                // the compaction on its way to a store, are given up by
                // nothing but more rows, so their count is given back now.
                if !has_issued_part && self.in_flight_writes.is_empty() {
                    self.find_multipart_upload(file_number)
                        .buffered_memory
                        .clear();
                }
            }
            Destination::Remote { bytes: held, .. } => {
                // An encoded page shares its ring buffers with whatever else
                // its encoder allocated, so holding the pages as they came
                // would keep many times their size in memory until the
                // upload. Copied into buffers of their own, a file's pages
                // take only their size.
                let allocator = self
                    .allocator
                    .get_or_insert_with(|| SlabAllocator::new(false));
                for run in bytes.runs() {
                    for piece in run.chunks(PACKED_PIECE_BYTES) {
                        let mut packed = allocator.get_slab_of_size(piece.len(), false);
                        packed.as_mut_slice().copy_from_slice(piece);
                        held.push_slab(packed);
                    }
                }
                // The row group waits only for more rows to end its file, so
                // its count is given back now (see the module).
                drop(memory);
            }
        }
        Ok(())
    }

    /// The multipart upload of output file `file_number`, which goes up as
    /// one and is not stored yet.
    fn find_multipart_upload(&mut self, file_number: usize) -> &mut MultipartUpload {
        self.unstored_files
            .get_mut(&file_number)
            .expect("an upload belongs to a file that is not stored yet")
            .multipart
            .as_mut()
            .expect("an S3 file goes up as a multipart upload")
    }

    /// Remember `write`, the request that was issued with the body `data`,
    /// until it is answered.
    fn record_issued_write(&mut self, data: Arc<FileBytes>, write: InFlightWrite) {
        self.unstored_files
            .get_mut(&write.file_number)
            .expect("a request belongs to a file that is not stored yet")
            .in_flight_writes += 1;
        self.in_flight_writes
            .insert(Arc::as_ptr(&data) as usize, write);
    }

    /// End `file` with its footer and issue what is left to write of it.
    fn close_file(&mut self, io: &mut OperatorIO, file: OpenOutputFile) -> UnaryResult<()> {
        let OpenOutputFile {
            number,
            path,
            key,
            builder,
            mut destination,
        } = file;
        let allocator = self
            .allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        let (footer, metadata) = builder.finish(allocator)?;
        self.append_to_file(io, number, &mut destination, footer, Vec::new())?;
        let size = match destination {
            Destination::Local { end, .. } => end,
            Destination::Multipart => {
                let upload = self.find_multipart_upload(number);
                upload.bytes_issued + upload.buffered.len() as u64
            }
            Destination::Remote { url, auth, bytes } => {
                let size = bytes.len() as u64;
                let remote = RemoteFile::open(url, auth, size)
                    .map(Arc::new)
                    .map_err(parquet_engine::op_err)?;
                // Recorded before the upload starts, so a compaction that
                // fails while it is in flight still knows to delete the
                // object.
                self.output.stored_files.push(path.clone());
                let data = Arc::new(bytes);
                io.write(OpenFile::Remote(remote), data.clone())?;
                let write = InFlightWrite {
                    file_number: number,
                    memory: Vec::new(),
                    kind: WriteKind::Bytes,
                };
                self.record_issued_write(data, write);
                size
            }
        };
        self.unstored_files
            .get_mut(&number)
            .expect("a closed file is not stored yet")
            .closed = Some(ClosedFile {
            file: FileRef { path, size },
            key,
            metadata,
        });
        // What is collected of an S3 file, which the footer now ends, goes up
        // as its last part if the store has named the upload.
        self.issue_next_upload_step(io, number)?;
        Ok(())
    }

    /// The store answered the request whose body is at address
    /// `data_address`, with the header `etag` and the body `answer` if it
    /// was sent to a remote one: give the count of the row groups it stored
    /// back to the compaction, and hand its file to the compaction's owner
    /// if that was the file's last request.
    fn record_landed_write(
        &mut self,
        data_address: usize,
        etag: Option<String>,
        answer: &[u8],
    ) -> UnaryResult<()> {
        let write = self
            .in_flight_writes
            .remove(&data_address)
            .expect("completion for a write not in flight");
        drop(write.memory);
        if self.in_flight_writes.is_empty() {
            let uploads = self.unstored_files.values_mut();
            for upload in uploads.filter_map(|unstored| unstored.multipart.as_mut()) {
                upload.buffered_memory.clear();
            }
        }
        self.unstored_files
            .get_mut(&write.file_number)
            .expect("a landed write belongs to a file that is not stored yet")
            .in_flight_writes -= 1;
        match write.kind {
            WriteKind::Bytes => {}
            WriteKind::Part(part_number) => {
                let etag = etag.ok_or_else(|| {
                    parquet_engine::op_err(std::io::Error::other(
                        "the store answered an uploaded part without an ETag",
                    ))
                })?;
                let upload = self.find_multipart_upload(write.file_number);
                upload.part_etags.push((part_number, etag));
            }
            WriteKind::UploadStart => self.record_started_upload(write.file_number, answer)?,
            WriteKind::UploadCompletion => {
                S3Store::check_completion_answer(answer).map_err(parquet_engine::op_err)?;
            }
        }
        let unstored = &self.unstored_files[&write.file_number];
        // A file that goes up as a multipart upload is stored once the
        // request that completes the upload is answered, which is issued
        // after its last part has landed.
        let is_stored = unstored.closed.is_some()
            && unstored.in_flight_writes == 0
            && (unstored.multipart.is_none() || matches!(write.kind, WriteKind::UploadCompletion));
        if !is_stored {
            return Ok(());
        }
        let closed = self
            .unstored_files
            .remove(&write.file_number)
            .and_then(|unstored| unstored.closed)
            .expect("the file was just found closed");
        let source = self
            .output
            .store
            .source(&closed.key)
            .map_err(parquet_engine::op_err)?;
        // Build the row groups from the footer the writer already produced;
        // only the location is bound now, since it names the stored file.
        let loaded = parquet_engine::file_row_groups_from_metadata(
            closed.file.clone(),
            source,
            closed.metadata,
            &parquet_engine::TableColumns::by_name(self.output.declared_columns.clone()),
        )
        .map_err(parquet_engine::op_err)?;
        self.output.uploaded_files.push(UploadedFile {
            file: closed.file,
            partition: self.output.partition.clone(),
            row_groups: loaded.row_groups,
        });
        Ok(())
    }
}

impl Unary<EncodedColumnChunks, RecordBatch> for OutputFileWriter {
    fn consume(
        &mut self,
        column_chunks: EncodedColumnChunks,
        _sender: &mut dyn Sender<RecordBatch>,
        io: &mut OperatorIO,
    ) -> UnaryResult<()> {
        let EncodedColumnChunks {
            chunks,
            ends_row_group,
            memory,
        } = column_chunks;
        let rows = chunks.count_rows();
        self.appended_bytes += chunks.count_bytes();
        let mut file = match self.open_file.take() {
            Some(file) => file,
            None => self.open_next_file(io)?,
        };
        let allocator = self
            .allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        let bytes = file.builder.append_column_chunks(chunks, allocator);
        self.append_to_file(io, file.number, &mut file.destination, bytes, vec![memory])?;
        if !ends_row_group {
            self.open_file = Some(file);
            return Ok(());
        }

        file.builder.end_row_group();
        self.appended_rows += rows;
        self.rows_left = self.rows_left.checked_sub(rows).ok_or_else(|| {
            parquet_engine::op_err(std::io::Error::other(
                "a compaction handed its writer more rows than its inputs hold",
            ))
        })?;
        let file_rows = file.builder.count_rows();
        if self.rows_left == 0 || file_rows >= self.count_rows_for_open_file(file_rows) {
            self.close_file(io, file)?;
        } else {
            self.open_file = Some(file);
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
        let file_numbers: Vec<usize> = self.unstored_files.keys().copied().collect();
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
        let mut unstored_files = self.unstored_files.values();
        !self.in_flight_writes.is_empty()
            || unstored_files.any(|unstored| unstored.find_next_upload_step().is_some())
    }

    fn finish(&mut self, _sender: &mut dyn Sender<RecordBatch>) -> UnaryResult<bool> {
        assert!(
            self.open_file.is_none() && self.unstored_files.is_empty(),
            "every file of the compaction was closed and written"
        );
        Ok(true)
    }
}
