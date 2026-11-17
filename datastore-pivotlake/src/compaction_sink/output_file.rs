//! One output file of a compaction, from its first row group to its last write.

use std::fs::OpenOptions;
use std::sync::Arc;

use dispatch::UnaryResult;
use dispatch::io::{LocalFile, OpenFile, OperatorIO, RemoteFile};
use dispatch::memory::{FileBytes, SlabAllocator};
use object_storage::{DataFileLocation, FileRef, ObjectPath, S3Store};
use parquet_engine::writing::{MemoryReservation, ParquetFileBuilder, WrittenFileMetadata};

use super::multipart::MultipartUpload;
use super::{CompactedFile, RowGroupWriter};

/// The largest piece an encoded row group is copied in when it is held for
/// the upload of a whole object, small enough that the pieces fill their ring
/// buffers.
pub(super) const PACKED_PIECE_BYTES: usize = 256 * 1024;

/// Where an output file's bytes go.
pub(super) enum Destination {
    /// A local file, written a piece at a time. `end` is where the next piece
    /// goes.
    Local { file: LocalFile, end: u64 },
    /// An object uploaded whole once the file is closed.
    Remote {
        url: url::Url,
        auth: Option<dispatch::io::AuthHeader>,
        bytes: FileBytes,
    },
    /// An S3 object uploaded a part at a time. The upload is kept with the
    /// file's writes, since its last requests go out after the file is
    /// closed.
    Multipart,
}

/// The file row groups are being appended to.
pub(super) struct OpenOutputFile {
    pub(super) number: usize,
    pub(super) rows: usize,
    pub(super) path: ObjectPath,
    pub(super) key: ObjectPath,
    pub(super) builder: ParquetFileBuilder,
    pub(super) destination: Destination,
}

/// A file whose last byte has been issued, waiting for its writes to land.
pub(super) struct ClosedFile {
    pub(super) file: FileRef,
    pub(super) key: ObjectPath,
    pub(super) metadata: WrittenFileMetadata,
}

/// The writes of one output file that have not landed.
#[derive(Default)]
pub(super) struct FileWrites {
    /// How many requests of the file are issued and not answered.
    pub(super) in_flight: usize,
    pub(super) closed: Option<ClosedFile>,
    /// The file's multipart upload, when it goes up as one.
    pub(super) multipart: Option<MultipartUpload>,
}

/// What a request of an output file does.
pub(super) enum WriteKind {
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
pub(super) struct InFlightWrite {
    pub(super) file_number: usize,
    /// The memory reserved for the rows this write puts in storage, given
    /// back to the compaction when it lands.
    pub(super) memory: Vec<Arc<MemoryReservation>>,
    pub(super) kind: WriteKind,
}

impl RowGroupWriter {
    /// Start the next output file.
    pub(super) fn open_file(&mut self, io: &mut OperatorIO) -> UnaryResult<OpenOutputFile> {
        let path = ObjectPath::new(format!("pivot-{}.parquet", uuid::Uuid::new_v4()));
        let key = self.output.location.resolve(&path);
        let number = self.next_file_number;
        self.next_file_number += 1;
        self.files.insert(number, FileWrites::default());
        let destination = match self.output.store.as_s3() {
            Some(_) => {
                self.files
                    .get_mut(&number)
                    .expect("the file was just started")
                    .multipart = Some(MultipartUpload::new(key.clone()));
                self.output.stored_files.push(path.clone());
                self.issue_upload_start(io, number, &key)?;
                Destination::Multipart
            }
            None => match self
                .output
                .store
                .sink(&key)
                .map_err(parquet_engine::op_err)?
            {
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
            },
        };
        Ok(OpenOutputFile {
            number,
            rows: 0,
            path,
            key,
            builder: ParquetFileBuilder::new(self.output.schema.clone()),
            destination,
        })
    }

    /// Issue `write`, which puts `bytes` in `open_file`, and remember it until
    /// it lands.
    pub(super) fn issue_write(
        &mut self,
        io: &mut OperatorIO,
        open_file: OpenFile,
        offset: u64,
        bytes: FileBytes,
        write: InFlightWrite,
    ) -> UnaryResult<()> {
        let data = Arc::new(bytes);
        let id = Arc::as_ptr(&data) as usize;
        match open_file {
            OpenFile::Local(file) => io.write_at(file, offset, data)?,
            remote @ OpenFile::Remote(_) => io.write(remote, data)?,
        };
        self.files
            .get_mut(&write.file_number)
            .expect("a write belongs to a file being written")
            .in_flight += 1;
        self.in_flight.insert(id, write);
        Ok(())
    }

    /// Issue `request`, a `POST` of `body` to a store, and remember it until
    /// it is answered.
    pub(super) fn issue_post(
        &mut self,
        io: &mut OperatorIO,
        request: Arc<RemoteFile>,
        body: FileBytes,
        write: InFlightWrite,
    ) -> UnaryResult<()> {
        let data = Arc::new(body);
        let id = Arc::as_ptr(&data) as usize;
        io.post(request, data)?;
        self.files
            .get_mut(&write.file_number)
            .expect("a request belongs to a file being written")
            .in_flight += 1;
        self.in_flight.insert(id, write);
        Ok(())
    }

    /// Put `bytes` at the end of output file `file_number`: written now to a
    /// local file, collected for the next part of an S3 object, kept for the
    /// upload of any other remote one. `memory` is what the compaction
    /// reserved for the rows they hold.
    pub(super) fn append_to_file(
        &mut self,
        io: &mut OperatorIO,
        file_number: usize,
        destination: &mut Destination,
        bytes: FileBytes,
        memory: Vec<Arc<MemoryReservation>>,
    ) -> UnaryResult<()> {
        match destination {
            Destination::Local { file, end } => {
                let offset = *end;
                *end += bytes.len() as u64;
                let file = file.clone();
                self.issue_write(
                    io,
                    OpenFile::Local(file),
                    offset,
                    bytes,
                    InFlightWrite {
                        file_number,
                        memory,
                        kind: WriteKind::Bytes,
                    },
                )
            }
            Destination::Multipart => {
                let upload = self
                    .files
                    .get_mut(&file_number)
                    .expect("bytes are appended to a file being written")
                    .multipart
                    .as_mut()
                    .expect("an S3 file goes up as a multipart upload");
                upload.buffered.append(bytes);
                upload.buffered_memory.extend(memory);
                let issued_part = self.issue_next_upload_step(io, file_number)?;
                // Bytes that wait for more to fill a part, with nothing of
                // the compaction on its way to a store, are given up by
                // nothing but more rows, so their rows' memory is given back
                // now.
                if !issued_part && self.in_flight.is_empty() {
                    self.files
                        .get_mut(&file_number)
                        .expect("bytes are appended to a file being written")
                        .multipart
                        .as_mut()
                        .expect("an S3 file goes up as a multipart upload")
                        .buffered_memory
                        .clear();
                }
                Ok(())
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
                drop(bytes);
                // The rows are encoded and wait only for their file's upload.
                drop(memory);
                Ok(())
            }
        }
    }

    /// End `file` with its footer and issue what is left to write of it.
    pub(super) fn close_file(
        &mut self,
        io: &mut OperatorIO,
        file: OpenOutputFile,
    ) -> UnaryResult<()> {
        let OpenOutputFile {
            number,
            path,
            key,
            builder,
            mut destination,
            ..
        } = file;
        let allocator = self
            .allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        let (footer, metadata) = builder.finish(allocator)?;
        self.append_to_file(io, number, &mut destination, footer, Vec::new())?;
        let size = match destination {
            Destination::Local { end, .. } => end,
            Destination::Multipart => {
                let upload = self
                    .files
                    .get(&number)
                    .expect("a closed file was being written")
                    .multipart
                    .as_ref()
                    .expect("an S3 file goes up as a multipart upload");
                upload.bytes_issued + upload.buffered.len() as u64
            }
            Destination::Remote { url, auth, bytes } => {
                let size = bytes.len() as u64;
                let remote = RemoteFile::open(url, auth, size)
                    .map(Arc::new)
                    .map_err(parquet_engine::op_err)?;
                // Recorded before the upload starts, so a compaction that fails
                // while it is in flight still knows to delete the object.
                self.output.stored_files.push(path.clone());
                self.issue_write(
                    io,
                    OpenFile::Remote(remote),
                    0,
                    bytes,
                    InFlightWrite {
                        file_number: number,
                        memory: Vec::new(),
                        kind: WriteKind::Bytes,
                    },
                )?;
                size
            }
        };
        self.files
            .get_mut(&number)
            .expect("a closed file was being written")
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

    /// The store answered one request, with the header `etag` and the body
    /// `answer` if it was sent to a remote one: give the memory of the rows
    /// it stored back to the compaction, and hand its file to the
    /// compaction's owner if that was the file's last request.
    pub(super) fn record_landed_write(
        &mut self,
        id: usize,
        etag: Option<String>,
        answer: &[u8],
    ) -> UnaryResult<()> {
        let write = self
            .in_flight
            .remove(&id)
            .expect("completion for a write not in flight");
        drop(write.memory);
        if self.in_flight.is_empty() {
            for upload in self
                .files
                .values_mut()
                .filter_map(|writes| writes.multipart.as_mut())
            {
                upload.buffered_memory.clear();
            }
        }
        self.files
            .get_mut(&write.file_number)
            .expect("a landed write belongs to a file being written")
            .in_flight -= 1;
        match write.kind {
            WriteKind::Bytes => {}
            WriteKind::Part(part_number) => {
                let etag = etag.ok_or_else(|| {
                    parquet_engine::op_err(std::io::Error::other(
                        "the store answered an uploaded part without an ETag",
                    ))
                })?;
                self.files
                    .get_mut(&write.file_number)
                    .expect("a landed write belongs to a file being written")
                    .multipart
                    .as_mut()
                    .expect("a part belongs to a multipart upload")
                    .part_etags
                    .push((part_number, etag));
            }
            WriteKind::UploadStart => self.record_started_upload(write.file_number, answer)?,
            WriteKind::UploadCompletion => {
                S3Store::check_completion_answer(answer).map_err(parquet_engine::op_err)?;
            }
        }
        let writes = self
            .files
            .get(&write.file_number)
            .expect("a landed write belongs to a file being written");
        // A file that goes up as a multipart upload is stored once the
        // request that completes the upload is answered, which is issued
        // after its last part has landed.
        let is_stored = writes.closed.is_some()
            && writes.in_flight == 0
            && (writes.multipart.is_none() || matches!(write.kind, WriteKind::UploadCompletion));
        if !is_stored {
            return Ok(());
        }
        let closed = self
            .files
            .remove(&write.file_number)
            .expect("the file is being written")
            .closed
            .expect("the file was closed");
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
        self.output.uploaded_files.push(CompactedFile {
            file: closed.file,
            partition: self.output.partition.clone(),
            row_groups: loaded.row_groups,
        });
        Ok(())
    }
}
