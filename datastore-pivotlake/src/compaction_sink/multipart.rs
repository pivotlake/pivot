//! The multipart upload of one S3 output file.
//!
//! An upload is three kinds of request, none of which the worker waits for.
//! One starts the upload, and the store's answer names it. Each part then goes
//! up under that name as soon as enough bytes are collected for it. Once the
//! file is closed and its last part has landed, one more request completes
//! the upload, and the object exists when the store has answered it. Row
//! groups that arrive before the upload has its name are collected like any
//! others, so the first part may be larger than the rest.

use std::sync::Arc;

use dispatch::UnaryResult;
use dispatch::io::{OpenFile, OperatorIO, RemoteFile};
use dispatch::memory::{FileBytes, SlabAllocator};
use object_storage::{ObjectPath, S3Store};
use parquet_engine::writing::MemoryReservation;

use super::output_file::{FileWrites, InFlightWrite, WriteKind};
use super::{RowGroupWriter, StartedUpload};

/// How many bytes of an S3 object are collected before they are uploaded as
/// one part. S3 takes no part but the last under 5 MiB and at most 10,000
/// parts, so an object can reach 160 GiB.
pub(super) const UPLOAD_PART_BYTES: usize = 16 * 1024 * 1024;

/// The multipart upload of one output file.
pub(super) struct MultipartUpload {
    /// Where the object is stored.
    pub(super) key: ObjectPath,
    /// The name the store gave the upload. `None` until the store has
    /// answered the request that starts it.
    pub(super) upload_id: Option<String>,
    /// The bytes that will be the next part.
    pub(super) buffered: FileBytes,
    /// The memory reserved for the rows in `buffered`.
    pub(super) buffered_memory: Vec<Arc<MemoryReservation>>,
    pub(super) parts_issued: usize,
    /// The bytes in the parts issued.
    pub(super) bytes_issued: u64,
    /// The number of every landed part and the `ETag` the store answered it
    /// with.
    pub(super) part_etags: Vec<(usize, String)>,
    /// Whether the request that completes the upload has been issued.
    pub(super) is_completion_issued: bool,
}

/// The next request of a multipart upload that can be issued.
pub(super) enum UploadStep {
    /// Upload what is collected as the next part.
    Part,
    /// Complete the upload: the file is closed and every part has landed.
    Completion,
}

impl MultipartUpload {
    /// An upload of the object at `key` that the store has not been asked
    /// for yet.
    pub(super) fn new(key: ObjectPath) -> Self {
        Self {
            key,
            upload_id: None,
            buffered: FileBytes::new(),
            buffered_memory: Vec::new(),
            parts_issued: 0,
            bytes_issued: 0,
            part_etags: Vec::new(),
            is_completion_issued: false,
        }
    }
}

impl FileWrites {
    /// The next request of the file's multipart upload that can be issued
    /// now, if it goes up as one and has one. Nothing can be issued before
    /// the store has named the upload.
    pub(super) fn find_next_upload_step(&self) -> Option<UploadStep> {
        let upload = self.multipart.as_ref()?;
        upload.upload_id.as_ref()?;
        let is_closed = self.closed.is_some();
        // What is collected when the file is closed, which the footer ends,
        // goes up as the last part. It may be smaller than every other part.
        if upload.buffered.len() >= UPLOAD_PART_BYTES || (is_closed && !upload.buffered.is_empty())
        {
            return Some(UploadStep::Part);
        }
        if is_closed && self.in_flight == 0 && !upload.is_completion_issued {
            return Some(UploadStep::Completion);
        }
        None
    }
}

impl RowGroupWriter {
    /// Ask the store to start the multipart upload of output file
    /// `file_number`, to be stored at `key`.
    pub(super) fn issue_upload_start(
        &mut self,
        io: &mut OperatorIO,
        file_number: usize,
        key: &ObjectPath,
    ) -> UnaryResult<()> {
        let url = self
            .output
            .store
            .as_s3()
            .expect("a multipart upload is only started in an S3 store")
            .presign_create_multipart_upload(key)
            .map_err(parquet_engine::op_err)?;
        let request = RemoteFile::open(url, None, 0)
            .map(Arc::new)
            .map_err(parquet_engine::op_err)?;
        self.issue_post(
            io,
            request,
            FileBytes::new(),
            InFlightWrite {
                file_number,
                memory: Vec::new(),
                kind: WriteKind::UploadStart,
            },
        )
    }

    /// Issue the next request of output file `file_number`'s multipart
    /// upload that can be issued now, and say whether there was one.
    pub(super) fn issue_next_upload_step(
        &mut self,
        io: &mut OperatorIO,
        file_number: usize,
    ) -> UnaryResult<bool> {
        let writes = self
            .files
            .get_mut(&file_number)
            .expect("an upload belongs to a file being written");
        let Some(step) = writes.find_next_upload_step() else {
            return Ok(false);
        };
        let upload = writes
            .multipart
            .as_mut()
            .expect("an upload step belongs to a multipart upload");
        let upload_id = upload
            .upload_id
            .clone()
            .expect("an upload step is found once the store has named the upload");
        let key = upload.key.clone();
        let s3 = self
            .output
            .store
            .as_s3()
            .expect("a multipart upload is only started in an S3 store");
        match step {
            UploadStep::Part => {
                let bytes = std::mem::take(&mut upload.buffered);
                let memory = std::mem::take(&mut upload.buffered_memory);
                upload.parts_issued += 1;
                upload.bytes_issued += bytes.len() as u64;
                let part_number = upload.parts_issued;
                let url = s3
                    .presign_upload_part(&key, &upload_id, part_number)
                    .map_err(parquet_engine::op_err)?;
                let part = RemoteFile::open(url, None, bytes.len() as u64)
                    .map(Arc::new)
                    .map_err(parquet_engine::op_err)?;
                self.issue_write(
                    io,
                    OpenFile::Remote(part),
                    0,
                    bytes,
                    InFlightWrite {
                        file_number,
                        memory,
                        kind: WriteKind::Part(part_number),
                    },
                )?;
            }
            UploadStep::Completion => {
                upload.is_completion_issued = true;
                upload
                    .part_etags
                    .sort_unstable_by_key(|(number, _)| *number);
                let part_etags: Vec<String> = upload
                    .part_etags
                    .iter()
                    .map(|(_, etag)| etag.clone())
                    .collect();
                let body = S3Store::build_completion_body(&part_etags);
                let url = s3
                    .presign_complete_multipart_upload(&key, &upload_id)
                    .map_err(parquet_engine::op_err)?;
                let request = RemoteFile::open(url, None, body.len() as u64)
                    .map(Arc::new)
                    .map_err(parquet_engine::op_err)?;
                // A request's body is sent from ring memory.
                let allocator = self
                    .allocator
                    .get_or_insert_with(|| SlabAllocator::new(false));
                let mut slab = allocator.get_slab_of_size(body.len(), false);
                slab.as_mut_slice().copy_from_slice(body.as_bytes());
                let mut bytes = FileBytes::new();
                bytes.push_slab(slab);
                self.issue_post(
                    io,
                    request,
                    bytes,
                    InFlightWrite {
                        file_number,
                        memory: Vec::new(),
                        kind: WriteKind::UploadCompletion,
                    },
                )?;
            }
        }
        Ok(true)
    }

    /// The store answered the request that starts output file
    /// `file_number`'s multipart upload with `answer`: take the upload's name
    /// from it.
    pub(super) fn record_started_upload(
        &mut self,
        file_number: usize,
        answer: &[u8],
    ) -> UnaryResult<()> {
        let upload_id = S3Store::parse_created_upload_id(answer).map_err(parquet_engine::op_err)?;
        let upload = self
            .files
            .get_mut(&file_number)
            .expect("an upload belongs to a file being written")
            .multipart
            .as_mut()
            .expect("an upload was started for a file that goes up as one");
        // Recorded before any part goes up, so a compaction that fails knows
        // to abort the upload, and to delete the object if the upload was
        // completed.
        self.output.started_uploads.push(StartedUpload {
            key: upload.key.clone(),
            upload_id: upload_id.clone(),
        });
        upload.upload_id = Some(upload_id);
        Ok(())
    }
}
