use crate::io::{IORequest, create_aligned_read_from_start_end};
use crate::memory::{FILE_CACHE, ReadBuffer};
use crate::operations::unary::parquet::types::metadata::{ColumnChunkMeta, QueryRowGroupMetadata};
use crate::operations::unary::parquet::types::projection::Projection;
use bytes::Bytes;
use std::os::fd::{AsRawFd, RawFd};
use crate::worker::WORKER_IDX;

/// Context attached to each `IORequest` so that when the IO completes we know which column and
/// buffer slot the returned data belongs to.
pub struct ColumnBufferContext {
    pub column_idx: usize,
    pub buffer_idx: usize,
}

/// Tracks the IO state for a single column chunk within a row group.
///
/// Because reads are aligned to `BUFFER_SIZE` boundaries, the raw buffers may contain leading
/// and trailing padding. `start_offset` and `end_offset` record where the actual column data
/// sits so we can trim the padding when producing the final `MultiBytes`.
struct ColumnRequest {
    /// Byte offset into the first buffer where column data begins (skips alignment padding).
    start_offset: usize,
    /// Byte offset in the last buffer where column data ends (trims alignment padding).
    end_offset: usize,
    /// One slot per aligned read. `None` until the corresponding IO completes.
    buffers: Vec<Option<ReadBuffer>>,
}

impl ColumnRequest {
    /// Build a `ColumnRequest` from column chunk metadata.
    ///
    /// Computes the aligned read range, then for each aligned block checks the `FILE_CACHE`.
    /// Cache hits are stored directly; misses are appended to `io_requests` for later submission.
    pub fn from(
        meta: &ColumnChunkMeta,
        fd: RawFd,
        column_identifier: usize,
        io_requests: &mut Vec<IORequest>,
    ) -> Self {
        let col_start = if let Some(dict_offset) = meta.dictionary_page_offset {
            dict_offset as u64
        } else {
            meta.data_page_offset as u64
        };

        let col_end = col_start + meta.total_compressed_size as u64;
        let read = create_aligned_read_from_start_end(fd, col_start as usize, col_end as usize);
        let mut buffers = Vec::with_capacity(read.locations.len());

        for (j, location) in read.locations.into_iter().enumerate() {
            match FILE_CACHE.get(&location) {
                None => {
                    io_requests.push(IORequest {
                        location,
                        ctx: Box::new(ColumnBufferContext {
                            column_idx: column_identifier,
                            buffer_idx: j,
                        }),
                    });
                    buffers.push(None);
                }
                r @ Some(_) => buffers.push(r),
            }
        }
        Self {
            start_offset: read.first_offset,
            end_offset: read.end_offset,
            buffers,
        }
    }

    fn insert(&mut self, idx: usize, buffer: ReadBuffer) {
        self.buffers[idx] = Some(buffer);
    }

    /// Consume this request into a trimmed `MultiBytes`.
    fn into_buffers(self) -> Vec<Bytes> {
        let mut data: Vec<_> = self
            .buffers
            .into_iter()
            .map(|c| Bytes::from_owner(c.unwrap()))
            .collect();
        // The end of the last buffer is trimmed first, then the start of the first buffer.
        // Order matters: when there is only a single buffer, slicing the end first keeps the
        // start offset valid.
        let length = data.len();
        data[length - 1] = data.last().unwrap().slice(..self.end_offset);
        data[0] = data[0].slice(self.start_offset..);
        data
    }
}

/// Tracks the IO state for an entire row group read.
///
/// Created by `from()`, which computes aligned reads for every projected column, checks the
/// file cache, and queues `IORequest`s for any misses. As IO completions arrive via
/// `enter_buffer_to_column`, the request counts down until `complete()` returns true.
pub struct RowGroupRequest {
    metadata: QueryRowGroupMetadata,
    column_requests: Vec<ColumnRequest>,
    /// IO requests not yet submitted to io-uring.
    pending_io: Vec<IORequest>,
    /// Outstanding IO count (pending + in-flight).
    remaining: usize,
}

impl RowGroupRequest {
    /// Build a request for all projected columns in the given row group.
    pub fn from(metadata_handle: QueryRowGroupMetadata, projection: &Projection) -> Self {
        let metadata = metadata_handle.get_metadata();
        let columns = metadata_handle.columns();

        let mut remaining_io = vec![];
        let indices: Vec<usize> = projection.indices().to_vec();

        let column_requests = indices
            .iter()
            .enumerate()
            .map(|(i, &col_idx)| {
                ColumnRequest::from(
                    &columns[col_idx],
                    metadata.file.as_raw_fd(),
                    i,
                    &mut remaining_io,
                )
            })
            .collect();

        Self {
            column_requests,
            metadata: metadata_handle,
            remaining: remaining_io.len(),
            pending_io: remaining_io,
        }
    }

    /// Record a completed IO buffer for the given column and buffer slot.
    pub fn enter_buffer_to_column(
        &mut self,
        column_idx: usize,
        buffer_idx: usize,
        buffer: ReadBuffer,
    ) {
        self.column_requests[column_idx].insert(buffer_idx, buffer);
        self.remaining -= 1;
    }

    /// True when all IO has completed and the row group is ready to be consumed.
    pub fn complete(&self) -> bool {
        self.remaining == 0
    }

    /// Returns IO requests that haven't been submitted yet.
    pub fn pending_io(&mut self) -> &mut Vec<IORequest> {
        &mut self.pending_io
    }

    /// Consume this request into a `RowGroupBuffer`
    pub fn into_row_group_buffer(self) -> RowGroupBuffer {
        RowGroupBuffer {
            metadata: self.metadata,
            columns: self
                .column_requests
                .into_iter()
                .map(|c| c.into_buffers())
                .collect(),
        }
    }
}

pub struct RowGroupBuffer {
    pub metadata: QueryRowGroupMetadata,
    /// 2d array, inner vec is chunks within column
    pub columns: Vec<Vec<Bytes>>,
}
