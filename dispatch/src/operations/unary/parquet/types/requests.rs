use crate::io::IORequest;
use crate::memory::{BUFFER_SIZE, CacheLookup, memory_ctx, region_base_of};
use crate::operations::unary::parquet::types::metadata::{ColumnChunkMeta, QueryRowGroupMetadata};
use crate::operations::unary::parquet::types::projection::Projection;
use bytes::Bytes;
use std::os::fd::{AsRawFd, RawFd};

/// Tracks the IO state for a single column chunk within a row group.
///
/// A column chunk's `[col_start, col_end)` byte range is split across the 2 MB
/// regions it spans, and each region is looked up in the file cache — yielding a
/// [`CacheLookup`] per region (resident ranges are `Hit`s; ranges with holes are
/// `Miss`es whose [`MissingBlock`](crate::memory::file_cache::MissingBlock)s are
/// queued for IO). Once every block has been filled, the parts are concatenated
/// in file order into the column's `Vec<Bytes>`.
struct ColumnRequest {
    parts: Vec<CacheLookup>,
}

impl ColumnRequest {
    /// Build a `ColumnRequest` from column chunk metadata, queueing the reads
    /// for any missing sub-blocks onto `io_requests`.
    fn from(meta: &ColumnChunkMeta, fd: RawFd, io_requests: &mut Vec<IORequest>) -> Self {
        let col_start = meta
            .dictionary_page_offset
            .unwrap_or(meta.data_page_offset) as usize;
        let col_end = col_start + meta.total_compressed_size as usize;

        let mut parts = Vec::new();
        let mut off = col_start;
        while off < col_end {
            let region = region_base_of(off);
            let seg_end = col_end.min(region + BUFFER_SIZE);
            let lookup = memory_ctx()
                .file_cache()
                .get(fd, region, off - region, seg_end - region);
            if let CacheLookup::Miss(miss) = &lookup {
                for block in miss.missing_blocks() {
                    io_requests.push(IORequest {
                        fd,
                        block: block.clone(),
                    });
                }
            }
            parts.push(lookup);
            off = seg_end;
        }
        Self { parts }
    }

    /// Consume this request into the column's data, in file order. Only valid
    /// once every queued block has been filled.
    fn into_buffers(self) -> Vec<Bytes> {
        self.parts
            .into_iter()
            .map(|p| match p {
                CacheLookup::Hit(b) => b,
                CacheLookup::Miss(m) => m.into_bytes(),
            })
            .collect()
    }
}

/// Tracks the IO state for an entire row group read.
///
/// Created by `from()`, which looks up every projected column in the file cache
/// and queues `IORequest`s for any missing sub-blocks. As completions arrive the
/// requester fills the blocks directly into their cache slots and the fetcher
/// counts down via [`complete_one`](Self::complete_one) until
/// [`complete`](Self::complete) holds.
pub struct RowGroupRequest {
    metadata: QueryRowGroupMetadata,
    column_requests: Vec<ColumnRequest>,
    /// IO requests not yet submitted to io-uring.
    pending_io: Vec<IORequest>,
    /// Outstanding read count (pending + in-flight).
    remaining: usize,
}

impl RowGroupRequest {
    /// Build a request for all projected columns in the given row group.
    pub fn from(metadata_handle: QueryRowGroupMetadata, projection: &Projection) -> Self {
        let fd = metadata_handle.get_metadata().file.as_raw_fd();
        let columns = metadata_handle.columns();

        let mut pending_io = vec![];
        let column_requests = projection
            .indices()
            .iter()
            .map(|&col_idx| ColumnRequest::from(&columns[col_idx], fd, &mut pending_io))
            .collect();

        Self {
            column_requests,
            remaining: pending_io.len(),
            pending_io,
            metadata: metadata_handle,
        }
    }

    /// Record that one queued read has landed (and been committed to its slot).
    pub fn complete_one(&mut self) {
        self.remaining -= 1;
    }

    /// True when every read has completed and the row group is ready to consume.
    pub fn complete(&self) -> bool {
        self.remaining == 0
    }

    /// Returns IO requests that haven't been submitted yet.
    pub fn pending_io(&mut self) -> &mut Vec<IORequest> {
        &mut self.pending_io
    }

    /// Consume this request into a `RowGroupBuffer`.
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
