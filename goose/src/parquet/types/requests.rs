use crate::parquet::types::metadata::{ColumnChunkMeta, QueryRowGroupMetadata};
use crate::parquet::types::projection::Projection;
use bytes::Bytes;
use dispatch::io::{FileLocation, FsRequest, HttpRequest};
use dispatch::memory::{CacheLookup, memory_ctx};

/// Tracks the IO state for a single column chunk within a row group.
///
/// The chunk's byte range is looked up in the file cache as one
/// [`CacheLookup`] per cache bucket it spans. Each lookup's
/// [`missing`](CacheLookup::missing)
/// [`MissingBlock`](dispatch::memory::file_cache::MissingBlock)s (empty when the
/// part is fully resident) are queued for IO. Once every block has been filled,
/// the parts' data is concatenated in file order into the column's `Vec<Bytes>`.
struct ColumnRequest {
    parts: Vec<CacheLookup>,
}

impl ColumnRequest {
    /// Build a `ColumnRequest` from column chunk metadata, queueing the reads
    /// for any missing sub-blocks onto the filesystem or HTTP request list
    /// according to where the file lives.
    fn from(
        meta: &ColumnChunkMeta,
        location: &FileLocation,
        fs_requests: &mut Vec<FsRequest>,
        http_requests: &mut Vec<HttpRequest>,
    ) -> Self {
        let col_start = meta.dictionary_page_offset.unwrap_or(meta.data_page_offset) as usize;
        let len = meta.total_compressed_size as usize;

        let parts = memory_ctx().file_cache().get(location, col_start, len);
        for lookup in &parts {
            for block in lookup.missing() {
                match location {
                    FileLocation::Local(fd) => fs_requests.push(FsRequest {
                        fd: *fd,
                        block: block.clone(),
                    }),
                    FileLocation::Remote(remote) => http_requests.push(HttpRequest {
                        remote: remote.clone(),
                        block: block.clone(),
                    }),
                }
            }
        }
        Self { parts }
    }

    /// Consume this request into the column's data, in file order. Only valid
    /// once every queued block has been filled.
    fn into_buffers(self) -> Vec<Bytes> {
        self.parts.into_iter().map(|p| p.into_data()).collect()
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
    /// Local filesystem read requests not yet submitted to io-uring.
    pending_fs: Vec<FsRequest>,
    /// Remote HTTP range-read requests not yet submitted to the ring.
    pending_http: Vec<HttpRequest>,
    /// Outstanding read count (pending + in-flight).
    remaining: usize,
}

impl RowGroupRequest {
    /// Build a request for all projected columns in the given row group.
    pub fn from(metadata_handle: QueryRowGroupMetadata, projection: &Projection) -> Self {
        let location = metadata_handle.get_metadata().source.location();
        let columns = metadata_handle.columns();

        let mut pending_fs = vec![];
        let mut pending_http = vec![];
        let column_requests = projection
            .indices()
            .iter()
            .map(|&col_idx| {
                ColumnRequest::from(
                    &columns[col_idx],
                    &location,
                    &mut pending_fs,
                    &mut pending_http,
                )
            })
            .collect();

        Self {
            column_requests,
            remaining: pending_fs.len() + pending_http.len(),
            pending_fs,
            pending_http,
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

    /// Returns local filesystem read requests that haven't been submitted yet.
    pub fn pending_fs(&mut self) -> &mut Vec<FsRequest> {
        &mut self.pending_fs
    }

    /// Returns remote HTTP range-read requests that haven't been submitted yet.
    pub fn pending_http(&mut self) -> &mut Vec<HttpRequest> {
        &mut self.pending_http
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
