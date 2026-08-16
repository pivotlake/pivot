//! The operator-facing side of the IO subsystem.
//!
//! An operator never talks to the caches or the ring itself: every IO it wants
//! is pushed into its node's [`OperatorIO`] and drained by the worker's
//! `register_pending_io` stage.
//!
//! - A **read** is a [`PendingIoRequest`] describing *where in the file* the
//!   bytes are - one file and the byte ranges needed, no memory addresses. The
//!   worker's [`RequestTracker`](super::request_tracker::RequestTracker)
//!   resolves each range against the decompressed and compressed caches,
//!   issues reads for whatever is missing, and hands the operator back a
//!   [`CompletedIoRequest`] (through `Operator::process_io_response`) once
//!   every byte is resident.
//! - A **write** ([`FsWriteRequest`]) or **upload** ([`HttpUploadRequest`])
//!   already carries its payload and needs no cache resolution; the worker
//!   submits it as-is and completes it back through
//!   `Operator::process_fs_write_response` /
//!   `Operator::process_http_upload_response`.
//!
//! Splitting "which file bytes" (here) from "which cache slots" (the tracker)
//! is deliberate: the request an operator stages carries no memory addresses,
//! so cache placement, deduplication of overlapping reads, and in-flight
//! bookkeeping all live in one place, per worker, instead of inside every
//! operator that reads.

use crate::io::{FsWriteRequest, HttpUploadRequest, OpenFile};
use bytes::Bytes;
use std::any::Any;

/// One contiguous byte range of a file.
#[derive(Clone, Copy, Debug)]
pub struct FileRange {
    pub offset: usize,
    pub len: usize,
}

/// Which caches a request's ranges may be served from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheTiers {
    /// Raw file bytes via the compressed cache only. For ranges whose consumer
    /// needs the bytes exactly as they are on disk (e.g. a parquet footer): a
    /// decompressed-cache hit would substitute decompressed page payloads for
    /// the raw bytes.
    CompressedOnly,
    /// Decompressed pages where the decompressed cache has them, raw file bytes
    /// for the gaps. Only sound for ranges that cover whole pages previously
    /// inserted into that cache (column chunk reads).
    DecompressedAndCompressed,
}

/// A read an operator stages against one file: the byte ranges it needs, with
/// no cache placement attached. The worker resolves the ranges against the
/// caches when it registers the request.
///
/// The file must already be registered with the compressed cache
/// ([`open_entry`](crate::memory::compressed_cache::CompressedCache::open_entry))
/// by the time the request is pushed.
pub struct PendingIoRequest {
    pub file: OpenFile,
    /// The byte ranges of `file` to read, in file order.
    pub ranges: Vec<FileRange>,
    /// Which caches may serve the ranges.
    pub tiers: CacheTiers,
    /// Operator-owned state that rides along with the request and comes back
    /// in the [`CompletedIoRequest`]; the operator downcasts it back in
    /// `process_io_response` to pick up where it left off.
    pub state: Box<dyn Any>,
}

/// One resolved piece of a requested range, in file order, named by the form
/// its bytes are in.
pub enum RangePart {
    /// A page served straight from the decompressed cache, no IO involved: its
    /// identity within the file (`offset`, `span`), the header bytes the
    /// inserter stored with it, and its decompressed bytes - all pinned for as
    /// long as this part is held.
    Decompressed {
        offset: usize,
        span: usize,
        header: Vec<Bytes>,
        data: Vec<Bytes>,
    },
    /// Raw file bytes via the compressed cache - hits and freshly landed reads
    /// alike. Concatenating `bytes` reproduces the file's bytes from `offset`.
    Compressed { offset: usize, bytes: Vec<Bytes> },
}

/// A [`PendingIoRequest`] whose every range has resolved, handed back to the
/// operator that pushed it.
pub struct CompletedIoRequest {
    /// The `state` the operator pushed, returned whole.
    pub state: Box<dyn Any>,
    /// The resolved parts of each requested range, aligned with the pushed
    /// `ranges`, each in file order.
    pub ranges: Vec<Vec<RangePart>>,
}

/// One staged filesystem operation: a cache-backed read awaiting registration,
/// or a write ready for submission.
pub enum PendingFsRequest {
    Read(PendingIoRequest),
    Write(FsWriteRequest),
}

/// One staged HTTP operation: a cache-backed read of a remote object awaiting
/// registration, or an upload ready for submission.
pub enum PendingHttpRequest {
    Read(PendingIoRequest),
    Upload(HttpUploadRequest),
}

/// The staging area each operator node owns for its IO. Operator methods
/// receive it as `io: &mut OperatorIO` and push reads, writes, and uploads;
/// the worker's `register_pending_io` stage drains it, budget permitting, into
/// its [`RequestTracker`](super::request_tracker::RequestTracker).
///
/// Filesystem and HTTP requests queue separately because the worker budgets
/// them separately: disk and object storage have very different latencies, so
/// each medium fills its own in-flight window without gating the other.
#[derive(Default)]
pub struct OperatorIO {
    /// Operations against local files, awaiting drain.
    pending_fs_requests: Vec<PendingFsRequest>,
    /// Operations against remote (HTTP) objects, awaiting drain.
    pending_http_requests: Vec<PendingHttpRequest>,
}

impl OperatorIO {
    /// Stage a cache-backed read, on the queue matching where its file lives.
    pub fn push_read(&mut self, request: PendingIoRequest) {
        // Flag the worker to walk the dataflows for staged IO on its next pass
        // (the walk is skipped entirely while nothing is staged).
        super::note_pending_io();
        match request.file {
            OpenFile::Local(_) => self
                .pending_fs_requests
                .push(PendingFsRequest::Read(request)),
            OpenFile::Remote(_) => self
                .pending_http_requests
                .push(PendingHttpRequest::Read(request)),
        }
    }

    /// Stage a filesystem write for submission.
    pub fn push_write(&mut self, write: FsWriteRequest) {
        super::note_pending_io();
        self.pending_fs_requests
            .push(PendingFsRequest::Write(write));
    }

    /// Stage an object upload for submission.
    pub fn push_upload(&mut self, upload: HttpUploadRequest) {
        super::note_pending_io();
        self.pending_http_requests
            .push(PendingHttpRequest::Upload(upload));
    }

    /// Take the next staged filesystem operation, oldest first. Operations
    /// drain in the order they were pushed, so a budget-limited drain makes
    /// progress on the front of the queue rather than starving it.
    pub(crate) fn pop_fs_request(&mut self) -> Option<PendingFsRequest> {
        if self.pending_fs_requests.is_empty() {
            None
        } else {
            Some(self.pending_fs_requests.remove(0))
        }
    }

    /// Take the next staged HTTP operation, oldest first.
    pub(crate) fn pop_http_request(&mut self) -> Option<PendingHttpRequest> {
        if self.pending_http_requests.is_empty() {
            None
        } else {
            Some(self.pending_http_requests.remove(0))
        }
    }

    pub(crate) fn has_fs_requests(&self) -> bool {
        !self.pending_fs_requests.is_empty()
    }

    pub(crate) fn has_http_requests(&self) -> bool {
        !self.pending_http_requests.is_empty()
    }

    /// Whether the next filesystem operation is a read (which the worker's
    /// read budget gates; a write drains regardless).
    pub(crate) fn fs_front_is_read(&self) -> bool {
        matches!(
            self.pending_fs_requests.first(),
            Some(PendingFsRequest::Read(_))
        )
    }

    /// Whether the next HTTP operation is a read.
    pub(crate) fn http_front_is_read(&self) -> bool {
        matches!(
            self.pending_http_requests.first(),
            Some(PendingHttpRequest::Read(_))
        )
    }

    /// Whether anything is still staged (registration ran out of budget before
    /// draining it).
    pub(crate) fn is_empty(&self) -> bool {
        self.pending_fs_requests.is_empty() && self.pending_http_requests.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::RemoteFile;
    use std::sync::Arc;

    fn request_for(file: OpenFile) -> PendingIoRequest {
        PendingIoRequest {
            file,
            ranges: vec![FileRange { offset: 0, len: 16 }],
            tiers: CacheTiers::CompressedOnly,
            state: Box::new(()),
        }
    }

    #[test]
    fn a_read_queues_on_the_lane_matching_its_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data");
        std::fs::write(&path, b"bytes").unwrap();
        let local = OpenFile::Local(Arc::new(std::fs::File::open(&path).unwrap()));
        let remote = OpenFile::Remote(Arc::new(
            RemoteFile::open(
                url::Url::parse("https://127.0.0.1:9000/object").unwrap(),
                None,
                16,
            )
            .unwrap(),
        ));

        let mut io = OperatorIO::default();
        io.push_read(request_for(local));
        io.push_read(request_for(remote));

        assert!(matches!(
            io.pop_fs_request(),
            Some(PendingFsRequest::Read(_))
        ));
        assert!(io.pop_fs_request().is_none());
        assert!(matches!(
            io.pop_http_request(),
            Some(PendingHttpRequest::Read(_))
        ));
        assert!(io.is_empty());
    }
}
