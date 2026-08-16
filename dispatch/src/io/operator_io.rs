//! The operator-facing side of the cache-backed read path.
//!
//! An operator that needs bytes from a file does not talk to the caches or the
//! ring itself. It describes *where in the file* the bytes are - a
//! [`PendingIoRequest`] naming one file and the byte ranges it needs - and
//! pushes it into its node's [`OperatorIO`]. The worker's `register_pending_io`
//! stage later drains those requests into its
//! [`RequestTracker`](super::request_tracker::RequestTracker), which resolves
//! each range against the decompressed and compressed caches, issues reads for
//! whatever is missing, and hands the operator back a [`CompletedIoRequest`]
//! (through `Operator::process_io_response`) once every byte is resident.
//!
//! Splitting "which file bytes" (here) from "which cache slots" (the tracker)
//! is deliberate: the request an operator stages carries no memory addresses,
//! so cache placement, deduplication of overlapping reads, and in-flight
//! bookkeeping all live in one place, per worker, instead of inside every
//! operator that reads.

use crate::io::OpenFile;
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

/// The staging area each operator node owns for cache-backed reads. Operator
/// methods receive it as `io: &mut OperatorIO` and push requests; the worker's
/// `register_pending_io` stage drains it, budget permitting, into its
/// [`RequestTracker`](super::request_tracker::RequestTracker).
///
/// Filesystem and HTTP requests queue separately because the worker budgets
/// them separately: disk and object storage have very different latencies, so
/// each medium fills its own in-flight window without gating the other.
#[derive(Default)]
pub struct OperatorIO {
    /// Requests against local files, awaiting registration.
    pending_fs_requests: Vec<PendingIoRequest>,
    /// Requests against remote (HTTP) objects, awaiting registration.
    pending_http_requests: Vec<PendingIoRequest>,
}

impl OperatorIO {
    /// Stage `request` for registration, on the queue matching where its file
    /// lives.
    pub fn push(&mut self, request: PendingIoRequest) {
        // Flag the worker to walk the dataflows for staged IO on its next pass
        // (the walk is skipped entirely while nothing is staged).
        super::note_pending_io();
        match request.file {
            OpenFile::Local(_) => self.pending_fs_requests.push(request),
            OpenFile::Remote(_) => self.pending_http_requests.push(request),
        }
    }

    /// Take the next staged filesystem request, oldest first.
    pub(crate) fn pop_fs_request(&mut self) -> Option<PendingIoRequest> {
        pop_front(&mut self.pending_fs_requests)
    }

    /// Take the next staged HTTP request, oldest first.
    pub(crate) fn pop_http_request(&mut self) -> Option<PendingIoRequest> {
        pop_front(&mut self.pending_http_requests)
    }

    pub(crate) fn has_fs_requests(&self) -> bool {
        !self.pending_fs_requests.is_empty()
    }

    pub(crate) fn has_http_requests(&self) -> bool {
        !self.pending_http_requests.is_empty()
    }

    /// Whether any request is still staged (registration ran out of budget
    /// before draining it).
    pub(crate) fn is_empty(&self) -> bool {
        self.pending_fs_requests.is_empty() && self.pending_http_requests.is_empty()
    }
}

/// Pop the oldest staged request. Requests are registered in the order they
/// were pushed, so a budget-limited drain makes progress on the front of the
/// queue rather than starving it.
fn pop_front(requests: &mut Vec<PendingIoRequest>) -> Option<PendingIoRequest> {
    if requests.is_empty() {
        None
    } else {
        Some(requests.remove(0))
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
    fn a_request_queues_on_the_lane_matching_its_file() {
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
        io.push(request_for(local));
        io.push(request_for(remote));

        assert!(io.pop_fs_request().is_some());
        assert!(io.pop_fs_request().is_none());
        assert!(io.pop_http_request().is_some());
        assert!(io.is_empty());
    }
}
