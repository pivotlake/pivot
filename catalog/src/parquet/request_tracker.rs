//! [`RequestTracker`]: the shared IO bookkeeping for a fetcher that pulls cache
//! blocks for many in-flight requests at once. Both the column-chunk scan
//! fetcher and the footer-metadata fetcher build on it.
//!
//! It owns the in-flight requests (slot-indexed, with a free list), the routing
//! from each scheduled read to the slots awaiting it, and the deduped outbox of
//! reads to submit. A request implements [`PendingRequest`] to expose the reads
//! it has generated; the tracker drains, dedups, routes, and counts them, and on
//! a completion hands back every slot waiting on that read so the fetcher can
//! advance it.

use dispatch::io::{FileLocation, FsRequest, HttpRequest};
use std::collections::HashMap;

/// Identity of one cache-block read: which file, and the exact byte run
/// (`offset`, `len`). Hashable, so the tracker can dedup identical reads and
/// route a completion to *every* waiter that needs exactly this run. Including
/// `len` keeps two reads at the same offset but different lengths distinct — a
/// short read never satisfies a waiter that needs a longer one.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct ReadRequest {
    pub location: FileLocation,
    pub offset: usize,
    pub len: usize,
}

impl ReadRequest {
    /// The read a local filesystem request fulfills.
    pub fn of_fs(req: &FsRequest) -> Self {
        Self {
            location: FileLocation::Local(req.file.clone()),
            offset: req.block.file_offset(),
            len: req.block.len(),
        }
    }

    /// The read a remote HTTP request fulfills.
    pub fn of_http(req: &HttpRequest) -> Self {
        Self {
            location: FileLocation::Remote(req.remote.clone()),
            offset: req.block.file_offset(),
            len: req.block.len(),
        }
    }
}

/// A request that exposes the not-yet-submitted reads it has generated, for the
/// tracker to drain. The two vectors are mutually exclusive in practice — a
/// request reads one file, which is local xor remote.
pub(crate) trait PendingRequest {
    fn pending_fs(&mut self) -> &mut Vec<FsRequest>;
    fn pending_http(&mut self) -> &mut Vec<HttpRequest>;
}

/// Tracks in-flight requests and their cache-block reads. See the module doc.
pub(crate) struct RequestTracker<T: PendingRequest> {
    /// In-flight requests, slot-indexed (`None` = free slot).
    in_flight: Vec<Option<T>>,
    /// Reusable freed slot indices.
    free_slots: Vec<usize>,
    /// Each scheduled read → the slots awaiting it. Doubles as the dedup index:
    /// a read already present is in flight, so a new waiter joins it rather than
    /// issuing the read again.
    routing: HashMap<ReadRequest, Vec<usize>>,
    /// Deduped reads queued for submission to the ring (drained by
    /// [`take_fs_requests`](Self::take_fs_requests) / `take_http_requests`).
    pending_fs: Vec<FsRequest>,
    pending_http: Vec<HttpRequest>,
    /// Submitted-but-not-landed reads, per medium — the fetcher caps disk and
    /// HTTP separately since their latencies differ.
    disk_in_flight: usize,
    http_in_flight: usize,
}

impl<T: PendingRequest> Default for RequestTracker<T> {
    fn default() -> Self {
        Self {
            in_flight: Vec::new(),
            free_slots: Vec::new(),
            routing: HashMap::new(),
            pending_fs: Vec::new(),
            pending_http: Vec::new(),
            disk_in_flight: 0,
            http_in_flight: 0,
        }
    }
}

impl<T: PendingRequest> RequestTracker<T> {
    /// Admit `request`: store it in a slot (reusing a freed one) and stage the
    /// reads it has already generated. Returns the slot, which the caller drives
    /// completion/advance against.
    pub fn admit_request(&mut self, request: T) -> usize {
        let slot = if let Some(slot) = self.free_slots.pop() {
            self.in_flight[slot] = Some(request);
            slot
        } else {
            self.in_flight.push(Some(request));
            self.in_flight.len() - 1
        };
        self.stage_reads_for_slot(slot);
        slot
    }

    /// Register the request at `slot`'s freshly-generated reads: drain them, and
    /// route each back to `slot`. A read already scheduled is **deduped** — the
    /// new waiter just joins it, no second read is issued; an unscheduled read is
    /// queued for submission and counted in flight.
    ///
    /// Admission stages a request's first reads via [`admit`](Self::admit_request); call
    /// this directly only when a request produces *more* reads mid-flight (e.g.
    /// the metadata fetcher's exact-footer re-read).
    pub fn stage_reads_for_slot(&mut self, slot: usize) {
        let (fs, http) = {
            let request = self.in_flight[slot].as_mut().expect("staging a free slot");
            (
                std::mem::take(request.pending_fs()),
                std::mem::take(request.pending_http()),
            )
        };
        for req in fs {
            let read = ReadRequest::of_fs(&req);
            if let Some(waiters) = self.routing.get_mut(&read) {
                waiters.push(slot);
            } else {
                self.routing.insert(read, vec![slot]);
                self.pending_fs.push(req);
                self.disk_in_flight += 1;
            }
        }
        for req in http {
            let read = ReadRequest::of_http(&req);
            if let Some(waiters) = self.routing.get_mut(&read) {
                waiters.push(slot);
            } else {
                self.routing.insert(read, vec![slot]);
                self.pending_http.push(req);
                self.http_in_flight += 1;
            }
        }
    }

    /// Drain the deduped reads queued for submission to the ring.
    pub fn take_fs_requests(&mut self) -> Vec<FsRequest> {
        std::mem::take(&mut self.pending_fs)
    }

    /// Drain the deduped remote reads queued for submission.
    pub fn take_http_requests(&mut self) -> Vec<HttpRequest> {
        std::mem::take(&mut self.pending_http)
    }

    /// A read landed: drop its in-flight charge (the medium is the read's own
    /// location) and return every slot waiting on it — the caller advances each.
    /// The read was registered when staged, so its entry is present.
    pub fn complete(&mut self, read: &ReadRequest) -> Vec<usize> {
        match read.location {
            FileLocation::Local(_) => self.disk_in_flight -= 1,
            FileLocation::Remote(_) => self.http_in_flight -= 1,
        }
        self.routing
            .remove(read)
            .expect("completion for an unrouted read")
    }

    /// The request in `slot`, or `None` if the slot is free.
    pub fn request_for_slot(&mut self, slot: usize) -> Option<&mut T> {
        self.in_flight[slot].as_mut()
    }

    /// Take the request out of `slot`, freeing it for reuse.
    pub fn take_request_at_slot(&mut self, slot: usize) -> T {
        let request = self.in_flight[slot].take().expect("freeing a free slot");
        self.free_slots.push(slot);
        request
    }

    /// Outstanding disk-backed reads.
    pub fn disk_in_flight(&self) -> usize {
        self.disk_in_flight
    }

    /// Outstanding remote-backed reads.
    pub fn http_in_flight(&self) -> usize {
        self.http_in_flight
    }

    /// No reads are in flight — every admitted request has drained.
    pub fn is_idle(&self) -> bool {
        self.disk_in_flight == 0 && self.http_in_flight == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dispatch::memory::{init_test_free_pool, memory_ctx};
    use std::sync::Arc;
    use tempfile::TempDir;

    /// A minimal [`PendingRequest`] for driving the tracker directly.
    struct TestRequest {
        pending_fs: Vec<FsRequest>,
        pending_http: Vec<HttpRequest>,
    }

    impl TestRequest {
        fn with_fs(reads: Vec<FsRequest>) -> Self {
            Self {
                pending_fs: reads,
                pending_http: vec![],
            }
        }
    }

    impl PendingRequest for TestRequest {
        fn pending_fs(&mut self) -> &mut Vec<FsRequest> {
            &mut self.pending_fs
        }
        fn pending_http(&mut self) -> &mut Vec<HttpRequest> {
            &mut self.pending_http
        }
    }

    /// A local file registered with the cache, so it can hand out reads. The
    /// returned `TempDir` must outlive the location.
    fn registered_file() -> (FileLocation, TempDir) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("data");
        std::fs::write(&path, vec![0u8; 1 << 20]).unwrap();
        let location = FileLocation::Local(Arc::new(std::fs::File::open(&path).unwrap()));
        memory_ctx().file_cache().open_entry(location.clone());
        (location, dir)
    }

    /// An fs read for `[offset, offset+len)` of a still-uncached (so still
    /// "missing") region — exactly the descriptor a column lookup would produce.
    fn fs_read(location: &FileLocation, offset: usize, len: usize) -> FsRequest {
        let lookups = memory_ctx().file_cache().get(location, offset, len);
        let block = lookups
            .iter()
            .flat_map(|lookup| lookup.missing())
            .next()
            .expect("an uncached range yields a missing block")
            .clone();
        match location {
            FileLocation::Local(file) => FsRequest {
                file: file.clone(),
                block,
            },
            FileLocation::Remote(_) => unreachable!("fs_read builds a local read"),
        }
    }

    #[test]
    fn stage_queues_each_distinct_read_and_counts_it() {
        init_test_free_pool(4);
        let (location, _dir) = registered_file();
        let reads = vec![fs_read(&location, 0, 4096), fs_read(&location, 8192, 4096)];
        let mut tracker = RequestTracker::default();
        tracker.admit_request(TestRequest::with_fs(reads));

        assert_eq!(tracker.take_fs_requests().len(), 2);
        assert_eq!(tracker.disk_in_flight(), 2);
    }

    #[test]
    fn stage_dedups_an_already_scheduled_read_and_credits_both_waiters() {
        init_test_free_pool(4);
        let (location, _dir) = registered_file();
        let shared = ReadRequest::of_fs(&fs_read(&location, 0, 4096));
        let mut tracker = RequestTracker::default();
        let first = tracker.admit_request(TestRequest::with_fs(vec![fs_read(&location, 0, 4096)]));
        let second = tracker.admit_request(TestRequest::with_fs(vec![fs_read(&location, 0, 4096)]));

        // One physical read submitted and counted, but both slots wait on it.
        assert_eq!(tracker.take_fs_requests().len(), 1);
        assert_eq!(tracker.disk_in_flight(), 1);
        assert_eq!(tracker.complete(&shared), vec![first, second]);
    }

    #[test]
    fn distinct_lengths_at_one_offset_stay_separate_reads() {
        init_test_free_pool(4);
        let (location, _dir) = registered_file();
        let short = ReadRequest::of_fs(&fs_read(&location, 0, 4096));
        let long = ReadRequest::of_fs(&fs_read(&location, 0, 8192));
        let mut tracker = RequestTracker::default();
        let short_slot =
            tracker.admit_request(TestRequest::with_fs(vec![fs_read(&location, 0, 4096)]));
        let long_slot =
            tracker.admit_request(TestRequest::with_fs(vec![fs_read(&location, 0, 8192)]));

        // Same offset, different length → two reads, each routed to its own slot.
        assert_eq!(tracker.take_fs_requests().len(), 2);
        assert_eq!(tracker.complete(&short), vec![short_slot]);
        assert_eq!(tracker.complete(&long), vec![long_slot]);
    }

    #[test]
    fn complete_drops_the_in_flight_charge() {
        init_test_free_pool(4);
        let (location, _dir) = registered_file();
        let read = fs_read(&location, 0, 4096);
        let key = ReadRequest::of_fs(&read);
        let mut tracker = RequestTracker::default();
        let slot = tracker.admit_request(TestRequest::with_fs(vec![read]));

        let waiters = tracker.complete(&key);

        assert_eq!(waiters, vec![slot]);
        assert!(tracker.is_idle());
    }

    #[test]
    fn a_freed_slot_is_reused() {
        let mut tracker = RequestTracker::<TestRequest>::default();
        let first = tracker.admit_request(TestRequest::with_fs(vec![]));
        let _second = tracker.admit_request(TestRequest::with_fs(vec![]));

        tracker.take_request_at_slot(first);
        let reused = tracker.admit_request(TestRequest::with_fs(vec![]));

        assert_eq!(reused, first);
    }
}
