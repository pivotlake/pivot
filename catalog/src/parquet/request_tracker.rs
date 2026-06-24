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

use dispatch::io::{FileLocation, FsRequest, HttpRequest, RING_SIZE};
use std::collections::HashMap;

/// Most reads to hand the worker for submission in one pass. The worker submits a
/// whole returned batch to its io_uring at once and only reaps (and submits the next
/// batch) once the ring drains, so a batch larger than the ring's completion queue
/// overflows it — and while the worker is busy finishing rather than entering the
/// ring, the overflowed completions are never flushed, so the fetcher waits forever
/// on reads that already landed. Containment dedup (see [`schedule_or_join`]) keeps
/// the common case small, but a genuine burst of distinct reads (a very wide table,
/// many disjoint ranges) could still overrun the queue, so this is the hard backstop:
/// it bounds outstanding reads regardless of how many a request generates.
///
/// Tied to the ring rather than a magic number: io_uring sizes the completion queue
/// at twice the submission queue ([`RING_SIZE`]), and the worker only submits a new
/// batch once the ring has drained, so an outstanding batch of `RING_SIZE` reads can
/// never overflow the CQ — it uses half its capacity, leaving headroom for the
/// engine's own cache-file reads on the shared ring. The remaining reads stay queued
/// and drain over the next passes.
///
/// [`schedule_or_join`]: RequestTracker::schedule_or_join
const MAX_SUBMIT_BATCH: usize = RING_SIZE as usize;

/// Identity of one cache-block read: which file, and the exact byte run
/// (`offset`, `len`). Hashable, so it can key the routing map and a completion can
/// be matched back to it. `len` is part of the identity (a completion must name
/// the exact run that was submitted); deduplication, though, is by *containment*,
/// not equality - see [`RequestTracker::find_containing_read`].
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
    /// In-flight reads grouped by file: each file → (each submitted read of it →
    /// the slots awaiting it). Grouping by file bounds the containment scan in
    /// [`find_containing_read`](Self::find_containing_read) to one file's reads
    /// rather than every file's - the footer fetcher keeps dozens of files in
    /// flight at once. A new read identical to, or fully contained in, one already
    /// here joins its waiter list rather than issuing its own IO (see
    /// [`schedule_or_join`](Self::schedule_or_join)).
    routing: HashMap<FileLocation, HashMap<ReadRequest, Vec<usize>>>,
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

    /// Register the request at `slot`'s freshly-generated reads: drain them, route
    /// each back to `slot`, and queue for submission only the ones not already
    /// covered by an in-flight read (see [`schedule_or_join`](Self::schedule_or_join)).
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
            if self.schedule_or_join(ReadRequest::of_fs(&req), slot) {
                self.pending_fs.push(req);
                self.disk_in_flight += 1;
            }
        }
        for req in http {
            if self.schedule_or_join(ReadRequest::of_http(&req), slot) {
                self.pending_http.push(req);
                self.http_in_flight += 1;
            }
        }
    }

    /// Register `slot` as a waiter on `read`, and report whether `read` is a *new*
    /// physical read that must be submitted (vs. one already covered by an in-flight
    /// read, which needs no IO of its own).
    ///
    /// Column chunks aren't 4 KB-aligned, so a column's first block is the same
    /// 4 KB block as the previous column's last. The cache returns that shared
    /// block as a tiny *refill* read that targets the previous column's slot - the
    /// very bytes that column's read is already fetching:
    ///
    /// ```text
    ///   column N   read:  file [4096 ................... 12288)   -> slot S
    ///   column N+1 head:  file           [8192 ......... 12288)   -> also slot S
    ///                                      \_______________/
    ///                          fully inside N's read, same memory: don't fetch it
    ///                          again - wait on N's read. Only N+1's body is new.
    /// ```
    ///
    /// So if `read` is identical to, or fully contained in, an in-flight read, we
    /// add `slot` to that read's waiter list (it is advanced once when that read
    /// lands) and submit nothing. Otherwise `read` is new: route it and submit it.
    fn schedule_or_join(&mut self, read: ReadRequest, slot: usize) -> bool {
        if let Some(container) = self.find_containing_read(&read) {
            self.routing
                .get_mut(&read.location)
                .and_then(|file| file.get_mut(&container))
                .expect("an in-flight read is always in `routing`")
                .push(slot);
            return false;
        }
        self.routing
            .entry(read.location.clone())
            .or_default()
            .insert(read, vec![slot]);
        true
    }

    /// The in-flight read of `read`'s file that fully contains it (`read`'s byte
    /// range inside it), or `None`. A contained read is satisfied by its container:
    /// it refills into the same slot the container is already filling, so the
    /// container landing fills its bytes too.
    ///
    /// A linear scan over that file's in-flight reads (the inner map). That set is
    /// small, and `routing` is the single source of truth for what's scheduled, so
    /// scanning it avoids a second index to keep in lock-step. Any container is
    /// equally correct: while a read is in flight its slot is pinned, so every read
    /// touching a given block maps to that block's one cache slot.
    fn find_containing_read(&self, read: &ReadRequest) -> Option<ReadRequest> {
        self.routing
            .get(&read.location)?
            .keys()
            .find(|scheduled| {
                scheduled.offset <= read.offset
                    && scheduled.offset + scheduled.len >= read.offset + read.len
            })
            .cloned()
    }

    /// Take up to [`MAX_SUBMIT_BATCH`] of the deduped reads queued for submission to
    /// the ring; the rest stay queued for the next pass so we never overrun the ring.
    pub fn take_fs_requests(&mut self) -> Vec<FsRequest> {
        let n = self.pending_fs.len().min(MAX_SUBMIT_BATCH);
        self.pending_fs.drain(..n).collect()
    }

    /// Take up to [`MAX_SUBMIT_BATCH`] of the deduped remote reads queued for
    /// submission; the rest stay queued for the next pass.
    pub fn take_http_requests(&mut self) -> Vec<HttpRequest> {
        let n = self.pending_http.len().min(MAX_SUBMIT_BATCH);
        self.pending_http.drain(..n).collect()
    }

    /// A read landed: drop its in-flight charge (the medium is the read's own
    /// location) and return every slot waiting on it — the caller advances each.
    /// The read was registered when staged, so its entry is present.
    pub fn complete(&mut self, read: &ReadRequest) -> Vec<usize> {
        match read.location {
            FileLocation::Local(_) => self.disk_in_flight -= 1,
            FileLocation::Remote(_) => self.http_in_flight -= 1,
        }
        let file = self
            .routing
            .get_mut(&read.location)
            .expect("completion for an unrouted file");
        let waiters = file.remove(read).expect("completion for an unrouted read");
        // Drop the file's bucket once its last read lands, so it doesn't accumulate
        // an empty map per file ever read.
        if file.is_empty() {
            self.routing.remove(&read.location);
        }
        waiters
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
        memory_ctx().file_memory_cache().open_entry(location.clone());
        (location, dir)
    }

    /// Wrap a cache missing-block as a local `FsRequest`.
    fn as_fs_request(location: &FileLocation, block: dispatch::memory::file_memory_cache::MissingBlock) -> FsRequest {
        match location {
            FileLocation::Local(file) => FsRequest {
                file: file.clone(),
                block,
            },
            FileLocation::Remote(_) => unreachable!("test builds local reads"),
        }
    }

    /// The first fs read a `get` of `[offset, offset+len)` produces — the
    /// descriptor a single-block column lookup would yield.
    fn fs_read(location: &FileLocation, offset: usize, len: usize) -> FsRequest {
        let lookups = memory_ctx().file_memory_cache().get(location, offset, len);
        let block = lookups
            .iter()
            .flat_map(|lookup| lookup.missing())
            .next()
            .expect("an uncached range yields a missing block")
            .clone();
        as_fs_request(location, block)
    }

    /// *All* fs reads a column-style `get` of `[offset, offset+len)` produces. A
    /// column chunk can split into a shared-boundary refill block (rounding into
    /// the previous column's last block) plus its own body, so this returns more
    /// than one when that happens.
    fn fs_reads(location: &FileLocation, offset: usize, len: usize) -> Vec<FsRequest> {
        memory_ctx()
            .file_memory_cache()
            .get(location, offset, len)
            .iter()
            .flat_map(|lookup| lookup.missing())
            .map(|block| as_fs_request(location, block.clone()))
            .collect()
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
    fn a_read_contained_in_a_scheduled_one_is_deduped() {
        init_test_free_pool(4);
        let (location, _dir) = registered_file();
        // Schedule [0,8192). A later [0,4096) is fully inside it (it refills into
        // the same slot), so it rides on the bigger read instead of issuing its own.
        let big = ReadRequest::of_fs(&fs_read(&location, 0, 8192));
        let mut tracker = RequestTracker::default();
        let big_slot = tracker.admit_request(TestRequest::with_fs(vec![fs_read(&location, 0, 8192)]));
        let inner_slot = tracker.admit_request(TestRequest::with_fs(vec![fs_read(&location, 0, 4096)]));

        // One physical read; both slots wait on it, and completing it advances both.
        assert_eq!(tracker.take_fs_requests().len(), 1);
        assert_eq!(tracker.disk_in_flight(), 1);
        assert_eq!(tracker.complete(&big), vec![big_slot, inner_slot]);
    }

    #[test]
    fn a_shared_column_boundary_block_rides_on_the_previous_column() {
        init_test_free_pool(8);
        let (location, _dir) = registered_file();
        // Column N covers blocks 0,1 (file [0,8192)). Column N+1 covers blocks 1,2
        // (file [4096,12288)), sharing block 1. The cache splits N+1 into a block-1
        // refill (into N's slot) + a block-2 body read; the refill is contained in
        // N's read, so only N's read and N+1's body are issued.
        let n_reads = fs_reads(&location, 0, 8192);
        let n = ReadRequest::of_fs(&n_reads[0]);
        let mut tracker = RequestTracker::default();
        let n_slot = tracker.admit_request(TestRequest::with_fs(n_reads));

        let n1_reads = fs_reads(&location, 4096, 8192);
        assert_eq!(n1_reads.len(), 2, "shared head + body");
        let n1_body = ReadRequest::of_fs(&n1_reads[1]);
        let n1_slot = tracker.admit_request(TestRequest::with_fs(n1_reads));

        // N's read + N+1's body = 2 reads; the shared block-1 refill folds in.
        assert_eq!(tracker.take_fs_requests().len(), 2);
        assert_eq!(tracker.disk_in_flight(), 2);
        // N's read advances N *and* N+1 (its shared block); the body advances N+1.
        assert_eq!(tracker.complete(&n), vec![n_slot, n1_slot]);
        assert_eq!(tracker.complete(&n1_body), vec![n1_slot]);
    }

    #[test]
    fn a_multi_block_shared_run_folds_in_as_one_unit() {
        init_test_free_pool(8);
        let (location, _dir) = registered_file();
        // N covers blocks 0..=3 (file [0,16384)). N+1 covers blocks 2,3,4 (file
        // [8192,20480)), sharing two whole blocks. The cache returns the shared
        // part as one refill run [2,3], fully inside N's read, so it folds in whole.
        let n_reads = fs_reads(&location, 0, 16384);
        let mut tracker = RequestTracker::default();
        tracker.admit_request(TestRequest::with_fs(n_reads));

        let n1_reads = fs_reads(&location, 8192, 12288);
        assert_eq!(n1_reads.len(), 2, "one shared run + body");
        tracker.admit_request(TestRequest::with_fs(n1_reads));

        // N's read + N+1's body; the two shared blocks were one folded-in refill.
        assert_eq!(tracker.take_fs_requests().len(), 2);
    }

    #[test]
    fn containment_does_not_cross_files() {
        init_test_free_pool(8);
        let (file_a, _a) = registered_file();
        let (file_b, _b) = registered_file();
        // The same byte range in two different files is two unrelated reads.
        let mut tracker = RequestTracker::default();
        tracker.admit_request(TestRequest::with_fs(vec![fs_read(&file_a, 0, 8192)]));
        tracker.admit_request(TestRequest::with_fs(vec![fs_read(&file_b, 0, 4096)]));

        assert_eq!(tracker.take_fs_requests().len(), 2);
        assert_eq!(tracker.disk_in_flight(), 2);
    }

    #[test]
    fn disjoint_reads_each_get_their_own_io() {
        init_test_free_pool(8);
        let (location, _dir) = registered_file();
        // Non-overlapping ranges - neither contains the other - stay separate.
        let mut tracker = RequestTracker::default();
        tracker.admit_request(TestRequest::with_fs(vec![
            fs_read(&location, 0, 4096),
            fs_read(&location, 65536, 4096),
        ]));

        assert_eq!(tracker.take_fs_requests().len(), 2);
        assert_eq!(tracker.disk_in_flight(), 2);
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

    /// A request with more queued reads than one submission batch holds is handed
    /// out in capped passes (never overrunning the ring) and fully drained across
    /// them. Regression for the `SELECT *` materialize livelock.
    #[test]
    fn take_fs_requests_caps_each_batch_and_drains_the_rest() {
        init_test_free_pool(8);
        let (location, _dir) = registered_file();
        let reads: Vec<FsRequest> = (0..MAX_SUBMIT_BATCH + 5)
            .map(|i| fs_read(&location, i * 4096, 4096))
            .collect();
        let mut tracker = RequestTracker::default();
        tracker.admit_request(TestRequest::with_fs(reads));

        let first = tracker.take_fs_requests();
        let second = tracker.take_fs_requests();
        let third = tracker.take_fs_requests();

        assert_eq!(first.len(), MAX_SUBMIT_BATCH);
        assert_eq!(second.len(), 5);
        assert!(third.is_empty());
    }
}
