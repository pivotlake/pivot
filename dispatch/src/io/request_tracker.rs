//! [`RequestTracker`]: the worker-side half of the cache-backed read path.
//!
//! Each worker owns one tracker. The worker's `register_pending_io` stage
//! drains every dataflow's staged [`PendingIoRequest`]s into it; registration
//! resolves each requested range against the caches, worker-side:
//!
//! 1. The decompressed cache first (when the request allows it): a page already
//!    decompressed there is pinned on the spot and needs no IO at all.
//! 2. The compressed cache for what's left: resident bytes are pinned, and each
//!    missing block gets a cache extent allocated and a read scheduled.
//!
//! A request whose every range resolved from the caches completes immediately.
//! Otherwise the tracker holds it - with the pinned cache lookups - until its
//! last scheduled read lands, then hands the worker a [`CompletedIoDelivery`]
//! to route back to the operator that pushed it.
//!
//! Because one tracker serves every request on the worker, scheduling dedups
//! reads across operators and dataflows alike: a read identical to, or fully
//! contained in, one already in flight joins its waiter list instead of issuing
//! its own IO (see [`RequestTracker::schedule_or_join`]).

use crate::Identifier;
use crate::io::operator_io::{
    CacheTiers, CompletedIoRequest, FileRange, PendingIoRequest, RangePart,
};
use crate::io::{
    DataFlowRequest, FsReadRequest, FsRequest, HttpGetRequest, HttpRequest, OpenFile, RING_SIZE,
};
use crate::memory::compressed_cache::MissingExtent;
use crate::memory::{CacheLookup, Segment, memory_ctx};
use bytes::Bytes;
use std::any::Any;
use std::collections::HashMap;

/// Most reads to hand the worker for submission in one pass. The worker submits
/// a whole batch to its io_uring at once and only reaps (and submits the next
/// batch) once the ring drains, so a batch larger than the ring's completion
/// queue overflows it - and while the worker is busy finishing rather than
/// entering the ring, the overflowed completions are never flushed, so the
/// tracker waits forever on reads that already landed. Containment dedup keeps
/// the common case small, but a genuine burst of distinct reads (a very wide
/// table, many disjoint ranges) could still overrun the queue, so this is the
/// hard backstop: it bounds a submission batch regardless of how many reads
/// registration generates.
///
/// Tied to the ring rather than a magic number: io_uring sizes the completion
/// queue at twice the submission queue ([`RING_SIZE`]), and the worker only
/// submits a new batch once the ring has drained, so an outstanding batch of
/// `RING_SIZE` reads can never overflow the CQ - it uses half its capacity,
/// leaving headroom for the engine's own cache-file reads on the shared ring.
/// The remaining reads stay queued and drain over the next passes.
const MAX_SUBMIT_BATCH: usize = RING_SIZE as usize;

/// Take up to [`MAX_SUBMIT_BATCH`] reads off the front of `queued`, leaving the
/// rest for the next pass. Takes the whole vec without copying when it already
/// fits in one batch (the common case); only a genuine overflow pays the
/// front-drain.
fn take_batch<R>(queued: &mut Vec<R>) -> Vec<R> {
    if queued.len() <= MAX_SUBMIT_BATCH {
        std::mem::take(queued)
    } else {
        queued.drain(..MAX_SUBMIT_BATCH).collect()
    }
}

/// Identity of one cache-block read: which file, the byte run (`offset`,
/// `len`), and `dest` - the cache-slot address the read fills. Hashable, so it
/// can key the routing map and a completion can be matched back to it.
///
/// `dest` is what makes the identity *physical*: two reads of the same file run
/// can land in different cache slots (e.g. a re-`open_entry`d file allocates
/// fresh), and only `dest` tells them apart. Deduplication is by *containment*
/// on `dest` (see [`RequestTracker::find_containing_read`]): a read is folded
/// onto another only when it fills the same bytes of the same slot, so it can
/// never be folded onto a read filling a different slot - which a
/// file-range-only match could do.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ReadKey {
    open_file: OpenFile,
    offset: usize,
    len: usize,
    /// Address of the cache slot region this read fills (`MissingExtent::dest`).
    dest: usize,
}

impl ReadKey {
    /// The read a local filesystem request fulfills.
    pub fn of_fs(request: &FsReadRequest) -> Self {
        Self {
            open_file: OpenFile::Local(request.file.clone()),
            offset: request.block.file_offset(),
            len: request.block.len(),
            dest: request.block.dest() as usize,
        }
    }

    /// The read a remote HTTP GET fulfills.
    pub fn of_http_get(request: &HttpGetRequest) -> Self {
        Self {
            open_file: OpenFile::Remote(request.remote.clone()),
            offset: request.block.file_offset(),
            len: request.block.len(),
            dest: request.block.dest() as usize,
        }
    }
}

/// One resolved piece of a requested range while its reads may still be in
/// flight. The IO-state twin of [`RangePart`]: a `Compressed` part holds its
/// [`CacheLookup`]s (whose pins keep the slots alive) until every read lands,
/// then [`into_ready`](Self::into_ready) resolves them into plain bytes.
enum PendingRangePart {
    Decompressed {
        offset: usize,
        span: usize,
        header: Vec<Bytes>,
        data: Vec<Bytes>,
    },
    Compressed {
        offset: usize,
        lookups: Vec<CacheLookup>,
    },
}

impl PendingRangePart {
    /// Resolve the IO state away. Only valid once every read filling this
    /// part's lookups has landed.
    fn into_ready(self) -> RangePart {
        match self {
            Self::Decompressed {
                offset,
                span,
                header,
                data,
            } => RangePart::Decompressed {
                offset,
                span,
                header,
                data,
            },
            Self::Compressed { offset, lookups } => RangePart::Compressed {
                offset,
                bytes: lookups.into_iter().map(CacheLookup::into_data).collect(),
            },
        }
    }
}

/// A registered request waiting on reads, with the routing needed to hand it
/// back once done.
struct RegisteredRequest {
    data_flow_id: Identifier,
    operator_idx: Identifier,
    /// The operator state riding along with the request.
    state: Box<dyn Any>,
    /// The resolving parts of each requested range, aligned with the pushed
    /// ranges.
    ranges: Vec<Vec<PendingRangePart>>,
    /// Scheduled reads this request still waits on. Each read it needs counts
    /// once - whether the read was newly issued or joined onto an in-flight
    /// one - and each completion credits it once per occurrence in the read's
    /// waiter list.
    remaining_reads: usize,
}

impl RegisteredRequest {
    /// Hand the request back, resolved. Only valid once `remaining_reads` is 0.
    fn into_completed(self) -> CompletedIoDelivery {
        CompletedIoDelivery {
            data_flow_id: self.data_flow_id,
            operator_idx: self.operator_idx,
            completed: CompletedIoRequest {
                state: self.state,
                ranges: self
                    .ranges
                    .into_iter()
                    .map(|parts| {
                        parts
                            .into_iter()
                            .map(PendingRangePart::into_ready)
                            .collect()
                    })
                    .collect(),
            },
        }
    }
}

/// The worker's registration budget, per medium: a dataflow's staged requests
/// stop being registered while a medium's scheduled-but-not-landed reads sit at
/// or above its cap (soft - the request that crosses it is not split up). The
/// unregistered rest stay staged in their [`OperatorIO`]s, holding no cache
/// slots, until in-flight reads land and free budget.
///
/// [`OperatorIO`]: crate::io::OperatorIO
pub struct RegistrationCaps {
    pub disk_reads: usize,
    pub http_reads: usize,
}

/// A completed request together with where to deliver it: the dataflow and
/// operator that pushed it.
pub struct CompletedIoDelivery {
    pub data_flow_id: Identifier,
    pub operator_idx: Identifier,
    pub completed: CompletedIoRequest,
}

/// Tracks every in-flight cache-backed read request on one worker. See the
/// module doc.
#[derive(Default)]
pub struct RequestTracker {
    /// In-flight requests, slot-indexed (`None` = free slot).
    in_flight: Vec<Option<RegisteredRequest>>,
    /// Reusable freed slot indices.
    free_slots: Vec<usize>,
    /// In-flight reads grouped by file: each file → (each scheduled read of it
    /// → the request slots awaiting it). Grouping by file bounds the
    /// containment scan in [`find_containing_read`](Self::find_containing_read)
    /// to one file's reads rather than every file's - a footer scan keeps
    /// dozens of files in flight at once.
    routing: HashMap<OpenFile, HashMap<ReadKey, Vec<usize>>>,
    /// Deduped local reads queued for submission, each tagged with the dataflow
    /// and operator whose request first scheduled it (for stats attribution).
    to_submit_fs: Vec<DataFlowRequest<FsRequest>>,
    /// Deduped remote reads queued for submission.
    to_submit_http: Vec<DataFlowRequest<HttpRequest>>,
    /// Scheduled-but-not-landed reads, per medium - the worker budgets disk and
    /// HTTP registration separately since their latencies differ.
    disk_in_flight: usize,
    http_in_flight: usize,
}

impl RequestTracker {
    /// Register `request` from `operator_idx` of `data_flow_id`: resolve each
    /// of its ranges against the caches and schedule reads for whatever is
    /// missing. Returns the completed request right away when every range was
    /// already resident; otherwise the tracker holds it until its last read
    /// lands.
    pub fn register(
        &mut self,
        data_flow_id: Identifier,
        operator_idx: Identifier,
        request: PendingIoRequest,
    ) -> Option<CompletedIoRequest> {
        let slot = self.claim_slot();
        let mut new_reads = Vec::new();
        let mut remaining_reads = 0;
        let ranges = request
            .ranges
            .iter()
            .map(|&range| {
                self.resolve_range(
                    &request.file,
                    range,
                    request.tiers,
                    slot,
                    &mut new_reads,
                    &mut remaining_reads,
                )
            })
            .collect();

        let registered = RegisteredRequest {
            data_flow_id,
            operator_idx,
            state: request.state,
            ranges,
            remaining_reads,
        };
        if registered.remaining_reads == 0 {
            self.free_slots.push(slot);
            return Some(registered.into_completed().completed);
        }
        self.in_flight[slot] = Some(registered);
        self.queue_new_reads(data_flow_id, operator_idx, &request.file, new_reads);
        None
    }

    /// Resolve one range: decompressed-cache pages first when the request
    /// allows them, compressed-cache bytes for the rest. Every block still
    /// missing is scheduled - `remaining_reads` counts what this request must
    /// wait for, and reads that need their own IO accumulate in `new_reads`.
    fn resolve_range(
        &mut self,
        file: &OpenFile,
        range: FileRange,
        tiers: CacheTiers,
        slot: usize,
        new_reads: &mut Vec<MissingExtent>,
        remaining_reads: &mut usize,
    ) -> Vec<PendingRangePart> {
        let segments =
            match tiers {
                CacheTiers::CompressedOnly => vec![Segment::Gap {
                    offset: range.offset,
                    len: range.len,
                }],
                CacheTiers::DecompressedAndCompressed => memory_ctx()
                    .decompressed_cache()
                    .get_range(file, range.offset, range.len),
            };
        segments
            .into_iter()
            .map(|segment| match segment {
                Segment::Cached {
                    offset,
                    span,
                    header,
                    data,
                } => PendingRangePart::Decompressed {
                    offset,
                    span,
                    header,
                    data,
                },
                Segment::Gap { offset, len } => {
                    let lookups = memory_ctx().compressed_cache().get(file, offset, len);
                    for lookup in &lookups {
                        if let Some(block) = lookup.missing() {
                            *remaining_reads += 1;
                            if self.schedule_or_join(read_key(file, block), slot) {
                                new_reads.push(block.clone());
                            }
                        }
                    }
                    PendingRangePart::Compressed { offset, lookups }
                }
            })
            .collect()
    }

    /// Register `slot` as a waiter on `read`, and report whether `read` is a
    /// *new* physical read that must be issued (vs. one already covered by an
    /// in-flight read, which needs no IO of its own).
    ///
    /// Column chunks aren't 4 KB-aligned, so a column's first block is the same
    /// 4 KB block as the previous column's last. The cache returns that shared
    /// block as a tiny *refill* read that targets the previous column's slot -
    /// the very bytes that column's read is already fetching:
    ///
    /// ```text
    ///   column N   read:  file [4096 ................... 12288)   -> slot S
    ///   column N+1 head:  file           [8192 ......... 12288)   -> also slot S
    ///                                      \_______________/
    ///                          fully inside N's read, same memory: don't fetch
    ///                          it again - wait on N's read. Only N+1's body is
    ///                          new.
    /// ```
    ///
    /// So if `read` is identical to, or fully contained in, an in-flight read,
    /// we add `slot` to that read's waiter list (it is credited when that read
    /// lands) and issue nothing. Otherwise `read` is new: route it and issue it.
    fn schedule_or_join(&mut self, read: ReadKey, slot: usize) -> bool {
        if let Some(waiters) = self.find_containing_read(&read) {
            waiters.push(slot);
            return false;
        }
        self.routing
            .entry(read.open_file.clone())
            .or_default()
            .insert(read, vec![slot]);
        true
    }

    /// The waiter list of the in-flight read of `read`'s file whose slot region
    /// fully contains `read`'s, or `None`. We fold `read` onto that read (issue
    /// no IO) and let it fill `read`'s bytes - which is sound only if they fill
    /// the *same* cache slot. Returns the list directly so the caller pushes
    /// its waiter without re-finding the entry.
    ///
    /// Containment is therefore on `dest` (the slot address), NOT the file
    /// range. They usually agree: a second reader of a block refills into the
    /// slot the first reader already mapped it to. But the file's extent map
    /// can be wiped (`open_entry` re-registers the file, `drop_cache` drains
    /// it) while a read is in flight, after which a re-read of the same file
    /// range *misses* and lands in a different slot - so file-range matching
    /// would wrongly fold it on:
    ///
    /// ```text
    ///   A: get(block0) -> miss -> slot S1; A's read in flight, S1 not filled
    ///   *** extent map wiped: "block0 -> S1" forgotten ***
    ///   B: get(block0) -> miss (block0's slot forgotten) -> fresh slot S2
    ///   file-range match: B's range == A's range -> fold B onto A   [WRONG]
    ///       A fills S1; S2 is never written; B reads S2 -> garbage.
    ///   dest match: A fills S1, B fills S2 -> disjoint -> B issued -> correct.
    /// ```
    ///
    /// A linear scan over that file's in-flight reads (the inner map). That set
    /// is small, and `routing` is the single source of truth for what's
    /// scheduled, so scanning it avoids a second index to keep in lock-step.
    fn find_containing_read(&mut self, read: &ReadKey) -> Option<&mut Vec<usize>> {
        self.routing
            .get_mut(&read.open_file)?
            .iter_mut()
            .find(|(scheduled, _)| {
                scheduled.dest <= read.dest
                    && scheduled.dest + scheduled.len >= read.dest + read.len
            })
            .map(|(_, waiters)| waiters)
    }

    /// Queue the newly scheduled reads for submission, on the queue matching
    /// where the file lives. Each is tagged with the dataflow and operator
    /// whose request first scheduled it, so the worker can attribute the IO in
    /// that dataflow's stats.
    fn queue_new_reads(
        &mut self,
        data_flow_id: Identifier,
        operator_idx: Identifier,
        file: &OpenFile,
        new_reads: Vec<MissingExtent>,
    ) {
        for block in new_reads {
            match file {
                OpenFile::Local(local) => {
                    self.disk_in_flight += 1;
                    self.to_submit_fs.push(DataFlowRequest::new(
                        data_flow_id,
                        operator_idx,
                        FsRequest::Read(FsReadRequest {
                            file: local.clone(),
                            block,
                        }),
                    ));
                }
                OpenFile::Remote(remote) => {
                    self.http_in_flight += 1;
                    self.to_submit_http.push(DataFlowRequest::new(
                        data_flow_id,
                        operator_idx,
                        HttpRequest::Get(HttpGetRequest {
                            remote: remote.clone(),
                            block,
                        }),
                    ));
                }
            }
        }
    }

    /// Take up to [`MAX_SUBMIT_BATCH`] of the deduped local reads queued for
    /// submission to the ring; the rest stay queued for the next pass so we
    /// never overrun the ring.
    pub fn take_fs_submissions(&mut self) -> Vec<DataFlowRequest<FsRequest>> {
        take_batch(&mut self.to_submit_fs)
    }

    /// Take up to [`MAX_SUBMIT_BATCH`] of the deduped remote reads queued for
    /// submission; the rest stay queued for the next pass.
    pub fn take_http_submissions(&mut self) -> Vec<DataFlowRequest<HttpRequest>> {
        take_batch(&mut self.to_submit_http)
    }

    /// Whether any deduped read is still queued for submission.
    pub fn has_reads_to_submit(&self) -> bool {
        !self.to_submit_fs.is_empty() || !self.to_submit_http.is_empty()
    }

    /// A read landed (its bytes committed to their cache slot): credit every
    /// request waiting on it and return the ones that are now complete, ready
    /// to deliver.
    pub fn complete(&mut self, read: &ReadKey) -> Vec<CompletedIoDelivery> {
        let waiters = self.remove_routed_read(read);
        let mut deliveries = Vec::new();
        for slot in waiters {
            // A slot freed by an earlier read failure (its request was dropped
            // and its owner cancelled) may still be referenced by this read's
            // waiter list; there is nothing left to credit.
            let Some(request) = self.in_flight[slot].as_mut() else {
                continue;
            };
            request.remaining_reads -= 1;
            if request.remaining_reads == 0 {
                let request = self.in_flight[slot].take().unwrap();
                self.free_slots.push(slot);
                deliveries.push(request.into_completed());
            }
        }
        deliveries
    }

    /// A read failed terminally: drop every request waiting on it and return
    /// the dataflows those requests belonged to (each listed once), so the
    /// worker can cancel them.
    pub fn fail(&mut self, read: &ReadKey) -> Vec<Identifier> {
        let waiters = self.remove_routed_read(read);
        let mut failed_data_flows = Vec::new();
        for slot in waiters {
            let Some(request) = self.in_flight[slot].take() else {
                continue;
            };
            self.free_slots.push(slot);
            if !failed_data_flows.contains(&request.data_flow_id) {
                failed_data_flows.push(request.data_flow_id);
            }
            // The dropped request may still be listed as a waiter on other
            // in-flight reads. Its slot is about to be reused, so purge those
            // references now - otherwise a later completion would credit
            // whatever new request lands in the slot.
            for file_reads in self.routing.values_mut() {
                for waiters in file_reads.values_mut() {
                    waiters.retain(|&waiter| waiter != slot);
                }
            }
        }
        failed_data_flows
    }

    /// Drop `read` from the routing map and return its waiters, releasing its
    /// in-flight charge. Only *submitted* reads ever come back - a deduped read
    /// never reaches the ring - so the routing entry is present.
    fn remove_routed_read(&mut self, read: &ReadKey) -> Vec<usize> {
        let file_reads = self
            .routing
            .get_mut(&read.open_file)
            .expect("completion for an unrouted file");
        let waiters = file_reads
            .remove(read)
            .expect("completion for an unrouted read");
        // Drop the file's bucket once its last read lands, so the map doesn't
        // accumulate an empty entry per file ever read.
        if file_reads.is_empty() {
            self.routing.remove(&read.open_file);
        }
        // Drop the charge only after confirming the read was routed, so the
        // counter can't underflow ahead of the guards above.
        match read.open_file {
            OpenFile::Local(_) => self.disk_in_flight -= 1,
            OpenFile::Remote(_) => self.http_in_flight -= 1,
        }
        waiters
    }

    /// Occupy a request slot, reusing a freed one when available.
    fn claim_slot(&mut self) -> usize {
        if let Some(slot) = self.free_slots.pop() {
            slot
        } else {
            self.in_flight.push(None);
            self.in_flight.len() - 1
        }
    }

    /// Scheduled-but-not-landed disk reads. The worker gates registration of
    /// further filesystem requests on this.
    pub fn disk_in_flight(&self) -> usize {
        self.disk_in_flight
    }

    /// Scheduled-but-not-landed remote reads. The worker gates registration of
    /// further HTTP requests on this.
    pub fn http_in_flight(&self) -> usize {
        self.http_in_flight
    }
}

/// The identity of the read filling `block` of `file`.
fn read_key(file: &OpenFile, block: &MissingExtent) -> ReadKey {
    ReadKey {
        open_file: file.clone(),
        offset: block.file_offset(),
        len: block.len(),
        dest: block.dest() as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::operator_io::CacheTiers;
    use crate::memory::init_test_free_pool;
    use std::sync::Arc;
    use tempfile::TempDir;

    /// A local file registered with the compressed cache, so lookups against it
    /// can hand out reads. The returned `TempDir` must outlive the file.
    fn registered_file() -> (OpenFile, TempDir) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("data");
        std::fs::write(&path, vec![0u8; 1 << 20]).unwrap();
        let open_file = OpenFile::Local(Arc::new(std::fs::File::open(&path).unwrap()));
        memory_ctx()
            .compressed_cache()
            .open_entry(open_file.clone());
        (open_file, dir)
    }

    /// A raw-bytes request for the given ranges of `file`.
    fn request_for(file: &OpenFile, ranges: Vec<FileRange>) -> PendingIoRequest {
        PendingIoRequest {
            file: file.clone(),
            ranges,
            tiers: CacheTiers::CompressedOnly,
            state: Box::new(()),
        }
    }

    fn range(offset: usize, len: usize) -> FileRange {
        FileRange { offset, len }
    }

    /// Simulate one submitted read landing: commit its block's bytes and
    /// complete it against the tracker, returning the delivered requests.
    fn land_one(
        tracker: &mut RequestTracker,
        submission: &DataFlowRequest<FsRequest>,
    ) -> Vec<CompletedIoDelivery> {
        let read = submission
            .request
            .as_read()
            .expect("the tracker submits only reads");
        read.block.commit();
        tracker.complete(&ReadKey::of_fs(read))
    }

    /// Land every submitted read, returning all delivered requests in landing
    /// order.
    fn land_all(
        tracker: &mut RequestTracker,
        submissions: &[DataFlowRequest<FsRequest>],
    ) -> Vec<CompletedIoDelivery> {
        submissions
            .iter()
            .flat_map(|submission| land_one(tracker, submission))
            .collect()
    }

    #[test]
    fn each_distinct_read_is_submitted_and_counted() {
        init_test_free_pool(4);
        let (file, _dir) = registered_file();
        let mut tracker = RequestTracker::default();

        let completed = tracker.register(
            0,
            0,
            request_for(&file, vec![range(0, 4096), range(65536, 4096)]),
        );

        assert!(completed.is_none());
        assert_eq!(tracker.take_fs_submissions().len(), 2);
        assert_eq!(tracker.disk_in_flight(), 2);
    }

    #[test]
    fn a_request_completes_only_once_its_last_read_lands() {
        init_test_free_pool(4);
        let (file, _dir) = registered_file();
        let mut tracker = RequestTracker::default();
        tracker.register(
            0,
            7,
            request_for(&file, vec![range(0, 4096), range(65536, 4096)]),
        );
        let submissions = tracker.take_fs_submissions();

        assert!(land_one(&mut tracker, &submissions[0]).is_empty());
        let delivered = land_one(&mut tracker, &submissions[1]);

        assert_eq!(delivered.len(), 1);
        assert_eq!(delivered[0].operator_idx, 7);
        assert_eq!(delivered[0].completed.ranges.len(), 2);
    }

    #[test]
    fn two_requests_for_the_same_range_share_one_read() {
        init_test_free_pool(4);
        let (file, _dir) = registered_file();
        let mut tracker = RequestTracker::default();
        tracker.register(1, 0, request_for(&file, vec![range(0, 4096)]));
        tracker.register(2, 0, request_for(&file, vec![range(0, 4096)]));

        let submissions = tracker.take_fs_submissions();

        // One physical read submitted and counted; its landing completes both
        // waiting requests.
        assert_eq!(submissions.len(), 1);
        assert_eq!(tracker.disk_in_flight(), 1);
        let owners: Vec<_> = land_all(&mut tracker, &submissions)
            .iter()
            .map(|delivery| delivery.data_flow_id)
            .collect();
        assert_eq!(owners, vec![1, 2]);
    }

    #[test]
    fn a_read_contained_in_a_scheduled_one_is_deduped() {
        init_test_free_pool(4);
        let (file, _dir) = registered_file();
        // Schedule [0,8192). A later [0,4096) is fully inside it (it refills
        // into the same slot), so it rides on the bigger read instead of
        // issuing its own.
        let mut tracker = RequestTracker::default();
        tracker.register(1, 0, request_for(&file, vec![range(0, 8192)]));
        tracker.register(2, 0, request_for(&file, vec![range(0, 4096)]));

        let submissions = tracker.take_fs_submissions();

        assert_eq!(submissions.len(), 1);
        assert_eq!(land_all(&mut tracker, &submissions).len(), 2);
    }

    #[test]
    fn a_reread_into_a_different_slot_is_not_folded_onto_the_stale_read() {
        init_test_free_pool(8);
        let (file, _dir) = registered_file();
        // Register a read of [0,4096), then re-register the file so the same
        // range re-reads into a different slot. Same file range, different
        // memory: the second read must NOT be folded onto the first - doing so
        // would credit it when the first's slot fills and leave its own slot
        // full of garbage.
        let mut tracker = RequestTracker::default();
        tracker.register(1, 0, request_for(&file, vec![range(0, 4096)]));
        memory_ctx().compressed_cache().open_entry(file.clone());
        tracker.register(2, 0, request_for(&file, vec![range(0, 4096)]));

        assert_eq!(tracker.take_fs_submissions().len(), 2);
        assert_eq!(tracker.disk_in_flight(), 2);
    }

    #[test]
    fn a_shared_column_boundary_block_rides_on_the_previous_column() {
        init_test_free_pool(8);
        let (file, _dir) = registered_file();
        // Column N covers blocks 0,1 (file [0,8192)). Column N+1 covers blocks
        // 1,2 (file [4096,12288)), sharing block 1. The cache splits N+1 into a
        // block-1 refill (into N's slot) plus a block-2 body read; the refill
        // is contained in N's read, so only N's read and N+1's body are issued.
        let mut tracker = RequestTracker::default();
        tracker.register(1, 0, request_for(&file, vec![range(0, 8192)]));
        tracker.register(2, 0, request_for(&file, vec![range(4096, 8192)]));

        let submissions = tracker.take_fs_submissions();

        // N's read credits N and N+1's shared block; N+1 still needs its body,
        // so it is delivered only after both reads land.
        assert_eq!(submissions.len(), 2);
        let owners: Vec<_> = land_all(&mut tracker, &submissions)
            .iter()
            .map(|delivery| delivery.data_flow_id)
            .collect();
        assert_eq!(owners, vec![1, 2]);
    }

    #[test]
    fn a_cached_range_completes_at_registration() {
        init_test_free_pool(4);
        let (file, _dir) = registered_file();
        // Land [0,4096) for one request, then register another for the same
        // range: everything is resident, so it completes with no IO.
        let mut tracker = RequestTracker::default();
        tracker.register(1, 0, request_for(&file, vec![range(0, 4096)]));
        let submissions = tracker.take_fs_submissions();
        land_all(&mut tracker, &submissions);

        let completed = tracker.register(2, 0, request_for(&file, vec![range(0, 4096)]));

        assert!(completed.is_some());
        assert!(tracker.take_fs_submissions().is_empty());
    }

    #[test]
    fn containment_does_not_cross_files() {
        init_test_free_pool(8);
        let (file_a, _a) = registered_file();
        let (file_b, _b) = registered_file();
        // The same byte range in two different files is two unrelated reads.
        let mut tracker = RequestTracker::default();
        tracker.register(1, 0, request_for(&file_a, vec![range(0, 8192)]));
        tracker.register(2, 0, request_for(&file_b, vec![range(0, 4096)]));

        assert_eq!(tracker.take_fs_submissions().len(), 2);
        assert_eq!(tracker.disk_in_flight(), 2);
    }

    #[test]
    fn a_failed_read_reports_every_waiting_dataflow_once() {
        init_test_free_pool(4);
        let (file, _dir) = registered_file();
        // Three requests wait on one read, two of them from the same dataflow.
        let mut tracker = RequestTracker::default();
        tracker.register(1, 0, request_for(&file, vec![range(0, 4096)]));
        tracker.register(1, 3, request_for(&file, vec![range(0, 4096)]));
        tracker.register(2, 0, request_for(&file, vec![range(0, 4096)]));
        let submissions = tracker.take_fs_submissions();
        assert_eq!(submissions.len(), 1);

        let read = ReadKey::of_fs(submissions[0].request.as_read().unwrap());

        assert_eq!(tracker.fail(&read), vec![1, 2]);
    }

    #[test]
    fn a_read_landing_after_its_waiter_failed_delivers_nothing() {
        init_test_free_pool(8);
        let (file, _dir) = registered_file();
        // One request waits on two reads. The first fails (killing the
        // request); the second still lands and must complete cleanly with
        // nobody left to deliver to.
        let mut tracker = RequestTracker::default();
        tracker.register(
            1,
            0,
            request_for(&file, vec![range(0, 4096), range(65536, 4096)]),
        );
        let submissions = tracker.take_fs_submissions();
        assert_eq!(submissions.len(), 2);

        let failed = ReadKey::of_fs(submissions[0].request.as_read().unwrap());

        assert_eq!(tracker.fail(&failed), vec![1]);
        assert!(land_one(&mut tracker, &submissions[1]).is_empty());
    }

    #[test]
    fn submissions_are_capped_per_batch_and_drain_across_passes() {
        init_test_free_pool(8);
        let (file, _dir) = registered_file();
        let ranges: Vec<FileRange> = (0..MAX_SUBMIT_BATCH + 5)
            .map(|i| range(i * 4096, 4096))
            .collect();
        let mut tracker = RequestTracker::default();
        tracker.register(0, 0, request_for(&file, ranges));

        let first = tracker.take_fs_submissions();
        let second = tracker.take_fs_submissions();
        let third = tracker.take_fs_submissions();

        assert_eq!(first.len(), MAX_SUBMIT_BATCH);
        assert_eq!(second.len(), 5);
        assert!(third.is_empty());
    }
}
