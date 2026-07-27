//! The [`RowGroupFetcher`]: drives the column-chunk IO for in-flight row groups.
//!
//! It accepts [`RowGroupRequest`]s and reads each one's projected column chunks
//! — local files via io_uring disk reads, remote objects via HTTP range reads on
//! the same ring — emitting a [`RowGroupBuffer`] once a row group's last block
//! lands. The slot/routing/dedup/in-flight bookkeeping lives in the shared
//! [`RequestTracker`]; this type adds only the row-group-specific bits: count a
//! landed block off the waiting row group, and emit it once complete.
//!
//! Disk and HTTP have very different latencies, so the tracker bounds the
//! outstanding reads of each kind independently: at most `MAX_DISK_IN_FLIGHT`
//! disk-backed and `MAX_HTTP_IN_FLIGHT` remote-backed blocks. The cap is soft
//! (a row group may push over it) and gates only whether the *next* row group is
//! admitted; a homogeneous scan only fills its medium's pool, so the other never
//! gates.

use crate::parquet::request_tracker::{ReadRequest, RequestTracker};
use crate::parquet::types::requests::{RowGroupBuffer, RowGroupRequest};
use dispatch::Sender;
use dispatch::Unary;
use dispatch::io::{FsReadRequest, FsRequest, HttpGetRequest, HttpRequest};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Disk-backed read blocks outstanding per worker before admitting another row
/// group. One keeps disk reads serial — io_uring gives a single read ample
/// depth.
const MAX_DISK_IN_FLIGHT: usize = 1;

/// Row groups a worker may hold "claimed but not yet fully decoded" before it
/// stops claiming more, when the table's files are local: the one it is
/// decoding plus one prefetched next, the minimum that keeps its pipeline
/// overlapped. A row group's page decode runs only on the worker that claimed
/// it (the decoder holds per-row-group state), so every claim beyond this
/// bound serializes decode work on one worker while others may sit idle. That
/// matters most when a scan has few row groups per worker: its wall time is
/// `max claims per worker x per-row-group cost`, and a single worker hoarding
/// a third row group while the queue runs dry sets the whole query's critical
/// path.
const MAX_PENDING_LOCAL_ROW_GROUPS: usize = 2;

/// The remote-file counterpart of [`MAX_PENDING_LOCAL_ROW_GROUPS`]. A remote
/// claim spends most of its life waiting on the network, not decoding, so the
/// hoarding concern above barely applies while read-ahead depth is the whole
/// throughput game: a claim's reads must be in flight long before its decode
/// is due, or every row group pays a full network round trip serially.
const MAX_PENDING_REMOTE_ROW_GROUPS: usize = 16;

/// How many undecoded row groups one worker may hold claims on while
/// scanning `table`. Local tables keep claims minimal for decode fairness;
/// tables with any remote file get read-ahead depth.
pub fn pending_claim_bound(table: &crate::parquet::ParquetTable) -> usize {
    let any_remote = table
        .row_groups()
        .iter()
        .any(|rg| matches!(rg.open_file, dispatch::io::OpenFile::Remote(_)));
    if any_remote {
        MAX_PENDING_REMOTE_ROW_GROUPS
    } else {
        MAX_PENDING_LOCAL_ROW_GROUPS
    }
}

pub struct RowGroupFetcher {
    tracker: RequestTracker<RowGroupRequest>,
    /// How many row groups this worker has claimed but not fully decoded.
    /// Incremented here per claim, decremented by the worker's `Decoder` when
    /// a row group finishes (or prunes), and consulted as claim backpressure.
    pending_row_groups: Arc<AtomicUsize>,
    /// The claim bound for this scan (see [`pending_claim_bound`]).
    max_pending_row_groups: usize,
}

impl RowGroupFetcher {
    pub fn new(pending_row_groups: Arc<AtomicUsize>, max_pending_row_groups: usize) -> Self {
        Self {
            tracker: RequestTracker::default(),
            pending_row_groups,
            max_pending_row_groups,
        }
    }

    /// A completed read: credit *every* row group waiting on it (they all need
    /// exactly that run, whose bytes are now committed), then emit any that are
    /// done. Credit all before emitting, so a row group listed twice — it needed
    /// this read for two columns — is fully counted before it emits.
    fn deliver<S: Sender<RowGroupBuffer>>(
        &mut self,
        read: ReadRequest,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        let waiters = self.tracker.complete(&read);
        for &slot in &waiters {
            self.tracker.request_for_slot(slot).unwrap().complete_one();
        }
        for slot in waiters {
            self.emit_if_complete(slot, sender)?;
        }
        Ok(())
    }

    /// If the row group in `slot` has all its blocks, free it and emit it.
    fn emit_if_complete<S: Sender<RowGroupBuffer>>(
        &mut self,
        slot: usize,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        if self
            .tracker
            .request_for_slot(slot)
            .is_some_and(|r| r.complete())
        {
            let request = self.tracker.take_request_at_slot(slot);
            sender.send(request.into_row_group_buffer())?;
        }
        Ok(())
    }
}

impl Unary<RowGroupRequest, RowGroupBuffer> for RowGroupFetcher {
    fn consume<S: Sender<RowGroupBuffer>>(
        &mut self,
        request: RowGroupRequest,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        self.pending_row_groups.fetch_add(1, Ordering::Relaxed);
        let slot = self.tracker.admit_request(request);
        // A fully-cached row group has no pending reads and is already done.
        self.emit_if_complete(slot, sender)?;
        Ok(())
    }

    fn next_fs_requests(&mut self) -> dispatch::UnaryResult<Vec<FsRequest>> {
        Ok(self.tracker.take_fs_requests())
    }

    fn next_http_requests(&mut self) -> dispatch::UnaryResult<Vec<HttpRequest>> {
        Ok(self.tracker.take_http_requests())
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.tracker.disk_in_flight() < MAX_DISK_IN_FLIGHT
            && self.tracker.http_in_flight() < crate::parquet::http_readahead()
            && self.pending_row_groups.load(Ordering::Relaxed) < self.max_pending_row_groups
    }

    fn process_fs_read_response<S: Sender<RowGroupBuffer>>(
        &mut self,
        sender: &mut S,
        request: FsReadRequest,
    ) -> dispatch::UnaryResult<()> {
        self.deliver(ReadRequest::of_fs(&request), sender)
    }

    fn process_http_get_response<S: Sender<RowGroupBuffer>>(
        &mut self,
        sender: &mut S,
        request: HttpGetRequest,
    ) -> dispatch::UnaryResult<()> {
        self.deliver(ReadRequest::of_http_get(&request), sender)
    }

    fn finish<S: Sender<RowGroupBuffer>>(
        &mut self,
        _sender: &mut S,
    ) -> dispatch::UnaryResult<bool> {
        Ok(self.tracker.is_idle())
    }
}
