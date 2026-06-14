//! The [`RowGroupFetcher`]: drives the column-chunk IO for in-flight row groups.
//!
//! It accepts [`RowGroupRequest`]s and reads each one's projected column chunks
//! — local files via io_uring disk reads, remote objects via HTTP range reads on
//! the same ring — emitting a [`RowGroupBuffer`] once a row group's last block
//! lands.
//!
//! Disk and HTTP have very different latencies, so the fetcher bounds how many
//! column-chunk reads (missing *blocks*) of each kind are outstanding
//! independently: at most [`MAX_DISK_IN_FLIGHT`] disk-backed and
//! [`MAX_HTTP_IN_FLIGHT`] remote-backed. A disk read on the ring is one SQE →
//! one CQE, so one at a time already keeps io_uring busy; an HTTP read is a
//! round trip (tens of ms), so we keep many in flight to hide latency. Counting
//! *blocks* (not row groups) makes the cap a soft one and bounds the actual
//! concurrent reads directly: a row group may stage more blocks than the cap,
//! but the fetcher then stops pulling the *next* row group until they drain. A
//! homogeneous scan only ever fills the pool for its medium, so the other never
//! gates (an all-remote scan keeps `disk_in_flight` at 0, an all-local scan
//! keeps `http_in_flight` at 0).
//!
//! Reads are keyed by [`RowGroupRead`] — `(location, offset, len)`, i.e. the
//! exact byte run, not just the offset. When a row group is admitted, each of
//! its reads is registered against that key; if the *same* read is already
//! scheduled (another waiter, or another column of the same row group), the new
//! waiter just joins the existing route and no duplicate read is issued. A
//! completion for a key then credits *every* waiter on it — sound because they
//! all need exactly that run, whose bytes are now committed. (Keying on offset
//! alone would let a short read wrongly satisfy a waiter needing a longer run at
//! the same offset.)

use crate::parquet::types::requests::{RowGroupBuffer, RowGroupRead, RowGroupRequest};
use dispatch::Sender;
use dispatch::Unary;
use dispatch::io::{FsRequest, HttpRequest};
use std::collections::HashMap;

/// Disk-backed read blocks outstanding per worker before admitting another row
/// group. One keeps disk reads serial — io_uring gives a single read ample
/// depth.
const MAX_DISK_IN_FLIGHT: usize = 1;
/// Remote-backed read blocks (≈ concurrent HTTP round trips) outstanding per
/// worker before admitting another row group. Larger, to hide HTTP latency;
/// also bounds pinned cache slots per worker.
const MAX_HTTP_IN_FLIGHT: usize = 32;

#[derive(Default)]
pub struct RowGroupFetcher {
    /// In-flight row groups, slot-indexed (`None` = free slot).
    in_flight: Vec<Option<RowGroupRequest>>,
    /// Reusable freed slot indices.
    free_slots: Vec<usize>,
    /// Each scheduled [`RowGroupRead`] → the slots awaiting it. Doubles as the
    /// dedup index: a read already present here is in flight, so a new waiter
    /// joins the list instead of issuing the read again.
    routing: HashMap<RowGroupRead, Vec<usize>>,
    /// Local filesystem reads queued but not yet submitted to the ring (deduped;
    /// drained by [`next_fs_requests`](Self::next_fs_requests)).
    pending_fs: Vec<FsRequest>,
    /// Remote HTTP reads queued but not yet submitted.
    pending_http: Vec<HttpRequest>,
    /// Outstanding disk-backed read blocks (soft-capped at [`MAX_DISK_IN_FLIGHT`]
    /// when admitting new row groups).
    disk_in_flight: usize,
    /// Outstanding remote-backed read blocks (soft-capped at
    /// [`MAX_HTTP_IN_FLIGHT`]).
    http_in_flight: usize,
}

impl RowGroupFetcher {
    fn alloc_slot(&mut self, request: RowGroupRequest) -> usize {
        if let Some(slot) = self.free_slots.pop() {
            self.in_flight[slot] = Some(request);
            slot
        } else {
            self.in_flight.push(Some(request));
            self.in_flight.len() - 1
        }
    }

    /// Register the row group in `slot`'s reads: route each back to `slot`, and
    /// for any read not already scheduled, queue it for submission and count it
    /// toward the in-flight cap. A read already in `routing` is deduped — the
    /// new waiter just joins it, no second read is issued or counted.
    ///
    /// Counting per submitted read (not per row group) makes the cap soft: a row
    /// group may push over it, but [`ready_for_more_work`](Self::ready_for_more_work)
    /// then stops admitting the *next* until they drain.
    fn stage_reads(&mut self, slot: usize) {
        let (fs, http) = {
            let rg = self.in_flight[slot].as_mut().unwrap();
            (
                std::mem::take(rg.pending_fs()),
                std::mem::take(rg.pending_http()),
            )
        };
        for req in fs {
            let read = RowGroupRead::of_fs(&req);
            if let Some(waiters) = self.routing.get_mut(&read) {
                waiters.push(slot);
            } else {
                self.routing.insert(read, vec![slot]);
                self.pending_fs.push(req);
                self.disk_in_flight += 1;
            }
        }
        for req in http {
            let read = RowGroupRead::of_http(&req);
            if let Some(waiters) = self.routing.get_mut(&read) {
                waiters.push(slot);
            } else {
                self.routing.insert(read, vec![slot]);
                self.pending_http.push(req);
                self.http_in_flight += 1;
            }
        }
    }

    /// If the row group in `slot` has all its blocks, free it and emit it.
    fn emit_if_complete<S: Sender<RowGroupBuffer>>(
        &mut self,
        slot: usize,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        if self.in_flight[slot].as_ref().is_some_and(|r| r.complete()) {
            let request = self.in_flight[slot].take().unwrap();
            self.free_slots.push(slot);
            sender.send(request.into_row_group_buffer())?;
        }
        Ok(())
    }

    /// A completed read (fs or http): its bytes are now committed in the cache,
    /// so credit *every* row group waiting on this exact read and emit any that
    /// are now done. The read was registered (and submitted once) in `consume`,
    /// and the ring delivers one completion per submitted read, so the entry is
    /// present. Credit all before emitting, so a slot listed twice (a row group
    /// that needed this read for two columns) is fully counted before it emits.
    fn process_completion<S: Sender<RowGroupBuffer>>(
        &mut self,
        read: RowGroupRead,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        let waiters = self
            .routing
            .remove(&read)
            .expect("completion for an unrouted read");
        for &slot in &waiters {
            self.in_flight[slot].as_mut().unwrap().complete_one();
        }
        for slot in waiters {
            self.emit_if_complete(slot, sender)?;
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
        let slot = self.alloc_slot(request);
        self.stage_reads(slot);
        // A fully-cached row group has no pending reads and is already done.
        self.emit_if_complete(slot, sender)?;
        Ok(())
    }

    fn next_fs_requests(&mut self) -> dispatch::UnaryResult<Vec<FsRequest>> {
        Ok(std::mem::take(&mut self.pending_fs))
    }

    fn next_http_requests(&mut self) -> dispatch::UnaryResult<Vec<HttpRequest>> {
        Ok(std::mem::take(&mut self.pending_http))
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.disk_in_flight < MAX_DISK_IN_FLIGHT && self.http_in_flight < MAX_HTTP_IN_FLIGHT
    }

    fn process_fs_response<S: Sender<RowGroupBuffer>>(
        &mut self,
        sender: &mut S,
        request: FsRequest,
    ) -> dispatch::UnaryResult<()> {
        self.disk_in_flight -= 1;
        self.process_completion(RowGroupRead::of_fs(&request), sender)
    }

    fn process_http_response<S: Sender<RowGroupBuffer>>(
        &mut self,
        sender: &mut S,
        request: HttpRequest,
    ) -> dispatch::UnaryResult<()> {
        self.http_in_flight -= 1;
        self.process_completion(RowGroupRead::of_http(&request), sender)
    }

    fn finish<S: Sender<RowGroupBuffer>>(
        &mut self,
        _sender: &mut S,
    ) -> dispatch::UnaryResult<bool> {
        Ok(self.disk_in_flight == 0 && self.http_in_flight == 0)
    }
}
