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
//! Completions are routed back to the in-flight row group that issued them by
//! their `(location, file offset)`. The route is a queue, not a single slot,
//! because a row group can issue more than one read at the same offset (two
//! small column chunks sharing a cache region) and completions for a key are
//! interchangeable (the bytes are already committed) — so any waiter claims one.

use crate::parquet::types::requests::{RowGroupBuffer, RowGroupRequest};
use dispatch::Sender;
use dispatch::Unary;
use dispatch::io::{FileLocation, FsRequest, HttpRequest};
use std::collections::HashMap;
use std::collections::VecDeque;

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
    /// `(location, file offset)` → slots awaiting a completion at that key. A
    /// queue, not a single slot: two column chunks sharing a cache region issue
    /// two reads at the same key, so a key can have several waiters.
    routing: HashMap<(FileLocation, usize), VecDeque<usize>>,
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

    /// Count the row group in `slot`'s reads toward the in-flight cap — one per
    /// missing block, not per row group — and route each block's completion back
    /// to `slot`. A request is local xor remote, so only one counter moves.
    ///
    /// Counting per block makes the cap soft: a row group may stage more blocks
    /// than the cap, but [`ready_for_more_work`](Self::ready_for_more_work) then
    /// stops admitting the *next* row group until they drain.
    fn stage_reads(&mut self, slot: usize) {
        let rg = self.in_flight[slot].as_mut().unwrap();
        let fs_keys: Vec<(FileLocation, usize)> = rg
            .pending_fs()
            .iter()
            .map(|fs| (FileLocation::Local(fs.file.clone()), fs.block.file_offset()))
            .collect();
        let http_keys: Vec<(FileLocation, usize)> = rg
            .pending_http()
            .iter()
            .map(|http| (FileLocation::Remote(http.remote.clone()), http.block.file_offset()))
            .collect();
        self.disk_in_flight += fs_keys.len();
        self.http_in_flight += http_keys.len();
        for key in fs_keys.into_iter().chain(http_keys) {
            self.routing.entry(key).or_default().push_back(slot);
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

    /// A completed read (fs or http): the requester already committed its bytes
    /// into the cache slot, so route it to the owning row group and count it off.
    ///
    /// Every submitted read registered a waiter in `consume`, and the ring
    /// delivers exactly one completion per read, so the key is always present
    /// with a non-empty queue and the slot is still in flight. The queue can hold
    /// more than one waiter (two column chunks sharing a cache region issue two
    /// reads at the same key); any waiter claims a completion, since the bytes are
    /// already committed.
    fn process_completion<S: Sender<RowGroupBuffer>>(
        &mut self,
        key: (FileLocation, usize),
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        let waiters = self
            .routing
            .get_mut(&key)
            .expect("completion for an unrouted read");
        let slot = waiters.pop_front().expect("empty routing queue");
        if waiters.is_empty() {
            self.routing.remove(&key);
        }
        self.in_flight[slot].as_mut().unwrap().complete_one();
        self.emit_if_complete(slot, sender)?;
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
        let mut all = Vec::new();
        for rg in self.in_flight.iter_mut().flatten() {
            all.append(rg.pending_fs());
        }
        Ok(all)
    }

    fn next_http_requests(&mut self) -> dispatch::UnaryResult<Vec<HttpRequest>> {
        let mut all = Vec::new();
        for rg in self.in_flight.iter_mut().flatten() {
            all.append(rg.pending_http());
        }
        Ok(all)
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
        let key = (
            FileLocation::Local(request.file),
            request.block.file_offset(),
        );
        self.process_completion(key, sender)
    }

    fn process_http_response<S: Sender<RowGroupBuffer>>(
        &mut self,
        sender: &mut S,
        request: HttpRequest,
    ) -> dispatch::UnaryResult<()> {
        self.http_in_flight -= 1;
        let key = (
            FileLocation::Remote(request.remote),
            request.block.file_offset(),
        );
        self.process_completion(key, sender)
    }

    fn finish<S: Sender<RowGroupBuffer>>(
        &mut self,
        _sender: &mut S,
    ) -> dispatch::UnaryResult<bool> {
        Ok(self.disk_in_flight == 0 && self.http_in_flight == 0)
    }
}
