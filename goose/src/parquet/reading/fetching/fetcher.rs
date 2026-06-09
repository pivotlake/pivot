//! The [`RowGroupFetcher`]: drives the column-chunk IO for in-flight row groups.
//!
//! It accepts [`RowGroupRequest`]s and reads each one's projected column chunks
//! — local files via io_uring disk reads, remote objects via HTTP range reads on
//! the same ring — emitting a [`RowGroupBuffer`] once a row group's last block
//! lands.
//!
//! Disk and HTTP have very different latencies, so the fetcher bounds how many
//! row groups of each kind are outstanding independently: at most
//! [`MAX_DISK_IN_FLIGHT`] disk-backed and [`MAX_HTTP_IN_FLIGHT`] remote-backed.
//! A disk read on the ring is one SQE → one CQE, so one row group at a time
//! already keeps io_uring busy; an HTTP read is a round trip (tens of ms), so we
//! keep many in flight to hide latency. It stops pulling new row groups when
//! *either* pool is full — a homogeneous scan only ever fills the pool for its
//! medium, so the other never gates (an all-remote scan keeps `disk_in_flight`
//! at 0, an all-local scan keeps `http_in_flight` at 0).
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

/// Disk-backed row groups outstanding per worker. One at a time — io_uring
/// already gives a single row group's reads ample depth.
const MAX_DISK_IN_FLIGHT: usize = 1;
/// Remote-backed row groups outstanding per worker. Larger, to hide HTTP
/// round-trip latency; bounds pinned cache slots per worker.
const MAX_HTTP_IN_FLIGHT: usize = 32;

#[derive(Default)]
pub struct RowGroupFetcher {
    /// In-flight row groups, slot-indexed (`None` = free slot).
    in_flight: Vec<Option<RowGroupRequest>>,
    /// Reusable freed slot indices.
    free_slots: Vec<usize>,
    /// `(location, file offset)` → slots awaiting a completion at that key.
    routing: HashMap<(FileLocation, usize), VecDeque<usize>>,
    /// Outstanding disk-backed row groups (capped at [`MAX_DISK_IN_FLIGHT`]).
    disk_in_flight: usize,
    /// Outstanding remote-backed row groups (capped at [`MAX_HTTP_IN_FLIGHT`]).
    http_in_flight: usize,
}

impl RowGroupFetcher {
    fn alloc_slot(&mut self, request: RowGroupRequest) -> usize {
        if request.is_remote() {
            self.http_in_flight += 1;
        } else {
            self.disk_in_flight += 1;
        }
        if let Some(slot) = self.free_slots.pop() {
            self.in_flight[slot] = Some(request);
            slot
        } else {
            self.in_flight.push(Some(request));
            self.in_flight.len() - 1
        }
    }

    /// If the row group in `slot` has all its blocks, free its pool charge and
    /// emit it.
    fn emit_if_complete<S: Sender<RowGroupBuffer>>(
        &mut self,
        slot: usize,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        if self.in_flight[slot].as_ref().is_some_and(|r| r.complete()) {
            let request = self.in_flight[slot].take().unwrap();
            if request.is_remote() {
                self.http_in_flight -= 1;
            } else {
                self.disk_in_flight -= 1;
            }
            self.free_slots.push(slot);
            sender.send(request.into_row_group_buffer())?;
        }
        Ok(())
    }

    /// A completed read (fs or http): the requester already committed its bytes
    /// into the cache slot, so route it to the owning row group and count it off.
    fn process_completion<S: Sender<RowGroupBuffer>>(
        &mut self,
        key: (FileLocation, usize),
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        let (slot, drained) = match self.routing.get_mut(&key) {
            Some(waiters) => (waiters.pop_front(), waiters.is_empty()),
            None => (None, false),
        };
        if drained {
            self.routing.remove(&key);
        }
        if let Some(slot) = slot {
            if let Some(rg) = self.in_flight[slot].as_mut() {
                rg.complete_one();
            }
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
        // Record where each queued read should route its completion. Peek the
        // pending reads now (they're drained later by `next_*_requests`).
        let mut keys: Vec<(FileLocation, usize)> = Vec::new();
        {
            let rg = self.in_flight[slot].as_mut().unwrap();
            for fs in rg.pending_fs().iter() {
                keys.push((FileLocation::Local(fs.fd), fs.block.file_offset()));
            }
        }
        {
            let rg = self.in_flight[slot].as_mut().unwrap();
            for http in rg.pending_http().iter() {
                keys.push((
                    FileLocation::Remote(http.remote.clone()),
                    http.block.file_offset(),
                ));
            }
        }
        for key in keys {
            self.routing.entry(key).or_default().push_back(slot);
        }
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
        let key = (FileLocation::Local(request.fd), request.block.file_offset());
        self.process_completion(key, sender)
    }

    fn process_http_response<S: Sender<RowGroupBuffer>>(
        &mut self,
        sender: &mut S,
        request: HttpRequest,
    ) -> dispatch::UnaryResult<()> {
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
