//! The [`RemoteRowGroupFetcher`]: like [`RowGroupFetcher`](super::fetcher::RowGroupFetcher)
//! but tuned for HTTP rather than disk.
//!
//! A disk read on the ring is one SQE → one CQE with negligible latency, so the
//! disk fetcher processes one row group at a time. An HTTP range read is a
//! round trip (tens of ms), so serializing row groups would leave the network
//! idle between them. This fetcher therefore keeps **many row groups in flight
//! per worker** — it accepts new input while up to `max_in_flight` are
//! outstanding, issues all their range reads together, and emits each row group
//! the moment its last block lands. Completions are routed back to the right
//! in-flight row group by their `(location, file offset)`.

use crate::parquet::types::requests::{RowGroupBuffer, RowGroupRequest};
use dispatch::Sender;
use dispatch::Unary;
use dispatch::io::{FileLocation, HttpRequest, IORequest};
use std::collections::HashMap;
use std::collections::VecDeque;

/// How many row groups a single worker keeps outstanding. HTTP latency is the
/// thing we're hiding, so this is comfortably larger than 1 (the disk fetcher's
/// effective depth); the cap bounds memory (pinned cache slots) per worker.
const DEFAULT_MAX_IN_FLIGHT: usize = 32;

pub struct RemoteRowGroupFetcher {
    /// In-flight row groups, slot-indexed (`None` = free slot).
    in_flight: Vec<Option<RowGroupRequest>>,
    /// Reusable freed slot indices.
    free_slots: Vec<usize>,
    /// Maps a queued read's `(location, file offset)` to the slots awaiting a
    /// completion at that key, so a completion is attributed to the right row
    /// group. It's a queue, not a single slot, because a row group can issue
    /// more than one read at the same offset (two small column chunks sharing a
    /// cache region), and completions for a key are interchangeable (the bytes
    /// are already committed) — so any waiter can claim one.
    routing: HashMap<(FileLocation, usize), VecDeque<usize>>,
    /// Count of occupied slots (cheap backpressure check).
    active: usize,
    max_in_flight: usize,
}

impl Default for RemoteRowGroupFetcher {
    fn default() -> Self {
        Self {
            in_flight: Vec::new(),
            free_slots: Vec::new(),
            routing: HashMap::new(),
            active: 0,
            max_in_flight: DEFAULT_MAX_IN_FLIGHT,
        }
    }
}

impl RemoteRowGroupFetcher {
    fn alloc_slot(&mut self, request: RowGroupRequest) -> usize {
        self.active += 1;
        if let Some(slot) = self.free_slots.pop() {
            self.in_flight[slot] = Some(request);
            slot
        } else {
            self.in_flight.push(Some(request));
            self.in_flight.len() - 1
        }
    }

    /// If the row group in `slot` has all its blocks, emit it and free the slot.
    fn emit_if_complete<S: Sender<RowGroupBuffer>>(
        &mut self,
        slot: usize,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        if self.in_flight[slot].as_ref().is_some_and(|r| r.complete()) {
            let request = self.in_flight[slot].take().unwrap();
            self.free_slots.push(slot);
            self.active -= 1;
            sender.send(request.into_row_group_buffer())?;
        }
        Ok(())
    }
}

impl Unary<RowGroupRequest, RowGroupBuffer> for RemoteRowGroupFetcher {
    fn consume<S: Sender<RowGroupBuffer>>(
        &mut self,
        request: RowGroupRequest,
        sender: &mut S,
    ) -> dispatch::UnaryResult<()> {
        let slot = self.alloc_slot(request);
        // Record where each queued read should route its completion. Peek the
        // pending reads now (they're drained later by `next_http_requests`).
        let keys: Vec<(FileLocation, usize)> = self.in_flight[slot]
            .as_mut()
            .unwrap()
            .pending_http()
            .iter()
            .map(|http| {
                (
                    FileLocation::Remote(http.remote.clone()),
                    http.block.file_offset(),
                )
            })
            .collect();
        for key in keys {
            self.routing.entry(key).or_default().push_back(slot);
        }
        // A fully-cached row group has no pending reads and is already done.
        self.emit_if_complete(slot, sender)?;
        Ok(())
    }

    fn next_http_requests(&mut self) -> dispatch::UnaryResult<Vec<HttpRequest>> {
        let mut all = Vec::new();
        for request in self.in_flight.iter_mut().flatten() {
            all.append(request.pending_http());
        }
        Ok(all)
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.active < self.max_in_flight
    }

    fn process_io_response<S: Sender<RowGroupBuffer>>(
        &mut self,
        sender: &mut S,
        request: IORequest,
    ) -> dispatch::UnaryResult<()> {
        // The requester already committed this block's bytes into its cache
        // slot; route the completion to the owning row group and count it off.
        let key = (request.location, request.block.file_offset());
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

    fn finish<S: Sender<RowGroupBuffer>>(
        &mut self,
        _sender: &mut S,
    ) -> dispatch::UnaryResult<bool> {
        Ok(self.active == 0)
    }
}
