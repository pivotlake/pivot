//! Holds a row group's rows back, undecoded, until the merge is about to
//! need them.
//!
//! A row group is fetched and decompressed ahead of the merge, and cut into
//! the ranges of rows it is decoded in. As pages it is small. Decoded, it is
//! many times larger, and the merge only needs the next rows of each file.
//! So the ranges wait at the [`DecodeGate`], and a range goes on to be
//! decoded only when the planner releases its rows.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use dispatch::waker::WakerSet;
use dispatch::{
    BuildContext, Built, BuiltUnary, DataFlowDispatcher, Sender, Unary, UnaryFactory, UnaryResult,
    WorkStatus,
};

use crate::DecodeRange;

/// How many rows of each input row group the planner has released to be
/// decoded, from the row group's first row.
pub(super) struct DecodeReleases {
    /// One count per row group of the table.
    rows_released: Vec<AtomicU32>,
    /// The worker the gate runs on, woken when rows are released.
    gate_worker: usize,
    waker_set: WakerSet,
}

impl DecodeReleases {
    /// Nothing released yet, of a table of `row_group_count` row groups,
    /// for a gate on `gate_worker`, a worker of `dispatcher`.
    pub(super) fn new(
        dispatcher: &DataFlowDispatcher,
        row_group_count: usize,
        gate_worker: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            rows_released: (0..row_group_count).map(|_| AtomicU32::new(0)).collect(),
            gate_worker,
            waker_set: dispatcher.waker_set(),
        })
    }

    /// Let the first `row_count` rows of `row_group` go on to be decoded,
    /// now or when they have been read.
    pub(super) fn release_rows(&self, row_group: usize, row_count: usize) {
        self.rows_released[row_group].store(row_count as u32, Ordering::Release);
        self.waker_set.notify_worker(self.gate_worker);
    }
}

/// Passes a range of rows on to be decoded once its rows are released, and
/// holds it until then. Every range is routed to one worker, the compaction
/// worker; the other workers' gates receive nothing.
pub(super) struct DecodeGate {
    releases: Arc<DecodeReleases>,
    /// The ranges whose rows are not released yet.
    held: Vec<DecodeRange>,
}

impl DecodeGate {
    /// One gate per worker.
    pub(super) fn factories(releases: &Arc<DecodeReleases>, worker_count: usize) -> Vec<Self> {
        (0..worker_count)
            .map(|_| Self {
                releases: releases.clone(),
                held: Vec::new(),
            })
            .collect()
    }

    /// Whether the rows of `range` are released. The planner releases whole
    /// ranges, so a range is released when its first row is.
    fn is_released(&self, range: &DecodeRange) -> bool {
        let row_group = range.metadata().row_group_index;
        range.rows().start < self.releases.rows_released[row_group].load(Ordering::Acquire)
    }
}

impl UnaryFactory<DecodeRange, DecodeRange> for DecodeGate {
    type Unary = Self;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self> {
        Built::new(self)
    }
}

impl Unary<DecodeRange, DecodeRange> for DecodeGate {
    fn consume(
        &mut self,
        range: DecodeRange,
        sender: &mut dyn Sender<DecodeRange>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        if self.is_released(&range) {
            sender.send(range)?;
        } else {
            self.held.push(range);
        }
        Ok(())
    }

    /// Send on the held ranges whose rows have been released since.
    fn run(
        &mut self,
        sender: &mut dyn Sender<DecodeRange>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<WorkStatus> {
        let mut sent = false;
        let mut index = 0;
        while index < self.held.len() {
            if self.is_released(&self.held[index]) {
                sender.send(self.held.swap_remove(index))?;
                sent = true;
            } else {
                index += 1;
            }
        }
        Ok(if sent {
            WorkStatus::Ran
        } else {
            WorkStatus::Pending
        })
    }

    /// Held ranges are still to go on.
    fn has_pending_work(&self) -> bool {
        !self.held.is_empty()
    }
}
