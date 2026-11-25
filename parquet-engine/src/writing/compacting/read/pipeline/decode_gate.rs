//! Holds a range of rows back, as decompressed pages, until the compacter
//! releases its rows.
//!
//! A page holds more rows than the merge needs next, so decompressing one
//! page makes several ranges of rows ready to decode. Decoded, rows are
//! larger again. So the ranges wait here, and a range goes on to be decoded
//! only when its rows are released (see [`GatedReads`]).

use std::collections::HashMap;
use std::sync::Arc;

use dispatch::{BuildContext, Built, BuiltUnary, Sender, Unary, UnaryFactory, UnaryResult};
use dispatch::{OperatorIO, WorkStatus};

use super::gated_reads::GatedReads;
use crate::DecodeRange;

/// Passes a range of rows on to be decoded once its rows are released, and
/// holds it until then.
///
/// Every worker has one, which holds the ranges that reached that worker
/// and passes them on to that worker's queue of ranges to decode.
pub(crate) struct DecodeGate {
    gated_reads: Arc<GatedReads>,
    /// The worker the gate runs on.
    worker: usize,
    /// The ranges that wait, by the number of their read.
    held_ranges: HashMap<usize, Vec<DecodeRange>>,
}

impl DecodeGate {
    /// One gate per worker.
    pub(crate) fn create_for_workers(
        gated_reads: &Arc<GatedReads>,
        worker_count: usize,
    ) -> Vec<Self> {
        (0..worker_count)
            .map(|worker| Self {
                gated_reads: gated_reads.clone(),
                worker,
                held_ranges: HashMap::new(),
            })
            .collect()
    }

    /// If the compacter has released rows since the gate last looked at the
    /// ranges it holds, send on those that are now released. Returns
    /// whether any range was sent on.
    ///
    /// The compacter releases whole ranges, so a range is released when its
    /// first row is. It wakes the waiting workers after it releases rows,
    /// which takes their registrations away. So a gate whose worker is
    /// still registered has nothing new to find. One that is not registers
    /// again before it looks, so that a release it does not see yet wakes
    /// it.
    fn pass_newly_released_ranges_on(
        &mut self,
        sender: &mut dyn Sender<DecodeRange>,
    ) -> UnaryResult<bool> {
        let gated_reads = &self.gated_reads;
        let waiting_gates = &gated_reads.waiting_decode_gates;
        if self.held_ranges.is_empty() || waiting_gates.is_registered(self.worker) {
            return Ok(false);
        }
        waiting_gates.register(self.worker);
        let mut released_ranges = Vec::new();
        self.held_ranges.retain(|&read, ranges| {
            released_ranges.extend(ranges.extract_if(.., |range| {
                gated_reads.is_row_released(read, range.rows().start)
            }));
            !ranges.is_empty()
        });
        let has_released_ranges = !released_ranges.is_empty();
        for range in released_ranges {
            sender.send(range)?;
        }
        Ok(has_released_ranges)
    }
}

impl UnaryFactory<DecodeRange, DecodeRange> for DecodeGate {
    type Unary = Self;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self> {
        Built::new(self)
    }
}

impl Unary<DecodeRange, DecodeRange> for DecodeGate {
    /// Take a range whose pages were decompressed. A range whose rows are
    /// released goes straight on.
    ///
    /// A range that is not released is held. Before that the gate registers
    /// its worker as waiting and looks once more: the compacter may have
    /// released the rows and woken the waiting workers in between.
    fn consume(
        &mut self,
        range: DecodeRange,
        sender: &mut dyn Sender<DecodeRange>,
        _io: &mut OperatorIO,
    ) -> UnaryResult<()> {
        self.pass_newly_released_ranges_on(sender)?;
        let read = self.gated_reads.find_read_index(range.metadata());
        let first_row = range.rows().start;
        let is_released = self.gated_reads.is_row_released(read, first_row) || {
            let waiting_gates = &self.gated_reads.waiting_decode_gates;
            waiting_gates.register(self.worker);
            self.gated_reads.is_row_released(read, first_row)
        };
        if is_released {
            sender.send(range)?;
        } else {
            self.held_ranges.entry(read).or_default().push(range);
        }
        Ok(())
    }

    /// Send on the held ranges whose rows have been released since.
    fn run(
        &mut self,
        sender: &mut dyn Sender<DecodeRange>,
        _io: &mut OperatorIO,
    ) -> UnaryResult<WorkStatus> {
        Ok(if self.pass_newly_released_ranges_on(sender)? {
            WorkStatus::Ran
        } else {
            WorkStatus::Pending
        })
    }

    /// Held ranges are still to go on.
    fn has_pending_work(&self) -> bool {
        !self.held_ranges.is_empty()
    }
}
