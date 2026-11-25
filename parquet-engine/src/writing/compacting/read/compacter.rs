//! The operator that runs a compaction's read in merge order.

use dispatch::{
    BuildContext, Built, BuiltUnary, OperatorIO, Sender, Unary, UnaryFactory, UnaryResult,
    WorkStatus,
};

use super::compaction_read::CompactionRead;
use super::output::MergedBatchColumn;
use crate::writing::compacting::read::input::decoded_values::DecodedValues;

/// Reads a compaction's input files in merge order (see the
/// [module](super) for the whole design).
///
/// The dataflow has one compacter per worker, but only the one on the
/// compaction worker holds a [`CompactionRead`]. All decoded values are
/// routed to that worker, so the other compacters receive nothing.
pub(crate) struct MergeSortCompacter {
    /// The read's state. `None` on every worker except the compaction
    /// worker.
    compaction_read: Option<CompactionRead>,
}

impl MergeSortCompacter {
    /// One compacter per worker. The compacter of `compaction_worker` holds
    /// `compaction_read`.
    pub(crate) fn create_for_workers(
        compaction_read: CompactionRead,
        worker_count: usize,
        compaction_worker: usize,
    ) -> Vec<Self> {
        let mut compaction_read = Some(compaction_read);
        (0..worker_count)
            .map(|worker| MergeSortCompacter {
                compaction_read: match worker == compaction_worker {
                    true => compaction_read.take(),
                    false => None,
                },
            })
            .collect()
    }
}

impl UnaryFactory<DecodedValues, MergedBatchColumn> for MergeSortCompacter {
    type Unary = Self;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self> {
        Built::new(self)
    }
}

impl Unary<DecodedValues, MergedBatchColumn> for MergeSortCompacter {
    /// Keep the decoded values that came back from the pipeline, then
    /// continue the read as far as they allow.
    fn consume(
        &mut self,
        decoded: DecodedValues,
        sender: &mut dyn Sender<MergedBatchColumn>,
        _io: &mut OperatorIO,
    ) -> UnaryResult<()> {
        let compaction_read = self
            .compaction_read
            .as_mut()
            .expect("decoded values are routed to the compaction worker, which holds the read");
        compaction_read.store_decoded_values(decoded)?;
        compaction_read.run_ready_steps(sender)?;
        Ok(())
    }

    /// Called when no decoded values are waiting. This starts the read (the
    /// first call makes the first requests) and continues it when memory
    /// was given back or fetched pages arrived.
    fn run(
        &mut self,
        sender: &mut dyn Sender<MergedBatchColumn>,
        _io: &mut OperatorIO,
    ) -> UnaryResult<WorkStatus> {
        let Some(compaction_read) = &mut self.compaction_read else {
            return Ok(WorkStatus::Pending);
        };
        Ok(if compaction_read.run_ready_steps(sender)? {
            WorkStatus::Ran
        } else {
            WorkStatus::Pending
        })
    }

    /// The operator is not finished while rows are still to be sent on.
    fn has_pending_work(&self) -> bool {
        self.compaction_read
            .as_ref()
            .is_some_and(|compaction_read| !compaction_read.has_sent_every_row())
    }
}
