//! Flat map: every input batch becomes a sequence of output batches.
//!
//! The caller's closure is handed the same input batch again and again and
//! returns the next output batch each time, or `None` once that input batch
//! has nothing more to give. The closure keeps its own place in the batch
//! between calls.
//!
//! One output batch goes out per turn, so an input batch that expands into a
//! great many rows is never resident all at once, and a downstream `LIMIT` can
//! stop it part-way.

use crate::api::BuildContext;
use crate::data_flow::WorkStatus;
use crate::operations::channels::Sender;
use crate::operations::unary::{self, Unary, UnaryFactory};
use crate::operations::{Built, BuiltUnary};
use arrow_array::RecordBatch;

/// Builds one worker's [`FlatMap`] from that worker's closure.
pub struct FlatMapFactory<F>(pub F);

impl<F> UnaryFactory<RecordBatch, RecordBatch> for FlatMapFactory<F>
where
    F: FnMut(&RecordBatch) -> unary::Result<Option<RecordBatch>> + Send + 'static,
{
    type Unary = FlatMap<F>;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self::Unary> {
        Built::new(FlatMap {
            produce_next_batch: self.0,
            held_input: None,
        })
    }
}

/// See the [module docs](self).
pub struct FlatMap<F> {
    produce_next_batch: F,
    /// The input batch whose output batches are still being produced.
    held_input: Option<RecordBatch>,
}

impl<F> FlatMap<F>
where
    F: FnMut(&RecordBatch) -> unary::Result<Option<RecordBatch>> + Send,
{
    /// Send the held input's next output batch, or let the input go when it
    /// has none left.
    fn advance_held_input(
        &mut self,
        input: RecordBatch,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        if let Some(output) = (self.produce_next_batch)(&input)? {
            sender.send(output)?;
            self.held_input = Some(input);
        }
        Ok(())
    }
}

impl<F> Unary<RecordBatch, RecordBatch> for FlatMap<F>
where
    F: FnMut(&RecordBatch) -> unary::Result<Option<RecordBatch>> + Send,
{
    fn consume(
        &mut self,
        batch: RecordBatch,
        sender: &mut dyn Sender<RecordBatch>,
        _io: &mut crate::io::OperatorIO,
    ) -> unary::Result<()> {
        assert!(
            self.held_input.is_none(),
            "a flat map takes one input batch at a time"
        );
        self.advance_held_input(batch, sender)
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.held_input.is_none()
    }

    fn run(
        &mut self,
        sender: &mut dyn Sender<RecordBatch>,
        _io: &mut crate::io::OperatorIO,
    ) -> unary::Result<WorkStatus> {
        let Some(input) = self.held_input.take() else {
            return Ok(WorkStatus::Pending);
        };
        self.advance_held_input(input, sender)?;
        Ok(WorkStatus::Ran)
    }

    /// A held input keeps the stage from finishing until its batches are out.
    fn has_pending_work(&self) -> bool {
        self.held_input.is_some()
    }
}
