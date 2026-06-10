//! Plain `LIMIT n [OFFSET m]` operator — no ordering involved.
//!
//! Keeps `limit` arbitrary rows after skipping `offset` arbitrary rows. SQL
//! leaves *which* rows entirely unspecified when there is no ORDER BY, so the
//! only obligation is the exact row count: `min(limit, total - offset)` rows
//! out (and an ordered input would arrive as a single batch from the sort's
//! sole outputter, so its order survives the funnel unchanged).
//!
//! ## Consume
//!
//! Each worker accumulates incoming batches, truncating once it holds
//! `limit + offset` rows ("fetch") — any `fetch` rows from one worker are
//! enough to cover the global window even if every other worker contributes
//! nothing. On finalization it sends its batches to a shared mpsc channel.
//!
//! ## Output
//!
//! The worker holding the receiver streams the collected batches, dropping the
//! first `offset` rows and emitting the next `limit`. Skipping must be globally
//! consistent (the same row must not be both skipped here and emitted there),
//! which funneling through this single output stage gives for free.

use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::factory::UnaryFactory;
use crate::operations::unary::pipeline_breaker::{Consumer, Outputter, PipelineBreaker};
use crate::worker::worker_waker;
use arrow_array::RecordBatch;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, TryRecvError};

/// Creates one [`Limit`] operator per worker with shared channel wiring.
///
/// The first factory receives the channel receiver; the rest get `None`. All
/// share a sender so per-worker batches funnel to a single collector, which
/// applies the offset and the final truncation.
pub struct LimitFactory {
    limit: usize,
    offset: usize,
    sender: mpsc::Sender<RecordBatch>,
    receiver: Option<Receiver<RecordBatch>>,
}

impl LimitFactory {
    /// Create `worker_count` factories sharing a single mpsc channel. Pass
    /// `usize::MAX` as `limit` for an unbounded `OFFSET`-only modifier.
    pub fn create_for_workers(
        limit: usize,
        offset: usize,
        worker_count: usize,
    ) -> impl IntoIterator<Item = LimitFactory> {
        let (tx, rx) = mpsc::channel();
        let mut rx_opt = Some(rx);

        (0..worker_count).map(move |_| LimitFactory {
            limit,
            offset,
            sender: tx.clone(),
            receiver: rx_opt.take(),
        })
    }
}

impl UnaryFactory<RecordBatch, RecordBatch> for LimitFactory {
    type Unary = PipelineBreaker<RecordBatch, RecordBatch, Limit>;

    fn build_unary(mut self) -> Self::Unary {
        PipelineBreaker::Consuming(Limit {
            fetch: self.limit.saturating_add(self.offset),
            limit: self.limit,
            offset: self.offset,
            rows_kept: 0,
            batches: vec![],
            sender: self.sender,
            receiver: self.receiver.take(),
        })
    }
}

/// Per-worker LIMIT consumer: keeps the first `limit + offset` rows it sees
/// (any rows are valid candidates — there is no ordering), drops the rest.
pub struct Limit {
    /// `limit + offset` (saturating): the per-worker keep budget.
    fetch: usize,
    limit: usize,
    offset: usize,
    rows_kept: usize,
    batches: Vec<RecordBatch>,
    sender: mpsc::Sender<RecordBatch>,
    receiver: Option<Receiver<RecordBatch>>,
}

impl Consumer<RecordBatch, RecordBatch> for Limit {
    type Outputter = LimitOutputter;

    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        _sender: &mut S,
    ) -> unary::Result<()> {
        if self.rows_kept >= self.fetch {
            return Ok(());
        }
        let keep = batch.num_rows().min(self.fetch - self.rows_kept);
        self.rows_kept += keep;
        self.batches.push(if keep == batch.num_rows() {
            batch
        } else {
            batch.slice(0, keep)
        });
        Ok(())
    }

    fn into_outputter(self) -> unary::Result<Option<Self::Outputter>> {
        let Limit {
            batches,
            sender,
            receiver,
            limit,
            offset,
            ..
        } = self;

        for batch in batches {
            sender.send(batch).expect("Receiver dropped!");
        }

        // Drop our sender *before* notifying, and notify *unconditionally* —
        // the receiver worker drains with a non-blocking `try_recv` while
        // parked on the shared waker, so it only observes `Disconnected` (all
        // senders dropped) when something wakes it. Same lost-wakeup hazard as
        // `OrderByLimit::into_outputter`.
        drop(sender);
        worker_waker().notify();

        Ok(receiver.map(|rx| LimitOutputter {
            rx,
            skip_left: offset,
            emit_left: limit,
        }))
    }
}

/// Output phase: streams the funneled batches, skipping the first `offset`
/// rows and forwarding the next `limit`.
pub struct LimitOutputter {
    rx: Receiver<RecordBatch>,
    skip_left: usize,
    emit_left: usize,
}

impl Outputter<RecordBatch> for LimitOutputter {
    fn output<S: Sender<RecordBatch>>(&mut self, sender: &mut S) -> unary::Result<bool> {
        match self.rx.try_recv() {
            Ok(batch) => {
                let skip = self.skip_left.min(batch.num_rows());
                self.skip_left -= skip;
                let keep = self.emit_left.min(batch.num_rows() - skip);
                self.emit_left -= keep;
                if keep > 0 {
                    sender.send(batch.slice(skip, keep))?;
                }
                Ok(false)
            }
            Err(TryRecvError::Empty) => Ok(false),
            Err(TryRecvError::Disconnected) => Ok(true),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::unary::test_utils::run_consumers;
    use arrow_array::{ArrayRef, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch(values: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let col: ArrayRef = Arc::new(Int32Array::from(values.to_vec()));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    fn consumers(limit: usize, offset: usize, workers: usize) -> Vec<Limit> {
        LimitFactory::create_for_workers(limit, offset, workers)
            .into_iter()
            .map(|f| match f.build_unary() {
                PipelineBreaker::Consuming(c) => c,
                _ => unreachable!(),
            })
            .collect()
    }

    #[test]
    fn emits_exactly_limit_rows_across_workers() {
        let sender = run_consumers(
            consumers(3, 0, 2),
            vec![vec![batch(&[1, 2]), batch(&[3, 4])], vec![batch(&[5, 6])]],
        );
        assert_eq!(sender.total_rows(), 3);
    }

    #[test]
    fn offset_skips_globally_before_emitting() {
        // 6 rows total, skip 4, emit min(3, 2) = 2.
        let sender = run_consumers(
            consumers(3, 4, 2),
            vec![vec![batch(&[1, 2]), batch(&[3, 4])], vec![batch(&[5, 6])]],
        );
        assert_eq!(sender.total_rows(), 2);
    }

    #[test]
    fn offset_beyond_input_emits_nothing() {
        let sender = run_consumers(consumers(5, 10, 1), vec![vec![batch(&[1, 2, 3])]]);
        assert_eq!(sender.total_rows(), 0);
    }

    #[test]
    fn unbounded_limit_with_offset() {
        // `OFFSET 1` with no LIMIT: everything but one row survives.
        let sender = run_consumers(
            consumers(usize::MAX, 1, 2),
            vec![vec![batch(&[1, 2, 3])], vec![batch(&[4, 5])]],
        );
        assert_eq!(sender.total_rows(), 4);
    }

    #[test]
    fn emitted_rows_are_input_rows() {
        let sender = run_consumers(consumers(2, 1, 1), vec![vec![batch(&[7, 8, 9, 10])]]);
        assert_eq!(sender.i32_column(0), vec![8, 9]);
    }
}
