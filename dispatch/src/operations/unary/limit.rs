//! Streaming `LIMIT … OFFSET` (no ORDER BY) with early termination.
//!
//! A pipeline breaker, but one that finalizes the *moment* it has enough rows
//! rather than waiting for its input to run dry. Each worker buffers rows into a
//! shared count; once that count reaches `fetch = limit + offset` across all
//! workers, the limit is globally satisfied and three things fire off the same
//! latched flag ([`reached`](Limit::reached)):
//!
//! 1. **Stop consuming**: [`ready_for_more_work`](Consumer::ready_for_more_work)
//!    goes `false`, so no more input is pulled.
//! 2. **Finish early**: [`finished_consuming`](Consumer::finished_consuming)
//!    goes `true`, letting `try_finish` finalize without draining the input.
//! 3. **Abandon the scan**: [`upstream_cancel_flag`](Consumer::upstream_cancel_flag)
//!    hands the flag to the `DataFlow`, which tears down every operator *upstream*
//!    of the limit (the scan/decode pipeline) and frees their in-flight buffers.
//!    Operators *downstream* of the limit are untouched, so a `GROUP BY` over a
//!    `LIMIT` subquery keeps running.
//!
//! ## Gathering
//!
//! As in `order_by_limit`, each worker funnels its buffer over an mpsc channel
//! to the single worker holding the receiver; that receiver worker concatenates,
//! skips `offset`, and emits `limit` rows. Because every worker caps its own
//! buffer at `fetch` and we count *buffered* (not merely seen) rows, the
//! receiver worker is guaranteed to hold at least `fetch` rows by the time
//! `reached` latches, so it never has to block waiting on a straggler.

use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::factory::UnaryFactory;
use crate::operations::unary::pipeline_breaker::{Consumer, Outputter, PipelineBreaker};
use crate::worker::worker_waker;
use arrow::compute::concat_batches;
use arrow_array::RecordBatch;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, mpsc};

/// Per-worker `LIMIT` consumer. See the [module docs](self).
pub struct Limit {
    /// Rows to emit after skipping `offset`.
    limit: usize,
    /// Rows to skip before emitting.
    offset: usize,
    /// `limit + offset`: how many rows must be buffered (globally) before the
    /// result is fully determined. Saturating, so an "unlimited" limit never
    /// latches `reached` and the scan runs to completion as usual.
    fetch: usize,
    /// This worker's buffered rows, each batch already sliced so the running
    /// total never exceeds `fetch`.
    buffer: Vec<RecordBatch>,
    buffered_rows: usize,
    /// Rows buffered across *all* workers. The trigger: once it reaches `fetch`
    /// the limit is satisfied and `reached` latches.
    rows_seen: Arc<AtomicUsize>,
    /// Latched once `rows_seen >= fetch`. Drives stop-consuming, finish-early,
    /// and abandon-upstream in lockstep (see the module docs).
    reached: Arc<AtomicBool>,
    /// Per-worker buffers funnel here so one worker assembles the final result.
    sender: mpsc::Sender<RecordBatch>,
    /// Held by exactly one worker (the receiver worker that gathers the result);
    /// `None` on the rest.
    receiver: Option<Receiver<RecordBatch>>,
}

impl Limit {
    pub fn new(
        limit: usize,
        offset: usize,
        rows_seen: Arc<AtomicUsize>,
        reached: Arc<AtomicBool>,
        sender: mpsc::Sender<RecordBatch>,
        receiver: Option<Receiver<RecordBatch>>,
    ) -> Self {
        Self {
            limit,
            offset,
            fetch: limit.saturating_add(offset),
            buffer: Vec::new(),
            buffered_rows: 0,
            rows_seen,
            reached,
            sender,
            receiver,
        }
    }
}

impl Consumer<RecordBatch, RecordBatch> for Limit {
    type Outputter = LimitOutputter;

    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        _sender: &mut S,
    ) -> unary::Result<()> {
        // Keep only what we still have room for. `fetch` rows total across all
        // workers answer any LIMIT/OFFSET, so we never hold more than that.
        let room = self.fetch - self.buffered_rows;
        if room == 0 {
            return Ok(());
        }
        let take = room.min(batch.num_rows());
        let kept = if take == batch.num_rows() {
            batch
        } else {
            batch.slice(0, take)
        };
        self.buffered_rows += take;
        self.buffer.push(kept);

        // Count rows we actually buffered (not merely saw): the global total then
        // equals the rows held across all workers, so the receiver worker is
        // guaranteed at least `fetch` rows once `reached` latches.
        let total = self.rows_seen.fetch_add(take, Ordering::Relaxed) + take;
        if total >= self.fetch && !self.reached.swap(true, Ordering::Relaxed) {
            // We are the worker that crossed the threshold. Wake the pool: a peer
            // that already parked (e.g. it got no input and is waiting on an
            // upstream stage that will now be abandoned rather than finished)
            // won't otherwise notice `reached`, so it would never abandon its own
            // upstream nor flush its buffer, and the receiver worker would wait
            // forever.
            worker_waker().notify();
        }
        Ok(())
    }

    fn ready_for_more_work(&mut self) -> bool {
        !self.reached.load(Ordering::Relaxed)
    }

    fn finished_consuming(&self) -> bool {
        self.reached.load(Ordering::Relaxed)
    }

    fn upstream_cancel_flag(&self) -> Option<Arc<AtomicBool>> {
        Some(self.reached.clone())
    }

    fn into_outputter(self) -> unary::Result<Option<Self::Outputter>> {
        let Limit {
            buffer,
            sender,
            receiver,
            limit,
            offset,
            ..
        } = self;

        // Hand this worker's buffer to the receiver worker. Sending one batch at
        // a time is bounded (the buffer is capped at `fetch`) and lets the
        // receiver worker do a single concat across all workers.
        for batch in buffer {
            sender.send(batch).expect("limit receiver dropped");
        }

        // Drop the sender *before* notifying, and notify unconditionally: the
        // receiver worker collects with a non-blocking `try_recv` while parked,
        // so a worker that buffered nothing must still wake it, or the final
        // disconnect could be slept through. (Same hazard as `order_by_limit`.)
        drop(sender);
        worker_waker().notify();

        Ok(receiver.map(|rx| LimitOutputter {
            rx,
            batches: vec![],
            limit,
            offset,
        }))
    }
}

/// Output phase: the receiver worker collects every worker's buffer, then
/// concatenates and applies `OFFSET`/`LIMIT` once all senders have disconnected.
pub struct LimitOutputter {
    rx: Receiver<RecordBatch>,
    batches: Vec<RecordBatch>,
    limit: usize,
    offset: usize,
}

impl Outputter<RecordBatch> for LimitOutputter {
    fn output<S: Sender<RecordBatch>>(&mut self, sender: &mut S) -> unary::Result<bool> {
        match self.rx.try_recv() {
            Ok(batch) => {
                self.batches.push(batch);
                Ok(false)
            }
            Err(TryRecvError::Empty) => Ok(false),
            Err(TryRecvError::Disconnected) => {
                if !self.batches.is_empty() {
                    let schema = self.batches[0].schema();
                    let merged = concat_batches(&schema, &self.batches)?;
                    // No ORDER BY, so any `limit` rows past `offset` satisfy the
                    // query. Both bounds are clamped to what we actually have.
                    let start = self.offset.min(merged.num_rows());
                    let len = self.limit.min(merged.num_rows() - start);
                    sender.send(merged.slice(start, len))?;
                }
                Ok(true)
            }
        }
    }
}

/// Builds one [`Limit`] per worker, all sharing the buffered-row counter, the
/// `reached` flag, and a single mpsc channel whose receiver lands on the first
/// worker (the receiver worker).
pub struct LimitFactory {
    limit: usize,
    offset: usize,
    rows_seen: Arc<AtomicUsize>,
    reached: Arc<AtomicBool>,
    sender: mpsc::Sender<RecordBatch>,
    receiver: Option<Receiver<RecordBatch>>,
}

impl LimitFactory {
    pub fn create_for_workers(
        limit: usize,
        offset: usize,
        worker_count: usize,
    ) -> impl IntoIterator<Item = LimitFactory> {
        let (tx, rx) = mpsc::channel();
        let mut rx_opt = Some(rx);
        let rows_seen = Arc::new(AtomicUsize::new(0));
        let reached = Arc::new(AtomicBool::new(false));

        (0..worker_count).map(move |_| LimitFactory {
            limit,
            offset,
            rows_seen: rows_seen.clone(),
            reached: reached.clone(),
            sender: tx.clone(),
            receiver: rx_opt.take(),
        })
    }
}

impl UnaryFactory<RecordBatch, RecordBatch> for LimitFactory {
    type Unary = PipelineBreaker<RecordBatch, RecordBatch, Limit>;

    fn build_unary(mut self) -> Self::Unary {
        PipelineBreaker::Consuming(Limit::new(
            self.limit,
            self.offset,
            self.rows_seen,
            self.reached,
            self.sender,
            self.receiver.take(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::unary::test_utils::{CollectSender, run_consumers};
    use arrow_array::{ArrayRef, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch(values: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let col: ArrayRef = Arc::new(Int32Array::from(values.to_vec()));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    fn run_limit(
        worker_batches: Vec<Vec<RecordBatch>>,
        limit: usize,
        offset: usize,
    ) -> CollectSender {
        let worker_count = worker_batches.len();
        let (tx, rx) = mpsc::channel();
        let mut rx_opt = Some(rx);
        let rows_seen = Arc::new(AtomicUsize::new(0));
        let reached = Arc::new(AtomicBool::new(false));

        let consumers: Vec<_> = (0..worker_count)
            .map(|_| {
                Limit::new(
                    limit,
                    offset,
                    rows_seen.clone(),
                    reached.clone(),
                    tx.clone(),
                    rx_opt.take(),
                )
            })
            .collect();
        drop(tx);

        run_consumers(consumers, worker_batches)
    }

    #[test]
    fn limit_smaller_than_input_truncates() {
        let sender = run_limit(vec![vec![batch(&[1, 2, 3, 4, 5])]], 2, 0);

        assert_eq!(sender.total_rows(), 2);
        assert_eq!(sender.i32_column(0), vec![1, 2]);
    }

    #[test]
    fn limit_larger_than_input_returns_all() {
        let sender = run_limit(vec![vec![batch(&[1, 2, 3])]], 100, 0);

        assert_eq!(sender.total_rows(), 3);
        assert_eq!(sender.i32_column(0), vec![1, 2, 3]);
    }

    #[test]
    fn offset_skips_then_limits() {
        let sender = run_limit(vec![vec![batch(&[1, 2, 3, 4, 5])]], 2, 1);

        assert_eq!(sender.total_rows(), 2);
        assert_eq!(sender.i32_column(0), vec![2, 3]);
    }

    #[test]
    fn offset_past_end_returns_empty() {
        let sender = run_limit(vec![vec![batch(&[1, 2])]], 5, 10);

        assert_eq!(sender.total_rows(), 0);
    }

    #[test]
    fn gathers_limit_rows_across_workers() {
        // Two workers, LIMIT 3: each caps its own buffer, the receiver worker
        // merges and keeps the first 3. Order across workers is arbitrary (no
        // ORDER BY), so assert on the multiset.
        let sender = run_limit(
            vec![vec![batch(&[1, 2, 3, 4])], vec![batch(&[5, 6, 7, 8])]],
            3,
            0,
        );

        assert_eq!(sender.total_rows(), 3);
    }

    #[test]
    fn reached_latches_and_stops_consuming() {
        crate::worker::install_test_worker_waker();
        let rows_seen = Arc::new(AtomicUsize::new(0));
        let reached = Arc::new(AtomicBool::new(false));
        let (tx, _rx) = mpsc::channel();
        let mut limit = Limit::new(3, 0, rows_seen, reached.clone(), tx, None);
        let mut sink = CollectSender::new();

        assert!(limit.ready_for_more_work());
        limit.consume(batch(&[1, 2]), &mut sink).unwrap();
        assert!(limit.ready_for_more_work(), "2 < 3 rows: still wants input");
        limit.consume(batch(&[3, 4]), &mut sink).unwrap();

        assert!(reached.load(Ordering::Relaxed));
        assert!(!limit.ready_for_more_work(), "limit met: stops consuming");
        assert!(limit.finished_consuming());
    }

    #[test]
    fn upstream_cancel_flag_is_the_reached_flag() {
        let reached = Arc::new(AtomicBool::new(false));
        let (tx, _rx) = mpsc::channel();
        let limit = Limit::new(
            1,
            0,
            Arc::new(AtomicUsize::new(0)),
            reached.clone(),
            tx,
            None,
        );

        let exposed = limit
            .upstream_cancel_flag()
            .expect("limit cancels upstream");
        exposed.store(true, Ordering::Relaxed);
        assert!(reached.load(Ordering::Relaxed), "same underlying flag");
    }
}
