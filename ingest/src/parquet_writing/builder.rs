//! Builds row groups out of a stream of record batches.
//!
//! A pipeline breaker (the [`OrderByLimit`](dispatch) shape): each worker
//! accumulates the batches it steals until it has a row group's worth, then
//! emits that row group mid-stream (stamped with the id that keeps its pages
//! together downstream). At finish, every worker sends its straddling
//! remainder to worker 0 over a side `mpsc` channel; worker 0 combines them
//! into the single final row group. Batch-native: a flush's items are
//! flattened by the upstream [`convert`](super::convert) stage, a compaction
//! feeds the scan's decoded batches straight in.

use std::mem;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender as StdSender, TryRecvError};

use arrow_array::RecordBatch;
use dispatch::{Consumer, Outputter, PipelineBreaker, Sender, UnaryFactory, UnaryResult};

use super::types::RowGroupBatch;

/// Factory for one worker's [`RowGroupBuilder`]. Only worker 0 receives the
/// side-channel receiver (`leftover_rx`).
pub(super) struct RowGroupBuilderFactory {
    target_rows: usize,
    worker_count: usize,
    next_rg_id: Arc<AtomicU64>,
    leftover_tx: StdSender<Vec<RecordBatch>>,
    leftover_rx: Option<Receiver<Vec<RecordBatch>>>,
}

impl UnaryFactory<RecordBatch, RowGroupBatch> for RowGroupBuilderFactory {
    type Unary = PipelineBreaker<RecordBatch, RowGroupBatch, RowGroupBuilder>;

    fn build_unary(self) -> Self::Unary {
        PipelineBreaker::Consuming(RowGroupBuilder {
            target_rows: self.target_rows,
            worker_count: self.worker_count,
            next_rg_id: self.next_rg_id,
            pending: Vec::new(),
            pending_rows: 0,
            leftover_tx: self.leftover_tx,
            leftover_rx: self.leftover_rx,
        })
    }
}

/// One factory per worker, sharing the row-group counter and the worker-0 side
/// channel.
pub(super) fn factories(
    target_rows: usize,
    worker_count: usize,
    next_rg_id: Arc<AtomicU64>,
) -> Vec<RowGroupBuilderFactory> {
    let (tx, rx) = mpsc::channel();
    let mut rx = Some(rx);
    (0..worker_count)
        .map(|_| RowGroupBuilderFactory {
            target_rows,
            worker_count,
            next_rg_id: next_rg_id.clone(),
            leftover_tx: tx.clone(),
            leftover_rx: rx.take(),
        })
        .collect()
}

/// Per-worker consumer: accumulate batches and emit full row groups; hand the
/// remainder to worker 0 at finish.
pub(super) struct RowGroupBuilder {
    target_rows: usize,
    worker_count: usize,
    next_rg_id: Arc<AtomicU64>,
    pending: Vec<RecordBatch>,
    pending_rows: usize,
    leftover_tx: StdSender<Vec<RecordBatch>>,
    leftover_rx: Option<Receiver<Vec<RecordBatch>>>,
}

impl Consumer<RecordBatch, RowGroupBatch> for RowGroupBuilder {
    type Outputter = RowGroupBuilderOutputter;

    fn consume<S: Sender<RowGroupBatch>>(
        &mut self,
        batch: RecordBatch,
        sender: &mut S,
    ) -> UnaryResult<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        self.pending_rows += batch.num_rows();
        self.pending.push(batch);
        if self.pending_rows >= self.target_rows {
            let batches = mem::take(&mut self.pending);
            self.pending_rows = 0;
            emit_row_group(sender, &self.next_rg_id, self.worker_count, batches)?;
        }
        Ok(())
    }

    fn into_outputter(mut self) -> UnaryResult<Option<Self::Outputter>> {
        if !self.pending.is_empty() {
            // Send our straddling remainder to worker 0 to combine.
            let _ = self.leftover_tx.send(mem::take(&mut self.pending));
        }
        Ok(self.leftover_rx.map(|rx| RowGroupBuilderOutputter {
            rx,
            next_rg_id: self.next_rg_id,
            worker_count: self.worker_count,
            acc: Vec::new(),
        }))
    }
}

/// Worker 0's output phase: gather every worker's remainder and emit it as the
/// single final row group once all senders disconnect.
pub(super) struct RowGroupBuilderOutputter {
    rx: Receiver<Vec<RecordBatch>>,
    next_rg_id: Arc<AtomicU64>,
    worker_count: usize,
    acc: Vec<RecordBatch>,
}

impl Outputter<RowGroupBatch> for RowGroupBuilderOutputter {
    fn output<S: Sender<RowGroupBatch>>(&mut self, sender: &mut S) -> UnaryResult<bool> {
        match self.rx.try_recv() {
            Ok(batches) => {
                self.acc.extend(batches);
                Ok(false)
            }
            Err(TryRecvError::Empty) => Ok(false),
            Err(TryRecvError::Disconnected) => {
                if !self.acc.is_empty() {
                    let batches = mem::take(&mut self.acc);
                    emit_row_group(sender, &self.next_rg_id, self.worker_count, batches)?;
                }
                Ok(true)
            }
        }
    }
}

/// Assign the next row-group id and emit `batches` as one row group.
fn emit_row_group<S: Sender<RowGroupBatch>>(
    sender: &mut S,
    next_rg_id: &AtomicU64,
    worker_count: usize,
    batches: Vec<RecordBatch>,
) -> UnaryResult<()> {
    let rg_id = next_rg_id.fetch_add(1, Ordering::Relaxed);
    let dest_worker = (rg_id as usize) % worker_count;
    let schema = batches[0].schema();
    sender.send(RowGroupBatch {
        rg_id,
        dest_worker,
        schema,
        batches,
    })?;
    Ok(())
}
