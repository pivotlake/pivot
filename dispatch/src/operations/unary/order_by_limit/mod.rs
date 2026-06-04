//! ORDER BY … LIMIT k operator.
//!
//! A pipeline breaker that keeps only the top-k rows per sort key across
//! all input batches. Like GROUP BY, it has two phases:
//!
//! ## Phase 1: Consume
//!
//! Each worker receives [`RecordBatch`]es and immediately reduces each one
//! to its local top-k via [`get_top_k_from_single`] (Arrow `lexsort_to_indices`
//! with a limit). The per-batch top-k results are accumulated in a `Vec`.
//!
//! When consumption finishes ([`into_outputter`](Consumer::into_outputter)),
//! the worker merges its accumulated top-k batches into a single local
//! top-k and sends it to a shared mpsc channel.
//!
//! ## Phase 2: Output
//!
//! The worker holding the channel receiver collects all per-worker top-k
//! batches, concatenates them, and performs a final global top-k sort.
//! The result is a single [`RecordBatch`] with at most `limit` rows in
//! sorted order, sent downstream.

use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::pipeline_breaker::{Consumer, Outputter};
use crate::worker::worker_waker;
use arrow::compute::kernels::cmp;
use arrow::compute::{SortColumn, lexsort_to_indices, take};
use arrow_array::{Array, ArrayRef, Datum, RecordBatch, Scalar};
use arrow_schema::{ArrowError, SortOptions};
use std::mem;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use thiserror::Error;
use tracing::debug;

mod factory;
pub use factory::OrderByLimitFactory;

/// A shared cell a Top-N operator publishes its current Nth-best sort key into,
/// for sibling scans to prune row groups against. One producer (this operator)
/// writes; any number of consumer scans read. Empty until the producer has seen
/// a full window; the value only ever tightens, so a stale read prunes less,
/// never more.
///
/// The comparison *direction* lives with the consumer (which carries the
/// comparison operator DuckDB chose); this cell holds only the boundary value.
pub type DynamicFilterSlot = RwLock<Option<Scalar<ArrayRef>>>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Arrow(#[from] ArrowError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A single sort-key specification: which column, sort direction, and null ordering.
#[derive(Clone)]
pub struct OrderBy {
    column_idx: usize,
    descending: bool,
    nulls_first: bool,
}

impl OrderBy {
    pub fn new(column_idx: usize, descending: bool, nulls_first: bool) -> Self {
        Self {
            column_idx,
            descending,
            nulls_first,
        }
    }
}

/// Merge multiple already-sorted top-k batches into a single global top-k.
///
/// Concatenates all batches, then sorts, skips `skip` rows, and keeps the next
/// `fetch - skip` rows. Local stages pass `skip = 0`; the final global stage
/// passes `skip = offset` to implement `LIMIT … OFFSET`.
fn get_top_k_from_top_ks(
    batches: Vec<RecordBatch>,
    order_by: &[OrderBy],
    fetch: usize,
    skip: usize,
) -> Result<RecordBatch> {
    if batches.is_empty() {
        panic!("get_top_k_from_top_ks called with no batches");
    }

    let schema = batches[0].schema();
    let final_pool = arrow::compute::concat_batches(&schema, &batches)?;
    debug!("Final pool size: {:?}", final_pool.num_rows());
    get_top_k_from_single(&final_pool, order_by, fetch, skip)
}

/// Sort a single batch by `order_by` keys, take the first `fetch` rows, then
/// drop the leading `skip` of them (so the result has at most `fetch - skip`
/// rows). `skip` is non-zero only at the final global stage, to honour OFFSET.
fn get_top_k_from_single(
    batch: &RecordBatch,
    order_by: &[OrderBy],
    fetch: usize,
    skip: usize,
) -> Result<RecordBatch> {
    let sort_columns: Vec<SortColumn> = order_by
        .iter()
        .map(|ob| {
            let values = batch.column(ob.column_idx).clone();
            SortColumn {
                values,
                options: Some(SortOptions {
                    descending: ob.descending,
                    nulls_first: ob.nulls_first,
                }),
            }
        })
        .collect();

    let indices = lexsort_to_indices(&sort_columns, Some(fetch))?;
    let len = indices.len();
    let skip = skip.min(len);
    let kept = indices.slice(skip, len - skip);

    let columns = batch
        .columns()
        .iter()
        .map(|c| take(c.as_ref(), &kept, None))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    Ok(unsafe { RecordBatch::new_unchecked(batch.schema(), columns, kept.len()) })
}

/// Per-worker ORDER BY LIMIT consumer.
///
/// Accumulates a local top-k from each incoming batch. On finalization,
/// merges them into one local top-k and sends it to the shared channel.
pub struct OrderByLimit {
    limit: usize,
    offset: usize,
    top_k_per_batch: Vec<RecordBatch>,
    order_by: Vec<OrderBy>,
    sender: mpsc::Sender<RecordBatch>,
    receiver: Option<Receiver<RecordBatch>>,
    /// When set, this worker is a dynamic-filter producer: after each batch it
    /// publishes its current Nth-best leading sort key here so sibling scans
    /// can prune row groups that can't reach the global top-N.
    dynamic_filter: Option<Arc<DynamicFilterSlot>>,
}

impl OrderByLimit {
    pub fn new(
        order_by: Vec<OrderBy>,
        limit: usize,
        offset: usize,
        sender: mpsc::Sender<RecordBatch>,
        receiver: Option<Receiver<RecordBatch>>,
        dynamic_filter: Option<Arc<DynamicFilterSlot>>,
    ) -> Self {
        Self {
            limit,
            offset,
            top_k_per_batch: vec![],
            order_by,
            sender,
            receiver,
            dynamic_filter,
        }
    }

    /// Publish this batch's Nth-best leading key into the shared slot, if the
    /// per-batch top-k is full and the new boundary is strictly tighter than
    /// what's there.
    ///
    /// The window keeps `limit + offset` rows, so the last (worst) of them is a
    /// valid bound on the *global* boundary: this batch alone already witnesses
    /// that many rows at least as good as it. We prune on the leading key only,
    /// so multi-key sorts publish `order_by[0]`'s value (ties on it are resolved
    /// by keeping the row group — see the consumer's strict comparison).
    fn publish_boundary(&self, batch_top_k: &RecordBatch) {
        let Some(slot) = &self.dynamic_filter else {
            return;
        };
        let window = self.limit + self.offset;
        if batch_top_k.num_rows() < window || self.order_by.is_empty() {
            return;
        }
        let leading = &self.order_by[0];
        let last_row = batch_top_k
            .column(leading.column_idx)
            .slice(batch_top_k.num_rows() - 1, 1);
        let new_boundary = Scalar::new(last_row);

        let mut guard = slot.write().expect("dynamic filter slot poisoned");
        let should_overwrite = match guard.as_ref() {
            None => true,
            Some(current) => is_tighter(&new_boundary, current, leading.descending),
        };
        if should_overwrite {
            *guard = Some(new_boundary);
        }
    }
}

/// Whether `new` is a strictly tighter boundary than `current` for a sort in
/// this direction:
/// * ascending — we keep the smallest N, the filter is `col < boundary`, so a
///   *smaller* boundary prunes more;
/// * descending — we keep the largest N, the filter is `col > boundary`, so a
///   *larger* boundary prunes more.
///
/// Returns `false` (don't overwrite) when the comparison kernel errors — e.g. a
/// type mismatch — so a slot can't get wedged on a value it can never improve.
fn is_tighter(new: &Scalar<ArrayRef>, current: &Scalar<ArrayRef>, descending: bool) -> bool {
    let kernel = if descending { cmp::gt } else { cmp::lt };
    match kernel(new as &dyn Datum, current as &dyn Datum) {
        Ok(arr) => arr.is_valid(0) && arr.value(0),
        Err(_) => false,
    }
}

impl Consumer<RecordBatch, RecordBatch> for OrderByLimit {
    type Outputter = OrderByLimitOutputter;

    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        _sender: &mut S,
    ) -> unary::Result<()> {
        debug!("Received batch of length {:?}", batch.num_rows());
        // Local stages keep `limit + offset` candidates (skip = 0); only the
        // final global merge skips `offset`, since which rows fall in the
        // offset window can only be decided once all workers' tops are merged.
        let fetch = self.limit + self.offset;
        let batch_top_k = get_top_k_from_single(&batch, &self.order_by, fetch, 0)?;
        self.publish_boundary(&batch_top_k);
        self.top_k_per_batch.push(batch_top_k);
        Ok(())
    }

    fn into_outputter(self) -> crate::operations::unary::Result<Option<Self::Outputter>> {
        let fetch = self.limit + self.offset;
        if !self.top_k_per_batch.is_empty() {
            let local_top_k =
                get_top_k_from_top_ks(self.top_k_per_batch, &self.order_by, fetch, 0)?;
            debug!("Sending on {:?}", local_top_k.num_rows());
            self.sender.send(local_top_k).expect("Receiver dropped!");
            worker_waker().notify();
        }

        Ok(self.receiver.map(|rx| OrderByLimitOutputter {
            rx,
            batches: vec![],
            order_by: self.order_by,
            limit: self.limit,
            offset: self.offset,
        }))
    }
}

/// Output phase: collects per-worker top-k batches from the channel, then
/// performs the final global top-k sort once all senders have disconnected.
pub struct OrderByLimitOutputter {
    rx: Receiver<RecordBatch>,
    batches: Vec<RecordBatch>,
    order_by: Vec<OrderBy>,
    limit: usize,
    offset: usize,
}

impl Outputter<RecordBatch> for OrderByLimitOutputter {
    fn output<S: Sender<RecordBatch>>(&mut self, sender: &mut S) -> unary::Result<bool> {
        match self.rx.try_recv() {
            Ok(c) => {
                self.batches.push(c);
                Ok(false)
            }
            Err(TryRecvError::Empty) => Ok(false),
            Err(TryRecvError::Disconnected) => {
                if !self.batches.is_empty() {
                    debug!("from {:?} batches", self.batches.len());
                    let start = Instant::now();
                    sender.send(get_top_k_from_top_ks(
                        mem::take(&mut self.batches),
                        &self.order_by,
                        self.limit + self.offset,
                        self.offset,
                    )?)?;
                    debug!("took {:?}", start.elapsed());
                }
                Ok(true)
            }
        }
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

    fn scalar32(v: i32) -> Scalar<ArrayRef> {
        Scalar::new(Arc::new(Int32Array::from(vec![v])) as ArrayRef)
    }

    fn slot_value(slot: &DynamicFilterSlot) -> Option<i32> {
        slot.read().unwrap().as_ref().map(|s| {
            s.get()
                .0
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0)
        })
    }

    #[test]
    fn is_tighter_respects_direction() {
        // Descending keeps the larger boundary (prunes more); ascending the smaller.
        assert!(is_tighter(&scalar32(50), &scalar32(20), true));
        assert!(!is_tighter(&scalar32(20), &scalar32(50), true));
        assert!(is_tighter(&scalar32(20), &scalar32(50), false));
        assert!(!is_tighter(&scalar32(50), &scalar32(20), false));
    }

    #[test]
    fn publishes_boundary_and_keeps_tighter() {
        let slot = Arc::new(DynamicFilterSlot::new(None));
        let (tx, _rx) = mpsc::channel();
        // ORDER BY v DESC LIMIT 2 → publish each batch's 2nd-largest, keep the max.
        let mut op = OrderByLimit::new(
            vec![OrderBy::new(0, true, false)],
            2,
            0,
            tx,
            None,
            Some(slot.clone()),
        );
        let mut sink = CollectSender::new();

        op.consume(batch(&[10, 20, 30]), &mut sink).unwrap();
        assert_eq!(slot_value(&slot), Some(20)); // top-2 of this batch is [30, 20]

        op.consume(batch(&[100, 5]), &mut sink).unwrap();
        assert_eq!(slot_value(&slot), Some(20)); // batch boundary 5 is looser, ignored

        op.consume(batch(&[50, 60]), &mut sink).unwrap();
        assert_eq!(slot_value(&slot), Some(50)); // batch boundary 50 is tighter
    }

    #[test]
    fn does_not_publish_before_window_is_full() {
        let slot = Arc::new(DynamicFilterSlot::new(None));
        let (tx, _rx) = mpsc::channel();
        let mut op = OrderByLimit::new(
            vec![OrderBy::new(0, true, false)],
            3,
            0,
            tx,
            None,
            Some(slot.clone()),
        );
        let mut sink = CollectSender::new();

        // Only 2 rows for a LIMIT 3 — no full window yet, nothing to bound on.
        op.consume(batch(&[10, 20]), &mut sink).unwrap();

        assert_eq!(slot_value(&slot), None);
    }

    fn run_order_by(
        worker_batches: Vec<Vec<RecordBatch>>,
        limit: usize,
        descending: bool,
    ) -> CollectSender {
        let worker_count = worker_batches.len();
        let (tx, rx) = mpsc::channel();
        let mut rx_opt = Some(rx);

        let consumers: Vec<_> = (0..worker_count)
            .map(|_| {
                OrderByLimit::new(
                    vec![OrderBy::new(0, descending, false)],
                    limit,
                    0,
                    tx.clone(),
                    rx_opt.take(),
                    None,
                )
            })
            .collect();
        drop(tx);

        run_consumers(consumers, worker_batches)
    }

    #[test]
    fn top_3_ascending() {
        let sender = run_order_by(vec![vec![batch(&[5, 3, 1, 4, 2])]], 3, false);

        assert_eq!(sender.total_rows(), 3);
        assert_eq!(sender.i32_column(0), vec![1, 2, 3]);
    }

    #[test]
    fn top_3_descending() {
        let sender = run_order_by(vec![vec![batch(&[5, 3, 1, 4, 2])]], 3, true);

        assert_eq!(sender.total_rows(), 3);
        assert_eq!(sender.i32_column(0), vec![5, 4, 3]);
    }

    #[test]
    fn limit_larger_than_input() {
        let sender = run_order_by(vec![vec![batch(&[3, 1, 2])]], 100, false);

        assert_eq!(sender.total_rows(), 3);
        assert_eq!(sender.i32_column(0), vec![1, 2, 3]);
    }

    #[test]
    fn multiple_batches_single_worker() {
        let sender = run_order_by(
            vec![vec![batch(&[10, 20]), batch(&[5, 15]), batch(&[1, 25])]],
            3,
            false,
        );

        assert_eq!(sender.total_rows(), 3);
        assert_eq!(sender.i32_column(0), vec![1, 5, 10]);
    }

    #[test]
    fn two_workers_ascending() {
        let sender = run_order_by(
            vec![vec![batch(&[10, 30, 50])], vec![batch(&[20, 40, 60])]],
            4,
            false,
        );

        assert_eq!(sender.total_rows(), 4);
        assert_eq!(sender.i32_column(0), vec![10, 20, 30, 40]);
    }

    #[test]
    fn two_workers_descending() {
        let sender = run_order_by(
            vec![vec![batch(&[10, 30, 50])], vec![batch(&[20, 40, 60])]],
            4,
            true,
        );

        assert_eq!(sender.total_rows(), 4);
        assert_eq!(sender.i32_column(0), vec![60, 50, 40, 30]);
    }

    #[test]
    fn limit_one() {
        let sender = run_order_by(vec![vec![batch(&[5, 3, 1, 4, 2])]], 1, false);

        assert_eq!(sender.total_rows(), 1);
        assert_eq!(sender.i32_column(0), vec![1]);
    }

    #[test]
    fn duplicates_preserved() {
        let sender = run_order_by(vec![vec![batch(&[3, 1, 1, 2, 2])]], 4, false);

        assert_eq!(sender.total_rows(), 4);
        assert_eq!(sender.i32_column(0), vec![1, 1, 2, 2]);
    }
}
