//! ORDER BY … LIMIT k operator.
//!
//! A pipeline breaker that keeps only the top-k rows per sort key across all
//! input batches, where k = `limit + offset` ("fetch").
//!
//! ## Consume
//!
//! Each worker merges every incoming batch into a single running top-k
//! ([`running_top_k`](OrderByLimit::running_top_k)), so it holds at most `fetch`
//! rows at a time. On finalization it sends that to a shared mpsc channel.
//!
//! ## Output
//!
//! The worker holding the receiver concatenates the per-worker top-ks and does
//! the final global top-k sort, emitting one [`RecordBatch`] of at most `limit`
//! rows in sorted order (after skipping `offset`).
//!
//! ## Rejecting batches before sorting
//!
//! Once the running window is full we cache its fetch-th-best key
//! ([`reject_boundary`](OrderByLimit::reject_boundary)) and test each new batch
//! against it, so rows that can't reach the global top-k are never sorted. Only
//! sound for a single nulls-last sort key (then that key is the whole key, and a
//! null can't enter a full window); multi-key / nulls-first sorts merge plainly.
//!
//! One vectorized `cmp(col, boundary)` gives both the reject *and* the count:
//! `n_keep` = rows that beat the boundary. If `n_keep == 0` the whole batch is
//! dropped with no sort. Otherwise the rows that beat the boundary are *exactly*
//! the `n_keep` best by key, so a single-key top-k capped at `min(fetch, n_keep)`
//! yields precisely the survivors — no `filter` pass, the specialized
//! single-column sort, and never more than `fetch` rows.
//!
//! Why this and not the shapes we tried:
//! * `min`/`max` value-reject skips the comparison mask but needs a vectorized
//!   reduction per type; strings have none, so it degrades to a scalar per-row
//!   `min` that is *slower* than `cmp` on the common rejected path. `cmp`
//!   prefix-compares against a constant and is uniformly fast.
//! * `filter`-then-sort materializes survivors into an intermediate batch; the
//!   `n_keep` cap gets the same survivor-limiting with one fewer pass.
//! * `lexsort([keep, key])` forces the generic multi-column comparator; sorting
//!   the key alone keeps the specialized single-column sort (the survivors are
//!   already the best `n_keep`, so the `keep` column buys nothing in the sort).
//!
//! Not handled yet: for a very large `fetch`, merging each accepted batch into
//! the running top-k re-sorts ~`fetch` rows (concat-and-re-sort). A linear merge
//! of the two already-sorted runs would make that O(fetch + survivors); deferred
//! until a large-`fetch` workload needs it.

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
/// Merges each incoming batch into one running top-k, rejecting batches that
/// can't reach it once the window is full (see the module docs), then sends the
/// running top-k to the shared channel on finalization.
pub struct OrderByLimit {
    limit: usize,
    offset: usize,
    order_by: Vec<OrderBy>,
    sender: mpsc::Sender<RecordBatch>,
    receiver: Option<Receiver<RecordBatch>>,
    /// When set, this worker is a dynamic-filter producer: after each batch it
    /// publishes its current Nth-best leading sort key here so sibling scans
    /// can prune row groups that can't reach the global top-N.
    dynamic_filter: Option<Arc<DynamicFilterSlot>>,
    /// The worker's running top-k: every batch is merged into this single
    /// `limit + offset`-row batch, so the worker holds one top-k at a time.
    running_top_k: Option<RecordBatch>,
    /// Whether to reject whole batches up front against the running boundary.
    /// Only sound for a single sort key (the leading key is the whole key) with
    /// nulls ordered last (a null can't enter a full window), so multi-key /
    /// nulls-first sorts fall back to plain merging.
    boundary_reject: bool,
    /// Cached fetch-th-best key: the reject boundary. Rebuilt only after a merge
    /// actually changes the running top-k, so the common rejected batch reads it
    /// by reference with no per-batch slice. `Some` only once the window is full
    /// (and that key is non-null) on a `boundary_reject` sort.
    reject_boundary: Option<Scalar<ArrayRef>>,
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
        let boundary_reject = order_by.len() == 1 && !order_by[0].nulls_first;
        Self {
            limit,
            offset,
            order_by,
            sender,
            receiver,
            dynamic_filter,
            running_top_k: None,
            boundary_reject,
            reject_boundary: None,
        }
    }

    /// Reduce an incoming batch to the rows worth merging into the running
    /// top-k: at most `fetch` already-sorted rows, or `None` if the whole batch
    /// is provably outside the running top-k (see the module docs).
    fn reduce_batch(
        &self,
        batch: &RecordBatch,
        fetch: usize,
    ) -> unary::Result<Option<RecordBatch>> {
        let Some(boundary) = self.reject_boundary.as_ref() else {
            // No boundary yet (window not full, or a multi-key / nulls-first sort
            // that never rejects): nothing to reject against, just take the top-k.
            return Ok(Some(get_top_k_from_single(
                batch,
                &self.order_by,
                fetch,
                0,
            )?));
        };
        let ordering = &self.order_by[0];
        let kernel = if ordering.descending {
            cmp::gt
        } else {
            cmp::lt
        };
        let column = batch.column(ordering.column_idx) as &dyn Datum;
        let keep = kernel(column, boundary as &dyn Datum).map_err(Error::Arrow)?;
        let n_keep = keep.true_count();
        if n_keep == 0 {
            // Nothing beats the boundary → the whole batch is out, with no sort.
            return Ok(None);
        }
        // The rows that beat the boundary are exactly the `n_keep` best by key, so
        // a single-key top-k capped at `n_keep` yields precisely the survivors —
        // no `filter` pass, the specialized single-column sort, and never more
        // than `fetch` rows even when `fetch` is large.
        Ok(Some(get_top_k_from_single(
            batch,
            &self.order_by,
            fetch.min(n_keep),
            0,
        )?))
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

        // Reduce the batch to its contribution (or drop it entirely if it can't
        // reach the running top-k); see `reduce_batch` / the module docs.
        let Some(candidates) = self.reduce_batch(&batch, fetch)? else {
            return Ok(());
        };

        // Merge into the running top-k. Candidates are few once the boundary is
        // tight, so this is a small sort; before the window fills it's the whole
        // batch's top-k, exactly as a plain per-batch top-k would be.
        let merged = match self.running_top_k.take() {
            None => candidates,
            Some(prev) => get_top_k_from_top_ks(vec![prev, candidates], &self.order_by, fetch, 0)?,
        };

        // Publish from the running top-k: its boundary is the worst of the best
        // `fetch` rows seen so far, so it's at least as tight as any single
        // batch's — tighter bounds prune more row groups in sibling scans.
        self.publish_boundary(&merged);
        // Refresh the cached reject boundary now the top-k has changed. Only the
        // surviving batches that reach here pay this; rejected batches don't.
        if self.boundary_reject && merged.num_rows() >= fetch {
            let last = merged
                .column(self.order_by[0].column_idx)
                .slice(merged.num_rows() - 1, 1);
            if last.is_valid(0) {
                self.reject_boundary = Some(Scalar::new(last));
            }
        }
        self.running_top_k = Some(merged);
        Ok(())
    }

    fn into_outputter(self) -> crate::operations::unary::Result<Option<Self::Outputter>> {
        let OrderByLimit {
            running_top_k,
            sender,
            receiver,
            order_by,
            limit,
            offset,
            ..
        } = self;

        if let Some(local_top_k) = running_top_k {
            debug!("Sending on {:?}", local_top_k.num_rows());
            sender.send(local_top_k).expect("Receiver dropped!");
        }

        // Drop our sender *before* notifying, and notify *unconditionally*. The
        // receiver worker collects the per-worker top-ks with a non-blocking
        // `try_recv` while parked on the shared waker, so it only observes the
        // channel reaching `Disconnected` (all senders dropped) when something
        // wakes it. A worker that consumed no rows has `running_top_k == None`
        // and would otherwise drop its sender silently — if that drop is the
        // one that disconnects the channel and the receiver is parked, it sleeps
        // forever. Dropping first means the wake reflects the post-drop state.
        drop(sender);
        worker_waker().notify();

        Ok(receiver.map(|rx| OrderByLimitOutputter {
            rx,
            batches: vec![],
            order_by,
            limit,
            offset,
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
    fn publishes_running_boundary() {
        let slot = Arc::new(DynamicFilterSlot::new(None));
        let (tx, _rx) = mpsc::channel();
        // ORDER BY v DESC LIMIT 2 takes the single-key fast path: each batch is
        // merged into a running top-2, and the boundary published is that
        // running top-2's 2nd-largest — i.e. the true global 2nd-largest so
        // far, which is at least as tight as any single batch's boundary.
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
        assert_eq!(slot_value(&slot), Some(20)); // running top-2 [30, 20]

        op.consume(batch(&[100, 5]), &mut sink).unwrap();
        assert_eq!(slot_value(&slot), Some(30)); // running top-2 now [100, 30]

        op.consume(batch(&[50, 60]), &mut sink).unwrap();
        assert_eq!(slot_value(&slot), Some(60)); // running top-2 now [100, 60]
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

    /// Regression: a worker that consumed no rows (`running_top_k == None`) must
    /// still wake the waker when it finalizes. The receiver worker collects the
    /// per-worker top-ks with a non-blocking `try_recv` while parked on the
    /// waker, so the *drop* of this worker's sender (which may disconnect the
    /// channel) has to be accompanied by a notify — otherwise a parked receiver
    /// can sleep through the final disconnect and the query hangs forever.
    #[test]
    fn into_outputter_notifies_even_with_no_local_top_k() {
        use crate::worker::{WorkerWaker, init_worker_waker};

        let waker = Arc::new(WorkerWaker::new());
        init_worker_waker(&waker);

        let (tx, rx) = mpsc::channel::<RecordBatch>();
        // Freshly built, nothing consumed -> `running_top_k` is None.
        let obl = OrderByLimit::new(
            vec![OrderBy::new(0, false, false)],
            10,
            0,
            tx,
            Some(rx),
            None,
        );

        let before = waker.wake_count();
        let _ = obl.into_outputter().unwrap();
        assert!(
            waker.wake_count() > before,
            "into_outputter must notify the waker even when it has no local top-k",
        );
    }
}
