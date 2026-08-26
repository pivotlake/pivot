//! ORDER BY … LIMIT k operator.
//!
//! A blocking unary operator that keeps only the top-k rows per sort key across all
//! input batches, where k = `limit + offset` ("fetch").
//!
//! ## Consume
//!
//! Each worker merges every incoming batch into a single running top-k
//! ([`running_top_k`](OrderByLimit::running_top_k)), so it holds at most `fetch`
//! rows at a time. On finalization it publishes that to a shared gather barrier.
//!
//! ## Output
//!
//! The final worker to reach the gather barrier concatenates the per-worker
//! top-ks and does the final global top-k sort, emitting one [`RecordBatch`] of
//! at most `limit` rows in sorted order (after skipping `offset`).
//!
//! ## Rejecting batches before sorting
//!
//! Once a full window has been witnessed we test each new batch against its
//! fetch-th-best *leading* key, so rows that can't reach the global top-k are
//! never sorted. The boundary comes from the shared key window when a
//! [`BoundarySlot`] is wired (it pools all workers' rows, so it is always
//! at least as tight as this worker's own), otherwise from the locally cached
//! [`reject_boundary`](OrderByLimit::reject_boundary). Requires nulls-last
//! ordering on the leading key (a null can't beat a full window). With a single
//! sort key, rows *tied* with the boundary are rejected too (equal keys are
//! interchangeable under a tie-ambiguous LIMIT); multi-key sorts keep ties,
//! since a row tied on the leading key can still win on a later key.
//!
//! One vectorized `cmp(col, boundary)` gives both the reject *and* the count:
//! `n_keep` = rows that beat the boundary. If `n_keep == 0` the whole batch is
//! dropped with no sort. Otherwise the rows that beat the boundary are *exactly*
//! the `n_keep` best rows, so a top-k capped at `min(fetch, n_keep)` yields
//! precisely the survivors - no `filter` pass, the specialized single-column
//! sort on single-key sorts, and never more than `fetch` rows.
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

use crate::boundary_slot::BoundarySlot;
use crate::gather_barrier::GatherBarrier;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::Unary;
use arrow::compute::kernels::cmp;
use arrow::compute::{SortColumn, lexsort_to_indices, take};
use arrow_array::{Array, ArrayRef, Datum, RecordBatch, Scalar};
use arrow_schema::{ArrowError, SortOptions};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use thiserror::Error;
use tracing::debug;

mod factory;
pub use factory::OrderByLimitFactory;

/// Largest `limit + offset` for which a Top-N pools per-batch keys into the
/// shared key window. Beyond this, merging that many keys under the window
/// mutex would cost more than boundary pruning is worth (a huge window prunes
/// little anyway), so larger fetches publish only from a full per-worker window.
const SHARED_WINDOW_MAX_FETCH: usize = 1024;

/// Offer one batch's surviving rows' leading sort keys to a Top-N's shared
/// key window, publishing into `slot` once the window fills. `keys` must be
/// non-empty, sorted best-first in the sort's direction, and contain no nulls,
/// and every source row may be offered at most once - a row counted twice
/// would fake a fuller window and over-tighten the boundary (wrong results,
/// not just weaker pruning). `boundary` is the caller's already-read
/// [`BoundarySlot::boundary`] (a stale read only weakens the prefix filter
/// below, never the window). Once `fetch` keys have pooled, the worst of them
/// is published as the boundary; each later offer that improves the window
/// tightens it.
fn offer_to_window(
    window: &Mutex<Option<ArrayRef>>,
    slot: &BoundarySlot,
    keys: ArrayRef,
    fetch: usize,
    descending: bool,
    boundary: Option<&Scalar<ArrayRef>>,
) -> Result<()> {
    debug_assert!(fetch > 0 && !keys.is_empty());
    // Only keys beating the current boundary can change a full window, and
    // being best-first they are a prefix of the offer. This keeps the
    // common armed-and-nothing-to-add case off the window mutex entirely.
    let keys = match boundary {
        None => keys,
        Some(current) => {
            let kernel = if descending { cmp::gt } else { cmp::lt };
            let better = kernel(&keys as &dyn Datum, current as &dyn Datum)?;
            match better.true_count() {
                0 => return Ok(()),
                n => keys.slice(0, n),
            }
        }
    };
    let mut window = window.lock().expect("top-n key window poisoned");
    let merged = match window.as_ref() {
        None => keys.slice(0, keys.len().min(fetch)),
        Some(pooled) => merge_top_keys(pooled, &keys, fetch, descending)?,
    };
    if merged.len() >= fetch {
        slot.publish_if_tighter(Scalar::new(merged.slice(merged.len() - 1, 1)), descending);
    }
    *window = Some(merged);
    Ok(())
}

/// Merge two best-first sorted key runs into the best `fetch` of their union.
/// Both runs hold at most `fetch` keys, so this is a tiny concat-and-sort.
fn merge_top_keys(a: &ArrayRef, b: &ArrayRef, fetch: usize, descending: bool) -> Result<ArrayRef> {
    let pool = arrow::compute::concat(&[a.as_ref(), b.as_ref()])?;
    let options = SortOptions {
        descending,
        nulls_first: false,
    };
    let indices = arrow::compute::sort_to_indices(&pool, Some(options), Some(fetch))?;
    Ok(take(&pool, &indices, None)?)
}

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

    pub fn column_idx(&self) -> usize {
        self.column_idx
    }

    pub fn descending(&self) -> bool {
        self.descending
    }

    pub fn nulls_first(&self) -> bool {
        self.nulls_first
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

    // A variant column arrives with a per-file physical layout, so top-ks from
    // different files cannot concatenate until their mismatched variant
    // columns are unshredded. A no-op when the layouts already agree.
    let batches = crate::arrays::variant::unify_variant_layouts(batches)?;
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
    // No sort keys → a plain LIMIT/OFFSET: any rows satisfy it, so keep the
    // first `fetch` rows of the batch then drop the leading `skip`, as a cheap
    // O(1) logical slice (no sort kernel — `lexsort_to_indices` rejects an empty
    // column list anyway). Downstream consumers handle logically-sliced arrays
    // (e.g. the materializer walks `RunEndBuffer::sliced_values()`).
    if order_by.is_empty() {
        let end = batch.num_rows().min(fetch);
        let skip = skip.min(end);
        return Ok(batch.slice(skip, end - skip));
    }

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
    gather: Arc<GatherBarrier<Option<RecordBatch>>>,
    /// When set, this worker is a boundary producer that pools each batch's
    /// keys into `key_window`, shared by every worker of this Top-N (see
    /// [`offer_to_window`]): the leading key orders nulls last and the fetch
    /// is small enough for window merges to stay cheap. Mutually exclusive
    /// with `publish_slot`.
    pooled_slot: Option<Arc<BoundarySlot>>,
    /// The key window the pooled workers merge their keys into: the
    /// up-to-`fetch` best non-null leading keys offered so far, sorted
    /// best-first in the sort's direction. Shared across this Top-N's workers
    /// because each sees only ~1/N of the stream: a per-worker window arms ~N
    /// times slower (on a selective filter maybe never), and near-immediate
    /// arming is what makes boundary pruning reliable. Unused without a
    /// `pooled_slot`.
    key_window: Arc<Mutex<Option<ArrayRef>>>,
    /// When set, this worker is a boundary producer whose fetch is too large
    /// for the shared window (or whose leading key orders nulls first): it
    /// publishes directly from its own full window instead (see
    /// [`Self::publish_boundary`]).
    publish_slot: Option<Arc<BoundarySlot>>,
    /// The worker's running top-k: every batch is merged into this single
    /// `limit + offset`-row batch, so the worker holds one top-k at a time.
    running_top_k: Option<RecordBatch>,
    /// Cached fetch-th-best leading key from this worker's own window. Rebuilt
    /// only after a merge actually changes the running top-k, so the common
    /// rejected batch reads it by reference with no per-batch slice. Used only
    /// without a `pooled_slot`, whose global boundary is always at least as
    /// tight.
    reject_boundary: Option<Scalar<ArrayRef>>,
}

impl OrderByLimit {
    pub(super) fn new(
        order_by: Vec<OrderBy>,
        limit: usize,
        offset: usize,
        gather: Arc<GatherBarrier<Option<RecordBatch>>>,
        boundary_slot: Option<Arc<BoundarySlot>>,
        key_window: Arc<Mutex<Option<ArrayRef>>>,
    ) -> Self {
        let nulls_last_leading = order_by.first().is_some_and(|leading| !leading.nulls_first);
        let pool = nulls_last_leading && (1..=SHARED_WINDOW_MAX_FETCH).contains(&(limit + offset));
        let (pooled_slot, publish_slot) = match boundary_slot {
            Some(slot) if pool => (Some(slot), None),
            other => (None, other),
        };
        Self {
            limit,
            offset,
            gather,
            pooled_slot,
            key_window,
            publish_slot,
            running_top_k: None,
            reject_boundary: None,
            order_by,
        }
    }

    /// Whether whole batches may be rejected up front against a full-window
    /// boundary (see the module docs). Requires nulls-last ordering on the
    /// leading key, since a null can't beat a full window.
    fn rejects_by_boundary(&self) -> bool {
        self.order_by
            .first()
            .is_some_and(|leading| !leading.nulls_first)
    }

    /// Whether rows *tied* with the boundary on the leading key are rejected
    /// too. Sound only for a single sort key (the leading key is the whole
    /// key, so equal-key rows are interchangeable under a tie-ambiguous
    /// LIMIT); with more keys a tied row can still win on a later key.
    fn rejects_ties(&self) -> bool {
        self.order_by.len() == 1
    }

    /// Reduce an incoming batch to the rows worth merging into the running
    /// top-k: at most `fetch` already-sorted rows, or `None` if the whole batch
    /// is provably outside the running top-k (see the module docs). `boundary`
    /// is the full-window boundary to reject against, or `None` while no full
    /// window has been witnessed (or on a nulls-first sort that never rejects).
    fn reduce_batch(
        &self,
        batch: &RecordBatch,
        fetch: usize,
        boundary: Option<&Scalar<ArrayRef>>,
    ) -> unary::Result<Option<RecordBatch>> {
        let Some(boundary) = boundary else {
            // Nothing to reject against, take the batch's top-k.
            return Ok(Some(get_top_k_from_single(
                batch,
                &self.order_by,
                fetch,
                0,
            )?));
        };
        let ordering = &self.order_by[0];
        // Strict comparison rejects boundary ties; multi-key sorts must keep
        // them (see `rejects_ties`).
        let kernel = match (ordering.descending, self.rejects_ties()) {
            (true, true) => cmp::gt,
            (true, false) => cmp::gt_eq,
            (false, true) => cmp::lt,
            (false, false) => cmp::lt_eq,
        };
        let column = batch.column(ordering.column_idx) as &dyn Datum;
        let keep = kernel(column, boundary as &dyn Datum).map_err(Error::Arrow)?;
        let n_keep = keep.true_count();
        if n_keep == 0 {
            // Nothing beats the boundary → the whole batch is out, with no sort.
            return Ok(None);
        }
        // Every surviving row sorts before every rejected one (a rejected row is
        // strictly worse on the leading key than all `n_keep` survivors), so a
        // top-k capped at `n_keep` yields precisely the survivors - no `filter`
        // pass, and never more than `fetch` rows even when `fetch` is large.
        Ok(Some(get_top_k_from_single(
            batch,
            &self.order_by,
            fetch.min(n_keep),
            0,
        )?))
    }

    /// Publish this worker's running-window Nth-best leading key directly into
    /// the `publish_slot`, once the window is full. This is the non-pooling
    /// path (a fetch too large for cheap window merges, or a nulls-first
    /// leading key); small fetches pool per-batch keys through the slot's
    /// key window in `consume` instead.
    ///
    /// The window keeps `limit + offset` rows, so the last (worst) of them is a
    /// valid bound on the *global* boundary: this worker alone already witnesses
    /// that many rows at least as good as it. We prune on the leading key only,
    /// so multi-key sorts publish `order_by[0]`'s value (ties on it are resolved
    /// by keeping the row group - see the consumer's comparison).
    fn publish_boundary(&self, batch_top_k: &RecordBatch) {
        let Some(slot) = &self.publish_slot else {
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
        // A null boundary can never prune, and comparisons against it are null,
        // so it could never be overwritten by a real key either - skip it.
        if !last_row.is_valid(0) {
            return;
        }
        slot.publish_if_tighter(Scalar::new(last_row), leading.descending);
    }
}

impl Unary<RecordBatch, RecordBatch> for OrderByLimit {
    fn consume(
        &mut self,
        batch: RecordBatch,
        _sender: &mut dyn Sender<RecordBatch>,
        _io: &mut crate::io::OperatorIO,
    ) -> unary::Result<()> {
        debug!("Received batch of length {:?}", batch.num_rows());
        // Local stages keep `limit + offset` candidates (skip = 0); only the
        // final global merge skips `offset`, since which rows fall in the
        // offset window can only be decided once all workers' tops are merged.
        let fetch = self.limit + self.offset;
        if fetch == 0 {
            // LIMIT 0 keeps nothing; consuming would only build empty windows.
            return Ok(());
        }

        // Pick the boundary to reject against: the pooled global one when a
        // pooled slot is wired (always at least as tight as this worker's
        // own, read once per batch and reused by the offer below), otherwise
        // the locally maintained one.
        let pooled_boundary = self.pooled_slot.as_ref().and_then(|slot| slot.boundary());
        let boundary = if self.pooled_slot.is_some() {
            pooled_boundary.as_ref()
        } else if self.rejects_by_boundary() {
            self.reject_boundary.as_ref()
        } else {
            None
        };

        // Reduce the batch to its contribution (or drop it entirely if it can't
        // reach the running top-k); see `reduce_batch` / the module docs.
        let Some(candidates) = self.reduce_batch(&batch, fetch, boundary)? else {
            return Ok(());
        };

        // Pool this batch's keys into the shared key window. Only the fresh
        // `candidates` are offered - never the running top-k, whose rows were
        // already offered once and would be double-counted (see
        // [`offer_to_window`]).
        if let Some(slot) = &self.pooled_slot {
            let leading = candidates.column(self.order_by[0].column_idx);
            // The leading key orders nulls last here (a `pooled_slot`
            // precondition), so the non-null keys are a best-first prefix.
            let valid = leading.len() - leading.null_count();
            if valid > 0 {
                offer_to_window(
                    &self.key_window,
                    slot,
                    leading.slice(0, valid),
                    fetch,
                    self.order_by[0].descending,
                    pooled_boundary.as_ref(),
                )?;
            }
        }

        // Merge into the running top-k. Candidates are few once the boundary is
        // tight, so this is a small sort; before the window fills it's the whole
        // batch's top-k, exactly as a plain per-batch top-k would be.
        let merged = match self.running_top_k.take() {
            None => candidates,
            Some(prev) => get_top_k_from_top_ks(vec![prev, candidates], &self.order_by, fetch, 0)?,
        };

        if self.pooled_slot.is_none() {
            // Without a pooled slot, publish this worker's full-window boundary.
            self.publish_boundary(&merged);
            // Refresh the cached local reject boundary now the top-k has
            // changed (a pooled slot reads the tighter global boundary per
            // batch instead). Only surviving batches pay this.
            if self.rejects_by_boundary() && merged.num_rows() >= fetch {
                let last = merged
                    .column(self.order_by[0].column_idx)
                    .slice(merged.num_rows() - 1, 1);
                if last.is_valid(0) {
                    self.reject_boundary = Some(Scalar::new(last));
                }
            }
        }
        self.running_top_k = Some(merged);
        Ok(())
    }

    fn finish(&mut self, sender: &mut dyn Sender<RecordBatch>) -> unary::Result<bool> {
        let order_by = &self.order_by;
        let limit = self.limit;
        let offset = self.offset;
        let completion =
            self.gather
                .arrive(self.running_top_k.take(), |values| -> unary::Result<()> {
                    let batches: Vec<RecordBatch> = values.into_iter().flatten().collect();
                    if batches.is_empty() {
                        return Ok(());
                    }
                    debug!("from {:?} batches", batches.len());
                    let start = Instant::now();
                    let output = get_top_k_from_top_ks(batches, order_by, limit + offset, offset)?;
                    sender.send(output)?;
                    debug!("took {:?}", start.elapsed());
                    Ok(())
                });
        if let Some(result) = completion {
            result?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::unary::test_utils::CollectSender;
    use crate::waker::install_test_worker_waker;
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

    fn slot_value(slot: &BoundarySlot) -> Option<i32> {
        slot.boundary().map(|s| {
            s.get()
                .0
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0)
        })
    }

    fn order_by_limit(
        order_by: Vec<OrderBy>,
        limit: usize,
        offset: usize,
        boundary_slot: Option<Arc<BoundarySlot>>,
    ) -> OrderByLimit {
        OrderByLimit::new(
            order_by,
            limit,
            offset,
            Arc::new(GatherBarrier::new(1)),
            boundary_slot,
            Arc::new(Mutex::new(None)),
        )
    }

    fn run_order_by_limit_workers(
        mut operators: Vec<OrderByLimit>,
        worker_batches: Vec<Vec<RecordBatch>>,
    ) -> CollectSender {
        install_test_worker_waker();
        let mut sender = CollectSender::new();

        for (worker_index, (operator, batches)) in
            operators.iter_mut().zip(worker_batches).enumerate()
        {
            crate::worker::WORKER_IDX.set(worker_index);
            for batch in batches {
                operator
                    .consume(
                        batch,
                        &mut sender,
                        &mut crate::io::TestOperatorIO::default().io(),
                    )
                    .unwrap();
            }
        }
        for (worker_index, operator) in operators.iter_mut().enumerate() {
            crate::worker::WORKER_IDX.set(worker_index);
            assert!(operator.finish(&mut sender).unwrap());
        }

        sender
    }

    #[test]
    fn publishes_running_boundary() {
        let slot = Arc::new(BoundarySlot::new());
        // ORDER BY v DESC LIMIT 2 takes the single-key fast path: each batch is
        // merged into a running top-2, and the boundary published is that
        // running top-2's 2nd-largest — i.e. the true global 2nd-largest so
        // far, which is at least as tight as any single batch's boundary.
        let mut op = order_by_limit(vec![OrderBy::new(0, true, false)], 2, 0, Some(slot.clone()));
        let mut sink = CollectSender::new();

        op.consume(
            batch(&[10, 20, 30]),
            &mut sink,
            &mut crate::io::TestOperatorIO::default().io(),
        )
        .unwrap();
        assert_eq!(slot_value(&slot), Some(20)); // running top-2 [30, 20]

        op.consume(
            batch(&[100, 5]),
            &mut sink,
            &mut crate::io::TestOperatorIO::default().io(),
        )
        .unwrap();
        assert_eq!(slot_value(&slot), Some(30)); // running top-2 now [100, 30]

        op.consume(
            batch(&[50, 60]),
            &mut sink,
            &mut crate::io::TestOperatorIO::default().io(),
        )
        .unwrap();
        assert_eq!(slot_value(&slot), Some(60)); // running top-2 now [100, 60]
    }

    #[test]
    fn does_not_publish_before_window_is_full() {
        let slot = Arc::new(BoundarySlot::new());
        let mut op = order_by_limit(vec![OrderBy::new(0, true, false)], 3, 0, Some(slot.clone()));
        let mut sink = CollectSender::new();

        // Only 2 rows for a LIMIT 3 — no full window yet, nothing to bound on.
        op.consume(
            batch(&[10, 20]),
            &mut sink,
            &mut crate::io::TestOperatorIO::default().io(),
        )
        .unwrap();

        assert_eq!(slot_value(&slot), None);
    }

    #[test]
    fn partial_windows_from_different_workers_arm_the_boundary() {
        let slot = Arc::new(BoundarySlot::new());
        let gather = Arc::new(GatherBarrier::new(2));
        let order_by = || vec![OrderBy::new(0, false, false)];
        // The window must be one shared pool for cross-worker arming to work.
        let window = Arc::new(Mutex::new(None));
        let mut first = OrderByLimit::new(
            order_by(),
            3,
            0,
            gather.clone(),
            Some(slot.clone()),
            window.clone(),
        );
        let mut second = OrderByLimit::new(order_by(), 3, 0, gather, Some(slot.clone()), window);
        let mut sink = CollectSender::new();

        // Neither worker alone sees 3 rows, but the pooled window does.
        first
            .consume(
                batch(&[10, 20]),
                &mut sink,
                &mut crate::io::TestOperatorIO::default().io(),
            )
            .unwrap();
        assert_eq!(slot_value(&slot), None);
        second
            .consume(
                batch(&[30, 5]),
                &mut sink,
                &mut crate::io::TestOperatorIO::default().io(),
            )
            .unwrap();

        // Pooled keys {10, 20, 30, 5}: best 3 ascending are [5, 10, 20].
        assert_eq!(slot_value(&slot), Some(20));
    }

    #[test]
    fn null_keys_are_not_pooled() {
        let slot = Arc::new(BoundarySlot::new());
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true)]));
        let col: ArrayRef = Arc::new(Int32Array::from(vec![Some(5), None, None]));
        let nullable = RecordBatch::try_new(schema, vec![col]).unwrap();
        let mut op = order_by_limit(
            vec![OrderBy::new(0, false, false)],
            2,
            0,
            Some(slot.clone()),
        );
        let mut sink = CollectSender::new();

        // Three rows for a LIMIT 2, but only one non-null key: nulls are no
        // evidence of rows beating a boundary, so the window must not arm.
        op.consume(
            nullable,
            &mut sink,
            &mut crate::io::TestOperatorIO::default().io(),
        )
        .unwrap();
        assert_eq!(slot_value(&slot), None);

        // Two more real keys arm it: pooled {5, 7, 9}, best 2 are [5, 7].
        op.consume(
            batch(&[9, 7]),
            &mut sink,
            &mut crate::io::TestOperatorIO::default().io(),
        )
        .unwrap();
        assert_eq!(slot_value(&slot), Some(7));
    }

    #[test]
    fn large_fetch_publishes_from_a_full_worker_window() {
        let slot = Arc::new(BoundarySlot::new());
        let fetch = SHARED_WINDOW_MAX_FETCH + 1;
        let mut op = order_by_limit(
            vec![OrderBy::new(0, false, false)],
            fetch,
            0,
            Some(slot.clone()),
        );
        let mut sink = CollectSender::new();

        let values: Vec<i32> = (0..fetch as i32 + 10).collect();
        op.consume(
            batch(&values),
            &mut sink,
            &mut crate::io::TestOperatorIO::default().io(),
        )
        .unwrap();

        // Too large for the pooled window, so the boundary comes from this
        // worker's own full window: its fetch-th smallest key.
        assert_eq!(slot_value(&slot), Some(fetch as i32 - 1));
    }

    fn two_key_batch(rows: &[(i32, i32)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let a: ArrayRef = Arc::new(Int32Array::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        ));
        let b: ArrayRef = Arc::new(Int32Array::from(
            rows.iter().map(|r| r.1).collect::<Vec<_>>(),
        ));
        RecordBatch::try_new(schema, vec![a, b]).unwrap()
    }

    #[test]
    fn multi_key_sort_publishes_leading_key_boundary() {
        let slot = Arc::new(BoundarySlot::new());
        let order_by = vec![OrderBy::new(0, false, false), OrderBy::new(1, false, false)];
        let mut op = order_by_limit(order_by, 2, 0, Some(slot.clone()));
        let mut sink = CollectSender::new();

        op.consume(
            two_key_batch(&[(5, 2), (1, 9), (3, 4)]),
            &mut sink,
            &mut crate::io::TestOperatorIO::default().io(),
        )
        .unwrap();

        // Top-2 rows are (1, 9) and (3, 4); the boundary is the leading key 3.
        assert_eq!(slot_value(&slot), Some(3));
    }

    #[test]
    fn multi_key_keeps_rows_tied_on_the_leading_key() {
        let slot = Arc::new(BoundarySlot::new());
        let order_by = vec![OrderBy::new(0, false, false), OrderBy::new(1, false, false)];
        let op = order_by_limit(order_by, 2, 0, Some(slot.clone()));

        // The first batch arms the boundary at leading key 3. The second's
        // (3, 1) ties the boundary on the leading key but wins on the second,
        // so batch rejection must let it through to displace (3, 8).
        let sender = run_order_by_limit_workers(
            vec![op],
            vec![vec![
                two_key_batch(&[(1, 9), (3, 8)]),
                two_key_batch(&[(3, 1), (7, 0)]),
            ]],
        );

        assert_eq!(sender.total_rows(), 2);
        assert_eq!(sender.i32_column(0), vec![1, 3]);
        assert_eq!(sender.i32_column(1), vec![9, 1]);
    }

    fn run_order_by(
        worker_batches: Vec<Vec<RecordBatch>>,
        limit: usize,
        descending: bool,
    ) -> CollectSender {
        let worker_count = worker_batches.len();
        let gather = Arc::new(GatherBarrier::new(worker_count));

        let consumers: Vec<_> = (0..worker_count)
            .map(|_| {
                OrderByLimit::new(
                    vec![OrderBy::new(0, descending, false)],
                    limit,
                    0,
                    gather.clone(),
                    None,
                    Arc::new(Mutex::new(None)),
                )
            })
            .collect();

        run_order_by_limit_workers(consumers, worker_batches)
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

    /// Regression: the final arrival must wake parked workers even when no
    /// worker produced a local top-k.
    #[test]
    fn finish_notifies_even_with_no_local_top_k() {
        use crate::waker::{WakerSet, WorkerWaker, init_waker_set, init_worker_waker};

        let waker = Arc::new(WorkerWaker::new(1));
        init_worker_waker(&waker);
        init_waker_set(WakerSet::new(vec![waker.clone()], 1));
        crate::worker::WORKER_IDX.set(0);

        // Freshly built, nothing consumed -> `running_top_k` is None.
        let mut obl = OrderByLimit::new(
            vec![OrderBy::new(0, false, false)],
            10,
            0,
            Arc::new(GatherBarrier::new(1)),
            None,
            Arc::new(Mutex::new(None)),
        );
        let mut sender = CollectSender::new();

        let before = waker.wake_count();
        assert!(obl.finish(&mut sender).unwrap());
        assert!(
            waker.wake_count() > before,
            "finish must notify the waker even when it has no local top-k",
        );
    }
}
