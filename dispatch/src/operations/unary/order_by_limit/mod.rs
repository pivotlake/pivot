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
//! Once a full window has been witnessed we test each new batch against its
//! fetch-th-best *leading* key, so rows that can't reach the global top-k are
//! never sorted. The boundary comes from the shared [`DynamicFilterSlot`]'s
//! arming window when one is wired (it pools all workers' rows, so it is always
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

use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::pipeline_breaker::{Consumer, Outputter};
use crate::worker::waker_set;
use arrow::compute::kernels::cmp;
use arrow::compute::{SortColumn, lexsort_to_indices, take};
use arrow_array::{Array, ArrayRef, Datum, RecordBatch, Scalar};
use arrow_schema::{ArrowError, SortOptions};
use std::mem;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use thiserror::Error;
use tracing::debug;

mod factory;
pub use factory::OrderByLimitFactory;

/// Largest `limit + offset` for which a Top-N pools per-batch keys into the
/// shared arming window. Beyond this, merging that many keys under the window
/// mutex would cost more than boundary pruning is worth (a huge window prunes
/// little anyway), so larger fetches publish only from a full per-worker window.
const SHARED_WINDOW_MAX_FETCH: usize = 1024;

/// The shared boundary a Top-N operator publishes its current Nth-best leading
/// sort key into, for sibling scans to prune row groups against. One producer
/// (the Top-N's workers) writes; any number of consumer scans read. Empty until
/// `fetch = limit + offset` rows have been witnessed; the value only ever
/// tightens, so a stale read prunes less, never more.
///
/// The boundary arms from a single *global* window: workers pool the leading
/// keys of each batch's surviving rows here, so it fills as soon as `fetch`
/// rows exist *anywhere*. With N workers each seeing ~1/N of a filtered
/// stream, waiting for one worker's private window to fill takes ~N times
/// longer (and on a selective filter may never happen before the scan ends) -
/// pooling makes arming near-immediate, which is what makes boundary pruning
/// reliable.
///
/// The comparison *direction* lives with the consumer (which carries the
/// comparison operator DuckDB chose); this slot holds only the boundary value.
pub struct DynamicFilterSlot {
    /// The published boundary, i.e. the `fetch`-th best leading key witnessed
    /// so far across all workers. Consumer scans read it for every row group
    /// they pull, so it lives in its own lock apart from the window mutex.
    boundary: RwLock<Option<Scalar<ArrayRef>>>,
    /// The arming window holds the up-to-`fetch` best non-null leading keys
    /// offered so far, sorted best-first in the sort's direction. `None`
    /// until the first offer.
    window: Mutex<Option<ArrayRef>>,
}

impl DynamicFilterSlot {
    pub fn new() -> Self {
        Self {
            boundary: RwLock::new(None),
            window: Mutex::new(None),
        }
    }

    /// The current boundary, or `None` while the window hasn't armed yet.
    pub fn boundary(&self) -> Option<Scalar<ArrayRef>> {
        self.boundary
            .read()
            .expect("dynamic filter slot poisoned")
            .clone()
    }

    /// Offer one batch's surviving rows' leading sort keys to the arming
    /// window. `keys` must be sorted best-first in the sort's direction and
    /// contain no nulls, and every source row may be offered at most once — a
    /// row counted twice would fake a fuller window and over-tighten the
    /// boundary (wrong results, not just weaker pruning). Once `fetch` keys
    /// have pooled, the worst of them is published as the boundary; each later
    /// offer that improves the window tightens it.
    fn offer(&self, keys: ArrayRef, fetch: usize, descending: bool) -> Result<()> {
        if fetch == 0 || keys.is_empty() {
            return Ok(());
        }
        // Only keys beating the current boundary can change a full window, and
        // being best-first they are a prefix of the offer. This keeps the
        // common armed-and-nothing-to-add case off the window mutex entirely.
        let keys = match self.boundary() {
            None => keys,
            Some(current) => {
                let kernel = if descending { cmp::gt } else { cmp::lt };
                let better = kernel(&keys as &dyn Datum, &current as &dyn Datum)?;
                match better.true_count() {
                    0 => return Ok(()),
                    n => keys.slice(0, n),
                }
            }
        };
        let mut window = self.window.lock().expect("dynamic filter window poisoned");
        let merged = match window.as_ref() {
            None => keys.slice(0, keys.len().min(fetch)),
            Some(pooled) => merge_top_keys(pooled, &keys, fetch, descending)?,
        };
        if merged.len() >= fetch {
            self.publish(Scalar::new(merged.slice(merged.len() - 1, 1)), descending);
        }
        *window = Some(merged);
        Ok(())
    }

    /// Publish `new_boundary` if it is strictly tighter than the current one.
    /// [`Self::offer`] publishes through here as its window arms and tightens;
    /// operators whose fetch is too large for the shared window call it
    /// directly when their own window fills.
    fn publish(&self, new_boundary: Scalar<ArrayRef>, descending: bool) {
        let mut guard = self.boundary.write().expect("dynamic filter slot poisoned");
        let tighter = match guard.as_ref() {
            None => true,
            Some(current) => is_tighter(&new_boundary, current, descending),
        };
        if tighter {
            *guard = Some(new_boundary);
        }
    }
}

impl Default for DynamicFilterSlot {
    fn default() -> Self {
        Self::new()
    }
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
    sender: mpsc::Sender<RecordBatch>,
    receiver: Option<Receiver<RecordBatch>>,
    /// When set, this worker is a dynamic-filter producer: it publishes into
    /// the shared slot so sibling scans can prune row groups that can't reach
    /// the global top-N — through the slot's pooled arming window when
    /// `shared_window` is set, or directly from its own full window otherwise.
    dynamic_filter: Option<Arc<DynamicFilterSlot>>,
    /// Whether this operator pools per-batch keys into the slot's shared
    /// arming window (see [`DynamicFilterSlot`]): a slot is wired, the leading
    /// key orders nulls last, and the fetch is small enough for window merges
    /// to stay cheap.
    shared_window: bool,
    /// The worker's running top-k: every batch is merged into this single
    /// `limit + offset`-row batch, so the worker holds one top-k at a time.
    running_top_k: Option<RecordBatch>,
    /// Whether to reject whole batches up front against a full-window boundary
    /// (see the module docs). Requires nulls-last ordering on the leading key,
    /// since a null can't beat a full window.
    boundary_reject: bool,
    /// Whether rows *tied* with the boundary on the leading key are rejected
    /// too. Sound only for a single sort key (the leading key is the whole
    /// key, so equal-key rows are interchangeable under a tie-ambiguous
    /// LIMIT); with more keys a tied row can still win on a later key.
    reject_ties: bool,
    /// Cached fetch-th-best leading key from this worker's own window. Rebuilt
    /// only after a merge actually changes the running top-k, so the common
    /// rejected batch reads it by reference with no per-batch slice. Used only
    /// without a `shared_window`, whose global boundary is always at least as
    /// tight.
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
        let nulls_last_leading = order_by.first().is_some_and(|leading| !leading.nulls_first);
        let shared_window = dynamic_filter.is_some()
            && nulls_last_leading
            && (1..=SHARED_WINDOW_MAX_FETCH).contains(&(limit + offset));
        Self {
            limit,
            offset,
            sender,
            receiver,
            dynamic_filter,
            shared_window,
            running_top_k: None,
            boundary_reject: nulls_last_leading,
            reject_ties: order_by.len() == 1,
            reject_boundary: None,
            order_by,
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
        // Reject against the pooled global boundary when the shared window is
        // in play (it is always at least as tight as this worker's own — see
        // `reject_boundary`); otherwise against the locally maintained one.
        let global_boundary;
        let boundary = if self.shared_window {
            global_boundary = self
                .dynamic_filter
                .as_ref()
                .expect("shared window without a slot")
                .boundary();
            global_boundary.as_ref()
        } else if self.boundary_reject {
            self.reject_boundary.as_ref()
        } else {
            None
        };
        let Some(boundary) = boundary else {
            // No boundary yet (no full window witnessed, or a nulls-first sort
            // that never rejects): nothing to reject against, take the top-k.
            return Ok(Some(get_top_k_from_single(
                batch,
                &self.order_by,
                fetch,
                0,
            )?));
        };
        let ordering = &self.order_by[0];
        // Strict comparison rejects boundary ties; multi-key sorts must keep
        // them (see `reject_ties`).
        let kernel = match (ordering.descending, self.reject_ties) {
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
    /// the shared slot, once the window is full. This is the non-pooling path
    /// (no `shared_window`: a fetch too large for cheap window merges, or a
    /// nulls-first leading key); small fetches pool per-batch keys through the
    /// slot's arming window in `consume` instead.
    ///
    /// The window keeps `limit + offset` rows, so the last (worst) of them is a
    /// valid bound on the *global* boundary: this worker alone already witnesses
    /// that many rows at least as good as it. We prune on the leading key only,
    /// so multi-key sorts publish `order_by[0]`'s value (ties on it are resolved
    /// by keeping the row group — see the consumer's comparison).
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
        // A null boundary can never prune, and comparisons against it are null,
        // so it could never be overwritten by a real key either - skip it.
        if !last_row.is_valid(0) {
            return;
        }
        slot.publish(Scalar::new(last_row), leading.descending);
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
        if fetch == 0 {
            // LIMIT 0 keeps nothing; consuming would only build empty windows.
            return Ok(());
        }

        // Reduce the batch to its contribution (or drop it entirely if it can't
        // reach the running top-k); see `reduce_batch` / the module docs.
        let Some(candidates) = self.reduce_batch(&batch, fetch)? else {
            return Ok(());
        };

        // Pool this batch's keys into the shared arming window. Only the fresh
        // `candidates` are offered — never the running top-k, whose rows were
        // already offered once and would be double-counted (see
        // [`DynamicFilterSlot::offer`]).
        if self.shared_window {
            let slot = self
                .dynamic_filter
                .as_ref()
                .expect("shared window without a slot");
            let leading = candidates.column(self.order_by[0].column_idx);
            // The leading key orders nulls last here (a `shared_window`
            // precondition), so the non-null keys are a best-first prefix.
            let valid = leading.len() - leading.null_count();
            if valid > 0 {
                slot.offer(leading.slice(0, valid), fetch, self.order_by[0].descending)?;
            }
        }

        // Merge into the running top-k. Candidates are few once the boundary is
        // tight, so this is a small sort; before the window fills it's the whole
        // batch's top-k, exactly as a plain per-batch top-k would be.
        let merged = match self.running_top_k.take() {
            None => candidates,
            Some(prev) => get_top_k_from_top_ks(vec![prev, candidates], &self.order_by, fetch, 0)?,
        };

        if !self.shared_window {
            // Without a pooled slot, publish this worker's full-window boundary.
            self.publish_boundary(&merged);
            // Refresh the cached local reject boundary now the top-k has
            // changed (a `shared_window` reads the tighter global boundary per
            // batch instead). Only surviving batches pay this.
            if self.boundary_reject && merged.num_rows() >= fetch {
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
        // The receiver may be parked on another node, so wake every node.
        drop(sender);
        waker_set().notify_all();

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
        slot.boundary().map(|s| {
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
        let slot = Arc::new(DynamicFilterSlot::new());
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
        let slot = Arc::new(DynamicFilterSlot::new());
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

    #[test]
    fn partial_windows_from_different_workers_arm_the_boundary() {
        let slot = Arc::new(DynamicFilterSlot::new());
        let (tx, _rx) = mpsc::channel();
        let order_by = || vec![OrderBy::new(0, false, false)];
        let mut first = OrderByLimit::new(order_by(), 3, 0, tx.clone(), None, Some(slot.clone()));
        let mut second = OrderByLimit::new(order_by(), 3, 0, tx, None, Some(slot.clone()));
        let mut sink = CollectSender::new();

        // Neither worker alone sees 3 rows, but the pooled window does.
        first.consume(batch(&[10, 20]), &mut sink).unwrap();
        assert_eq!(slot_value(&slot), None);
        second.consume(batch(&[30, 5]), &mut sink).unwrap();

        // Pooled keys {10, 20, 30, 5}: best 3 ascending are [5, 10, 20].
        assert_eq!(slot_value(&slot), Some(20));
    }

    #[test]
    fn null_keys_are_not_pooled() {
        let slot = Arc::new(DynamicFilterSlot::new());
        let (tx, _rx) = mpsc::channel();
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true)]));
        let col: ArrayRef = Arc::new(Int32Array::from(vec![Some(5), None, None]));
        let nullable = RecordBatch::try_new(schema, vec![col]).unwrap();
        let mut op = OrderByLimit::new(
            vec![OrderBy::new(0, false, false)],
            2,
            0,
            tx,
            None,
            Some(slot.clone()),
        );
        let mut sink = CollectSender::new();

        // Three rows for a LIMIT 2, but only one non-null key: nulls are no
        // evidence of rows beating a boundary, so the window must not arm.
        op.consume(nullable, &mut sink).unwrap();
        assert_eq!(slot_value(&slot), None);

        // Two more real keys arm it: pooled {5, 7, 9}, best 2 are [5, 7].
        op.consume(batch(&[9, 7]), &mut sink).unwrap();
        assert_eq!(slot_value(&slot), Some(7));
    }

    #[test]
    fn large_fetch_publishes_from_a_full_worker_window() {
        let slot = Arc::new(DynamicFilterSlot::new());
        let (tx, _rx) = mpsc::channel();
        let fetch = SHARED_WINDOW_MAX_FETCH + 1;
        let mut op = OrderByLimit::new(
            vec![OrderBy::new(0, false, false)],
            fetch,
            0,
            tx,
            None,
            Some(slot.clone()),
        );
        let mut sink = CollectSender::new();

        let values: Vec<i32> = (0..fetch as i32 + 10).collect();
        op.consume(batch(&values), &mut sink).unwrap();

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
        let slot = Arc::new(DynamicFilterSlot::new());
        let (tx, _rx) = mpsc::channel();
        let order_by = vec![OrderBy::new(0, false, false), OrderBy::new(1, false, false)];
        let mut op = OrderByLimit::new(order_by, 2, 0, tx, None, Some(slot.clone()));
        let mut sink = CollectSender::new();

        op.consume(two_key_batch(&[(5, 2), (1, 9), (3, 4)]), &mut sink)
            .unwrap();

        // Top-2 rows are (1, 9) and (3, 4); the boundary is the leading key 3.
        assert_eq!(slot_value(&slot), Some(3));
    }

    #[test]
    fn multi_key_keeps_rows_tied_on_the_leading_key() {
        let slot = Arc::new(DynamicFilterSlot::new());
        let (tx, rx) = mpsc::channel();
        let order_by = vec![OrderBy::new(0, false, false), OrderBy::new(1, false, false)];
        let op = OrderByLimit::new(order_by, 2, 0, tx, Some(rx), Some(slot.clone()));

        // The first batch arms the boundary at leading key 3. The second's
        // (3, 1) ties the boundary on the leading key but wins on the second,
        // so batch rejection must let it through to displace (3, 8).
        let sender = run_consumers(
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
        use crate::worker::{WakerSet, WorkerWaker, init_waker_set, init_worker_waker};

        let waker = Arc::new(WorkerWaker::new(1));
        init_worker_waker(&waker);
        init_waker_set(WakerSet::new(vec![waker.clone()], 1));

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
