//! Global aggregate operator (no GROUP BY) computing one or more column
//! aggregates in a single pass.
//!
//! A pipeline breaker (see [`pipeline_breaker`](super::pipeline_breaker)): each
//! worker keeps a local running accumulator per requested aggregate while
//! consuming batches, then on finalization sends its partials down a shared mpsc
//! channel. One worker holds the receiver, drains every sibling's partials, sums
//! them, and emits the single-row result with one output column per aggregate.
//! The cross-worker merge therefore needs no shared lock — it mirrors
//! [`OrderByLimit`](super::order_by_limit), which combines worker partials the
//! same way.
//!
//! The aggregate *kind* is the group module's [`AggregationKind`] /
//! [`AggregationSlot`], shared with GROUP BY, and so is the accumulator width `A`
//! ([`NumericCell`]):
//! the operator is generic over `i64`/`i128`, chosen by the same column-width rule
//! as the grouped path (`i128` only when a sum reads a 64-bit column).
//! `Sum` emits `Int64` or `Decimal128(38, 0)` accordingly (the
//! latter matching DuckDB's `HUGEINT`); `Count` always emits `Int64`. There is no
//! `Avg` kind: DuckDB lowers `AVG(x)` to `sum(x) / count(x)` over two aggregates
//! plus a divide projection, so an average reaches this operator as a `Sum` slot
//! and a `Count` slot.

use crate::operations::channels::Sender;
use crate::operations::unary::group::{AggregationKind, AggregationSlot, Numeric};
use crate::operations::unary::pipeline_breaker::{Consumer, Outputter, PipelineBreaker};
use crate::operations::unary::{self, UnaryFactory};
use crate::worker::worker_waker;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int16Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;
use std::sync::mpsc;

/// Factory for the aggregate operator. All workers share one mpsc channel; the
/// first factory gets the receiver, the rest get `None`, so per-worker partials
/// flow to a single collector (the same wiring as [`OrderByLimitFactory`]).
///
/// [`OrderByLimitFactory`]: super::order_by_limit
pub struct AggregateFactory<A: Numeric> {
    slots: Arc<Vec<AggregationSlot>>,
    sender: mpsc::Sender<Vec<Option<A>>>,
    receiver: Option<mpsc::Receiver<Vec<Option<A>>>>,
}

impl<A: Numeric> AggregateFactory<A> {
    /// Create one factory per worker, all sharing the same partials channel.
    pub fn create_for_workers(
        slots: Vec<AggregationSlot>,
        worker_count: usize,
    ) -> impl IntoIterator<Item = AggregateFactory<A>> {
        let slots = Arc::new(slots);
        let (tx, rx) = mpsc::channel();
        let mut rx_opt = Some(rx);

        (0..worker_count).map(move |_| AggregateFactory {
            slots: slots.clone(),
            sender: tx.clone(),
            receiver: rx_opt.take(),
        })
    }
}

impl<A: Numeric> UnaryFactory<RecordBatch, RecordBatch> for AggregateFactory<A> {
    type Unary = PipelineBreaker<RecordBatch, RecordBatch, Aggregate<A>>;

    fn build_unary(mut self) -> Self::Unary {
        PipelineBreaker::Consuming(Aggregate::new(
            self.slots,
            self.sender,
            self.receiver.take(),
        ))
    }
}

/// Per-worker aggregate consumer. Accumulates local partials, then sends them
/// down the shared channel on finalization.
pub struct Aggregate<A: Numeric> {
    slots: Arc<Vec<AggregationSlot>>,
    local: Vec<Option<A>>,
    sender: mpsc::Sender<Vec<Option<A>>>,
    receiver: Option<mpsc::Receiver<Vec<Option<A>>>>,
}

impl<A: Numeric> Aggregate<A> {
    fn new(
        slots: Arc<Vec<AggregationSlot>>,
        sender: mpsc::Sender<Vec<Option<A>>>,
        receiver: Option<mpsc::Receiver<Vec<Option<A>>>>,
    ) -> Self {
        // No identity seed — each slot takes its first contribution (`None` until
        // a row is seen), so an aggregate over zero rows is a SQL `NULL` rather
        // than a fabricated `0`/bound, and `MIN`/`MAX` need no width extreme.
        let local = vec![None; slots.len()];
        Aggregate {
            slots,
            local,
            sender,
            receiver,
        }
    }
}

/// Fold a fresh contribution into a running (possibly empty) accumulator —
/// take-first: the first value seeds, the rest fold via the slot's op `merge`.
#[inline(always)]
fn fold_in<A: Numeric>(kind: AggregationKind, acc: Option<A>, c: Option<A>) -> Option<A> {
    match (acc, c) {
        (Some(a), Some(b)) => Some(merge_pair(kind, a, b)),
        (Some(a), None) => Some(a),
        (None, c) => c,
    }
}

/// Combine two partials by the slot's op `merge` — the single definition of the
/// additive / extreme fold, shared with the grouped path. The integer ops'
/// `merge` is width-independent (it folds two already-materialised `A`s), so they
/// are named at an arbitrary width. A global string extreme is rejected during
/// planning, so it never reaches here.
#[inline(always)]
fn merge_pair<A: Numeric>(kind: AggregationKind, a: A, b: A) -> A {
    use crate::operations::unary::group::{Count, FoldAcc, Max, Min, Sum};
    match kind {
        AggregationKind::CountStar | AggregationKind::Count => Count::<A>::merge(a, b, &()),
        AggregationKind::Sum => Sum::<A>::merge(a, b, &()),
        AggregationKind::Min => Min::<A>::merge(a, b, &()),
        AggregationKind::Max => Max::<A>::merge(a, b, &()),
        AggregationKind::StrMin | AggregationKind::StrMax => {
            unreachable!("global string extreme is rejected during planning")
        }
    }
}

/// Reduce one batch's integer column to a partial of `kind` (`SUM`/`MIN`/`MAX`),
/// or `None` if the column is empty. `SUM`/`MIN`/`MAX` read the column identically
/// (the widened value); only the fold differs ([`AggregationKind::combine`]).
fn reduce_column<A: Numeric>(kind: AggregationKind, arr: &dyn Array) -> Option<A> {
    macro_rules! reduce_primitive {
        ($ty:ty) => {{
            let a = arr.as_primitive::<$ty>();
            let mut acc: Option<A> = None;
            for i in 0..a.len() {
                let v = A::from(unsafe { a.value_unchecked(i) } as i64);
                acc = fold_in(kind, acc, Some(v));
            }
            acc
        }};
    }

    match arr.data_type() {
        DataType::Int16 => reduce_primitive!(Int16Type),
        DataType::Int32 => reduce_primitive!(Int32Type),
        DataType::Int64 => reduce_primitive!(Int64Type),
        other => panic!("aggregate: unsupported column type {other:?}"),
    }
}

/// Build the single output column for one aggregate slot from its accumulator.
/// `None` means zero rows were aggregated: a SQL `NULL` for `SUM`/`MIN`/`MAX`,
/// `0` for `COUNT`.
fn result_column<A: Numeric>(kind: AggregationKind, total: Option<A>) -> (Field, ArrayRef) {
    match kind {
        // `Int64` or `Decimal128(38, 0)` per the accumulator width; nullable, so
        // an empty input emits `NULL` (DuckDB's `SUM`/`MIN`/`MAX` of nothing).
        AggregationKind::Sum | AggregationKind::Min | AggregationKind::Max => {
            let name = match kind {
                AggregationKind::Min => "min",
                AggregationKind::Max => "max",
                _ => "sum",
            };
            (
                Field::new(name, A::data_type(), true),
                A::scalar_array(total),
            )
        }
        // A global string extreme isn't supported (only grouped is) and is
        // rejected during planning, so it never reaches the numeric operator.
        AggregationKind::StrMin | AggregationKind::StrMax => {
            unreachable!("global string extreme is rejected during planning")
        }
        // A count is `0` over zero rows (never NULL) and fits i64; the checked
        // narrowing panics on the impossible overflow rather than truncating.
        AggregationKind::Count | AggregationKind::CountStar => {
            let count = total
                .map(|v| i64::try_from(v.into()).expect("count exceeds i64::MAX"))
                .unwrap_or(0);
            (
                Field::new("count", DataType::Int64, false),
                Arc::new(Int64Array::from(vec![count])),
            )
        }
    }
}

impl<A: Numeric> Consumer<RecordBatch, RecordBatch> for Aggregate<A> {
    type Outputter = AggregateOutputter<A>;

    fn consume<OP: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        _output: &mut OP,
    ) -> unary::Result<()> {
        for (i, slot) in self.slots.iter().enumerate() {
            // This batch's partial for the slot (`None` if it contributes
            // nothing — an empty SUM/MIN/MAX column), folded into the running
            // local by take-first.
            let contribution = match slot.kind {
                // COUNT(*) counts every row and never reads a column.
                AggregationKind::CountStar => Some(A::from(batch.num_rows() as i64)),
                // COUNT(c) needs only the non-null count, not the values — read
                // it straight off the null bitmap instead of summing the column.
                AggregationKind::Count => {
                    let col = batch.column(slot.column);
                    Some(A::from((col.len() - col.null_count()) as i64))
                }
                AggregationKind::Sum | AggregationKind::Min | AggregationKind::Max => {
                    reduce_column::<A>(slot.kind, batch.column(slot.column))
                }
                // Rejected during planning (only grouped string extremes exist).
                AggregationKind::StrMin | AggregationKind::StrMax => {
                    unreachable!("global string extreme is rejected during planning")
                }
            };
            self.local[i] = fold_in(slot.kind, self.local[i], contribution);
        }
        Ok(())
    }

    fn into_outputter(self) -> unary::Result<Option<Self::Outputter>> {
        // Every worker sends its partials (even all-zero, so the merge always
        // sees a contribution per worker) and wakes the collector. Notifying
        // unconditionally after the send avoids the lost-wakeup that bites when
        // a finishing worker drops its sender while the collector is parked.
        self.sender
            .send(self.local)
            .expect("aggregate collector dropped");
        worker_waker().notify();

        let totals = vec![None; self.slots.len()];
        Ok(self.receiver.map(|rx| AggregateOutputter {
            rx,
            slots: self.slots,
            totals,
        }))
    }
}

/// Output phase (one worker only): drains every sibling's partials from the
/// channel, sums them, then emits the single-row result.
pub struct AggregateOutputter<A: Numeric> {
    rx: mpsc::Receiver<Vec<Option<A>>>,
    slots: Arc<Vec<AggregationSlot>>,
    totals: Vec<Option<A>>,
}

impl<A: Numeric> Outputter<RecordBatch> for AggregateOutputter<A> {
    fn output<OP: Sender<RecordBatch>>(&mut self, output: &mut OP) -> unary::Result<bool> {
        loop {
            match self.rx.try_recv() {
                Ok(partial) => {
                    // Fold each worker's partial with the slot's kind-aware
                    // combine — additive for SUM/COUNT, the extreme for MIN/MAX.
                    for (i, p) in partial.iter().enumerate() {
                        self.totals[i] = fold_in(self.slots[i].kind, self.totals[i], *p);
                    }
                }
                // Some siblings haven't finished yet; resume when re-driven.
                Err(mpsc::TryRecvError::Empty) => return Ok(false),
                // All senders dropped → every worker's partial is folded in.
                Err(mpsc::TryRecvError::Disconnected) => {
                    let (fields, columns): (Vec<Field>, Vec<ArrayRef>) = self
                        .slots
                        .iter()
                        .zip(&self.totals)
                        .map(|(slot, total)| result_column(slot.kind, *total))
                        .unzip();
                    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
                    output.send(batch)?;
                    return Ok(true);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::unary::test_utils::run_consumers;
    use arrow_array::{Decimal128Array, Int32Array};

    fn make_batch(values: &[i32]) -> RecordBatch {
        let array = Int32Array::from(values.to_vec());
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)])),
            vec![Arc::new(array)],
        )
        .unwrap()
    }

    /// Build `n` channel-wired aggregate consumers sharing one partials channel
    /// (the first holds the receiver), mirroring the factory's wiring.
    fn build<A: Numeric>(n: usize, slots: Vec<AggregationSlot>) -> Vec<Aggregate<A>> {
        let slots = Arc::new(slots);
        let (tx, rx) = mpsc::channel();
        let mut rx_opt = Some(rx);
        (0..n)
            .map(move |_| Aggregate::new(slots.clone(), tx.clone(), rx_opt.take()))
            .collect()
    }

    fn col_i64(batch: &RecordBatch, i: usize) -> i64 {
        batch.column(i).as_primitive::<Int64Type>().value(0)
    }
    fn col_i128(batch: &RecordBatch, i: usize) -> i128 {
        batch
            .column(i)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap()
            .value(0)
    }

    fn slot(kind: AggregationKind, column: usize) -> AggregationSlot {
        AggregationSlot::new(kind, column)
    }

    #[test]
    fn single_worker_sum() {
        // Int32 column → i64 accumulator → Int64 output.
        let ops = build::<i64>(1, vec![slot(AggregationKind::Sum, 0)]);
        let out = run_consumers(ops, vec![vec![make_batch(&[1, 2, 3]), make_batch(&[4, 5])]]);
        assert_eq!(out.items.len(), 1);
        assert_eq!(col_i64(&out.items[0], 0), 15);
    }

    #[test]
    fn sum_and_count_together() {
        let ops = build::<i64>(
            1,
            vec![
                slot(AggregationKind::Sum, 0),
                slot(AggregationKind::Count, 0),
            ],
        );
        let out = run_consumers(ops, vec![vec![make_batch(&[2, 4, 6, 8])]]);
        assert_eq!(out.items.len(), 1);
        assert_eq!(col_i64(&out.items[0], 0), 20); // sum
        assert_eq!(col_i64(&out.items[0], 1), 4); // count
    }

    #[test]
    fn multiple_workers_sum_merge() {
        let ops = build::<i64>(3, vec![slot(AggregationKind::Sum, 0)]);
        let out = run_consumers(
            ops,
            vec![
                vec![make_batch(&[10])],
                vec![make_batch(&[20, 5])],
                vec![make_batch(&[1])],
            ],
        );
        assert_eq!(out.items.len(), 1);
        assert_eq!(col_i64(&out.items[0], 0), 36);
    }

    #[test]
    fn large_i64_sum_is_exact() {
        // Three i64::MAX values sum past i64::MAX; an i128 accumulator over an
        // Int64 column must keep the full value and emit it as Decimal128.
        let schema = Arc::new(Schema::new(vec![Field::new("u", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow_array::Int64Array::from(vec![
                i64::MAX,
                i64::MAX,
                i64::MAX,
            ]))],
        )
        .unwrap();
        let ops = build::<i128>(1, vec![slot(AggregationKind::Sum, 0)]);
        let out = run_consumers(ops, vec![vec![batch]]);
        assert_eq!(col_i128(&out.items[0], 0), 3 * i64::MAX as i128);
    }

    #[test]
    fn count_star_counts_all_rows() {
        let ops = build::<i64>(1, vec![slot(AggregationKind::CountStar, 0)]);
        let out = run_consumers(ops, vec![vec![make_batch(&[5, 6, 7]), make_batch(&[8])]]);
        assert_eq!(col_i64(&out.items[0], 0), 4);
    }

    #[test]
    fn min_and_max_over_batches() {
        // MIN/MAX fold the extreme across rows and across batches, not a sum.
        let ops = build::<i64>(
            1,
            vec![slot(AggregationKind::Min, 0), slot(AggregationKind::Max, 0)],
        );
        let out = run_consumers(
            ops,
            vec![vec![make_batch(&[5, 2, 9]), make_batch(&[7, 1, 8])]],
        );
        assert_eq!(out.items.len(), 1);
        assert_eq!(col_i64(&out.items[0], 0), 1); // min
        assert_eq!(col_i64(&out.items[0], 1), 9); // max
    }

    #[test]
    fn min_max_merge_across_workers() {
        // Each worker's partial extreme combines (not adds) at the collector, so
        // the global MIN/MAX is the extreme of all workers' extremes — including
        // negatives, which the MAX identity (i64::MIN) must not swallow.
        let ops = build::<i64>(
            3,
            vec![slot(AggregationKind::Min, 0), slot(AggregationKind::Max, 0)],
        );
        let out = run_consumers(
            ops,
            vec![
                vec![make_batch(&[10, 4])],
                vec![make_batch(&[-3, 20])],
                vec![make_batch(&[7])],
            ],
        );
        assert_eq!(out.items.len(), 1);
        assert_eq!(col_i64(&out.items[0], 0), -3); // min
        assert_eq!(col_i64(&out.items[0], 1), 20); // max
    }

    #[test]
    fn empty_input_sum_is_zero() {
        let ops = build::<i64>(1, vec![slot(AggregationKind::Sum, 0)]);
        let out = run_consumers(ops, vec![vec![]]);
        assert_eq!(out.items.len(), 1);
        assert_eq!(col_i64(&out.items[0], 0), 0);
    }
}
