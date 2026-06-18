//! Global aggregate operator (no GROUP BY) computing one or more column
//! aggregates in a single pass.
//!
//! A pipeline breaker (see [`pipeline_breaker`](super::pipeline_breaker)): each
//! worker keeps a local running accumulator per requested aggregate while
//! consuming batches, then on finalization sends its partials down a shared mpsc
//! channel. One worker holds the receiver, drains every sibling's partials, folds
//! them, and emits the single-row result with one output column per aggregate.
//! The cross-worker merge therefore needs no shared lock — it mirrors
//! [`OrderByLimit`](super::order_by_limit), which combines worker partials the
//! same way.
//!
//! The aggregate *kind* is the group module's [`AggregationKind`] /
//! [`AggregationSlot`], shared with GROUP BY. Each slot accumulates as a
//! [`Partial`]: a numeric reduction (`COUNT`/`SUM`/`MIN`/`MAX` over an integer
//! column, in the width `A` the planner picks — `i128` only when a sum reads a
//! 64-bit column) or a string extreme (`MIN`/`MAX` over a `Utf8` column, kept as
//! an owned `String`). Unlike the grouped string path there's no value arena: a
//! global string extreme is a single winner per slot, so the owned string is
//! cheaper than an `ArenaKey`. `Sum` emits `Int64`/`Decimal128(38, 0)` (the latter
//! matching DuckDB's `HUGEINT`), `Count` always `Int64`, a string extreme
//! `Utf8View`. There is no `Avg` kind: DuckDB lowers `AVG(x)` to `sum(x) /
//! count(x)`, so an average reaches this operator as a `Sum` slot and a `Count`
//! slot.

use crate::operations::channels::Sender;
use crate::operations::unary::group::{
    AggregationKind, AggregationSlot, Count, FoldAcc, Max, Min, Numeric, Sum,
};
use crate::operations::unary::pipeline_breaker::{Consumer, Outputter, PipelineBreaker};
use crate::operations::unary::{self, UnaryFactory};
use crate::worker::worker_waker;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int16Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;
use std::sync::mpsc;

/// One slot's running accumulator. `None` until the slot sees its first
/// contribution, so an aggregate over zero rows stays a SQL `NULL` (for
/// `SUM`/`MIN`/`MAX`) rather than a fabricated `0`/bound — `Count` renders `None`
/// as `0`. The variant is fixed by the slot kind: integer aggregates accumulate
/// in `A`, string extremes in an owned `String`.
#[derive(Clone)]
enum Partial<A> {
    Num(Option<A>),
    Str(Option<String>),
}

impl<A: Numeric> Partial<A> {
    /// The empty accumulator for a slot of `kind`.
    fn empty(kind: AggregationKind) -> Self {
        if kind.is_string_extreme() {
            Partial::Str(None)
        } else {
            Partial::Num(None)
        }
    }

    /// Fold a contribution — a batch's partial, or a sibling worker's — into this
    /// accumulator, take-first: the first value seeds, the rest combine by the
    /// slot's op (additive for `COUNT`/`SUM`, the extreme for `MIN`/`MAX`). The
    /// variant always matches `kind`, both built from the same slot.
    fn fold(&mut self, kind: AggregationKind, other: Partial<A>) {
        match (self, other) {
            (Partial::Num(acc), Partial::Num(c)) => {
                *acc = fold_first(*acc, c, |a, b| merge_num(kind, a, b));
            }
            (Partial::Str(acc), Partial::Str(c)) => {
                let is_max = matches!(kind, AggregationKind::StrMax);
                // Plain `Ord` extreme, not the arena-keyed StrMin/StrMax::merge
                // (see `reduce_str_column`).
                *acc = fold_first(
                    acc.take(),
                    c,
                    |a, b| if is_max { a.max(b) } else { a.min(b) },
                );
            }
            _ => unreachable!("partial variant must match the slot kind"),
        }
    }

    /// Build the single output column for this slot from its accumulator. A
    /// numeric slot renders its width's Arrow type; a string extreme a
    /// zero-or-one-row `Utf8View`. `None` is `NULL` for everything but `COUNT`,
    /// which is `0`.
    fn into_column(self, kind: AggregationKind) -> (Field, ArrayRef) {
        match self {
            Partial::Num(total) => num_column::<A>(kind, total),
            Partial::Str(total) => {
                let name = if matches!(kind, AggregationKind::StrMax) {
                    "max"
                } else {
                    "min"
                };
                let arr: ArrayRef = Arc::new(StringViewArray::from_iter(std::iter::once(
                    total.as_deref(),
                )));
                (Field::new(name, DataType::Utf8View, true), arr)
            }
        }
    }
}

/// Take-first fold over two optional partials: the first present value seeds, a
/// later one combines, and an absent one is dropped. Used at the batch and
/// cross-worker levels (once per slot, not per row), so the `Option` match here is
/// cold — the hot per-row reductions ([`reduce_int_column`]/[`reduce_str_column`])
/// seed from their first element and carry no such check.
#[inline(always)]
fn fold_first<T>(acc: Option<T>, c: Option<T>, combine: impl FnOnce(T, T) -> T) -> Option<T> {
    match (acc, c) {
        (Some(a), Some(b)) => Some(combine(a, b)),
        (Some(a), None) => Some(a),
        (None, c) => c,
    }
}

/// Combine two numeric partials by the slot's op `merge` — the single definition
/// of the additive / extreme fold, shared with the grouped path. The integer ops'
/// `merge` is width-independent (it folds two already-materialised `A`s).
#[inline(always)]
fn merge_num<A: Numeric>(kind: AggregationKind, a: A, b: A) -> A {
    match kind {
        AggregationKind::CountStar | AggregationKind::Count => Count::<A>::merge(a, b, &()),
        AggregationKind::Sum => Sum::<A>::merge(a, b, &()),
        AggregationKind::Min => Min::<A>::merge(a, b, &()),
        AggregationKind::Max => Max::<A>::merge(a, b, &()),
        AggregationKind::StrMin | AggregationKind::StrMax => {
            unreachable!("string extreme accumulates as Partial::Str, not numeric")
        }
    }
}

/// Reduce one batch's integer column to a partial of `kind` (`SUM`/`MIN`/`MAX`),
/// or `None` if the column is empty. The width is hoisted out of the row loop, and
/// the loop seeds from the first element then folds the rest — so there is no
/// per-row take-first (`Option`) check; only `merge_num`'s kind match, which is
/// loop-invariant and lifts out.
fn reduce_int_column<A: Numeric>(kind: AggregationKind, arr: &dyn Array) -> Option<A> {
    macro_rules! reduce_primitive {
        ($ty:ty) => {{
            let a = arr.as_primitive::<$ty>();
            if a.is_empty() {
                return None;
            }
            let mut acc = A::from(unsafe { a.value_unchecked(0) } as i64);
            for i in 1..a.len() {
                let v = A::from(unsafe { a.value_unchecked(i) } as i64);
                acc = merge_num(kind, acc, v);
            }
            Some(acc)
        }};
    }

    match arr.data_type() {
        DataType::Int16 => reduce_primitive!(Int16Type),
        DataType::Int32 => reduce_primitive!(Int32Type),
        DataType::Int64 => reduce_primitive!(Int64Type),
        other => panic!("aggregate: unsupported column type {other:?}"),
    }
}

/// Reduce one batch's `Utf8View` column to its extreme (`is_max` picks `MAX` vs
/// `MIN`), or `None` if empty. Seeds from the first row and keeps a *borrowed*
/// `&str` winner through the loop, allocating the owned `String` once at the end —
/// no per-row take-first check and no per-row allocation.
fn reduce_str_column(is_max: bool, arr: &dyn Array) -> Option<String> {
    let a = arr.as_string_view();
    if a.is_empty() {
        return None;
    }
    // Keep a borrowed winner through the loop, allocating once at the end. The
    // grouped `StrMin`/`StrMax::merge` can't be reused — it folds arena keys
    // resolved through the value arena, which the global path has no arena for —
    // so the extreme is a plain `Ord` compare on the `&str` itself.
    let mut acc: &str = unsafe { a.value_unchecked(0) };
    for i in 1..a.len() {
        let v = unsafe { a.value_unchecked(i) };
        acc = if is_max { acc.max(v) } else { acc.min(v) };
    }
    Some(acc.to_string())
}

/// Build the single output column for one numeric aggregate slot. `None` means
/// zero rows were aggregated: a SQL `NULL` for `SUM`/`MIN`/`MAX`, `0` for `COUNT`.
fn num_column<A: Numeric>(kind: AggregationKind, total: Option<A>) -> (Field, ArrayRef) {
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
        AggregationKind::StrMin | AggregationKind::StrMax => {
            unreachable!("string extreme renders through Partial::Str")
        }
    }
}

/// Factory for the aggregate operator. All workers share one mpsc channel; the
/// first factory gets the receiver, the rest get `None`, so per-worker partials
/// flow to a single collector (the same wiring as [`OrderByLimitFactory`]).
///
/// [`OrderByLimitFactory`]: super::order_by_limit
pub struct AggregateFactory<A: Numeric> {
    slots: Arc<Vec<AggregationSlot>>,
    sender: mpsc::Sender<Vec<Partial<A>>>,
    receiver: Option<mpsc::Receiver<Vec<Partial<A>>>>,
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
    local: Vec<Partial<A>>,
    sender: mpsc::Sender<Vec<Partial<A>>>,
    receiver: Option<mpsc::Receiver<Vec<Partial<A>>>>,
}

impl<A: Numeric> Aggregate<A> {
    fn new(
        slots: Arc<Vec<AggregationSlot>>,
        sender: mpsc::Sender<Vec<Partial<A>>>,
        receiver: Option<mpsc::Receiver<Vec<Partial<A>>>>,
    ) -> Self {
        // Empty (take-first) per slot, so an aggregate over zero rows is `NULL`
        // (or `0` for COUNT) rather than a fabricated seed.
        let local = slots.iter().map(|s| Partial::empty(s.kind)).collect();
        Aggregate {
            slots,
            local,
            sender,
            receiver,
        }
    }

    /// This batch's contribution for one slot (`None` if it contributes nothing —
    /// an empty column).
    fn contribution(slot: &AggregationSlot, batch: &RecordBatch) -> Partial<A> {
        match slot.kind {
            // COUNT(*) counts every row and never reads a column.
            AggregationKind::CountStar => Partial::Num(Some(A::from(batch.num_rows() as i64))),
            // COUNT(c) needs only the non-null count, not the values — read it
            // straight off the null bitmap instead of scanning the column.
            AggregationKind::Count => {
                let col = batch.column(slot.column);
                Partial::Num(Some(A::from((col.len() - col.null_count()) as i64)))
            }
            AggregationKind::Sum | AggregationKind::Min | AggregationKind::Max => {
                Partial::Num(reduce_int_column::<A>(slot.kind, batch.column(slot.column)))
            }
            AggregationKind::StrMin | AggregationKind::StrMax => {
                let is_max = matches!(slot.kind, AggregationKind::StrMax);
                Partial::Str(reduce_str_column(is_max, batch.column(slot.column)))
            }
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
        for (local, slot) in self.local.iter_mut().zip(self.slots.iter()) {
            local.fold(slot.kind, Self::contribution(slot, &batch));
        }
        Ok(())
    }

    fn into_outputter(self) -> unary::Result<Option<Self::Outputter>> {
        // Every worker sends its partials (even all-empty, so the merge always
        // sees a contribution per worker) and wakes the collector. Notifying
        // unconditionally after the send avoids the lost-wakeup that bites when
        // a finishing worker drops its sender while the collector is parked.
        self.sender
            .send(self.local)
            .expect("aggregate collector dropped");
        worker_waker().notify();

        let totals = self.slots.iter().map(|s| Partial::empty(s.kind)).collect();
        Ok(self.receiver.map(|rx| AggregateOutputter {
            rx,
            slots: self.slots,
            totals,
        }))
    }
}

/// Output phase (one worker only): drains every sibling's partials from the
/// channel, folds them, then emits the single-row result.
pub struct AggregateOutputter<A: Numeric> {
    rx: mpsc::Receiver<Vec<Partial<A>>>,
    slots: Arc<Vec<AggregationSlot>>,
    totals: Vec<Partial<A>>,
}

impl<A: Numeric> Outputter<RecordBatch> for AggregateOutputter<A> {
    fn output<OP: Sender<RecordBatch>>(&mut self, output: &mut OP) -> unary::Result<bool> {
        loop {
            match self.rx.try_recv() {
                Ok(partial) => {
                    // Fold each worker's partial with the slot's take-first combine.
                    for (i, p) in partial.into_iter().enumerate() {
                        let kind = self.slots[i].kind;
                        self.totals[i].fold(kind, p);
                    }
                }
                // Some siblings haven't finished yet; resume when re-driven.
                Err(mpsc::TryRecvError::Empty) => return Ok(false),
                // All senders dropped → every worker's partial is folded in.
                Err(mpsc::TryRecvError::Disconnected) => {
                    let (fields, columns): (Vec<Field>, Vec<ArrayRef>) =
                        std::mem::take(&mut self.totals)
                            .into_iter()
                            .zip(self.slots.iter())
                            .map(|(total, slot)| total.into_column(slot.kind))
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

    fn make_str_batch(values: &[&str]) -> RecordBatch {
        let array = StringViewArray::from(values.to_vec());
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "s",
                DataType::Utf8View,
                false,
            )])),
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
    fn col_str(batch: &RecordBatch, i: usize) -> String {
        batch.column(i).as_string_view().value(0).to_string()
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

    #[test]
    fn string_min_and_max_over_batches() {
        // String MIN/MAX fold the byte-lexicographic extreme across rows/batches.
        let ops = build::<i64>(
            1,
            vec![
                slot(AggregationKind::StrMin, 0),
                slot(AggregationKind::StrMax, 0),
            ],
        );
        let out = run_consumers(
            ops,
            vec![vec![
                make_str_batch(&["banana", "apple", "cherry"]),
                make_str_batch(&["date", "avocado"]),
            ]],
        );
        assert_eq!(out.items.len(), 1);
        assert_eq!(col_str(&out.items[0], 0), "apple"); // min
        assert_eq!(col_str(&out.items[0], 1), "date"); // max
    }

    #[test]
    fn string_extreme_merges_across_workers() {
        let ops = build::<i64>(
            3,
            vec![
                slot(AggregationKind::StrMin, 0),
                slot(AggregationKind::StrMax, 0),
            ],
        );
        let out = run_consumers(
            ops,
            vec![
                vec![make_str_batch(&["mango"])],
                vec![make_str_batch(&["apple", "zebra"])],
                vec![make_str_batch(&["kiwi"])],
            ],
        );
        assert_eq!(out.items.len(), 1);
        assert_eq!(col_str(&out.items[0], 0), "apple"); // min
        assert_eq!(col_str(&out.items[0], 1), "zebra"); // max
    }

    #[test]
    fn string_extreme_mixed_with_numeric() {
        // A string MIN beside an integer SUM — the heterogeneous mix the
        // per-slot `Partial` unlocks.
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8View, false),
            Field::new("n", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringViewArray::from(vec!["pear", "apple", "fig"])),
                Arc::new(Int32Array::from(vec![1, 2, 3])),
            ],
        )
        .unwrap();
        let ops = build::<i64>(
            1,
            vec![
                slot(AggregationKind::StrMin, 0),
                slot(AggregationKind::Sum, 1),
            ],
        );
        let out = run_consumers(ops, vec![vec![batch]]);
        assert_eq!(col_str(&out.items[0], 0), "apple");
        assert_eq!(col_i64(&out.items[0], 1), 6);
    }
}
