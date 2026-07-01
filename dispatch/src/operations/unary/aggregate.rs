//! Global aggregate operator (no GROUP BY) computing one or more column
//! aggregates in a single pass.
//!
//! A pipeline breaker (see [`pipeline_breaker`](super::pipeline_breaker)): each
//! worker keeps a local [`Slot`] accumulator per requested aggregate while
//! consuming batches, then on finalization sends its slots down a shared mpsc
//! channel. One worker holds the receiver, drains every sibling's slots, merges
//! them, and emits the single-row result with one output column per aggregate.
//! The cross-worker merge therefore needs no shared lock - it mirrors
//! [`OrderByLimit`](super::order_by_limit), which combines worker partials the
//! same way.
//!
//! A [`Slot`] owns one aggregate's whole lifecycle - fold batches in, merge a
//! sibling worker's accumulator, render the output column - so each op's logic
//! reads in one place. The integer ops reuse the grouped path's [`Fold`] impls
//! ([`Sum`]/[`Min`]/[`Max`]); a float column accumulates in `f64`; the string
//! extremes keep an owned `String` winner instead of the grouped path's arena key,
//! since a global extreme is a single value per slot. A `SUM`/`MIN`/`MAX`'s value
//! family (integer / float / string) is picked from the slot's declared
//! `output_type`, not the `MIN`/`MAX` kind. There is no `Avg`: DuckDB lowers
//! `AVG(x)` to `sum(x) / count(x)`, so an average arrives as a `Sum` slot and a
//! `Count` slot.

use crate::operations::channels::Sender;
use crate::operations::unary::group::{
    AggregationKind, AggregationSlot, Fold, Max, Min, Numeric, Sum, cast_value_column,
};
use crate::operations::unary::pipeline_breaker::{Consumer, Outputter, PipelineBreaker};
use crate::operations::unary::{self, UnaryFactory};
use crate::worker::worker_waker;
use arrow_array::cast::AsArray;
use arrow_array::types::{
    ArrowPrimitiveType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    UInt8Type, UInt16Type, UInt32Type,
};
use arrow_array::{
    Array, ArrayRef, Float64Array, Int64Array, PrimitiveArray, RecordBatch, StringViewArray,
};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;
use std::sync::mpsc;

/// A numeric aggregate op over an integer column, dispatched once per batch
/// (never per row) into the corresponding monomorphic [`Fold`].
#[derive(Clone, Copy)]
enum NumOp {
    Sum,
    Min,
    Max,
}

impl NumOp {
    /// Combine two accumulators by this op - additive for `Sum`, the extreme
    /// for `Min`/`Max`. Delegates to the grouped path's [`Fold::merge`].
    fn merge<A: Numeric>(self, a: A, b: A) -> A {
        match self {
            NumOp::Sum => Sum::<A>::merge(a, b),
            NumOp::Min => Min::<A>::merge(a, b),
            NumOp::Max => Max::<A>::merge(a, b),
        }
    }

    /// The output column name for this op.
    fn column_name(self) -> &'static str {
        match self {
            NumOp::Sum => "sum",
            NumOp::Min => "min",
            NumOp::Max => "max",
        }
    }

    /// Reduce one batch's integer column by this op, or `None` if the column is
    /// empty. Dispatches the column width here, once per batch, into a
    /// monomorphic loop.
    fn reduce_column<A: Numeric>(self, arr: &dyn Array) -> Option<A> {
        match arr.data_type() {
            DataType::Int8 => self.reduce_primitive::<A, Int8Type>(arr.as_primitive()),
            DataType::Int16 => self.reduce_primitive::<A, Int16Type>(arr.as_primitive()),
            DataType::Int32 => self.reduce_primitive::<A, Int32Type>(arr.as_primitive()),
            DataType::Int64 => self.reduce_primitive::<A, Int64Type>(arr.as_primitive()),
            DataType::UInt8 => self.reduce_primitive::<A, UInt8Type>(arr.as_primitive()),
            DataType::UInt16 => self.reduce_primitive::<A, UInt16Type>(arr.as_primitive()),
            DataType::UInt32 => self.reduce_primitive::<A, UInt32Type>(arr.as_primitive()),
            other => panic!("aggregate: unsupported column type {other:?}"),
        }
    }

    /// Dispatch this op into its [`Fold`], hoisting the op match out of the row
    /// loop: each `(width, op)` pair below is a separate monomorphic
    /// [`fold_primitive_column`] LLVM vectorises (a per-row op match leaves a
    /// loop-carried branch the autovectoriser will not lift - ~2x slower on a
    /// full-column SUM).
    fn reduce_primitive<A: Numeric, T: ArrowPrimitiveType>(
        self,
        arr: &PrimitiveArray<T>,
    ) -> Option<A>
    where
        T::Native: Into<i64>,
    {
        assert_eq!(
            arr.null_count(),
            0,
            "aggregate input must not contain NULLs"
        );
        match self {
            NumOp::Sum => fold_primitive_column::<A, Sum<A>, T>(arr),
            NumOp::Min => fold_primitive_column::<A, Min<A>, T>(arr),
            NumOp::Max => fold_primitive_column::<A, Max<A>, T>(arr),
        }
    }

    /// Combine two `f64` accumulators by this op. `MIN`/`MAX` order with
    /// [`f64::total_cmp`] so the extreme is deterministic regardless of fold order
    /// (NaN sorts greatest, matching DuckDB).
    fn merge_float(self, a: f64, b: f64) -> f64 {
        match self {
            NumOp::Sum => a + b,
            NumOp::Min => {
                if b.total_cmp(&a).is_lt() {
                    b
                } else {
                    a
                }
            }
            NumOp::Max => {
                if b.total_cmp(&a).is_gt() {
                    b
                } else {
                    a
                }
            }
        }
    }

    /// Reduce one batch's float column by this op, or `None` if the column is
    /// empty. The float counterpart of [`reduce_column`](NumOp::reduce_column);
    /// `REAL`/`DOUBLE` both accumulate in `f64`.
    fn reduce_float_column(self, arr: &dyn Array) -> Option<f64> {
        match arr.data_type() {
            DataType::Float32 => self.reduce_float_primitive::<Float32Type>(arr.as_primitive()),
            DataType::Float64 => self.reduce_float_primitive::<Float64Type>(arr.as_primitive()),
            other => panic!("float aggregate: unsupported column type {other:?}"),
        }
    }

    /// The float row loop: fold every value of a float column into an `f64` by this
    /// op, or `None` if the column is empty.
    fn reduce_float_primitive<T: ArrowPrimitiveType>(self, arr: &PrimitiveArray<T>) -> Option<f64>
    where
        T::Native: Into<f64>,
    {
        assert_eq!(
            arr.null_count(),
            0,
            "aggregate input must not contain NULLs"
        );
        let mut values = arr.values().iter().map(|&v| v.into());
        let first = values.next()?;
        Some(values.fold(first, |a, b| self.merge_float(a, b)))
    }
}

/// The hot numeric loop: fold every value of an integer column into an
/// accumulator by the op `F`, or `None` if the column is empty. Monomorphic per
/// `(width, op)`, iterating the raw native slice.
fn fold_primitive_column<A: Numeric, F: Fold<Val = i64, Acc = A>, T: ArrowPrimitiveType>(
    arr: &PrimitiveArray<T>,
) -> Option<A>
where
    T::Native: Into<i64>,
{
    let mut values = arr.values().iter().map(|&v| v.into());
    let first = values.next()?;
    Some(values.fold(F::seed(first), F::update))
}

/// One aggregate's running accumulator and its whole lifecycle: fold batches in
/// ([`consume`](Self::consume)), combine a sibling worker's accumulator
/// ([`merge`](Self::merge)), render the output column
/// ([`into_column`](Self::into_column)). The op is the variant, so it is matched
/// in one place per lifecycle step and can never disagree with the accumulator's
/// shape.
///
/// The numeric and string accumulators start `None` so an aggregate over zero
/// rows stays a SQL `NULL` rather than a fabricated `0`/bound; a count over zero
/// rows is `0`, so it needs no such distinction.
enum Slot<A: Numeric> {
    /// `COUNT(*)` / `COUNT(col)` - with no NULLs, both are the row count.
    Count { count: i64 },
    /// `SUM`/`MIN`/`MAX` over an integer column, accumulating in the width `A`
    /// the planner picks (`i128` only when a sum reads a 64-bit column).
    Num {
        op: NumOp,
        column: usize,
        acc: Option<A>,
    },
    /// `SUM`/`MIN`/`MAX` over a `REAL`/`DOUBLE` column, accumulating in `f64`
    /// (independent of the integer width `A`). The output narrows to `Float32` for
    /// a `REAL` slot via the slot's declared `output_type` cast.
    Float {
        op: NumOp,
        column: usize,
        acc: Option<f64>,
    },
    /// `MIN`/`MAX` over a `Utf8View` column.
    Str {
        is_max: bool,
        column: usize,
        acc: Option<String>,
    },
}

impl<A: Numeric> Slot<A> {
    /// The empty accumulator for `spec`. A `SUM`/`MIN`/`MAX`'s value family is
    /// decided by the column's declared `output_type`, not the kind: a `Utf8View`
    /// extreme keeps an owned `String`, a floating column an `f64`, else the integer
    /// width `A`.
    fn build(spec: &AggregationSlot) -> Self {
        let column = spec.column;
        let op = match spec.kind {
            AggregationKind::CountStar | AggregationKind::Count => return Slot::Count { count: 0 },
            AggregationKind::Sum => NumOp::Sum,
            AggregationKind::Min => NumOp::Min,
            AggregationKind::Max => NumOp::Max,
        };
        if spec.is_string_extreme() {
            Slot::Str {
                is_max: matches!(op, NumOp::Max),
                column,
                acc: None,
            }
        } else if spec.output_type.is_floating() {
            Slot::Float {
                op,
                column,
                acc: None,
            }
        } else {
            Slot::Num {
                op,
                column,
                acc: None,
            }
        }
    }

    /// Fold one batch's contribution into this accumulator.
    fn consume(&mut self, batch: &RecordBatch) {
        match self {
            Slot::Count { count } => *count += batch.num_rows() as i64,
            Slot::Num { op, column, acc } => {
                let op = *op;
                let reduced = op.reduce_column::<A>(batch.column(*column).as_ref());
                fold_into(acc, reduced, |a, b| op.merge(a, b));
            }
            Slot::Float { op, column, acc } => {
                let op = *op;
                let reduced = op.reduce_float_column(batch.column(*column).as_ref());
                fold_into(acc, reduced, |a, b| op.merge_float(a, b));
            }
            Slot::Str {
                is_max,
                column,
                acc,
            } => {
                let reduced = reduce_str_column(*is_max, batch.column(*column).as_ref());
                fold_into(acc, reduced, str_extreme_fn(*is_max));
            }
        }
    }

    /// Fold a sibling worker's finished accumulator into this one. Every worker
    /// builds its slots from the same specs, so the variants always match.
    fn merge(&mut self, other: Slot<A>) {
        match (self, other) {
            (Slot::Count { count }, Slot::Count { count: other }) => *count += other,
            (Slot::Num { op, acc, .. }, Slot::Num { acc: other, .. }) => {
                let op = *op;
                fold_into(acc, other, |a, b| op.merge(a, b));
            }
            (Slot::Float { op, acc, .. }, Slot::Float { acc: other, .. }) => {
                let op = *op;
                fold_into(acc, other, |a, b| op.merge_float(a, b));
            }
            (Slot::Str { is_max, acc, .. }, Slot::Str { acc: other, .. }) => {
                fold_into(acc, other, str_extreme_fn(*is_max));
            }
            _ => unreachable!("every worker builds its slots from the same specs"),
        }
    }

    /// Build the single-row output column from the finished accumulator. `None`
    /// (zero rows aggregated) renders as SQL `NULL`; a count renders its plain
    /// total (`0` over zero rows, never NULL).
    fn into_column(self) -> (Field, ArrayRef) {
        match self {
            Slot::Count { count } => (
                Field::new("count", DataType::Int64, false),
                Arc::new(Int64Array::from(vec![count])),
            ),
            Slot::Num { op, acc, .. } => (
                Field::new(op.column_name(), A::data_type(), true),
                A::scalar_array(acc),
            ),
            // Renders `Float64`; the outputter casts to the slot's declared
            // `output_type` (narrowing to `Float32` for a `REAL` slot).
            Slot::Float { op, acc, .. } => (
                Field::new(op.column_name(), DataType::Float64, true),
                Arc::new(Float64Array::from(vec![acc])),
            ),
            Slot::Str { is_max, acc, .. } => (
                Field::new(if is_max { "max" } else { "min" }, DataType::Utf8View, true),
                Arc::new(StringViewArray::from_iter(std::iter::once(acc.as_deref()))),
            ),
        }
    }
}

/// Fold an optional contribution (a batch's reduction, or a sibling worker's
/// accumulator) into `acc`: the first present value seeds, later ones combine,
/// an absent one (an empty batch) is dropped. Called once per slot per
/// batch/worker, so the `Option` check is cold - the hot per-row reductions seed
/// from their first element and carry no such check.
fn fold_into<T>(acc: &mut Option<T>, contribution: Option<T>, combine: impl FnOnce(T, T) -> T) {
    if let Some(v) = contribution {
        *acc = Some(match acc.take() {
            Some(a) => combine(a, v),
            None => v,
        });
    }
}

/// The string extreme combine: a plain `Ord` compare on the strings themselves.
/// The grouped `StrMin`/`StrMax::merge` folds arena keys resolved through the
/// value arena, which the global path has no arena for. `MAX` is const so the
/// per-row loop carries no branch.
fn str_extreme<T: Ord, const MAX: bool>(a: T, b: T) -> T {
    if MAX { a.max(b) } else { a.min(b) }
}

/// Select the extreme combine for a runtime `is_max` - for the cold fold sites
/// (once per batch / per sibling worker), where the branch costs nothing.
fn str_extreme_fn<T: Ord>(is_max: bool) -> fn(T, T) -> T {
    if is_max {
        str_extreme::<T, true>
    } else {
        str_extreme::<T, false>
    }
}

/// Reduce one batch's `Utf8View` column to its extreme (`is_max` picks `MAX` vs
/// `MIN`), or `None` if the column is empty. Keeps a *borrowed* `&str` winner
/// through the loop, allocating the owned `String` once at the end - no per-row
/// allocation.
fn reduce_str_column(is_max: bool, arr: &dyn Array) -> Option<String> {
    let a = arr.as_string_view();
    assert_eq!(a.null_count(), 0, "aggregate input must not contain NULLs");
    if is_max {
        reduce_str_extreme::<true>(a)
    } else {
        reduce_str_extreme::<false>(a)
    }
}

/// The hot string loop, monomorphic per extreme.
fn reduce_str_extreme<const MAX: bool>(a: &StringViewArray) -> Option<String> {
    if a.is_empty() {
        return None;
    }
    let mut acc: &str = unsafe { a.value_unchecked(0) };
    for i in 1..a.len() {
        let v = unsafe { a.value_unchecked(i) };
        acc = str_extreme::<_, MAX>(acc, v);
    }
    Some(acc.to_string())
}

/// Factory for the aggregate operator. All workers share one mpsc channel; the
/// first factory gets the receiver, the rest get `None`, so per-worker slots
/// flow to a single collector (the same wiring as [`OrderByLimitFactory`]).
///
/// [`OrderByLimitFactory`]: super::order_by_limit
pub struct AggregateFactory<A: Numeric> {
    specs: Arc<Vec<AggregationSlot>>,
    sender: mpsc::Sender<Vec<Slot<A>>>,
    receiver: Option<mpsc::Receiver<Vec<Slot<A>>>>,
}

impl<A: Numeric> AggregateFactory<A> {
    /// Create one factory per worker, all sharing the same slots channel.
    pub fn create_for_workers(
        specs: Vec<AggregationSlot>,
        worker_count: usize,
    ) -> impl IntoIterator<Item = AggregateFactory<A>> {
        let specs = Arc::new(specs);
        let (tx, rx) = mpsc::channel();
        let mut rx_opt = Some(rx);

        (0..worker_count).map(move |_| AggregateFactory {
            specs: specs.clone(),
            sender: tx.clone(),
            receiver: rx_opt.take(),
        })
    }
}

impl<A: Numeric> UnaryFactory<RecordBatch, RecordBatch> for AggregateFactory<A> {
    type Unary = PipelineBreaker<RecordBatch, RecordBatch, Aggregate<A>>;

    fn build_unary(mut self) -> Self::Unary {
        PipelineBreaker::Consuming(Aggregate::new(
            self.specs,
            self.sender,
            self.receiver.take(),
        ))
    }
}

/// Per-worker aggregate consumer. Accumulates local slots, then sends them down
/// the shared channel on finalization.
pub struct Aggregate<A: Numeric> {
    specs: Arc<Vec<AggregationSlot>>,
    local: Vec<Slot<A>>,
    sender: mpsc::Sender<Vec<Slot<A>>>,
    receiver: Option<mpsc::Receiver<Vec<Slot<A>>>>,
}

impl<A: Numeric> Aggregate<A> {
    fn new(
        specs: Arc<Vec<AggregationSlot>>,
        sender: mpsc::Sender<Vec<Slot<A>>>,
        receiver: Option<mpsc::Receiver<Vec<Slot<A>>>>,
    ) -> Self {
        let local = specs.iter().map(Slot::build).collect();
        Aggregate {
            specs,
            local,
            sender,
            receiver,
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
        for slot in &mut self.local {
            slot.consume(&batch);
        }
        Ok(())
    }

    fn into_outputter(self) -> unary::Result<Option<Self::Outputter>> {
        // Every worker sends its slots (even untouched, so the merge always sees
        // a contribution per worker) and wakes the collector. Notifying
        // unconditionally after the send avoids the lost-wakeup that bites when
        // a finishing worker drops its sender while the collector is parked.
        self.sender
            .send(self.local)
            .expect("aggregate collector dropped");
        worker_waker().notify();

        let totals = self.specs.iter().map(Slot::build).collect();
        Ok(self.receiver.map(|rx| AggregateOutputter {
            rx,
            specs: self.specs,
            totals,
        }))
    }
}

/// Output phase (one worker only): drains every sibling's slots from the
/// channel, merges them, then emits the single-row result.
pub struct AggregateOutputter<A: Numeric> {
    rx: mpsc::Receiver<Vec<Slot<A>>>,
    specs: Arc<Vec<AggregationSlot>>,
    totals: Vec<Slot<A>>,
}

impl<A: Numeric> Outputter<RecordBatch> for AggregateOutputter<A> {
    fn output<OP: Sender<RecordBatch>>(&mut self, output: &mut OP) -> unary::Result<bool> {
        loop {
            match self.rx.try_recv() {
                Ok(worker_slots) => {
                    for (total, slot) in self.totals.iter_mut().zip(worker_slots) {
                        total.merge(slot);
                    }
                }
                // Some siblings haven't finished yet; resume when re-driven.
                Err(mpsc::TryRecvError::Empty) => return Ok(false),
                // All senders dropped → every worker's slots are merged in.
                Err(mpsc::TryRecvError::Disconnected) => {
                    let (fields, columns): (Vec<Field>, Vec<ArrayRef>) =
                        std::mem::take(&mut self.totals)
                            .into_iter()
                            .zip(self.specs.iter())
                            .map(|(slot, spec)| {
                                let (field, column) = slot.into_column();
                                cast_value_column(field, column, &spec.output_type)
                            })
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

    /// Build `n` channel-wired aggregate consumers sharing one slots channel
    /// (the first holds the receiver), mirroring the factory's wiring.
    fn build<A: Numeric>(n: usize, specs: Vec<AggregationSlot>) -> Vec<Aggregate<A>> {
        let specs = Arc::new(specs);
        let (tx, rx) = mpsc::channel();
        let mut rx_opt = Some(rx);
        (0..n)
            .map(move |_| Aggregate::new(specs.clone(), tx.clone(), rx_opt.take()))
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
        AggregationSlot::new(kind, column, test_output_type(kind))
    }

    /// A `MIN`/`MAX` over a string column, declared as `Utf8View` (which is how the
    /// operator picks the string extreme over the numeric one).
    fn str_slot(kind: AggregationKind, column: usize) -> AggregationSlot {
        AggregationSlot::new(kind, column, arrow_schema::DataType::Utf8View)
    }

    /// The output type a numeric test slot of `kind` declares: `Decimal128` for a
    /// sum, `Int64` otherwise (the accumulator's natural width for a count or numeric
    /// extreme over the int columns tests use).
    fn test_output_type(kind: AggregationKind) -> arrow_schema::DataType {
        match kind {
            AggregationKind::Sum => arrow_schema::DataType::Decimal128(38, 0),
            _ => arrow_schema::DataType::Int64,
        }
    }

    #[test]
    fn single_worker_sum() {
        // Int32 column → i64 accumulator → Decimal128 output (a SUM is a HUGEINT).
        let ops = build::<i64>(1, vec![slot(AggregationKind::Sum, 0)]);

        let out = run_consumers(ops, vec![vec![make_batch(&[1, 2, 3]), make_batch(&[4, 5])]]);

        assert_eq!(out.items.len(), 1);
        assert_eq!(col_i128(&out.items[0], 0), 15);
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
        assert_eq!(col_i128(&out.items[0], 0), 20); // sum (Decimal128)
        assert_eq!(col_i64(&out.items[0], 1), 4); // count (Int64)
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
        assert_eq!(col_i128(&out.items[0], 0), 36);
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
        // the global MIN/MAX is the extreme of all workers' extremes - including
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
    fn empty_input_sum_is_null() {
        let ops = build::<i64>(1, vec![slot(AggregationKind::Sum, 0)]);

        let out = run_consumers(ops, vec![vec![]]);

        assert_eq!(out.items.len(), 1);
        assert!(out.items[0].column(0).is_null(0)); // SUM of zero rows is NULL
    }

    #[test]
    fn string_min_and_max_over_batches() {
        // String MIN/MAX fold the byte-lexicographic extreme across rows/batches.
        let ops = build::<i64>(
            1,
            vec![
                str_slot(AggregationKind::Min, 0),
                str_slot(AggregationKind::Max, 0),
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
                str_slot(AggregationKind::Min, 0),
                str_slot(AggregationKind::Max, 0),
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
        // A string MIN beside an integer SUM - a heterogeneous slot mix.
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
                str_slot(AggregationKind::Min, 0),
                slot(AggregationKind::Sum, 1),
            ],
        );

        let out = run_consumers(ops, vec![vec![batch]]);

        assert_eq!(col_str(&out.items[0], 0), "apple");
        assert_eq!(col_i128(&out.items[0], 1), 6); // SUM is Decimal128
    }

    fn make_f64_batch(values: &[f64]) -> RecordBatch {
        let array = Float64Array::from(values.to_vec());
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)])),
            vec![Arc::new(array)],
        )
        .unwrap()
    }

    /// A float slot declares a floating `output_type`, which is how the operator
    /// accumulates it in `f64`.
    fn float_slot(kind: AggregationKind, column: usize) -> AggregationSlot {
        AggregationSlot::new(kind, column, DataType::Float64)
    }

    fn col_f64(batch: &RecordBatch, i: usize) -> f64 {
        batch.column(i).as_primitive::<Float64Type>().value(0)
    }

    #[test]
    fn float_sum_over_batches() {
        let ops = build::<i64>(1, vec![float_slot(AggregationKind::Sum, 0)]);

        let out = run_consumers(
            ops,
            vec![vec![make_f64_batch(&[1.5, 2.25]), make_f64_batch(&[0.25])]],
        );

        assert_eq!(out.items.len(), 1);
        assert_eq!(col_f64(&out.items[0], 0), 4.0);
    }

    #[test]
    fn float_min_max_merge_across_workers() {
        let ops = build::<i64>(
            3,
            vec![
                float_slot(AggregationKind::Min, 0),
                float_slot(AggregationKind::Max, 0),
            ],
        );

        let out = run_consumers(
            ops,
            vec![
                vec![make_f64_batch(&[10.5, 4.5])],
                vec![make_f64_batch(&[-3.5, 20.5])],
                vec![make_f64_batch(&[7.0])],
            ],
        );

        assert_eq!(col_f64(&out.items[0], 0), -3.5); // min
        assert_eq!(col_f64(&out.items[0], 1), 20.5); // max
    }

    #[test]
    fn empty_input_float_sum_is_null() {
        let ops = build::<i64>(1, vec![float_slot(AggregationKind::Sum, 0)]);

        let out = run_consumers(ops, vec![vec![]]);

        assert!(out.items[0].column(0).is_null(0));
    }
}
