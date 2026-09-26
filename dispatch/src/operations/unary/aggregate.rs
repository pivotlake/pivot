//! Global aggregate operator (no GROUP BY) computing one or more column
//! aggregates in a single pass.
//!
//! A pipeline breaker (see [`pipeline_breaker`](super::pipeline_breaker)): each
//! worker keeps a local [`Slot`] accumulator per requested aggregate while
//! consuming batches, then on finalization hands its slots to a gather barrier
//! channel. One worker holds the receiver, drains every sibling's slots, merges
//! them, and emits the single-row result with one output column per aggregate.
//! The cross-worker merge therefore needs no shared lock - it mirrors
//! [`OrderByLimit`](super::order_by_limit), which combines worker partials the
//! same way.
//!
//! A [`Slot`] owns one aggregate's whole lifecycle - fold batches in, merge a
//! sibling worker's accumulator, render the output column - so each op's logic
//! reads in one place. Both numeric families reuse the grouped path's ops: the
//! integer ops its [`Fold`] impls ([`Sum`]/[`Min`]/[`Max`]), the float ops its
//! [`F64Sum`]/[`F64Min`]/[`F64Max`] (an `f64` bit-punned into the same cell). The
//! string extremes keep an owned `String` winner instead of the grouped path's arena
//! key, since a global extreme is a single value per slot. A `SUM`/`MIN`/`MAX`'s value
//! family (integer / float / string) is picked from the slot's declared
//! `output_type`, not the `MIN`/`MAX` kind. There is no `Avg`: DuckDB lowers
//! `AVG(x)` to `sum(x) / count(x)`, so an average arrives as a `Sum` slot and a
//! `Count` slot.

use crate::cpu_features::multitarget_kernel;
use crate::gather_barrier::GatherBarrier;
use crate::operations::channels::Sender;
use crate::operations::unary::group::{
    AggregationKind, AggregationSlot, F64Cell, F64Max, F64Min, F64Sum, Fold, IntCell, Max, Min,
    Sum, U128Max, U128Min, U128Sum, WideCell, cast_value_column,
};
use crate::operations::unary::pipeline_breaker::{Consumer, Outputter, PipelineBreaker};
use crate::operations::unary::{self, UnaryFactory};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    ArrowPrimitiveType, Decimal64Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, UInt8Type, UInt16Type, UInt32Type,
};
use arrow_array::{
    Array, ArrayRef, Float64Array, Int64Array, PrimitiveArray, RecordBatch, StringViewArray,
};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

/// A numeric aggregate op (`SUM`/`MIN`/`MAX`) shared by the integer and float
/// slots, dispatched once per batch (never per row) into its monomorphic op.
#[derive(Clone, Copy)]
enum NumOp {
    Sum,
    Min,
    Max,
}

impl NumOp {
    /// The output column name for this op.
    fn column_name(self) -> &'static str {
        match self {
            NumOp::Sum => "sum",
            NumOp::Min => "min",
            NumOp::Max => "max",
        }
    }

    // --- integer columns: reduce a batch, then combine partials ---

    /// Reduce one batch's integer column by this op, or `None` if the column is
    /// empty. Dispatches the column width here, once per batch, into a
    /// monomorphic loop. A `Decimal64` column folds its raw unscaled `i64`s
    /// through the integer ops like any other 64-bit width. A `Decimal128`
    /// column (a wide decimal's raw unscaled values, or a re-read wide partial)
    /// folds its full `i128`s through the wide ops into the wide cell the
    /// planner picked.
    fn reduce_int_column<A: IntCell + WideCell>(self, arr: &dyn Array) -> Option<A> {
        match arr.data_type() {
            DataType::Int8 => self.reduce_int_primitive::<A, Int8Type>(arr.as_primitive()),
            DataType::Int16 => self.reduce_int_primitive::<A, Int16Type>(arr.as_primitive()),
            DataType::Int32 => self.reduce_int_primitive::<A, Int32Type>(arr.as_primitive()),
            DataType::Int64 => self.reduce_int_primitive::<A, Int64Type>(arr.as_primitive()),
            DataType::UInt8 => self.reduce_int_primitive::<A, UInt8Type>(arr.as_primitive()),
            DataType::UInt16 => self.reduce_int_primitive::<A, UInt16Type>(arr.as_primitive()),
            DataType::UInt32 => self.reduce_int_primitive::<A, UInt32Type>(arr.as_primitive()),
            DataType::Decimal64(_, _) => {
                self.reduce_int_primitive::<A, Decimal64Type>(arr.as_primitive())
            }
            DataType::Decimal128(_, _) => self.reduce_wide_primitive::<A>(arr.as_primitive()),
            other => panic!("aggregate: unsupported column type {other:?}"),
        }
    }

    /// Dispatch this op into its [`Fold`]'s `seed`/`update`, hoisting the op match out
    /// of the row loop: each `(width, op)` pair below is a separate monomorphic
    /// [`fold_primitive_column`] LLVM vectorises (a per-row op match leaves a
    /// loop-carried branch the autovectoriser will not lift - ~2x slower on a
    /// full-column SUM).
    fn reduce_int_primitive<A: IntCell, T: ArrowPrimitiveType>(
        self,
        arr: &PrimitiveArray<T>,
    ) -> Option<A>
    where
        T::Native: Into<i64>,
    {
        match self {
            NumOp::Sum => fold_primitive_column(arr, Sum::<A>::seed, Sum::<A>::update),
            NumOp::Min => fold_primitive_column(arr, Min::<A>::seed, Min::<A>::update),
            NumOp::Max => fold_primitive_column(arr, Max::<A>::seed, Max::<A>::update),
        }
    }

    /// Fold a `Decimal128` column's raw `i128` values through the wide
    /// [`U128Sum`]/[`U128Min`]/[`U128Max`] ops into the wide cell, hoisting the op
    /// match out of the row loop exactly as
    /// [`reduce_int_primitive`](NumOp::reduce_int_primitive) does. The values stay
    /// unscaled integers throughout; the cell holds the full `i128`, so a total
    /// past `i64` stays exact. Reaching this with a narrow `A` is a planning bug
    /// ([`WideCell`]'s `i64` arms panic): the planner widens every signature that
    /// aggregates a decimal.
    fn reduce_wide_primitive<A: WideCell>(
        self,
        arr: &PrimitiveArray<arrow_array::types::Decimal128Type>,
    ) -> Option<A> {
        match self {
            NumOp::Sum => fold_primitive_column(arr, U128Sum::<A>::seed, U128Sum::<A>::update),
            NumOp::Min => fold_primitive_column(arr, U128Min::<A>::seed, U128Min::<A>::update),
            NumOp::Max => fold_primitive_column(arr, U128Max::<A>::seed, U128Max::<A>::update),
        }
    }

    /// Combine two accumulators by this op - additive for `Sum`, the extreme
    /// for `Min`/`Max`. Delegates to the grouped path's [`Fold::merge`].
    fn merge<A: IntCell>(self, a: A, b: A) -> A {
        match self {
            NumOp::Sum => Sum::<A>::merge(a, b),
            NumOp::Min => Min::<A>::merge(a, b),
            NumOp::Max => Max::<A>::merge(a, b),
        }
    }

    // --- float columns: the same three steps, over an `f64` punned into the cell ---

    /// Reduce one batch's float column by this op into a punned `f64` cell, or `None`
    /// if the column is empty. The float counterpart of
    /// [`reduce_int_column`](NumOp::reduce_int_column); dispatches the column width
    /// here, once per batch, into a monomorphic loop.
    fn reduce_float_column<A: F64Cell>(self, arr: &dyn Array) -> Option<A> {
        match arr.data_type() {
            DataType::Float32 => self.reduce_float_primitive::<A, Float32Type>(arr.as_primitive()),
            DataType::Float64 => self.reduce_float_primitive::<A, Float64Type>(arr.as_primitive()),
            other => panic!("float aggregate: unsupported column type {other:?}"),
        }
    }

    /// Dispatch this op into its [`F64Sum`]/[`F64Min`]/[`F64Max`] `seed`/`update` out
    /// of the row loop, hoisting the op match exactly as
    /// [`reduce_int_primitive`](NumOp::reduce_int_primitive) does - so each
    /// `(width, op)` is a separate monomorphic [`fold_primitive_column`] with no
    /// per-row op branch.
    fn reduce_float_primitive<A: F64Cell, T: ArrowPrimitiveType>(
        self,
        arr: &PrimitiveArray<T>,
    ) -> Option<A>
    where
        T::Native: Into<f64>,
    {
        match self {
            NumOp::Sum => fold_primitive_column(arr, F64Sum::<A>::seed, F64Sum::<A>::update),
            NumOp::Min => fold_primitive_column(arr, F64Min::<A>::seed, F64Min::<A>::update),
            NumOp::Max => fold_primitive_column(arr, F64Max::<A>::seed, F64Max::<A>::update),
        }
    }

    /// Combine two float accumulators by this op, reusing the grouped path's
    /// [`F64Sum`]/[`F64Min`]/[`F64Max`] `merge` - the single source of the float
    /// `SUM`/`MIN`/`MAX` rule, exactly as [`merge`](NumOp::merge) reuses the integer
    /// ops. The accumulator is an `f64` bit-punned into the cell `A` (via [`F64Cell`]),
    /// so this mirrors the integer merge with no separate combine logic.
    fn merge_float<A: F64Cell>(self, a: A, b: A) -> A {
        match self {
            NumOp::Sum => F64Sum::<A>::merge(a, b),
            NumOp::Min => F64Min::<A>::merge(a, b),
            NumOp::Max => F64Max::<A>::merge(a, b),
        }
    }
}

multitarget_kernel! {
    /// Hot primitive reduction loop, runtime-dispatched to the baseline or Ice
    /// Lake clone. Function-item callbacks make each width and operation
    /// monomorphic, with no indirect call or per-row branch. Integer folds can
    /// vectorize; floating folds remain scalar without fast-math. Returns `None`
    /// when no values remain.
    fn fold_primitive_column[A, V, T](
        arr: &PrimitiveArray<T>,
        seed: impl Fn(V) -> A,
        update: impl Fn(A, V) -> A,
    ) -> Option<A>
    where [T: ArrowPrimitiveType, T::Native: Into<V>]
    {
        if arr.null_count() == 0 {
            let mut values = arr.values().iter().map(|&v| v.into());
            let first = values.next()?;
            Some(values.fold(seed(first), update))
        } else {
            // NULLs contribute nothing to a SUM/MIN/MAX; a column with only NULLs
            // reduces to `None`, exactly like an empty column.
            let mut values = arr.iter().flatten().map(|v| v.into());
            let first = values.next()?;
            Some(values.fold(seed(first), update))
        }
    }
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
enum Slot<A: IntCell> {
    /// `COUNT(*)` (`column: None`, all rows) or `COUNT(col)` (`column: Some`,
    /// only the rows where `col` is non-NULL).
    Count { column: Option<usize>, count: i64 },
    /// `SUM`/`MIN`/`MAX` over an integer column, accumulating in the width `A`
    /// the planner picks (`i128` only when a sum reads a 64-bit column).
    Int {
        op: NumOp,
        column: usize,
        acc: Option<A>,
    },
    /// `SUM`/`MIN`/`MAX` over a `REAL`/`DOUBLE` column, accumulating an `f64`
    /// bit-punned into the same cell `A` the integer path uses (via [`F64Cell`]),
    /// exactly as a string extreme puns its `ArenaKey`. The output narrows to
    /// `Float32` for a `REAL` slot via the slot's declared `output_type` cast.
    Float {
        op: NumOp,
        column: usize,
        acc: Option<A>,
    },
    /// `MIN`/`MAX` over a `Utf8View` column.
    Str {
        is_max: bool,
        column: usize,
        acc: Option<String>,
    },
    /// `FIRST(col)`: the first value this worker saw, kept as a one-row slice
    /// of its column so any value type rides through unchanged (NULL included -
    /// `FIRST` keeps the first *row's* value, unlike the NULL-skipping folds).
    /// The declared output type shapes the NULL emitted over zero rows.
    First {
        column: usize,
        output_type: DataType,
        acc: Option<ArrayRef>,
    },
}

impl<A: IntCell + F64Cell + WideCell> Slot<A> {
    /// The empty accumulator for `spec`. A `SUM`/`MIN`/`MAX`'s value family is
    /// decided by the column's declared `output_type`, not the kind: a `Utf8View`
    /// extreme keeps an owned `String`, a floating column an `f64` punned into `A`,
    /// else the integer width `A`.
    fn build(spec: &AggregationSlot) -> Self {
        let column = spec.column;
        let op = match spec.kind {
            AggregationKind::CountStar => {
                return Slot::Count {
                    column: None,
                    count: 0,
                };
            }
            AggregationKind::Count => {
                return Slot::Count {
                    column: Some(column),
                    count: 0,
                };
            }
            AggregationKind::Sum => NumOp::Sum,
            AggregationKind::Min => NumOp::Min,
            AggregationKind::Max => NumOp::Max,
            AggregationKind::First => {
                return Slot::First {
                    column,
                    output_type: spec.output_type.clone(),
                    acc: None,
                };
            }
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
            Slot::Int {
                op,
                column,
                acc: None,
            }
        }
    }

    /// Fold one batch's contribution into this accumulator.
    fn consume(&mut self, batch: &RecordBatch) {
        match self {
            Slot::Count { column, count } => {
                *count += batch.num_rows() as i64;
                if let Some(column) = column {
                    *count -= batch.column(*column).null_count() as i64;
                }
            }
            Slot::Int { op, column, acc } => {
                let op = *op;
                let reduced = op.reduce_int_column::<A>(batch.column(*column).as_ref());
                fold_into(acc, reduced, |a, b| op.merge(a, b));
            }
            Slot::Float { op, column, acc } => {
                let op = *op;
                let reduced = op.reduce_float_column::<A>(batch.column(*column).as_ref());
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
            Slot::First { column, acc, .. } => {
                if acc.is_none() && batch.num_rows() > 0 {
                    *acc = Some(batch.column(*column).slice(0, 1));
                }
            }
        }
    }

    /// Fold a sibling worker's finished accumulator into this one. Every worker
    /// builds its slots from the same specs, so the variants always match.
    fn merge(&mut self, other: Slot<A>) {
        match (self, other) {
            (Slot::Count { count, .. }, Slot::Count { count: other, .. }) => *count += other,
            (Slot::Int { op, acc, .. }, Slot::Int { acc: other, .. }) => {
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
            (Slot::First { acc, .. }, Slot::First { acc: other, .. }) => {
                fold_into(acc, other, |mine, _theirs| mine);
            }
            _ => unreachable!("every worker builds its slots from the same specs"),
        }
    }

    /// Build the single-row output column from the finished accumulator. `None`
    /// (zero rows aggregated) renders as SQL `NULL`; a count renders its plain
    /// total (`0` over zero rows, never NULL).
    fn into_column(self) -> (Field, ArrayRef) {
        match self {
            Slot::Count { count, .. } => (
                Field::new("count", DataType::Int64, false),
                Arc::new(Int64Array::from(vec![count])),
            ),
            Slot::Int { op, acc, .. } => (
                Field::new(op.column_name(), A::data_type(), true),
                A::scalar_array(acc),
            ),
            // Unpun the cell back to `f64` and render `Float64`; the outputter casts
            // to the slot's declared `output_type` (narrowing to `Float32` for a
            // `REAL` slot).
            Slot::Float { op, acc, .. } => (
                Field::new(op.column_name(), DataType::Float64, true),
                Arc::new(Float64Array::from(vec![acc.map(F64Cell::into_f64)])),
            ),
            Slot::Str { is_max, acc, .. } => (
                Field::new(if is_max { "max" } else { "min" }, DataType::Utf8View, true),
                Arc::new(StringViewArray::from_iter(std::iter::once(acc.as_deref()))),
            ),
            Slot::First {
                output_type, acc, ..
            } => {
                let value = acc.unwrap_or_else(|| arrow_array::new_null_array(&output_type, 1));
                (Field::new("first", value.data_type().clone(), true), value)
            }
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
    if is_max {
        reduce_str_extreme::<true>(a)
    } else {
        reduce_str_extreme::<false>(a)
    }
}

/// The hot string loop, monomorphic per extreme. NULL rows contribute nothing;
/// an all-NULL column reduces to `None`, exactly like an empty one.
fn reduce_str_extreme<const MAX: bool>(a: &StringViewArray) -> Option<String> {
    if a.null_count() == 0 {
        if a.is_empty() {
            return None;
        }
        let mut acc: &str = unsafe { a.value_unchecked(0) };
        for i in 1..a.len() {
            let v = unsafe { a.value_unchecked(i) };
            acc = str_extreme::<_, MAX>(acc, v);
        }
        Some(acc.to_string())
    } else {
        let mut acc: Option<&str> = None;
        for i in 0..a.len() {
            if a.is_valid(i) {
                let v = unsafe { a.value_unchecked(i) };
                acc = Some(match acc {
                    Some(best) => str_extreme::<_, MAX>(best, v),
                    None => v,
                });
            }
        }
        acc.map(str::to_string)
    }
}

/// Factory for the aggregate operator. All workers share one gather barrier:
/// each publishes its slots into its own slot of the barrier, and the last
/// worker to finish merges them and emits the result. A channel every worker
/// sends into at the same moment would serialise the whole pool on one cache
/// line at the end of every aggregate query.
pub struct AggregateFactory<A: IntCell> {
    specs: Arc<Vec<AggregationSlot>>,
    gather: Arc<GatherBarrier<Vec<Slot<A>>>>,
}

impl<A: IntCell> AggregateFactory<A> {
    /// Create one factory per worker, all sharing the same gather barrier.
    pub fn create_for_workers(
        specs: Vec<AggregationSlot>,
        worker_count: usize,
    ) -> impl IntoIterator<Item = AggregateFactory<A>> {
        let specs = Arc::new(specs);
        let gather = Arc::new(GatherBarrier::new(worker_count));

        (0..worker_count).map(move |_| AggregateFactory {
            specs: specs.clone(),
            gather: gather.clone(),
        })
    }
}

impl<A: IntCell + F64Cell + WideCell> UnaryFactory<RecordBatch, RecordBatch>
    for AggregateFactory<A>
{
    type Unary = PipelineBreaker<RecordBatch, RecordBatch, Aggregate<A>>;

    fn build_unary(self) -> Self::Unary {
        PipelineBreaker::Consuming(Aggregate::new(self.specs, self.gather))
    }
}

/// Per-worker aggregate consumer. Accumulates local slots, then publishes
/// them at the shared gather barrier on finalization.
pub struct Aggregate<A: IntCell> {
    specs: Arc<Vec<AggregationSlot>>,
    local: Vec<Slot<A>>,
    gather: Arc<GatherBarrier<Vec<Slot<A>>>>,
}

impl<A: IntCell + F64Cell + WideCell> Aggregate<A> {
    fn new(specs: Arc<Vec<AggregationSlot>>, gather: Arc<GatherBarrier<Vec<Slot<A>>>>) -> Self {
        let local = specs.iter().map(Slot::build).collect();
        Aggregate {
            specs,
            local,
            gather,
        }
    }
}

impl<A: IntCell + F64Cell + WideCell> Consumer<RecordBatch, RecordBatch> for Aggregate<A> {
    type Outputter = AggregateOutputter<A>;

    fn consume(
        &mut self,
        batch: RecordBatch,
        _output: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        for slot in &mut self.local {
            slot.consume(&batch);
        }
        Ok(())
    }

    fn into_outputter(self) -> unary::Result<Option<Self::Outputter>> {
        // Every worker publishes its slots (even untouched, so the merge always
        // sees a contribution per worker); the last to arrive merges them all
        // and becomes the one outputter. The barrier wakes the pool for it.
        let specs = self.specs;
        let merged = self.gather.arrive(self.local, |all_slots| {
            let mut totals: Vec<Slot<A>> = specs.iter().map(Slot::build).collect();
            for worker_slots in all_slots {
                for (total, slot) in totals.iter_mut().zip(worker_slots) {
                    total.merge(slot);
                }
            }
            totals
        });
        Ok(merged.map(|totals| AggregateOutputter { specs, totals }))
    }
}

/// Output phase (the last worker to finish only): emits the single-row
/// result merged from every worker's slots.
pub struct AggregateOutputter<A: IntCell> {
    specs: Arc<Vec<AggregationSlot>>,
    totals: Vec<Slot<A>>,
}

impl<A: IntCell + F64Cell + WideCell> Outputter<RecordBatch> for AggregateOutputter<A> {
    fn output(&mut self, output: &mut dyn Sender<RecordBatch>) -> unary::Result<bool> {
        let (fields, columns): (Vec<Field>, Vec<ArrayRef>) = std::mem::take(&mut self.totals)
            .into_iter()
            .zip(self.specs.iter())
            .map(|(slot, spec)| {
                let (field, column) = slot.into_column();
                cast_value_column(field, column, &spec.output_type)
            })
            .unzip();
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
        output.send(batch)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::unary::test_utils::run_consumers;
    use arrow_array::{Decimal64Array, Decimal128Array, Int32Array};

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

    /// Build `n` aggregate consumers sharing one gather barrier, mirroring the
    /// factory's wiring.
    fn build<A: IntCell + F64Cell + WideCell>(
        n: usize,
        specs: Vec<AggregationSlot>,
    ) -> Vec<Aggregate<A>> {
        let specs = Arc::new(specs);
        let gather = Arc::new(GatherBarrier::new(n));
        (0..n)
            .map(move |_| Aggregate::new(specs.clone(), gather.clone()))
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

    /// A DECIMAL(10,2) column of [10.50, 2.50, 4.00] as raw unscaled i64s in a
    /// single-column batch (a precision of 10 rides the Decimal64 carrier).
    fn make_decimal64_batch() -> RecordBatch {
        let array = Decimal64Array::from(vec![1050i64, 250, 400])
            .with_precision_and_scale(10, 2)
            .unwrap();
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "price",
                DataType::Decimal64(10, 2),
                false,
            )])),
            vec![Arc::new(array)],
        )
        .unwrap()
    }

    fn col_dec64(batch: &RecordBatch, i: usize) -> i64 {
        batch
            .column(i)
            .as_any()
            .downcast_ref::<Decimal64Array>()
            .unwrap()
            .value(0)
    }

    #[test]
    fn decimal_sum_min_max_restamp_not_rescale() {
        // The sum's declared type is DECIMAL(38,2) (a Decimal128 column), the
        // extremes keep the input's Decimal64(10,2); the raw values must come
        // through untouched (a rescaling cast would multiply them by 100). The
        // SUM forces the wide i128 accumulator, so the extremes render wide and
        // narrow back to Decimal64 on output.
        let ops = build::<i128>(
            1,
            vec![
                AggregationSlot::new(AggregationKind::Sum, 0, DataType::Decimal128(38, 2)),
                AggregationSlot::new(AggregationKind::Min, 0, DataType::Decimal64(10, 2)),
                AggregationSlot::new(AggregationKind::Max, 0, DataType::Decimal64(10, 2)),
            ],
        );

        let out = run_consumers(ops, vec![vec![make_decimal64_batch()]]);

        let batch = &out.items[0];
        assert_eq!(batch.column(0).data_type(), &DataType::Decimal128(38, 2));
        assert_eq!(batch.column(1).data_type(), &DataType::Decimal64(10, 2));
        assert_eq!(col_i128(batch, 0), 1700); // 17.00
        assert_eq!(col_dec64(batch, 1), 250); // 2.50
        assert_eq!(col_dec64(batch, 2), 1050); // 10.50
    }

    #[test]
    fn decimal64_min_max_ride_the_narrow_lane() {
        // With no SUM forcing the wide accumulator, Decimal64 extremes fold in
        // plain i64 cells; the Int64 they render as is restamped (same buffer,
        // never cast) to the declared Decimal64(10,2).
        let ops = build::<i64>(
            1,
            vec![
                AggregationSlot::new(AggregationKind::Min, 0, DataType::Decimal64(10, 2)),
                AggregationSlot::new(AggregationKind::Max, 0, DataType::Decimal64(10, 2)),
            ],
        );

        let out = run_consumers(ops, vec![vec![make_decimal64_batch()]]);

        let batch = &out.items[0];
        assert_eq!(batch.column(0).data_type(), &DataType::Decimal64(10, 2));
        assert_eq!(col_dec64(batch, 0), 250); // 2.50
        assert_eq!(col_dec64(batch, 1), 1050); // 10.50
    }

    #[test]
    fn wide_decimal_sum_min_max_stay_decimal128() {
        // A precision past 18 rides the Decimal128 carrier end to end: the
        // extremes keep the declared (20,2) and the raw i128s are untouched.
        let array = Decimal128Array::from(vec![1050i128, 250, 400])
            .with_precision_and_scale(20, 2)
            .unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "price",
                DataType::Decimal128(20, 2),
                false,
            )])),
            vec![Arc::new(array)],
        )
        .unwrap();
        let ops = build::<i128>(
            1,
            vec![
                AggregationSlot::new(AggregationKind::Sum, 0, DataType::Decimal128(38, 2)),
                AggregationSlot::new(AggregationKind::Min, 0, DataType::Decimal128(20, 2)),
                AggregationSlot::new(AggregationKind::Max, 0, DataType::Decimal128(20, 2)),
            ],
        );

        let out = run_consumers(ops, vec![vec![batch]]);

        let batch = &out.items[0];
        assert_eq!(batch.column(0).data_type(), &DataType::Decimal128(38, 2));
        assert_eq!(batch.column(1).data_type(), &DataType::Decimal128(20, 2));
        assert_eq!(col_i128(batch, 0), 1700); // 17.00
        assert_eq!(col_i128(batch, 1), 250); // 2.50
        assert_eq!(col_i128(batch, 2), 1050); // 10.50
    }

    #[test]
    fn first_keeps_the_first_value_seen() {
        let ops = build::<i64>(
            1,
            vec![AggregationSlot::new(
                AggregationKind::First,
                0,
                DataType::Int32,
            )],
        );

        let out = run_consumers(ops, vec![vec![make_batch(&[7, 2]), make_batch(&[9])]]);

        assert_eq!(out.items.len(), 1);
        assert_eq!(
            out.items[0].column(0).as_primitive::<Int32Type>().value(0),
            7
        );
    }

    #[test]
    fn first_over_zero_rows_is_null() {
        let ops = build::<i64>(
            1,
            vec![AggregationSlot::new(
                AggregationKind::First,
                0,
                DataType::Int32,
            )],
        );

        let out = run_consumers(ops, vec![vec![]]);

        assert!(out.items[0].column(0).is_null(0));
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

    fn make_nullable_batch(values: &[Option<i32>]) -> RecordBatch {
        let array = Int32Array::from(values.to_vec());
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)])),
            vec![Arc::new(array)],
        )
        .unwrap()
    }

    #[test]
    fn nulls_are_skipped_by_sum_min_max() {
        let ops = build::<i64>(
            1,
            vec![
                slot(AggregationKind::Sum, 0),
                slot(AggregationKind::Min, 0),
                slot(AggregationKind::Max, 0),
            ],
        );

        let out = run_consumers(
            ops,
            vec![vec![make_nullable_batch(&[Some(5), None, Some(2), None])]],
        );

        assert_eq!(col_i128(&out.items[0], 0), 7);
        assert_eq!(col_i64(&out.items[0], 1), 2);
        assert_eq!(col_i64(&out.items[0], 2), 5);
    }

    #[test]
    fn count_column_skips_nulls_count_star_does_not() {
        let ops = build::<i64>(
            1,
            vec![
                slot(AggregationKind::Count, 0),
                slot(AggregationKind::CountStar, 0),
            ],
        );

        let out = run_consumers(
            ops,
            vec![vec![make_nullable_batch(&[Some(1), None, None, Some(4)])]],
        );

        assert_eq!(col_i64(&out.items[0], 0), 2);
        assert_eq!(col_i64(&out.items[0], 1), 4);
    }

    #[test]
    fn all_null_input_sum_is_null() {
        let ops = build::<i64>(1, vec![slot(AggregationKind::Sum, 0)]);

        let out = run_consumers(ops, vec![vec![make_nullable_batch(&[None, None])]]);

        assert!(out.items[0].column(0).is_null(0));
    }

    #[test]
    fn string_extreme_skips_nulls() {
        let array = StringViewArray::from(vec![Some("pear"), None, Some("apple"), None]);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8View, true)])),
            vec![Arc::new(array)],
        )
        .unwrap();
        let ops = build::<i64>(
            1,
            vec![
                str_slot(AggregationKind::Min, 0),
                str_slot(AggregationKind::Max, 0),
            ],
        );

        let out = run_consumers(ops, vec![vec![batch]]);

        assert_eq!(col_str(&out.items[0], 0), "apple");
        assert_eq!(col_str(&out.items[0], 1), "pear");
    }
}
