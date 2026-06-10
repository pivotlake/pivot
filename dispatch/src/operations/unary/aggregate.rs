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
//! The aggregate *kind* and per-row contribution are the group module's
//! [`AggregationSlot`] / [`Aggregate`](RowAggregate) ops — the single source of
//! truth shared with GROUP BY — and so is the accumulator width [`A`](Accumulator):
//! the operator is generic over `i64`/`i128`, chosen by the same column-width rule
//! as the grouped path (`i128` only when a sum reads a 64-bit column).
//! `Sum` emits `Int64` or `Decimal128(38, 0)` accordingly (the
//! latter matching DuckDB's `HUGEINT`); `Count` always emits `Int64`. There is no
//! `Avg` kind: DuckDB lowers `AVG(x)` to `sum(x) / count(x)` over two aggregates
//! plus a divide projection, so an average reaches this operator as a `Sum` slot
//! and a `Count` slot.

use crate::operations::channels::Sender;
use crate::operations::unary::group::{
    Accumulator, Aggregate as RowAggregate, AggregationKind, AggregationSlot, Sum,
};
use crate::operations::unary::pipeline_breaker::{Consumer, Outputter, PipelineBreaker};
use crate::operations::unary::{self, UnaryFactory};
use crate::worker::worker_waker;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type, UInt16Type, UInt32Type};
use arrow_array::{Array, ArrayRef, Int64Array, PrimitiveArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;
use std::sync::mpsc;

/// Factory for the aggregate operator. All workers share one mpsc channel; the
/// first factory gets the receiver, the rest get `None`, so per-worker partials
/// flow to a single collector (the same wiring as [`OrderByLimitFactory`]).
///
/// [`OrderByLimitFactory`]: super::order_by_limit
pub struct AggregateFactory<A: Accumulator> {
    slots: Arc<Vec<AggregationSlot>>,
    sender: mpsc::Sender<Partials<A>>,
    receiver: Option<mpsc::Receiver<Partials<A>>>,
}

impl<A: Accumulator> AggregateFactory<A> {
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

impl<A: Accumulator> UnaryFactory<RecordBatch, RecordBatch> for AggregateFactory<A> {
    type Unary = PipelineBreaker<RecordBatch, RecordBatch, Aggregate<A>>;

    fn build_unary(mut self) -> Self::Unary {
        PipelineBreaker::Consuming(Aggregate::new(
            self.slots,
            self.sender,
            self.receiver.take(),
        ))
    }
}

/// A worker's partials: one accumulator per slot, plus the rows it saw —
/// `MIN`/`MAX` over zero rows is `NULL`, which the identity accumulators
/// (`i64::MAX` / `i64::MIN`) can't express on their own.
pub struct Partials<A> {
    totals: Vec<A>,
    rows: u64,
}

/// The identity element a slot's fold starts from: 0 for the additive kinds,
/// the saturating extreme for `MIN`/`MAX`. (Min/max inputs are at most 64-bit
/// columns, so `i64`'s extremes are valid identities at any accumulator width.)
fn identity<A: Accumulator>(kind: AggregationKind) -> A {
    match kind {
        AggregationKind::Min => A::from(i64::MAX),
        AggregationKind::Max => A::from(i64::MIN),
        _ => A::default(),
    }
}

/// Fold a slot's incoming partial into its running total.
#[inline(always)]
fn fold<A: Accumulator>(kind: AggregationKind, total: A, partial: A) -> A {
    match kind {
        AggregationKind::Min => total.min(partial),
        AggregationKind::Max => total.max(partial),
        _ => {
            let mut t = total;
            t += partial;
            t
        }
    }
}

/// Per-worker aggregate consumer. Accumulates local partials, then sends them
/// down the shared channel on finalization.
pub struct Aggregate<A: Accumulator> {
    slots: Arc<Vec<AggregationSlot>>,
    local: Vec<A>,
    rows: u64,
    sender: mpsc::Sender<Partials<A>>,
    receiver: Option<mpsc::Receiver<Partials<A>>>,
}

impl<A: Accumulator> Aggregate<A> {
    fn new(
        slots: Arc<Vec<AggregationSlot>>,
        sender: mpsc::Sender<Partials<A>>,
        receiver: Option<mpsc::Receiver<Partials<A>>>,
    ) -> Self {
        let local = slots.iter().map(|s| identity::<A>(s.kind)).collect();
        Aggregate {
            slots,
            local,
            rows: 0,
            sender,
            receiver,
        }
    }
}

/// Sum an integer primitive column into the accumulator `A`.
///
/// The per-row value read (downcast + widen to `i64`) is delegated to the GROUP
/// BY [`Sum`] op — the single source of truth for what `SUM` contributes per
/// row. The fold accumulates into `A`: `i64` for 16/32-bit columns (a full scan
/// can't overflow it) and `i128` for a 64-bit column (whose values can approach
/// `i64::MAX`), the width the planner picks.
fn sum_column<A: Accumulator>(arr: &dyn Array) -> A {
    macro_rules! sum_primitive {
        ($ty:ty) => {{
            let a = arr.as_primitive::<$ty>();
            let mut acc = A::default();
            for i in 0..a.len() {
                acc += A::from(Sum::<$ty>::contribution(&a, i));
            }
            acc
        }};
    }

    match arr.data_type() {
        DataType::Int16 => sum_primitive!(Int16Type),
        DataType::Int32 => sum_primitive!(Int32Type),
        DataType::Int64 => sum_primitive!(Int64Type),
        other => panic!("aggregate: unsupported column type {other:?}"),
    }
}

/// The batch-local extreme of an integer primitive column, widened to `i64` —
/// over any storage width an integer/date column scans as.
fn extreme_column(arr: &dyn Array, min: bool) -> i64 {
    macro_rules! extreme_primitive {
        ($ty:ty) => {{
            let a = arr.as_primitive::<$ty>();
            let mut acc = if min { i64::MAX } else { i64::MIN };
            for i in 0..a.len() {
                let v = unsafe { a.value_unchecked(i) } as i64;
                acc = if min { acc.min(v) } else { acc.max(v) };
            }
            acc
        }};
    }

    match arr.data_type() {
        DataType::Int8 => extreme_primitive!(Int8Type),
        DataType::Int16 => extreme_primitive!(Int16Type),
        DataType::Int32 => extreme_primitive!(Int32Type),
        DataType::Int64 => extreme_primitive!(Int64Type),
        DataType::UInt16 => extreme_primitive!(UInt16Type),
        DataType::UInt32 => extreme_primitive!(UInt32Type),
        other => panic!("aggregate MIN/MAX: unsupported column type {other:?}"),
    }
}

/// Build the single output column for one aggregate slot from its accumulator.
/// `rows` is the total row count: a `MIN`/`MAX` over zero rows is `NULL`.
fn result_column<A: Accumulator>(kind: AggregationKind, value: A, rows: u64) -> (Field, ArrayRef) {
    match kind {
        // `Int64` or `Decimal128(38, 0)` per the accumulator width.
        AggregationKind::Sum => {
            let array = A::finalize(Arc::new(PrimitiveArray::<A::Arrow>::from_iter_values([
                value,
            ])));
            (Field::new("sum", array.data_type().clone(), false), array)
        }
        // A count can't exceed the row count, so it always fits i64; the checked
        // narrowing panics on the impossible overflow rather than truncating.
        AggregationKind::Count | AggregationKind::CountStar => {
            let count = i64::try_from(value.into()).expect("count exceeds i64::MAX");
            (
                Field::new("count", DataType::Int64, false),
                Arc::new(Int64Array::from(vec![count])),
            )
        }
        AggregationKind::Min | AggregationKind::Max => {
            let name = if kind == AggregationKind::Min {
                "min"
            } else {
                "max"
            };
            let array: ArrayRef = if rows == 0 {
                Arc::new(Int64Array::from(vec![None::<i64>]))
            } else {
                let v = i64::try_from(value.into()).expect("extreme exceeds i64 range");
                Arc::new(Int64Array::from(vec![v]))
            };
            (Field::new(name, DataType::Int64, true), array)
        }
        AggregationKind::MinStr | AggregationKind::MaxStr => {
            panic!("global string MIN/MAX is not supported")
        }
    }
}

impl<A: Accumulator> Consumer<RecordBatch, RecordBatch> for Aggregate<A> {
    type Outputter = AggregateOutputter<A>;

    fn consume<OP: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        _output: &mut OP,
    ) -> unary::Result<()> {
        for (i, slot) in self.slots.iter().enumerate() {
            let partial = match slot.kind {
                // COUNT(*) counts every row and never reads a column.
                AggregationKind::CountStar => A::from(batch.num_rows() as i64),
                // COUNT(c) needs only the non-null count, not the values — read
                // it straight off the null bitmap instead of summing the column.
                AggregationKind::Count => {
                    let col = batch.column(slot.column);
                    A::from((col.len() - col.null_count()) as i64)
                }
                AggregationKind::Sum => sum_column::<A>(batch.column(slot.column)),
                AggregationKind::Min => A::from(extreme_column(batch.column(slot.column), true)),
                AggregationKind::Max => A::from(extreme_column(batch.column(slot.column), false)),
                AggregationKind::MinStr | AggregationKind::MaxStr => {
                    panic!("global string MIN/MAX is not supported")
                }
            };
            self.local[i] = fold(slot.kind, self.local[i], partial);
        }
        self.rows += batch.num_rows() as u64;
        Ok(())
    }

    fn into_outputter(self) -> unary::Result<Option<Self::Outputter>> {
        // Every worker sends its partials (even all-zero, so the merge always
        // sees a contribution per worker) and wakes the collector. Notifying
        // unconditionally after the send avoids the lost-wakeup that bites when
        // a finishing worker drops its sender while the collector is parked.
        self.sender
            .send(Partials {
                totals: self.local,
                rows: self.rows,
            })
            .expect("aggregate collector dropped");
        worker_waker().notify();

        let totals = self.slots.iter().map(|s| identity::<A>(s.kind)).collect();
        Ok(self.receiver.map(|rx| AggregateOutputter {
            rx,
            slots: self.slots,
            totals,
            rows: 0,
        }))
    }
}

/// Output phase (one worker only): drains every sibling's partials from the
/// channel, sums them, then emits the single-row result.
pub struct AggregateOutputter<A: Accumulator> {
    rx: mpsc::Receiver<Partials<A>>,
    slots: Arc<Vec<AggregationSlot>>,
    totals: Vec<A>,
    rows: u64,
}

impl<A: Accumulator> Outputter<RecordBatch> for AggregateOutputter<A> {
    fn output<OP: Sender<RecordBatch>>(&mut self, output: &mut OP) -> unary::Result<bool> {
        loop {
            match self.rx.try_recv() {
                Ok(partial) => {
                    for (i, (total, p)) in self.totals.iter_mut().zip(&partial.totals).enumerate() {
                        *total = fold(self.slots[i].kind, *total, *p);
                    }
                    self.rows += partial.rows;
                }
                // Some siblings haven't finished yet; resume when re-driven.
                Err(mpsc::TryRecvError::Empty) => return Ok(false),
                // All senders dropped → every worker's partial is folded in.
                Err(mpsc::TryRecvError::Disconnected) => {
                    let (fields, columns): (Vec<Field>, Vec<ArrayRef>) = self
                        .slots
                        .iter()
                        .zip(&self.totals)
                        .map(|(slot, total)| result_column(slot.kind, *total, self.rows))
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
    fn build<A: Accumulator>(n: usize, slots: Vec<AggregationSlot>) -> Vec<Aggregate<A>> {
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
    fn min_max_across_workers() {
        let ops = build::<i64>(
            2,
            vec![slot(AggregationKind::Min, 0), slot(AggregationKind::Max, 0)],
        );
        let out = run_consumers(
            ops,
            vec![vec![make_batch(&[5, -3, 9])], vec![make_batch(&[7, 0])]],
        );
        assert_eq!(col_i64(&out.items[0], 0), -3);
        assert_eq!(col_i64(&out.items[0], 1), 9);
    }

    #[test]
    fn min_over_empty_input_is_null() {
        let ops = build::<i64>(1, vec![slot(AggregationKind::Min, 0)]);
        let out = run_consumers(ops, vec![vec![]]);
        assert!(out.items[0].column(0).is_null(0));
    }

    #[test]
    fn empty_input_sum_is_zero() {
        let ops = build::<i64>(1, vec![slot(AggregationKind::Sum, 0)]);
        let out = run_consumers(ops, vec![vec![]]);
        assert_eq!(out.items.len(), 1);
        assert_eq!(col_i64(&out.items[0], 0), 0);
    }
}
