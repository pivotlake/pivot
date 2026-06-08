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
//! truth shared with GROUP BY — so this operator only adds the global concerns:
//! null skipping, an `i128` batch accumulator (a full billion-row scan of a
//! 16/32/64-bit integer column never overflows it), and the single-row output.
//! `Sum` emits its full `i128` as a `Decimal128(38, 0)` cell (matching DuckDB's
//! `HUGEINT` result type) so large sums over a 64-bit column are exact; `Count`
//! emits `Int64`. There is no `Avg` kind: DuckDB lowers `AVG(x)` to
//! `sum(x) / count(x)` over two aggregates plus a divide projection (the divide
//! casts both operands to `f64`, so the full-precision `Decimal128` sum yields a
//! correct average), so an average reaches this operator as a `Sum` slot and a
//! `Count` slot, never as a dedicated kind.

use crate::operations::channels::Sender;
use crate::operations::unary::group::{Aggregate as RowAggregate, AggregationKind, AggregationSlot, Sum};
use crate::operations::unary::pipeline_breaker::{Consumer, Outputter, PipelineBreaker};
use crate::operations::unary::{self, UnaryFactory};
use crate::worker::worker_waker;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int16Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, Decimal128Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;
use std::sync::mpsc;

/// One partial accumulator: running `i128` sum and non-null count. A `Sum` slot
/// uses `sum`; a `Count`/`CountStar` slot uses `count`.
#[derive(Default, Clone, Copy)]
struct Acc {
    sum: i128,
    count: u64,
}

impl Acc {
    fn merge(&mut self, other: &Acc) {
        self.sum += other.sum;
        self.count += other.count;
    }
}

/// Factory for the aggregate operator. All workers share one mpsc channel; the
/// first factory gets the receiver, the rest get `None`, so per-worker partials
/// flow to a single collector (the same wiring as [`OrderByLimitFactory`]).
///
/// [`OrderByLimitFactory`]: super::order_by_limit
pub struct AggregateFactory {
    slots: Arc<Vec<AggregationSlot>>,
    sender: mpsc::Sender<Vec<Acc>>,
    receiver: Option<mpsc::Receiver<Vec<Acc>>>,
}

impl AggregateFactory {
    /// Create one factory per worker, all sharing the same partials channel.
    pub fn create_for_workers(
        slots: Vec<AggregationSlot>,
        worker_count: usize,
    ) -> impl IntoIterator<Item = AggregateFactory> {
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

impl UnaryFactory<RecordBatch, RecordBatch> for AggregateFactory {
    type Unary = PipelineBreaker<RecordBatch, RecordBatch, Aggregate>;

    fn build_unary(mut self) -> Self::Unary {
        PipelineBreaker::Consuming(Aggregate::new(self.slots, self.sender, self.receiver.take()))
    }
}

/// Per-worker aggregate consumer. Accumulates local partials, then sends them
/// down the shared channel on finalization.
pub struct Aggregate {
    slots: Arc<Vec<AggregationSlot>>,
    local: Vec<Acc>,
    sender: mpsc::Sender<Vec<Acc>>,
    receiver: Option<mpsc::Receiver<Vec<Acc>>>,
}

impl Aggregate {
    fn new(
        slots: Arc<Vec<AggregationSlot>>,
        sender: mpsc::Sender<Vec<Acc>>,
        receiver: Option<mpsc::Receiver<Vec<Acc>>>,
    ) -> Self {
        let local = vec![Acc::default(); slots.len()];
        Aggregate {
            slots,
            local,
            sender,
            receiver,
        }
    }
}

/// Sum the non-null values of an integer primitive column as `i128`, returning
/// `(sum, non_null_count)`.
///
/// The per-row value read (downcast + widen to `i64`) is delegated to the GROUP
/// BY [`Sum`] op — the single source of truth for what `SUM` contributes per
/// row — so this path only layers on the two concerns that op leaves out: null
/// skipping, and an accumulator wide enough for a whole batch.
///
/// `$acc` is the per-batch accumulator type. For 16/32-bit columns an `i64`
/// batch fold is safe (a single batch can't overflow it) and fast. For 64-bit
/// columns the values themselves can approach `i64::MAX`, so a
/// batch sum must accumulate in `i128` or it overflows within one batch.
fn sum_column(arr: &dyn Array) -> (i128, u64) {
    macro_rules! sum_primitive {
        ($ty:ty, $acc:ty) => {{
            let a = arr.as_primitive::<$ty>();
            let non_null = (a.len() - a.null_count()) as u64;
            let contribution = |i: usize| Sum::<$ty>::contribution(&a, i) as $acc;
            let batch_sum: $acc = if a.null_count() == 0 {
                (0..a.len()).map(contribution).sum()
            } else {
                (0..a.len()).filter(|&i| a.is_valid(i)).map(contribution).sum()
            };
            (batch_sum as i128, non_null)
        }};
    }

    match arr.data_type() {
        DataType::Int16 => sum_primitive!(Int16Type, i64),
        DataType::Int32 => sum_primitive!(Int32Type, i64),
        DataType::Int64 => sum_primitive!(Int64Type, i128),
        other => panic!("aggregate: unsupported column type {other:?}"),
    }
}

/// Build the single output column for one aggregate slot from its accumulator.
fn result_column(kind: AggregationKind, acc: Acc) -> (Field, ArrayRef) {
    match kind {
        // Emit the full i128 sum as Decimal128(38, 0) (scale 0 = a plain
        // integer) so large sums survive without the i64 truncation that an
        // Int64 output column would force. Precision 38 is the max Decimal128
        // holds and comfortably covers any i128 a real scan produces.
        AggregationKind::Sum => (
            Field::new("sum", DataType::Decimal128(38, 0), false),
            Arc::new(
                Decimal128Array::from(vec![acc.sum])
                    .with_precision_and_scale(38, 0)
                    .unwrap(),
            ),
        ),
        AggregationKind::Count | AggregationKind::CountStar => (
            Field::new("count", DataType::Int64, false),
            Arc::new(Int64Array::from(vec![acc.count as i64])),
        ),
    }
}

impl Consumer<RecordBatch, RecordBatch> for Aggregate {
    type Outputter = AggregateOutputter;

    fn consume<OP: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        _output: &mut OP,
    ) -> unary::Result<()> {
        for (i, slot) in self.slots.iter().enumerate() {
            match slot.kind {
                // COUNT(*) counts every row and never reads a column.
                AggregationKind::CountStar => self.local[i].count += batch.num_rows() as u64,
                // COUNT(c) needs only the non-null count, not the values — read
                // it straight off the null bitmap instead of summing the column.
                AggregationKind::Count => {
                    let col = batch.column(slot.column);
                    self.local[i].count += (col.len() - col.null_count()) as u64;
                }
                AggregationKind::Sum => {
                    let (sum, count) = sum_column(batch.column(slot.column));
                    self.local[i].sum += sum;
                    self.local[i].count += count;
                }
            }
        }
        Ok(())
    }

    fn into_outputter(self) -> unary::Result<Option<Self::Outputter>> {
        // Every worker sends its partials (even all-zero, so the merge always
        // sees a contribution per worker) and wakes the collector. Notifying
        // unconditionally after the send avoids the lost-wakeup that bites when
        // a finishing worker drops its sender while the collector is parked.
        self.sender.send(self.local).expect("aggregate collector dropped");
        worker_waker().notify();

        let totals = vec![Acc::default(); self.slots.len()];
        Ok(self.receiver.map(|rx| AggregateOutputter {
            rx,
            slots: self.slots,
            totals,
        }))
    }
}

/// Output phase (one worker only): drains every sibling's partials from the
/// channel, sums them, then emits the single-row result.
pub struct AggregateOutputter {
    rx: mpsc::Receiver<Vec<Acc>>,
    slots: Arc<Vec<AggregationSlot>>,
    totals: Vec<Acc>,
}

impl Outputter<RecordBatch> for AggregateOutputter {
    fn output<OP: Sender<RecordBatch>>(&mut self, output: &mut OP) -> unary::Result<bool> {
        loop {
            match self.rx.try_recv() {
                Ok(partial) => {
                    for (total, p) in self.totals.iter_mut().zip(&partial) {
                        total.merge(p);
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
                        .map(|(slot, acc)| result_column(slot.kind, *acc))
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
    use arrow_array::Int32Array;

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
    fn build(n: usize, slots: Vec<AggregationSlot>) -> Vec<Aggregate> {
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
        let ops = build(1, vec![slot(AggregationKind::Sum, 0)]);
        let out = run_consumers(ops, vec![vec![make_batch(&[1, 2, 3]), make_batch(&[4, 5])]]);
        assert_eq!(out.items.len(), 1);
        assert_eq!(col_i128(&out.items[0], 0), 15);
    }

    #[test]
    fn sum_and_count_together() {
        let ops = build(
            1,
            vec![
                slot(AggregationKind::Sum, 0),
                slot(AggregationKind::Count, 0),
            ],
        );
        let out = run_consumers(ops, vec![vec![make_batch(&[2, 4, 6, 8])]]);
        assert_eq!(out.items.len(), 1);
        assert_eq!(col_i128(&out.items[0], 0), 20); // sum
        assert_eq!(col_i64(&out.items[0], 1), 4); // count
    }

    #[test]
    fn multiple_workers_sum_merge() {
        let ops = build(3, vec![slot(AggregationKind::Sum, 0)]);
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
        // Three i64::MAX values sum past i64::MAX; the Decimal128 output must
        // keep the full i128 instead of truncating to i64.
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
        let ops = build(1, vec![slot(AggregationKind::Sum, 0)]);
        let out = run_consumers(ops, vec![vec![batch]]);
        assert_eq!(col_i128(&out.items[0], 0), 3 * i64::MAX as i128);
    }

    #[test]
    fn count_star_counts_all_rows() {
        let ops = build(1, vec![slot(AggregationKind::CountStar, 0)]);
        let out = run_consumers(ops, vec![vec![make_batch(&[5, 6, 7]), make_batch(&[8])]]);
        assert_eq!(col_i64(&out.items[0], 0), 4);
    }

    #[test]
    fn empty_input_sum_is_zero() {
        let ops = build(1, vec![slot(AggregationKind::Sum, 0)]);
        let out = run_consumers(ops, vec![vec![]]);
        assert_eq!(out.items.len(), 1);
        assert_eq!(col_i128(&out.items[0], 0), 0);
    }
}
