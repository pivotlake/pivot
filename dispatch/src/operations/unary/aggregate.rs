//! Global aggregate operator (no GROUP BY) computing one or more column
//! aggregates in a single pass.
//!
//! Mirrors [`Count`](super::count): each worker keeps a local running
//! accumulator per requested aggregate while consuming batches. On
//! [`finish`](Unary::finish) every worker folds its partials into a shared
//! accumulator; the last worker to finish emits the single-row result with one
//! output column per aggregate.
//!
//! The sum is accumulated in `i128` so a full billion-row scan of a
//! 16/32/64-bit integer column never overflows. `Sum` emits that full `i128`
//! as a `Decimal128(38, 0)` cell (matching DuckDB's `HUGEINT` result type) so
//! large sums over a 64-bit column are exact; `Count` emits `Int64` and `Avg` a
//! `Float64`. (DuckDB lowers `AVG(x)` to `sum(x) / count(x)` over two
//! aggregates plus a divide projection; that divide casts both operands to
//! `f64`, so the full-precision `Decimal128` sum yields a correct average and
//! the `Avg` kind here is only used for an undecomposed average.)

use crate::operations::channels::Sender;
use crate::operations::unary::group::{Aggregate as RowAggregate, Sum};
use crate::operations::unary::{self, Unary, UnaryFactory};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int16Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, Decimal128Array, Float64Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Which aggregate to compute for a given column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggKind {
    Sum,
    Count,
    Avg,
    /// `COUNT(*)` — counts every row regardless of value (column is ignored).
    CountStar,
}

/// One requested aggregate: a kind plus the input column it reads.
#[derive(Clone, Copy, Debug)]
pub struct AggSpec {
    pub kind: AggKind,
    pub column: usize,
}

impl AggSpec {
    pub fn new(kind: AggKind, column: usize) -> Self {
        Self { kind, column }
    }
}

/// One partial accumulator: running `i128` sum and non-null count.
#[derive(Default, Clone, Copy)]
struct Acc {
    sum: i128,
    count: u64,
}

/// Factory for the aggregate operator. All workers share the same `shared`
/// accumulators and `siblings_left` counter.
pub struct AggregateFactory {
    specs: Arc<Vec<AggSpec>>,
    siblings_left: Arc<AtomicUsize>,
    shared: Arc<Mutex<Vec<Acc>>>,
}

impl AggregateFactory {
    /// Create one factory per worker, all sharing the same accumulators/counter.
    pub fn create_for_workers(
        specs: Vec<AggSpec>,
        worker_count: usize,
    ) -> impl IntoIterator<Item = AggregateFactory> {
        let siblings_left = Arc::new(AtomicUsize::new(worker_count));
        let shared = Arc::new(Mutex::new(vec![Acc::default(); specs.len()]));
        let specs = Arc::new(specs);

        (0..worker_count).map(move |_| AggregateFactory {
            specs: specs.clone(),
            siblings_left: siblings_left.clone(),
            shared: shared.clone(),
        })
    }
}

impl UnaryFactory<RecordBatch, RecordBatch> for AggregateFactory {
    type Unary = Aggregate;

    fn build_unary(self) -> Self::Unary {
        let local = vec![Acc::default(); self.specs.len()];
        Aggregate {
            specs: self.specs,
            local,
            wrote_shared: false,
            siblings_left: self.siblings_left,
            shared: self.shared,
        }
    }
}

/// Per-worker aggregate state. Folds its partials into `shared` once at finish.
pub struct Aggregate {
    specs: Arc<Vec<AggSpec>>,
    local: Vec<Acc>,
    wrote_shared: bool,
    siblings_left: Arc<AtomicUsize>,
    shared: Arc<Mutex<Vec<Acc>>>,
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

/// Build the single output column for one aggregate spec from its accumulator.
fn result_column(kind: AggKind, acc: Acc) -> (Field, ArrayRef) {
    match kind {
        // Emit the full i128 sum as Decimal128(38, 0) (scale 0 = a plain
        // integer) so large sums survive without the i64 truncation that an
        // Int64 output column would force. Precision 38 is the max Decimal128
        // holds and comfortably covers any i128 a real scan produces.
        AggKind::Sum => (
            Field::new("sum", DataType::Decimal128(38, 0), false),
            Arc::new(
                Decimal128Array::from(vec![acc.sum])
                    .with_precision_and_scale(38, 0)
                    .unwrap(),
            ),
        ),
        AggKind::Count | AggKind::CountStar => (
            Field::new("count", DataType::Int64, false),
            Arc::new(Int64Array::from(vec![acc.count as i64])),
        ),
        AggKind::Avg => {
            let avg = if acc.count == 0 {
                0.0
            } else {
                acc.sum as f64 / acc.count as f64
            };
            (
                Field::new("avg", DataType::Float64, false),
                Arc::new(Float64Array::from(vec![avg])),
            )
        }
    }
}

impl Unary<RecordBatch, RecordBatch> for Aggregate {
    fn consume<OP: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        _output: &mut OP,
    ) -> unary::Result<()> {
        for (i, spec) in self.specs.iter().enumerate() {
            match spec.kind {
                // COUNT(*) counts every row and never reads a column.
                AggKind::CountStar => self.local[i].count += batch.num_rows() as u64,
                // COUNT(c) needs only the non-null count, not the values — read
                // it straight off the null bitmap instead of summing the column.
                // This keeps `AVG(c)` (lowered to `Sum(c)`/`Count(c)`) to a
                // single value-summing pass instead of scanning twice.
                AggKind::Count => {
                    let col = batch.column(spec.column);
                    self.local[i].count += (col.len() - col.null_count()) as u64;
                }
                AggKind::Sum | AggKind::Avg => {
                    let (sum, count) = sum_column(batch.column(spec.column));
                    self.local[i].sum += sum;
                    self.local[i].count += count;
                }
            }
        }
        Ok(())
    }

    fn finish<OP: Sender<RecordBatch>>(&mut self, output: &mut OP) -> unary::Result<bool> {
        if self.wrote_shared {
            return Ok(true);
        }
        self.wrote_shared = true;

        let totals = {
            let mut shared = self.shared.lock().unwrap();
            for (i, acc) in self.local.iter().enumerate() {
                shared[i].sum += acc.sum;
                shared[i].count += acc.count;
            }
            shared.clone()
        };

        // Last worker emits the single-row result, one column per spec.
        if self.siblings_left.fetch_sub(1, Ordering::SeqCst) == 1 {
            let (fields, columns): (Vec<Field>, Vec<ArrayRef>) = self
                .specs
                .iter()
                .zip(totals.iter())
                .map(|(spec, acc)| result_column(spec.kind, *acc))
                .unzip();
            let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
            output.send(batch)?;
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::unary::test_utils::{CollectSender, run_unary_to_completion};
    use arrow_array::Int32Array;

    fn make_batch(values: &[i32]) -> RecordBatch {
        let array = Int32Array::from(values.to_vec());
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)])),
            vec![Arc::new(array)],
        )
        .unwrap()
    }

    fn build(n: usize, specs: Vec<AggSpec>) -> Vec<Aggregate> {
        AggregateFactory::create_for_workers(specs, n)
            .into_iter()
            .map(|f| f.build_unary())
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
    fn col_f64(batch: &RecordBatch, i: usize) -> f64 {
        batch
            .column(i)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0)
    }

    #[test]
    fn single_worker_sum() {
        let op = build(1, vec![AggSpec::new(AggKind::Sum, 0)]).pop().unwrap();
        let results =
            run_unary_to_completion(op, vec![make_batch(&[1, 2, 3]), make_batch(&[4, 5])]);
        assert_eq!(results.len(), 1);
        assert_eq!(col_i128(&results[0], 0), 15);
    }

    #[test]
    fn sum_and_count_together() {
        let op = build(
            1,
            vec![
                AggSpec::new(AggKind::Sum, 0),
                AggSpec::new(AggKind::Count, 0),
            ],
        )
        .pop()
        .unwrap();
        let results = run_unary_to_completion(op, vec![make_batch(&[2, 4, 6, 8])]);
        assert_eq!(results.len(), 1);
        assert_eq!(col_i128(&results[0], 0), 20); // sum
        assert_eq!(col_i64(&results[0], 1), 4); // count
    }

    #[test]
    fn single_worker_avg() {
        let op = build(1, vec![AggSpec::new(AggKind::Avg, 0)]).pop().unwrap();
        let results = run_unary_to_completion(op, vec![make_batch(&[2, 4, 6])]);
        assert_eq!(results.len(), 1);
        assert_eq!(col_f64(&results[0], 0), 4.0);
    }

    #[test]
    fn multiple_workers_sum_merge() {
        let mut ops = build(3, vec![AggSpec::new(AggKind::Sum, 0)]);
        let mut sender = CollectSender::new();
        ops[0].consume(make_batch(&[10]), &mut sender).unwrap();
        ops[1].consume(make_batch(&[20, 5]), &mut sender).unwrap();
        ops[2].consume(make_batch(&[1]), &mut sender).unwrap();
        ops[0].finish(&mut sender).unwrap();
        ops[1].finish(&mut sender).unwrap();
        assert!(sender.items.is_empty());
        ops[2].finish(&mut sender).unwrap();
        assert_eq!(sender.items.len(), 1);
        assert_eq!(col_i128(&sender.items[0], 0), 36);
    }

    #[test]
    fn large_i64_sum_does_not_overflow() {
        // A batch of large BIGINTs overflows an i64 accumulator
        // within a single batch; the i128 fold must keep it exact.
        let schema = Arc::new(Schema::new(vec![Field::new("u", DataType::Int64, false)]));
        let vals = vec![i64::MAX, i64::MAX, i64::MAX];
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow_array::Int64Array::from(vals.clone()))],
        )
        .unwrap();
        // Avg of three i64::MAX values is i64::MAX exactly.
        let op = build(1, vec![AggSpec::new(AggKind::Avg, 0)]).pop().unwrap();
        let results = run_unary_to_completion(op, vec![batch]);
        assert_eq!(col_f64(&results[0], 0), i64::MAX as f64);
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
        let op = build(1, vec![AggSpec::new(AggKind::Sum, 0)]).pop().unwrap();
        let results = run_unary_to_completion(op, vec![batch]);
        assert_eq!(col_i128(&results[0], 0), 3 * i64::MAX as i128);
    }

    #[test]
    fn count_star_counts_all_rows() {
        let op = build(1, vec![AggSpec::new(AggKind::CountStar, 0)])
            .pop()
            .unwrap();
        let results = run_unary_to_completion(op, vec![make_batch(&[5, 6, 7]), make_batch(&[8])]);
        assert_eq!(col_i64(&results[0], 0), 4);
    }

    #[test]
    fn empty_input_sum_is_zero() {
        let op = build(1, vec![AggSpec::new(AggKind::Sum, 0)]).pop().unwrap();
        let results = run_unary_to_completion(op, vec![]);
        assert_eq!(results.len(), 1);
        assert_eq!(col_i128(&results[0], 0), 0);
    }
}
