//! Count operator: counts total rows across all workers.
//!
//! Each worker maintains a local count during the consuming phase. On [`finish`](Unary::finish),
//! each worker atomically adds its local count to a shared total. The last worker to
//! finish (determined by a `siblings_left` counter) emits a single `RecordBatch` with
//! one row containing the total count.

use crate::operations::channels::Sender;
use crate::operations::unary::{self, Unary, UnaryFactory};
use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Factory for the count operator. All workers share the same `shared_count` and
/// `siblings_left` atomics, created by [`create_for_workers`](CountFactory::create_for_workers).
pub struct CountFactory {
    siblings_left: Arc<AtomicUsize>,
    shared_count: Arc<AtomicUsize>,
}

impl CountFactory {
    /// Create one `CountFactory` per worker, all sharing the same atomic counters.
    pub fn create_for_workers(worker_count: usize) -> impl IntoIterator<Item = CountFactory> {
        let siblings_left = Arc::new(AtomicUsize::new(worker_count));
        let shared_count: Arc<AtomicUsize> = Arc::new(Default::default());

        (0..worker_count).map(move |_| CountFactory {
            siblings_left: siblings_left.clone(),
            shared_count: shared_count.clone(),
        })
    }
}

impl UnaryFactory<RecordBatch, RecordBatch> for CountFactory {
    type Unary = Count;

    fn build_unary(self) -> Self::Unary {
        Count {
            internal_count: 0,
            wrote_shared_count: false,
            siblings_left: self.siblings_left,
            shared_count: self.shared_count,
        }
    }
}

/// Your average, everyday `Count` (count(*)). This friendly operation
/// continuously counts incoming rows until `finish` time comes, where it synchronizes with its
/// fellow counts and the lucky leader gets to output.
pub struct Count {
    internal_count: usize,
    wrote_shared_count: bool,
    siblings_left: Arc<AtomicUsize>,
    shared_count: Arc<AtomicUsize>,
}

impl Unary<RecordBatch, RecordBatch> for Count {
    fn consume<OP: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        _output: &mut OP,
    ) -> unary::Result<()> {
        self.internal_count += batch.num_rows();
        Ok(())
    }

    fn finish<OP: Sender<RecordBatch>>(&mut self, output: &mut OP) -> unary::Result<bool> {
        if self.wrote_shared_count {
            return Ok(true);
        }

        self.shared_count
            .fetch_add(self.internal_count, Ordering::SeqCst);
        self.wrote_shared_count = true;

        if self.siblings_left.fetch_sub(1, Ordering::SeqCst) == 1 {
            let array = UInt64Array::from(vec![self.shared_count.load(Ordering::SeqCst) as u64]);
            let schema = Arc::new(Schema::new(vec![Field::new(
                "count",
                DataType::UInt64,
                false,
            )]));
            let batch = RecordBatch::try_new(schema, vec![Arc::new(array)])?;
            output.send(batch)?;
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::unary::UnaryFactory;
    use crate::operations::unary::test_utils::{CollectSender, run_unary_to_completion};
    use arrow_array::UInt64Array;

    fn extract_count(batch: &RecordBatch) -> u64 {
        batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0)
    }

    fn make_batch(num_rows: usize) -> RecordBatch {
        let array = arrow_array::Int32Array::from(vec![0i32; num_rows]);
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)])),
            vec![Arc::new(array)],
        )
        .unwrap()
    }

    fn build_counts(n: usize) -> Vec<Count> {
        CountFactory::create_for_workers(n)
            .into_iter()
            .map(|f| f.build_unary())
            .collect()
    }

    #[test]
    fn single_worker_counts_rows() {
        let count = build_counts(1).pop().unwrap();
        let results = run_unary_to_completion(count, vec![make_batch(3), make_batch(7)]);

        assert_eq!(results.len(), 1);
        assert_eq!(extract_count(&results[0]), 10);
    }

    #[test]
    fn multiple_workers_sum_counts() {
        let mut counts = build_counts(3);
        let mut sender = CollectSender::new();

        counts[0].consume(make_batch(2), &mut sender).unwrap();
        counts[1].consume(make_batch(5), &mut sender).unwrap();
        counts[2].consume(make_batch(3), &mut sender).unwrap();

        // First two workers finish — no output yet.
        counts[0].finish(&mut sender).unwrap();
        assert!(sender.items.is_empty());

        counts[1].finish(&mut sender).unwrap();
        assert!(sender.items.is_empty());

        // Last worker emits the total.
        counts[2].finish(&mut sender).unwrap();
        assert_eq!(sender.items.len(), 1);
        assert_eq!(extract_count(&sender.items[0]), 10);
    }

    #[test]
    fn zero_rows_produces_zero_count() {
        let count = build_counts(1).pop().unwrap();
        let results = run_unary_to_completion(count, vec![]);

        assert_eq!(results.len(), 1);
        assert_eq!(extract_count(&results[0]), 0);
    }

    #[test]
    fn finish_called_twice_does_not_double_count() {
        let mut counts = build_counts(2);
        let mut sender = CollectSender::new();

        counts[0].consume(make_batch(5), &mut sender).unwrap();
        counts[1].consume(make_batch(3), &mut sender).unwrap();

        // Worker 0 finishes twice (the bug: second call re-adds internal_count).
        counts[0].finish(&mut sender).unwrap();
        counts[0].finish(&mut sender).unwrap();

        // Worker 1 is the last to finish and emits the total.
        counts[1].finish(&mut sender).unwrap();
        assert_eq!(sender.items.len(), 1);
        assert_eq!(extract_count(&sender.items[0]), 8); // not 13
    }
}
