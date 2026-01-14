use crate::operations::channels::Sender;
use crate::operations::unary::{self, Unary, UnaryFactory};
use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::debug;

pub struct CountFactory {
    siblings_left: Arc<AtomicUsize>,
    shared_count: Arc<AtomicUsize>,
}

impl CountFactory {
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
            siblings_left: self.siblings_left,
            shared_count: self.shared_count,
        }
    }
}

/// You're average, every day `Count` (count(*)). This friendly operation (and pipeline breaker!)
/// continuously counts incoming rows until `finish` time comes, where it synchronizes with its
/// fellow counts and the lucky leader gets to output.
pub struct Count {
    internal_count: usize,
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
        self.shared_count
            .fetch_add(self.internal_count, Ordering::Relaxed);

        if self.siblings_left.fetch_sub(1, Ordering::AcqRel) == 1 {
            debug!("Sending coutn!!");
            let array = UInt64Array::from(vec![self.shared_count.load(Ordering::Relaxed) as u64]);
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
