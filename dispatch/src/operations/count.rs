use crate::io::OperationIOSubmitter;
use crate::operations::Operation;
use crate::{ConsumeContext, Output, PipelineBreaker};
use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};

/// You're average, every day `Count` (count(*)). This friendly operation (and pipeline breaker!)
/// continuously counts incoming rows until `output` time comes, where it synchronizes with it's
/// fellow counts and the lucky leader gets to output.
pub struct Count {
    internal_count: usize,
    shared_count: Arc<AtomicUsize>,
    barrier: Arc<Barrier>,
    output: Box<dyn Output>,
}

impl Count {
    pub fn new(
        shared_count: Arc<AtomicUsize>,
        barrier: Arc<Barrier>,
        output: Box<dyn Output>,
    ) -> Self {
        Self {
            internal_count: 0,
            shared_count,
            barrier,
            output,
        }
    }
}
impl Operation for Count {
    fn consume(
        &mut self,
        _: &ConsumeContext,
        _: OperationIOSubmitter,
        batch: &RecordBatch,
    ) -> super::Result<Option<RecordBatch>> {
        self.internal_count += batch.num_rows();
        Ok(None)
    }
}

impl PipelineBreaker for Count {
    fn output(mut self: Box<Self>) -> super::Result<()> {
        self.shared_count
            .fetch_add(self.internal_count, Ordering::Relaxed);
        if self.barrier.wait().is_leader() {
            let array = UInt64Array::from(vec![self.shared_count.load(Ordering::Relaxed) as u64]);
            let schema = Arc::new(Schema::new(vec![Field::new(
                "count",
                DataType::UInt64,
                false,
            )]));
            let batch = RecordBatch::try_new(schema, vec![Arc::new(array)])?;
            self.output.write(batch);
        }
        self.output.finish();
        Ok(())
    }
}
