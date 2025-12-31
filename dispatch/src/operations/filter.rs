use crate::ConsumeContext;
use crate::io::OperationIOSubmitter;
use crate::operations::Operation;
use arrow::compute::filter_record_batch;
use arrow_array::{BooleanArray, RecordBatch};

pub struct Filter<F>
where
    F: Fn(&RecordBatch) -> BooleanArray + Send + Sync + 'static,
{
    func: F,
}

impl<F> Filter<F>
where
    F: Fn(&RecordBatch) -> BooleanArray + Send + Sync + 'static,
{
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F> Operation for Filter<F>
where
    F: Fn(&RecordBatch) -> BooleanArray + Send + Sync + 'static,
{
    fn consume(
        &mut self,
        _: &ConsumeContext,
        _: OperationIOSubmitter,
        batch: &RecordBatch,
    ) -> super::Result<Option<RecordBatch>> {
        let mask = (self.func)(batch);
        debug_assert_eq!(mask.len(), batch.num_rows());

        if mask.true_count() == 0 {
            return Ok(None);
        }
        Ok(Some(filter_record_batch(batch, &mask)?))
    }
}
