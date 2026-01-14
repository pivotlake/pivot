use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::{Unary, UnaryFactory};
use arrow::compute::filter_record_batch;
use arrow_array::{BooleanArray, RecordBatch};

pub struct FilterFactory<F>(pub F);

impl<F> UnaryFactory<RecordBatch, RecordBatch> for FilterFactory<F>
where
    F: FnMut(&RecordBatch) -> BooleanArray + Send + 'static,
{
    type Unary = Filter<F>;

    fn build_unary(self) -> Self::Unary {
        Filter { func: self.0 }
    }
}

pub struct Filter<F>
where
    F: FnMut(&RecordBatch) -> BooleanArray + Send,
{
    func: F,
}

impl<F> Unary<RecordBatch, RecordBatch> for Filter<F>
where
    F: FnMut(&RecordBatch) -> BooleanArray + Send,
{
    fn consume<OP: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        output: &mut OP,
    ) -> unary::Result<()> {
        let mask = (self.func)(&batch);
        debug_assert_eq!(mask.len(), batch.num_rows());

        if mask.true_count() == 0 {
            return Ok(());
        }
        output.send(filter_record_batch(&batch, &mask)?)?;
        Ok(())
    }
}
