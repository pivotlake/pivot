use crate::io::OperationIOSubmitter;
use crate::{ConsumeContext, Operation, Output, PipelineBreaker};
use arrow_array::RecordBatch;
use tracing::debug;

pub struct OutputOperation {
    output: Box<dyn Output>,
}

impl OutputOperation {
    pub fn new(output: Box<dyn Output>) -> Self {
        Self { output }
    }
}
impl Operation for OutputOperation {
    fn consume(
        &mut self,
        _: &ConsumeContext,
        _: OperationIOSubmitter,
        batch: &RecordBatch,
    ) -> super::Result<Option<RecordBatch>> {
        self.output.write(batch.clone());
        Ok(None)
    }
}

impl PipelineBreaker for OutputOperation {
    fn finish(mut self: Box<Self>) -> super::Result<()> {
        debug!("Finishing output op");
        self.output.finish();
        Ok(())
    }
}
