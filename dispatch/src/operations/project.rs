use crate::io::OperationIOSubmitter;
use crate::operations::{ConsumeContext, Operation};
use arrow_array::RecordBatch;

pub struct Project<P: FnMut(&RecordBatch) -> RecordBatch + Send> {
    projector: P,
}

impl<P> Project<P>
where
    P: FnMut(&RecordBatch) -> RecordBatch + Send,
{
    pub fn new(projector: P) -> Self {
        Self { projector }
    }
}

impl<P> Operation for Project<P>
where
    P: FnMut(&RecordBatch) -> RecordBatch + Send,
{
    fn consume(
        &mut self,
        _: &ConsumeContext,
        _: OperationIOSubmitter,
        batch: &RecordBatch,
    ) -> super::Result<Option<RecordBatch>> {
        Ok(Some((self.projector)(batch)))
    }
}
