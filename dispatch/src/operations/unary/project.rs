use crate::operations::channels::Sender;
use crate::operations::unary::Unary;
use crate::operations::unary::factory::UnaryFactory;
use arrow_array::RecordBatch;

pub struct ProjectFactory<P>(pub P);

impl<P> UnaryFactory<RecordBatch, RecordBatch> for ProjectFactory<P>
where
    P: FnMut(&RecordBatch) -> RecordBatch + Send + 'static,
{
    type Unary = Project<P>;

    fn build_unary(self) -> Self::Unary {
        Project { projector: self.0 }
    }
}

pub struct Project<P: FnMut(&RecordBatch) -> RecordBatch + Send> {
    projector: P,
}

impl<P> Unary<RecordBatch, RecordBatch> for Project<P>
where
    P: FnMut(&RecordBatch) -> RecordBatch + Send,
{
    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        sender: &mut S,
    ) -> crate::operations::unary::Result<()> {
        sender.send((self.projector)(&batch))?;
        Ok(())
    }
}
