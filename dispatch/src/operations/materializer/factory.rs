use crate::api::{Chain, OperatorFactory};
use crate::operations::ChannelFactory;
use crate::operations::channels::Sender;
use crate::operations::unary::UnaryFactory;
use crate::table::input::RowGroupBuffer;
use crate::table::{Projection, Table};
use arrow_array::RecordBatch;
use std::sync::Arc;

use super::{MaterializeJobGenerator, MaterializeRequest, Materializer};

pub struct MaterializeJobGeneratorFactory {
    projection: Option<Projection>,
    table: Arc<Table>,
}

impl MaterializeJobGeneratorFactory {
    pub fn new(projection: Option<Projection>, table: Arc<Table>) -> Self {
        Self { projection, table }
    }
}

impl UnaryFactory<RecordBatch, MaterializeRequest> for MaterializeJobGeneratorFactory {
    type Unary = MaterializeJobGenerator;

    fn build_unary(self) -> MaterializeJobGenerator {
        MaterializeJobGenerator::new(self.projection, self.table)
    }
}

pub struct MaterializerFactory<
    OF: OperatorFactory<MaterializeRequest>,
    C: ChannelFactory<MaterializeRequest>,
> {
    head: OF,
    channel_factory: C,
}

impl<OF: OperatorFactory<MaterializeRequest>, C: ChannelFactory<MaterializeRequest>>
    MaterializerFactory<OF, C>
{
    pub fn new(head: OF, channel_factory: C) -> Self {
        Self {
            head,
            channel_factory,
        }
    }
}

impl<OF: OperatorFactory<MaterializeRequest>, C: ChannelFactory<MaterializeRequest>>
    OperatorFactory<RowGroupBuffer> for MaterializerFactory<OF, C>
{
    fn build<S: Sender<RowGroupBuffer> + 'static>(self: Box<Self>, sender: S) -> Chain {
        let (tx, rx) = self.channel_factory.build();
        let chain = Box::new(self.head).build(tx);
        chain.with(Box::new(Materializer::new(rx, sender)))
    }
}
