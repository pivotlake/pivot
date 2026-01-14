use crate::api::{Chain, OperatorFactory};
use crate::operations::Sender;
use crate::table::input::{RowGroupBuffer, TableInput};
use crate::table::{Projection, Table, TableSource};
use std::sync::Arc;

pub struct TableInputFactory {
    source: Arc<TableSource>,
    table: Arc<Table>,
    projection: Option<Projection>,
}

impl TableInputFactory {
    pub fn new(
        source: Arc<TableSource>,
        table: Arc<Table>,
        projection: Option<Projection>,
    ) -> Self {
        Self {
            source,
            table,
            projection,
        }
    }
}

impl OperatorFactory<RowGroupBuffer> for TableInputFactory {
    fn build<S: Sender<RowGroupBuffer> + 'static>(self: Box<Self>, sender: S) -> Chain {
        Chain::root(Box::new(TableInput::new(
            sender,
            self.source,
            self.table,
            self.projection,
        )))
    }
}
