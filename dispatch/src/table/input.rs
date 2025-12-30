use crate::input::Input;
use crate::table::RowGroupMetadataHandle;
use crate::table::source::TableSource;
use parquetd::Projection;
use std::sync::Arc;

pub struct TableInput {
    source: Arc<TableSource>,
    projection: Option<Projection>,
}

impl TableInput {
    pub fn new(source: Arc<TableSource>, projection: Option<Projection>) -> Self {
        Self { source, projection }
    }
}

impl Input for TableInput {
    fn source_finished(&self) -> bool {
        self.source.is_empty()
    }

    fn poll_io(&self) -> Option<(RowGroupMetadataHandle, Option<Projection>)> {
        Some((self.source.pop_row_group()?, self.projection.clone()))
    }
}
