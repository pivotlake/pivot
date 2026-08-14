//! [`DropTable`]: remove one table from a datastore catalog.

use crate::catalog::{CatalogTransaction, DropTableRequest};
use crate::compile::Error;
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use std::fmt;

/// `DROP TABLE [IF EXISTS] name [CASCADE | RESTRICT]`.
#[derive(Debug)]
pub struct DropTable {
    pub request: DropTableRequest,
}

impl fmt::Display for DropTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DropTable({}, if_exists: {}, cascade: {})",
            self.request.schema_qualified_name(),
            self.request.if_exists,
            self.request.cascade,
        )
    }
}

impl DropTable {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        transaction: &dyn CatalogTransaction,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let table_drop = transaction
            .bind_drop_table(self.request.clone())
            .map_err(Error::DropTable)?;
        table_drop.compile(dispatcher).map_err(Error::DropTable)
    }
}
