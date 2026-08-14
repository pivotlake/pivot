//! [`DropTable`]: `DROP TABLE`, removing a table from its datastore's catalog.

use crate::catalog::{CatalogTransaction, DropTableRequest};
use crate::compile::Error;
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use std::fmt;

/// DROP TABLE.
#[derive(Debug)]
pub struct DropTable {
    pub request: DropTableRequest,
    pub cascade: bool,
}

impl fmt::Display for DropTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DropTable({})", self.request.name)
    }
}

impl DropTable {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        transaction: &dyn CatalogTransaction,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // There are no dependent objects (views, foreign keys) to cascade over,
        // so accepting CASCADE would promise semantics that don't exist.
        if self.cascade {
            return Err(Error::UnsupportedDropTableCascade);
        }

        let drop = transaction
            .bind_drop_table(self.request.clone())
            .map_err(Error::DropTable)?;
        drop.compile(dispatcher).map_err(Error::DropTable)
    }
}
