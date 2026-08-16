//! [`DropSchema`]: `DROP SCHEMA`, removing a schema from its datastore's
//! catalog.

use crate::catalog::{CatalogTransaction, DropSchemaRequest};
use crate::compile::Error;
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use std::fmt;

/// DROP SCHEMA.
#[derive(Debug)]
pub struct DropSchema {
    pub request: DropSchemaRequest,
}

impl fmt::Display for DropSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DropSchema({})", self.request.name)
    }
}

impl DropSchema {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        transaction: &dyn CatalogTransaction,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // The default schema is what unqualified table names resolve in, and
        // every datastore is required to define it, so no datastore may drop
        // it; rejected here once rather than in each datastore.
        if self.request.name == crate::DEFAULT_SCHEMA_NAME {
            return Err(Error::DropDefaultSchema(self.request.name.clone()));
        }

        let drop = transaction
            .bind_drop_schema(self.request.clone())
            .map_err(Error::DropSchema)?;
        drop.compile(dispatcher).map_err(Error::DropSchema)
    }
}
