//! [`CreateSchema`]: `CREATE SCHEMA`, a namespace for tables inside a datastore.

use crate::catalog::{CatalogTransaction, CreateSchemaRequest};
use crate::compile::Error;
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use std::fmt;

/// CREATE SCHEMA.
#[derive(Debug)]
pub struct CreateSchema {
    pub request: CreateSchemaRequest,
    pub or_replace: bool,
}

impl fmt::Display for CreateSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CreateSchema({})", self.request.name)
    }
}

impl CreateSchema {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        transaction: &dyn CatalogTransaction,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        if self.or_replace {
            return Err(Error::UnsupportedCreateSchemaOrReplace);
        }

        let creation = transaction
            .bind_create_schema(self.request.clone())
            .map_err(Error::CreateSchema)?;
        creation.compile(dispatcher).map_err(Error::CreateSchema)
    }
}
