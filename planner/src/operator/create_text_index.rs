//! [`CreateTextIndex`]: `CREATE TEXT INDEX`, an index of the tokens of a text
//! column that the table's files are written with from then on.

use crate::catalog::{CatalogTransaction, CreateTextIndexRequest};
use crate::compile::Error;
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use std::fmt;

/// `CREATE TEXT INDEX [IF NOT EXISTS] <name> ON <table> (<column>)
/// [WITH (case_sensitive = true)]`.
///
/// A statement, not a query: it produces no rows. Like a `CREATE SCHEMA`, it
/// compiles through the transaction into a dataflow that only stages the
/// creation; the index is durably added when the transaction commits.
#[derive(Debug, PartialEq, Eq)]
pub struct CreateTextIndex {
    pub request: CreateTextIndexRequest,
}

impl CreateTextIndex {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        transaction: &dyn CatalogTransaction,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let creation = transaction
            .bind_create_text_index(self.request.clone())
            .map_err(Error::CreateTextIndex)?;
        creation.compile(dispatcher).map_err(Error::CreateTextIndex)
    }
}

impl fmt::Display for CreateTextIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CreateTextIndex({} ON {}.{}.{}({}))",
            self.request.index_name,
            self.request.datastore_name,
            self.request.schema_name,
            self.request.table_name,
            self.request.column
        )
    }
}
