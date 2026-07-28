//! `INSERT INTO table VALUES ...`.

use std::fmt;

use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};

use crate::catalog::{CatalogTransaction, Table};
use crate::compile::Error;

#[derive(Debug)]
pub struct Insert {
    pub table: Box<dyn Table>,
    /// The target's name, as the statement referenced it. Kept so the planner
    /// can bind the target's generated columns against it (see
    /// [`Planner::plan`](crate::Planner::plan)), which happens after the walk
    /// that builds this operator.
    pub table_name: String,
}

impl fmt::Display for Insert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Insert")
    }
}

impl Insert {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
        dispatcher: &DataFlowDispatcher,
        transaction: &dyn CatalogTransaction,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        self.table
            .compile_insert(input, dispatcher, transaction)
            .map_err(Error::Insert)
    }
}
