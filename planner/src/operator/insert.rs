//! `INSERT INTO table VALUES ...`.

use std::fmt;

use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};

use crate::catalog::{CatalogTransaction, Column, TableReference};
use crate::compile::Error;

/// Writes its input's rows into a table.
///
/// Holds the target's *name* and the column schema the statement was planned
/// against, not a binding: a binding is wired to one transaction's staging
/// (its uploaded-files injector) and one catalog snapshot, so a plan that
/// outlives its planning transaction (a cached prepared statement) cannot
/// carry one. Compile rebinds through the executing transaction instead, so
/// every execution stages its rows into its own transaction's commit.
#[derive(Debug)]
pub struct Insert {
    pub reference: TableReference,
    /// The target's columns as planned against, verified on rebind so a
    /// schema change between planning and execution fails loudly instead of
    /// writing misshapen rows.
    pub columns: Vec<Column>,
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
        let table = transaction
            .bind_table(&self.reference)
            .ok_or_else(|| Error::InsertTargetMissing(self.reference.clone()))?;
        if table.columns() != self.columns {
            return Err(Error::InsertTargetSchemaChanged(self.reference.clone()));
        }
        table
            .compile_insert(input, dispatcher)
            .map_err(Error::Insert)
    }
}
