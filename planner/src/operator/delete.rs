//! `DELETE FROM table [WHERE ...]`.

use std::fmt;

use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};

use crate::catalog::BoundTable;
use crate::compile::Error;
use crate::expression::Expression;

/// Deletes the rows of `table` every one of `conditions` is true of. It reads
/// no input: the table is handed the conditions and decides how the rows go.
#[derive(Debug)]
pub struct Delete {
    pub table: Box<dyn BoundTable>,
    /// What a row is deleted for, each over the table's columns: a
    /// reference's index is a position in the table's schema. Without any,
    /// every row is deleted.
    pub conditions: Vec<Expression>,
}

impl fmt::Display for Delete {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let conditions = self
            .conditions
            .iter()
            .map(|condition| condition.to_string())
            .collect::<Vec<_>>()
            .join(" AND ");
        write!(f, "Delete({conditions})")
    }
}

impl Delete {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        self.table
            .compile_delete(&self.conditions, dispatcher)
            .map_err(Error::Delete)
    }
}
