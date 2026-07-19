//! `INSERT INTO table VALUES ...`.

use std::fmt;

use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};

use crate::catalog::{CatalogTransaction, Error as CatalogError};
use crate::compile::Error;

#[derive(Debug)]
pub struct Insert {
    pub table: String,
}

impl fmt::Display for Insert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Insert({})", self.table)
    }
}

impl Insert {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
        dispatcher: &DataFlowDispatcher,
        transaction: &dyn CatalogTransaction,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // Writes dispatch through the target table itself, mirroring how a read
        // compiles through `Table::compile_scan`.
        let table = transaction.table(&self.table).ok_or_else(|| {
            Error::Insert(CatalogError::Other(
                format!("cannot INSERT into unknown table {}", self.table).into(),
            ))
        })?;
        table
            .compile_insert(input, dispatcher, transaction)
            .map_err(Error::Insert)
    }
}
