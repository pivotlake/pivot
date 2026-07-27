//! `INSERT INTO table VALUES ...`.

use std::fmt;

use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};

use crate::catalog::Table;
use crate::compile::Error;

#[derive(Debug)]
pub struct Insert {
    pub table: Box<dyn Table>,
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
    ) -> Result<RecordBatchOperatorSpec, Error> {
        self.table
            .compile_insert(input, dispatcher)
            .map_err(Error::Insert)
    }
}
