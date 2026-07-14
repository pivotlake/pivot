//! Temporary INSERT compatibility operator for Kafka Connect integration.
//!
//! DuckDB still binds the complete statement (target table, column names, and
//! VALUES types), but this operator intentionally discards the values. It
//! exists only so the pgwire/JDBC/transaction integration can be completed and
//! tested independently of the real transactional writer being developed on a
//! separate branch.

use crate::compile::Error;
use arrow_array::RecordBatch;
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use std::fmt;

#[derive(Debug)]
pub struct FakeInsert {
    pub table: String,
    pub affected_rows: usize,
}

impl fmt::Display for FakeInsert {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "FakeInsert({}, rows: {})",
            self.table, self.affected_rows
        )
    }
}

impl FakeInsert {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        Ok(dispatch::values_input(dispatcher, Vec::<RecordBatch>::new()).record_batches())
    }
}
