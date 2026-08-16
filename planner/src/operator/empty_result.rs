//! [`EmptyResult`], a source whose schema is known but which emits no rows.

use crate::compile::Error;
use crate::types::Type;
use arrow_array::RecordBatch;
use dispatch::{DataFlowDispatcher, OneShotNullaryFactory, RecordBatchOperatorSpec};
use std::fmt;

/// An input DuckDB proved empty during optimization.
#[derive(Debug)]
pub struct EmptyResult {
    pub output_types: Vec<Type>,
}

impl fmt::Display for EmptyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EmptyResult")
    }
}

impl EmptyResult {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        Ok(RecordBatchOperatorSpec::from_nullary(
            dispatcher,
            (0..dispatcher.worker_count())
                .map(|_| OneShotNullaryFactory::new(|| None::<RecordBatch>)),
        ))
    }
}
