//! [`EmptyResult`] — a source that emits no rows.
//!
//! DuckDB's optimizer replaces a subtree it proves returns nothing (e.g. a
//! filter whose predicate folds to false, as when a NULL is compared) with
//! `LOGICAL_EMPTY_RESULT`, keeping only the subtree's output types. Those
//! types are carried here so passes over the plan can still ask what the
//! (empty) subtree produces; at run time it is a `VALUES` source with no rows.

use std::fmt;

use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};

use super::values::Values;
use crate::compile::Error;
use crate::types::Type;

/// A source that emits no rows, with the output types of the subtree it
/// replaced.
#[derive(Debug)]
pub struct EmptyResult {
    pub types: Vec<Type>,
}

impl fmt::Display for EmptyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EmptyResult(columns: {})", self.types.len())
    }
}

impl EmptyResult {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        Values { rows: Vec::new() }.compile(dispatcher)
    }
}
