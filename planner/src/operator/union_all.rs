//! [`UnionAll`] — the rows of every input, concatenated.
//!
//! Lowered from DuckDB's `UNION` set operation, whose children all emit the
//! union's column list (the binder casts each side to the common types). A
//! plain `UNION` arrives as a [`Distinct`](super::Distinct) above this node,
//! so the node itself always keeps every row. Rows come out in whatever order
//! the inputs produce them.

use crate::compile::Error;
use dispatch::RecordBatchOperatorSpec;
use std::fmt;

/// Concatenation of its inputs' rows, column for column.
#[derive(Debug)]
pub struct UnionAll;

impl fmt::Display for UnionAll {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("UnionAll")
    }
}

impl UnionAll {
    /// Lower onto the compiled inputs, in input order. `nullable` is the
    /// union's own output nullability: whether any input can hold a NULL in
    /// each column.
    pub(crate) fn compile(
        &self,
        mut inputs: Vec<RecordBatchOperatorSpec>,
        nullable: Vec<bool>,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let first = inputs.remove(0);
        Ok(first.union_all(inputs, nullable))
    }
}
