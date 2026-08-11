//! [`Distinct`] — the distinct values of a tuple of input columns.
//!
//! Emits each distinct key tuple once, as the key columns alone, in `keys`
//! order. Synthesized by the delim join translation as the
//! duplicate-elimination stage feeding the subquery side's delim scans; there
//! is no user-facing operator that lowers to it directly (DuckDB plans
//! `SELECT DISTINCT` as an aggregate with no aggregates).

use crate::compile::Error;
use crate::operator::aggregate::grouped::build_dedup_operator;
use crate::types::Type;
use dispatch::RecordBatchOperatorSpec;
use std::fmt;

/// A keys-only dedup: `GROUP BY keys` emitting the keys themselves.
#[derive(Debug)]
pub struct Distinct {
    /// The deduplicated columns: each an index into the input's output plus
    /// its type, in output order.
    pub keys: Vec<(usize, Type)>,
}

impl fmt::Display for Distinct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let columns: Vec<String> = self.keys.iter().map(|(idx, _)| idx.to_string()).collect();
        write!(f, "Distinct(keys: [{}])", columns.join(", "))
    }
}

impl Distinct {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
        input_nullability: Vec<bool>,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        build_dedup_operator(input, &self.keys, &input_nullability)
    }
}
