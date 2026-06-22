//! [`Limit`] — a bare `LIMIT … OFFSET …` with no ORDER BY.
//!
//! DuckDB fuses an `ORDER BY … LIMIT` into a [`TopN`](crate::operator::TopN); a
//! `LIMIT` that reaches here therefore has no ordering, so it keeps an arbitrary
//! `limit` rows after skipping `offset` (SQL leaves which rows undefined).

use crate::compile::Error;
use dispatch::RecordBatchOperatorSpec;
use duckdb_planner::operator as duckdb_operator;
use std::fmt;

/// `LIMIT … OFFSET …` with no ORDER BY.
#[derive(Debug)]
pub struct Limit {
    /// `None` for an offset-only query (`OFFSET n` with no upper bound).
    pub limit: Option<usize>,
    pub offset: usize,
}

impl TryFrom<duckdb_operator::Limit> for Limit {
    type Error = super::Error;
    fn try_from(l: duckdb_operator::Limit) -> Result<Self, Self::Error> {
        Ok(Limit {
            limit: l.limit,
            offset: l.offset.unwrap_or(0),
        })
    }
}

impl fmt::Display for Limit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let limit = self
            .limit
            .map(|n| n.to_string())
            .unwrap_or_else(|| "ALL".to_string());
        write!(f, "Limit(limit: {limit}, offset: {})", self.offset)
    }
}

impl Limit {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // An absent upper bound (`OFFSET` only) keeps everything past the offset.
        // The dispatch limit operator takes a concrete row count, so model
        // "unbounded" as keeping every remaining row (usize::MAX - offset can't
        // overflow the slice math since the input never has usize::MAX rows).
        let limit = self.limit.unwrap_or(usize::MAX - self.offset);
        Ok(input.limit(limit, self.offset))
    }
}
