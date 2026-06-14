//! [`Ref`] — a bound column reference.

use super::Error;
use crate::compile::{self, ExprFn, ExprResult, stateless_expr};
use crate::types::{Type, type_from_logical};
use arrow_array::RecordBatch;
use duckdb_planner::expression as duckdb_expression;
use std::fmt::{self, Display};

/// A bound column reference (points at a column by index in the input).
#[derive(Debug, Clone)]
pub struct Ref {
    pub column_idx: usize,
    pub return_type: Type,
}

impl TryFrom<duckdb_expression::Ref> for Ref {
    type Error = Error;
    fn try_from(r: duckdb_expression::Ref) -> Result<Self, Self::Error> {
        Ok(Ref {
            column_idx: r.column_idx,
            return_type: type_from_logical(r.return_type)?,
        })
    }
}

impl Display for Ref {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}:{}", self.column_idx, self.return_type)
    }
}

impl Ref {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let column_idx = self.column_idx;
        Ok(stateless_expr(move |batch: &RecordBatch| {
            ExprResult::Array(batch.column(column_idx).clone())
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn projects_column(mut testing_planner: TestingPlanner) {
        let mut rows = run(&mut testing_planner, "SELECT a FROM example_table");

        rows.sort_by_key(|r| r["a"].as_i64().unwrap());

        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0]["a"], 1);
        assert_eq!(rows[4]["a"], 5);
    }
}
