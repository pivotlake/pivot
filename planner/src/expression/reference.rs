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
    /// The column's source name from DuckDB's binding, when known. Display-only;
    /// `None` for synthesized references, which fall back to the `#idx` form.
    pub name: Option<String>,
}

impl TryFrom<duckdb_expression::Ref> for Ref {
    type Error = Error;
    fn try_from(r: duckdb_expression::Ref) -> Result<Self, Self::Error> {
        Ok(Ref {
            column_idx: r.column_idx,
            return_type: type_from_logical(r.return_type)?,
            // Drop DuckDB's positional aliases (e.g. "0" for an unnamed computed
            // group key): an all-digit name carries no more than the index does
            // and reads as a constant in a plan dump, so fall back to `#idx`.
            name: r
                .name
                .filter(|n| !n.is_empty() && !n.bytes().all(|b| b.is_ascii_digit())),
        })
    }
}

impl Display for Ref {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.name {
            Some(name) => write!(f, "{name}:{}", self.return_type),
            None => write!(f, "#{}:{}", self.column_idx, self.return_type),
        }
    }
}

impl Ref {
    /// The reference shown without its type: the source name when known, else
    /// the positional `#idx` form. Used where a bare column reference reads
    /// better than a full `name:Type` (e.g. inside an aggregate call).
    pub fn name_or_index(&self) -> String {
        match &self.name {
            Some(name) => name.clone(),
            None => format!("#{}", self.column_idx),
        }
    }

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
