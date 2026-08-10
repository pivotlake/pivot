//! [`Ref`] — a bound column reference.

use crate::compile::{self, ExprFn, ExprResult, stateless_expr};
use crate::types::Type;
use arrow_array::RecordBatch;
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

impl Display for Ref {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.name {
            Some(name) => write!(f, "{name}:{}", self.return_type),
            None => write!(f, "#{}:{}", self.column_idx, self.return_type),
        }
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
