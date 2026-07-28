//! [`Conjunction`] — a boolean `AND`/`OR` over child predicates.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow::compute::kernels::boolean::{and_kleene, or_kleene};
use arrow_array::{ArrayRef, BooleanArray, RecordBatch};
use arrow_schema::ArrowError;
use std::fmt::{self, Display};
use std::sync::Arc;

/// Whether a [`Conjunction`] combines its children with boolean `AND` or `OR`.
#[derive(Debug, Clone, Copy)]
pub enum ConjunctionOp {
    And,
    Or,
}

/// A boolean `AND`/`OR` over two or more child predicates. DuckDB rewrites a
/// small `x IN (a, b)` into the `OR` form, so this is how most membership tests
/// reach the executor; it also covers any explicit `AND`/`OR` in a `WHERE`.
#[derive(Debug, Clone)]
pub struct Conjunction {
    pub op: ConjunctionOp,
    pub children: Vec<Expression>,
}

impl Display for Conjunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = match self.op {
            ConjunctionOp::And => "AND",
            ConjunctionOp::Or => "OR",
        };
        let parts: Vec<String> = self.children.iter().map(|p| p.to_string()).collect();
        write!(f, "({})", parts.join(&format!(" {op} ")))
    }
}

impl Conjunction {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        // Compile each child predicate once, then per batch reduce their boolean
        // masks with the conjunction's kernel (`AND`/`OR`). The Kleene kernels
        // keep SQL three-valued logic: `TRUE OR NULL` is `TRUE` and
        // `FALSE AND NULL` is `FALSE`, which the plain kernels would turn NULL.
        type BoolKernel = fn(&BooleanArray, &BooleanArray) -> Result<BooleanArray, ArrowError>;
        let kernel: BoolKernel = match self.op {
            ConjunctionOp::And => and_kleene,
            ConjunctionOp::Or => or_kleene,
        };
        let child_builders = self
            .children
            .iter()
            .map(|c| c.compile())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Box::new(move || {
            let mut child_exprs: Vec<ExprEvalFn> = child_builders.iter().map(|b| b()).collect();
            Box::new(move |batch: &RecordBatch| {
                let mask = child_exprs
                    .iter_mut()
                    .map(|c| {
                        let result = c(batch);
                        let (arr, _) = result.as_datum().get();
                        arr.as_any().downcast_ref::<BooleanArray>().unwrap().clone()
                    })
                    .reduce(|left, right| kernel(&left, &right).unwrap())
                    .expect("conjunction always has at least two children");
                ExprResult::Array(Arc::new(mask) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn true_or_null_keeps_the_row(mut testing_planner: TestingPlanner) {
        // Row a = 6 has s = 'z' but b NULL: `b = 10` is NULL there, and SQL's
        // `TRUE OR NULL` is TRUE, so the row must be kept.
        let mut rows = run(
            &mut testing_planner,
            "SELECT a FROM nullable_table WHERE s = 'z' OR b = 10",
        );

        rows.sort_by_key(|r| r["a"].as_i64().unwrap());
        assert_eq!(
            rows.iter()
                .map(|r| r["a"].as_i64().unwrap())
                .collect::<Vec<_>>(),
            vec![1, 6]
        );
    }

    #[rstest]
    fn or_of_two_equalities(mut testing_planner: TestingPlanner) {
        let mut rows = run(
            &mut testing_planner,
            "SELECT a FROM example_table WHERE a = 1 OR a = 5",
        );

        rows.sort_by_key(|r| r["a"].as_i64().unwrap());

        assert_eq!(
            rows.iter()
                .map(|r| r["a"].as_i64().unwrap())
                .collect::<Vec<_>>(),
            vec![1, 5]
        );
    }
}
