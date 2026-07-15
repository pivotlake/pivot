//! [`Like`] — SQL `LIKE` / `NOT LIKE` pattern matching (`%` and `_` wildcards).
//!
//! DuckDB rewrites the simple patterns (`'foo%'`, `'%foo%'`) into
//! `prefix`/`contains`; a multi-wildcard pattern such as `'%a%b%'` stays a
//! `~~` / `!~~` function call and lands here, evaluated by arrow's `like`
//! kernel against a constant pattern.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow_array::{ArrayRef, Datum, RecordBatch, Scalar};
use arrow_string::like::{like, nlike};
use std::fmt::{self, Display};
use std::sync::Arc;

/// SQL `haystack LIKE pattern` (`negated` for `NOT LIKE`).
#[derive(Debug, Clone)]
pub struct Like {
    pub haystack: Box<Expression>,
    pub pattern: Box<Expression>,
    pub negated: bool,
}

impl Display for Like {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = if self.negated { "NOT LIKE" } else { "LIKE" };
        write!(f, "{} {op} {}", self.haystack, self.pattern)
    }
}

impl Like {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let haystack_builder = self.haystack.compile()?;
        let pattern: Scalar<ArrayRef> = match self.pattern.as_ref() {
            Expression::Constant(scalar) => {
                let (arr, _) = scalar.get();
                Scalar::new(arrow_array::make_array(arrow_array::Array::to_data(arr)))
            }
            expr => {
                return Err(compile::Error::UnsupportedExpressionForLikePattern(
                    expr.clone(),
                ));
            }
        };
        let negated = self.negated;

        Ok(Box::new(move || {
            let mut haystack_expr = haystack_builder();
            let pattern = pattern.clone();
            Box::new(move |batch: &RecordBatch| {
                let haystack = haystack_expr(batch);
                let matched = if negated {
                    nlike(haystack.as_datum(), &pattern)
                } else {
                    like(haystack.as_datum(), &pattern)
                }
                .expect("LIKE kernel failed");
                ExprResult::Array(Arc::new(matched) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn filters_by_multi_wildcard_pattern(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE name LIKE '%al%ce%'",
        );

        assert!(!rows.is_empty());
        assert!(rows.iter().all(|r| r["name"] == "alice"));
    }

    #[rstest]
    fn negated_pattern_keeps_the_complement(mut testing_planner: TestingPlanner) {
        let all = run(&mut testing_planner, "SELECT name FROM example_table");
        let kept = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE name NOT LIKE '%al%ce%'",
        );

        assert!(kept.len() < all.len());
        assert!(kept.iter().all(|r| r["name"] != "alice"));
    }
}
