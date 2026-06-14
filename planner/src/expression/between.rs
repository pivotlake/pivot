//! [`Between`] — a `BETWEEN` range test.

use super::shared::{CmpKernel, compare_coerced};
use super::{Error, Expression};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow::compute::kernels::boolean::and;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_ord::cmp::{gt, gt_eq, lt, lt_eq};
use duckdb_planner::expression as duckdb_expression;
use std::fmt::{self, Display};
use std::sync::Arc;

/// A `BETWEEN` range test (`input BETWEEN lower AND upper`). DuckDB folds
/// `x >= a AND x <= b` into this; we compile it back to the conjunction of two
/// comparisons (respecting the inclusive/exclusive flags).
#[derive(Debug, Clone)]
pub struct Between {
    pub input: Box<Expression>,
    pub lower: Box<Expression>,
    pub upper: Box<Expression>,
    pub lower_inclusive: bool,
    pub upper_inclusive: bool,
}

impl TryFrom<duckdb_expression::Between> for Between {
    type Error = Error;
    fn try_from(b: duckdb_expression::Between) -> Result<Self, Self::Error> {
        Ok(Between {
            input: Box::<Expression>::try_from(b.input)?,
            lower: Box::<Expression>::try_from(b.lower)?,
            upper: Box::<Expression>::try_from(b.upper)?,
            lower_inclusive: b.lower_inclusive,
            upper_inclusive: b.upper_inclusive,
        })
    }
}

impl Display for Between {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} BETWEEN {} AND {}",
            self.input, self.lower, self.upper
        )
    }
}

impl Between {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        // `input BETWEEN lower AND upper` == `input >= lower AND input <= upper`
        // (the inclusive/exclusive flags swap >=/> and <=/<).
        let lower_kernel: CmpKernel = if self.lower_inclusive { gt_eq } else { gt };
        let upper_kernel: CmpKernel = if self.upper_inclusive { lt_eq } else { lt };
        let input_builder = self.input.compile()?;
        let lower_builder = self.lower.compile()?;
        let upper_builder = self.upper.compile()?;
        Ok(Box::new(move || {
            let mut input_expr = input_builder();
            let mut lower_expr = lower_builder();
            let mut upper_expr = upper_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let lower = lower_expr(batch);
                let upper = upper_expr(batch);
                let ge = compare_coerced(input.as_datum(), lower.as_datum(), lower_kernel);
                let le = compare_coerced(input.as_datum(), upper.as_datum(), upper_kernel);
                ExprResult::Array(Arc::new(and(&ge, &le).unwrap()) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn filters_inclusive_range(mut testing_planner: TestingPlanner) {
        let mut rows = run(
            &mut testing_planner,
            "SELECT a FROM example_table WHERE a BETWEEN 2 AND 4",
        );

        rows.sort_by_key(|r| r["a"].as_i64().unwrap());

        assert_eq!(
            rows.iter()
                .map(|r| r["a"].as_i64().unwrap())
                .collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
    }
}
