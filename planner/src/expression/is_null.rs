//! [`IsNull`]: `expr IS NULL` / `expr IS NOT NULL`.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow::compute::kernels::boolean::{is_not_null, is_null};
use arrow_array::{ArrayRef, BooleanArray, RecordBatch};
use arrow_schema::ArrowError;
use std::fmt::{self, Display};
use std::sync::Arc;

/// A NULL test. DuckDB lowers both forms as a `BoundOperatorExpression`
/// (`OPERATOR_IS_NULL` / `OPERATOR_IS_NOT_NULL`) with a single child;
/// `negated` distinguishes them. Unlike every other predicate, the result mask
/// is never NULL itself: a NULL input row yields `false`/`true`, not NULL.
#[derive(Debug, Clone)]
pub struct IsNull {
    pub negated: bool,
    pub input: Box<Expression>,
}

impl Display for IsNull {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = if self.negated {
            "IS NOT NULL"
        } else {
            "IS NULL"
        };
        write!(f, "({} {op})", self.input)
    }
}

impl IsNull {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        type NullKernel = fn(&dyn arrow_array::Array) -> Result<BooleanArray, ArrowError>;
        let kernel: NullKernel = if self.negated { is_not_null } else { is_null };
        let input_builder = self.input.compile()?;
        Ok(Box::new(move || {
            let mut input_expr = input_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (arr, _) = input.as_datum().get();
                ExprResult::Array(Arc::new(kernel(arr).unwrap()) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn is_null_keeps_only_null_rows(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT a FROM nullable_table WHERE b IS NULL",
        );

        assert_eq!(rows.len(), 3);
    }

    #[rstest]
    fn is_not_null_drops_null_rows(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT a FROM nullable_table WHERE b IS NOT NULL",
        );

        assert_eq!(rows.len(), 3);
    }
}
