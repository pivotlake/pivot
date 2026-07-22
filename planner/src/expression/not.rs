//! [`Not`] — logical negation (`NOT expr`).

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow::compute::kernels::boolean::not;
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, RecordBatch};
use std::fmt::{self, Display};
use std::sync::Arc;

/// Logical negation (`NOT expr`). DuckDB lowers it as a
/// `BoundOperatorExpression` of type `OPERATOR_NOT` with a single child.
#[derive(Debug, Clone)]
pub struct Not {
    pub input: Box<Expression>,
}

impl Display for Not {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NOT({})", self.input)
    }
}

impl Not {
    pub fn compile(&self, parameters: &compile::BoundParameters) -> Result<ExprFn, compile::Error> {
        let input_builder = self.input.compile(parameters)?;
        Ok(Box::new(move || {
            let mut input_expr = input_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (arr, _) = input.as_datum().get();
                ExprResult::Array(Arc::new(not(arr.as_boolean()).unwrap()) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn negates_predicate(mut testing_planner: TestingPlanner) {
        // Names containing 'a' are alice/charlie/dave; NOT leaves only bob.
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE NOT contains(name, 'a')",
        );

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["name"], "bob");
    }
}
