//! [`Contains`] — SQL `contains(haystack, needle)` substring search.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::Type;
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, Datum, RecordBatch};
use dispatch::Contains as DispatchContains;
use std::fmt::{self, Display};
use std::sync::Arc;

/// SQL `contains(haystack, needle)`.
#[derive(Debug, Clone)]
pub struct Contains {
    pub needle: Box<Expression>,
    pub haystack: Box<Expression>,
}

impl Display for Contains {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "contains({}, {})", self.haystack, self.needle)
    }
}

impl Contains {
    pub fn compile(&self, parameters: &compile::BoundParameters) -> Result<ExprFn, compile::Error> {
        match self.haystack.as_ref() {
            Expression::Ref(r) if r.return_type == Type::Utf8 => {}
            expr => {
                return Err(compile::Error::UnsupportedExpressionForContainsHaystack(
                    expr.clone(),
                ));
            }
        }

        let haystack_builder = self.haystack.compile(parameters)?;

        // Extract the needle string from the constant expression
        let needle_str: String = match self.needle.as_ref() {
            Expression::Constant(scalar) => {
                let (arr, _) = scalar.get();
                arr.as_string_view_opt()
                    .ok_or_else(|| {
                        compile::Error::FailedToDowncastScalarIntoString(scalar.clone())
                    })?
                    .value(0)
                    .to_string()
            }
            expr => {
                return Err(compile::Error::UnsupportedExpressionForContainsNeedle(
                    expr.clone(),
                ));
            }
        };
        Ok(Box::new(move || {
            let mut haystack_expr = haystack_builder();
            let mut contains = DispatchContains::new(&needle_str);
            Box::new(move |batch: &RecordBatch| {
                let haystack = haystack_expr(batch);
                let (arr, _) = haystack.as_datum().get();
                let col = arr
                    .as_any()
                    .downcast_ref::<arrow_array::StringViewArray>()
                    .unwrap();
                ExprResult::Array(Arc::new(contains.run(col)) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn filters_substring(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE contains(name, 'ali')",
        );

        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r["name"] == "alice"));
    }
}
