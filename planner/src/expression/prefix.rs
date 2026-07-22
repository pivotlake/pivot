//! [`Prefix`] — SQL `prefix(string, search)` starts-with matching, the function
//! DuckDB's optimizer rewrites `LIKE 'foo%'` patterns into.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::Type;
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, Datum, RecordBatch};
use dispatch::Prefix as DispatchPrefix;
use std::fmt::{self, Display};
use std::sync::Arc;

/// SQL `prefix(haystack, prefix)`.
#[derive(Debug, Clone)]
pub struct Prefix {
    pub haystack: Box<Expression>,
    pub prefix: Box<Expression>,
}

impl Display for Prefix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "prefix({}, {})", self.haystack, self.prefix)
    }
}

impl Prefix {
    pub fn compile(&self, parameters: &compile::BoundParameters) -> Result<ExprFn, compile::Error> {
        match self.haystack.as_ref() {
            Expression::Ref(r) if r.return_type == Type::Utf8 => {}
            expr => {
                return Err(compile::Error::UnsupportedExpressionForPrefixHaystack(
                    expr.clone(),
                ));
            }
        }

        let haystack_builder = self.haystack.compile(parameters)?;

        // Extract the prefix string from the constant expression
        let prefix_str: String = match self.prefix.as_ref() {
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
                return Err(compile::Error::UnsupportedExpressionForPrefixPattern(
                    expr.clone(),
                ));
            }
        };
        Ok(Box::new(move || {
            let mut haystack_expr = haystack_builder();
            let prefix = DispatchPrefix::new(&prefix_str);
            Box::new(move |batch: &RecordBatch| {
                let haystack = haystack_expr(batch);
                let (arr, _) = haystack.as_datum().get();
                let col = arr
                    .as_any()
                    .downcast_ref::<arrow_array::StringViewArray>()
                    .unwrap();
                ExprResult::Array(Arc::new(prefix.run(col)) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn filters_by_prefix_function(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE prefix(name, 'ali')",
        );

        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r["name"] == "alice"));
    }

    #[rstest]
    fn filters_by_like_prefix_pattern(mut testing_planner: TestingPlanner) {
        // DuckDB's optimizer rewrites a trailing-% LIKE into prefix().
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE name LIKE 'ali%'",
        );

        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r["name"] == "alice"));
    }
}
