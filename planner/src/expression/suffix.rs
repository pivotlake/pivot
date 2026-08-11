//! [`Suffix`] — SQL `suffix(string, search)` ends-with matching, the function
//! DuckDB's optimizer rewrites `LIKE '%foo'` patterns into.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::Type;
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, Datum, RecordBatch};
use dispatch::Suffix as DispatchSuffix;
use std::fmt::{self, Display};
use std::sync::Arc;

/// SQL `suffix(haystack, suffix)`.
#[derive(Debug, Clone)]
pub struct Suffix {
    pub haystack: Box<Expression>,
    pub suffix: Box<Expression>,
}

impl Display for Suffix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "suffix({}, {})", self.haystack, self.suffix)
    }
}

impl Suffix {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        match self.haystack.as_ref() {
            Expression::Ref(r) if r.return_type == Type::Utf8 => {}
            expr => {
                return Err(compile::Error::UnsupportedExpressionForSuffixHaystack(
                    expr.clone(),
                ));
            }
        }

        let haystack_builder = self.haystack.compile()?;

        // Extract the suffix string from the constant expression
        let suffix_str: String = match self.suffix.as_ref() {
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
                return Err(compile::Error::UnsupportedExpressionForSuffixPattern(
                    expr.clone(),
                ));
            }
        };
        Ok(Box::new(move || {
            let mut haystack_expr = haystack_builder();
            let suffix = DispatchSuffix::new(&suffix_str);
            Box::new(move |batch: &RecordBatch| {
                let haystack = haystack_expr(batch);
                let (arr, _) = haystack.as_datum().get();
                let col = arr
                    .as_any()
                    .downcast_ref::<arrow_array::StringViewArray>()
                    .unwrap();
                ExprResult::Array(Arc::new(suffix.run(col)) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn filters_by_suffix_function(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE suffix(name, 'ice')",
        );

        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r["name"] == "alice"));
    }

    #[rstest]
    fn filters_by_like_suffix_pattern(mut testing_planner: TestingPlanner) {
        // DuckDB's optimizer rewrites a leading-% LIKE into suffix().
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE name LIKE '%ice'",
        );

        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r["name"] == "alice"));
    }
}
