//! [`Contains`] — SQL `contains(haystack, needle)` substring search.

use super::{Error, Expression};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult, SelectedEvalFn, SelectedFn};
use crate::types::Type;
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, BooleanArray, Datum, RecordBatch};
use dispatch::Contains as DispatchContains;
use duckdb_planner::expression as duckdb_expression;
use std::fmt::{self, Display};
use std::sync::Arc;

/// SQL `contains(haystack, needle)`.
#[derive(Debug, Clone)]
pub struct Contains {
    pub needle: Box<Expression>,
    pub haystack: Box<Expression>,
}

impl TryFrom<duckdb_expression::Function> for Contains {
    type Error = Error;
    fn try_from(mut f: duckdb_expression::Function) -> Result<Self, Self::Error> {
        if f.params.len() != 2 {
            let actual = f.params.len();
            return Err(Error::InvalidParameterCount {
                function: f.function,
                expected: 2,
                actual,
            });
        }
        let needle = Box::new(Expression::try_from(f.params.remove(1))?);
        let haystack = Box::new(Expression::try_from(f.params.remove(0))?);
        Ok(Contains { needle, haystack })
    }
}

impl Display for Contains {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "contains({}, {})", self.haystack, self.needle)
    }
}

impl Contains {
    /// Validate the haystack (a `Utf8` column reference) and extract the
    /// constant needle, returning the haystack's builder and the needle string.
    /// Shared by [`compile`](Self::compile) and
    /// [`compile_selected`](Self::compile_selected).
    fn validated_parts(&self) -> Result<(ExprFn, String), compile::Error> {
        match self.haystack.as_ref() {
            Expression::Ref(r) if r.return_type == Type::Utf8 => {}
            expr => {
                return Err(compile::Error::UnsupportedExpressionForContainsHaystack(
                    expr.clone(),
                ));
            }
        }

        let haystack_builder = self.haystack.compile()?;

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

        Ok((haystack_builder, needle_str))
    }

    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let (haystack_builder, needle_str) = self.validated_parts()?;
        Ok(Box::new(move || {
            let mut haystack_expr = haystack_builder();
            let mut contains = DispatchContains::new(&needle_str);
            Box::new(move |batch: &RecordBatch| {
                let haystack = haystack_expr(batch);
                let (arr, _) = haystack.as_datum().get();
                let col = arr.as_string_view();
                ExprResult::Array(Arc::new(contains.run(col)) as ArrayRef)
            }) as ExprEvalFn
        }))
    }

    /// Selection-aware variant of [`compile`](Self::compile): the substring scan
    /// runs only over the rows set in the filter mask, returning `selection AND
    /// contains`. A `None` selection scans every row (no mask yet). See
    /// [`SelectedFn`].
    pub fn compile_selected(&self) -> Result<SelectedFn, compile::Error> {
        let (haystack_builder, needle_str) = self.validated_parts()?;
        Ok(Box::new(move || {
            let mut haystack_expr = haystack_builder();
            let mut contains = DispatchContains::new(&needle_str);
            Box::new(move |batch: &RecordBatch, selection: Option<&BooleanArray>| {
                let haystack = haystack_expr(batch);
                let (arr, _) = haystack.as_datum().get();
                let col = arr.as_string_view();
                match selection {
                    Some(selection) => contains.run_selected(col, selection),
                    None => contains.run(col),
                }
            }) as SelectedEvalFn
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
