//! [`Concat`] — the SQL `left || right` string concatenation operator.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow_array::builder::StringViewBuilder;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch};
use std::fmt::{self, Display};
use std::sync::Arc;

/// SQL `left || right`, DuckDB's `BOUND_FUNCTION` named `||`.
///
/// Only the string form is accepted (DuckDB's binder casts the operands to
/// VARCHAR); `||` over blobs or lists is rejected at plan time. A NULL on
/// either side yields NULL, which is the operator's contract as opposed to
/// the variadic `concat(...)` function's treat-NULL-as-empty one.
#[derive(Debug, Clone)]
pub struct Concat {
    pub left: Box<Expression>,
    pub right: Box<Expression>,
}

impl Display for Concat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} || {})", self.left, self.right)
    }
}

impl Concat {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let left_builder = self.left.compile()?;
        let right_builder = self.right.compile()?;
        Ok(Box::new(move || {
            let mut left_expr = left_builder();
            let mut right_expr = right_builder();
            // One reused buffer joins each row's halves so the builder gets a
            // single contiguous value; no per-row allocation.
            let mut scratch = String::new();
            Box::new(move |batch: &RecordBatch| {
                let left = left_expr(batch);
                let right = right_expr(batch);
                let (left_arr, left_is_scalar) = left.as_datum().get();
                let (right_arr, right_is_scalar) = right.as_datum().get();
                let left_strings = left_arr.as_string_view();
                let right_strings = right_arr.as_string_view();
                let mut out = StringViewBuilder::with_capacity(batch.num_rows());
                for row in 0..batch.num_rows() {
                    let left_idx = if left_is_scalar { 0 } else { row };
                    let right_idx = if right_is_scalar { 0 } else { row };
                    if left_strings.is_null(left_idx) || right_strings.is_null(right_idx) {
                        out.append_null();
                        continue;
                    }
                    scratch.clear();
                    scratch.push_str(left_strings.value(left_idx));
                    scratch.push_str(right_strings.value(right_idx));
                    out.append_value(&scratch);
                }
                ExprResult::Array(Arc::new(out.finish()) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::{ArrayRef, StringViewArray};
    use rstest::rstest;
    use std::sync::Arc;

    fn people_table(testing_planner: &mut TestingPlanner) {
        testing_planner.add_table(
            "people",
            &[
                (
                    "first",
                    Type::Utf8,
                    Arc::new(StringViewArray::from(vec![
                        Some("ada"),
                        Some("brian"),
                        None,
                    ])) as ArrayRef,
                ),
                (
                    "last",
                    Type::Utf8,
                    Arc::new(StringViewArray::from(vec![
                        Some("lovelace"),
                        None,
                        Some("kernighan"),
                    ])) as ArrayRef,
                ),
            ],
        );
    }

    #[rstest]
    fn concatenates_a_constant_and_a_column(mut testing_planner: TestingPlanner) {
        people_table(&mut testing_planner);

        let rows = run(
            &mut testing_planner,
            "SELECT 'hi ' || first AS greeting FROM people WHERE first = 'ada'",
        );

        assert_eq!(
            rows,
            serde_json::json!([{"greeting": "hi ada"}])
                .as_array()
                .unwrap()
                .clone()
        );
    }

    #[rstest]
    fn concatenates_two_columns(mut testing_planner: TestingPlanner) {
        people_table(&mut testing_planner);

        let rows = run(
            &mut testing_planner,
            "SELECT first || ' ' || last AS full FROM people WHERE first = 'ada'",
        );

        assert_eq!(
            rows,
            serde_json::json!([{"full": "ada lovelace"}])
                .as_array()
                .unwrap()
                .clone()
        );
    }

    #[rstest]
    fn a_null_operand_yields_null(mut testing_planner: TestingPlanner) {
        people_table(&mut testing_planner);

        let rows = run(
            &mut testing_planner,
            "SELECT count(first || last) AS both_named FROM people",
        );

        assert_eq!(
            rows,
            serde_json::json!([{"both_named": 1}])
                .as_array()
                .unwrap()
                .clone()
        );
    }
}
