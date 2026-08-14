//! [`Compare`] — a binary comparison (`=`, `<>`, `<`, …) and its [`CompareType`].

use super::Error;
use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::Type;
use arrow_array::{ArrayRef, BooleanArray, Datum, RecordBatch};
use arrow_ord::cmp::{eq, gt, gt_eq, lt, lt_eq, neq};
use arrow_schema::ArrowError;
use duckdb_planner::duckdb_bridge::duckdb_types::ExpressionType;
use std::fmt::{self, Display};
use std::sync::Arc;

/// Signature shared by arrow's scalar comparison kernels, used here and by
/// [`Between`](super::Between).
pub(crate) type CmpKernel =
    fn(&dyn Datum, &dyn Datum) -> std::result::Result<BooleanArray, ArrowError>;

#[derive(Debug, Clone, Copy)]
pub enum CompareType {
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
}

impl TryFrom<ExpressionType> for CompareType {
    type Error = Error;
    fn try_from(c: ExpressionType) -> Result<Self, Self::Error> {
        match c {
            ExpressionType::COMPARE_EQUAL => Ok(CompareType::Equal),
            ExpressionType::COMPARE_NOTEQUAL => Ok(CompareType::NotEqual),
            ExpressionType::COMPARE_LESSTHAN => Ok(CompareType::Less),
            ExpressionType::COMPARE_GREATERTHAN => Ok(CompareType::Greater),
            ExpressionType::COMPARE_LESSTHANOREQUALTO => Ok(CompareType::LessEqual),
            ExpressionType::COMPARE_GREATERTHANOREQUALTO => Ok(CompareType::GreaterEqual),
            _ => Err(Error::UnsupportedComparisonType(c)),
        }
    }
}

impl Display for CompareType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompareType::Equal => f.write_str("="),
            CompareType::NotEqual => f.write_str("<>"),
            CompareType::Less => f.write_str("<"),
            CompareType::Greater => f.write_str(">"),
            CompareType::LessEqual => f.write_str("<="),
            CompareType::GreaterEqual => f.write_str(">="),
        }
    }
}

/// A binary comparison expression (e.g. `<>`, `=`).
#[derive(Debug, Clone)]
pub struct Compare {
    pub left: Box<Expression>,
    pub right: Box<Expression>,
    pub compare_type: CompareType,
    pub return_type: Type,
}

impl Display for Compare {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} {} -> {}",
            self.left, self.compare_type, self.right, self.return_type
        )
    }
}

impl Compare {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let kernel: CmpKernel = match self.compare_type {
            CompareType::Equal => eq,
            CompareType::NotEqual => neq,
            CompareType::Less => lt,
            CompareType::Greater => gt,
            CompareType::LessEqual => lt_eq,
            CompareType::GreaterEqual => gt_eq,
        };
        let left_builder = self.left.compile()?;
        let right_builder = self.right.compile()?;
        Ok(Box::new(move || {
            let mut left_expr = left_builder();
            let mut right_expr = right_builder();
            Box::new(move |batch: &RecordBatch| {
                let left = left_expr(batch);
                let right = right_expr(batch);
                let mask = kernel(left.as_datum(), right.as_datum())
                    .expect("comparison operands share a type");
                ExprResult::Array(Arc::new(mask) as ArrayRef)
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

    /// A string column mixing empty values, values that inline in a view, and
    /// values too long to, so one table exercises all three constant shapes.
    fn phrases_table(testing_planner: &mut TestingPlanner) {
        testing_planner.add_table(
            "phrases",
            &[(
                "s",
                Type::Utf8,
                Arc::new(StringViewArray::from(vec![
                    "",
                    "short",
                    "",
                    "a phrase far longer than inlines",
                    "a phrase far longer than inlinez",
                ])) as ArrayRef,
            )],
        );
    }

    fn phrase_rows(testing_planner: &mut TestingPlanner, sql: &str) -> Vec<String> {
        let mut rows: Vec<String> = run(testing_planner, sql)
            .iter()
            .map(|r| r["s"].as_str().unwrap().to_string())
            .collect();
        rows.sort();
        rows
    }

    #[rstest]
    fn filters_string_not_equal_to_empty(mut testing_planner: TestingPlanner) {
        phrases_table(&mut testing_planner);

        let rows = phrase_rows(&mut testing_planner, "SELECT s FROM phrases WHERE s <> ''");

        assert_eq!(
            rows,
            vec![
                "a phrase far longer than inlines",
                "a phrase far longer than inlinez",
                "short"
            ]
        );
    }

    #[rstest]
    fn filters_string_equal_to_empty(mut testing_planner: TestingPlanner) {
        phrases_table(&mut testing_planner);

        let rows = phrase_rows(&mut testing_planner, "SELECT s FROM phrases WHERE s = ''");

        assert_eq!(rows, vec!["", ""]);
    }

    #[rstest]
    fn filters_string_equal_to_an_inline_constant(mut testing_planner: TestingPlanner) {
        phrases_table(&mut testing_planner);

        let rows = phrase_rows(
            &mut testing_planner,
            "SELECT s FROM phrases WHERE s = 'short'",
        );

        assert_eq!(rows, vec!["short"]);
    }

    /// The two long values share every byte but the last, so a match decided on
    /// the view's length and four-byte prefix alone would take both.
    #[rstest]
    fn filters_string_equal_to_a_constant_past_the_inline_limit(
        mut testing_planner: TestingPlanner,
    ) {
        phrases_table(&mut testing_planner);

        let rows = phrase_rows(
            &mut testing_planner,
            "SELECT s FROM phrases WHERE s = 'a phrase far longer than inlines'",
        );

        assert_eq!(rows, vec!["a phrase far longer than inlines"]);
    }

    /// `s <> 'x'` is NULL, not true, on a NULL row, so the filter drops it.
    #[rstest]
    fn string_not_equal_drops_null_rows(mut testing_planner: TestingPlanner) {
        let mut rows = run(
            &mut testing_planner,
            "SELECT a FROM nullable_table WHERE s <> 'x'",
        );

        rows.sort_by_key(|r| r["a"].as_i64().unwrap());

        assert_eq!(
            rows.iter()
                .map(|r| r["a"].as_i64().unwrap())
                .collect::<Vec<_>>(),
            vec![3, 6]
        );
    }

    #[rstest]
    fn filters_string_with_the_constant_on_the_left(mut testing_planner: TestingPlanner) {
        phrases_table(&mut testing_planner);

        let rows = phrase_rows(&mut testing_planner, "SELECT s FROM phrases WHERE '' <> s");

        assert_eq!(
            rows,
            vec![
                "a phrase far longer than inlines",
                "a phrase far longer than inlinez",
                "short"
            ]
        );
    }

    #[rstest]
    fn filters_not_equal(mut testing_planner: TestingPlanner) {
        let mut rows = run(
            &mut testing_planner,
            "SELECT a FROM example_table WHERE a <> 3",
        );

        rows.sort_by_key(|r| r["a"].as_i64().unwrap());

        assert_eq!(
            rows.iter()
                .map(|r| r["a"].as_i64().unwrap())
                .collect::<Vec<_>>(),
            vec![1, 2, 4, 5]
        );
    }
}
