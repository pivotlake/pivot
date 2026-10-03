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

impl From<CompareType> for pruning::Comparison {
    fn from(compare: CompareType) -> Self {
        match compare {
            CompareType::Equal => Self::Equal,
            CompareType::NotEqual => Self::NotEqual,
            CompareType::Less => Self::Less,
            CompareType::Greater => Self::Greater,
            CompareType::LessEqual => Self::LessEqual,
            CompareType::GreaterEqual => Self::GreaterEqual,
        }
    }
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
    use rstest::rstest;

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
