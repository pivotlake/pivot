//! Expressions used inside [`Operator`](crate::operator::Operator)s — column
//! references, comparisons, aggregates, scalar functions, and constants.

use crate::types;
use crate::types::{Type, build_scalar_value, type_from_logical};
use arrow_array::{ArrayRef, Scalar};
use duckdb_planner::duckdb_bridge::duckdb_types::ExpressionType;
use duckdb_planner::expression as duckdb_expression;
use std::fmt::Display;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    TypeError(#[from] types::Error),
    #[error("Equality comparisons are not supported yet")]
    UnsupportedEqualityComparison,
    #[error("Unsupported comparison type: {0:?}")]
    UnsupportedComparisonType(ExpressionType),
    #[error("Unsupported aggregate function: {0}")]
    UnsupportedAggregateFunction(String),
    #[error("Unsupported scalar function: {0}")]
    UnsupportedScalarFunction(String),
    #[error("Invalid parameter count for {function}: expected {expected}, got {actual}")]
    InvalidParameterCount {
        function: String,
        expected: usize,
        actual: usize,
    },
}

/// A bound column reference (points at a column by index in the input).
#[derive(Debug, Clone)]
pub struct Ref {
    pub column_idx: usize,
    pub return_type: Type,
}

impl TryFrom<duckdb_expression::Ref> for Ref {
    type Error = Error;
    fn try_from(r: duckdb_expression::Ref) -> Result<Self, Self::Error> {
        Ok(Ref {
            column_idx: r.column_idx,
            return_type: type_from_logical(r.return_type)?,
        })
    }
}

#[derive(Debug, Clone)]
pub enum CompareType {
    NotEqual,
}

impl TryFrom<ExpressionType> for CompareType {
    type Error = Error;
    fn try_from(c: ExpressionType) -> Result<Self, Self::Error> {
        match c {
            ExpressionType::COMPARE_NOTEQUAL => Ok(CompareType::NotEqual),
            ExpressionType::COMPARE_EQUAL => Err(Error::UnsupportedEqualityComparison),
            _ => Err(Error::UnsupportedComparisonType(c)),
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

impl TryFrom<duckdb_expression::Compare> for Compare {
    type Error = Error;
    fn try_from(c: duckdb_expression::Compare) -> Result<Self, Self::Error> {
        Ok(Compare {
            left: Box::<Expression>::try_from(c.left)?,
            right: Box::<Expression>::try_from(c.right)?,
            compare_type: c.compare_type.try_into()?,
            return_type: type_from_logical(c.return_type)?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct CountStar {
    pub params: Vec<Expression>,
}

impl TryFrom<duckdb_expression::AggregateFunc> for CountStar {
    type Error = Error;
    fn try_from(a: duckdb_expression::AggregateFunc) -> Result<Self, Self::Error> {
        if a.aggregate_function != "count_star" {
            return Err(Error::UnsupportedAggregateFunction(a.aggregate_function));
        }
        Ok(CountStar {
            params: a
                .params
                .into_iter()
                .map(Expression::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

/// An aggregate function call (e.g. `SUM`, `COUNT`).
#[derive(Debug, Clone)]
pub enum AggregateFunc {
    CountStar(CountStar),
}

impl TryFrom<duckdb_expression::AggregateFunc> for AggregateFunc {
    type Error = Error;
    fn try_from(a: duckdb_expression::AggregateFunc) -> Result<Self, Self::Error> {
        match a.aggregate_function.as_str() {
            "count_star" => Ok(AggregateFunc::CountStar(a.try_into()?)),
            _ => Err(Error::UnsupportedAggregateFunction(a.aggregate_function)),
        }
    }
}

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

/// A scalar function call (e.g. `year`, `substring`).
#[derive(Debug, Clone)]
pub enum Function {
    Contains(Contains),
}

impl TryFrom<duckdb_expression::Function> for Function {
    type Error = Error;
    fn try_from(f: duckdb_expression::Function) -> Result<Self, Self::Error> {
        match f.function.as_str() {
            "contains" => Ok(Function::Contains(f.try_into()?)),
            _ => Err(Error::UnsupportedScalarFunction(f.function)),
        }
    }
}

/// An expression in the logical plan. Discriminated by DuckDB's [`ExpressionType`].
#[derive(Debug, Clone)]
pub enum Expression {
    Ref(Ref),
    Compare(Compare),
    Constant(Scalar<ArrayRef>),
    AggregateFunc(AggregateFunc),
    Function(Function),
}

impl Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl TryFrom<duckdb_expression::Expression> for Expression {
    type Error = Error;
    fn try_from(e: duckdb_expression::Expression) -> Result<Self, Self::Error> {
        Ok(match e {
            duckdb_expression::Expression::Ref(r) => Expression::Ref(r.try_into()?),
            duckdb_expression::Expression::Compare(c) => Expression::Compare(c.try_into()?),
            duckdb_expression::Expression::Constant(c) => {
                Expression::Constant(build_scalar_value(c)?)
            }
            duckdb_expression::Expression::AggregateFunc(a) => {
                Expression::AggregateFunc(a.try_into()?)
            }
            duckdb_expression::Expression::Function(f) => Expression::Function(f.try_into()?),
        })
    }
}

impl TryFrom<Box<duckdb_planner::expression::Expression>> for Box<Expression> {
    type Error = Error;
    fn try_from(e: Box<duckdb_planner::expression::Expression>) -> Result<Self, Self::Error> {
        Ok(Box::new((*e).try_into()?))
    }
}

/// A filter that was pushed down into a table scan.
#[derive(Debug)]
pub enum TableFilter {
    Expression(Box<Expression>),
}

impl TryFrom<duckdb_expression::TableFilter> for TableFilter {
    type Error = Error;
    fn try_from(f: duckdb_expression::TableFilter) -> Result<Self, Self::Error> {
        Ok(match f {
            duckdb_expression::TableFilter::Expression(e) => TableFilter::Expression(e.try_into()?),
        })
    }
}
