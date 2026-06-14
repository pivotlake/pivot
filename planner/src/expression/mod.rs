//! Expressions used inside [`Operator`](crate::operator::Operator)s — column
//! references, comparisons, aggregates, scalar functions, and constants.
//!
//! Each expression *kind* lives in its own submodule co-locating the AST type,
//! its `TryFrom` from the DuckDB expression, its `Display`, and (for the
//! evaluable kinds) its `compile` impl producing an
//! [`ExprFn`](crate::compile::ExprFn). This module holds the cross-cutting
//! pieces: the [`Expression`] / [`Function`] enums that tie the kinds together,
//! the conversion [`Error`], the pushed-down [`TableFilter`], and the small
//! `Display`/constant helpers shared across kinds. Coercion and civil-date
//! helpers used by more than one `compile` impl live in [`shared`].

mod aggregate;
mod arithmetic;
mod between;
mod case;
mod compare;
mod conjunction;
mod contains;
mod date_part;
mod date_trunc;
mod divide;
mod function;
mod in_list;
mod length;
mod not;
mod reference;
mod regexp;
mod shared;

pub use aggregate::{AggregateFunc, CountStar, NumericAggregate};
pub use arithmetic::{Arithmetic, ArithmeticOp};
pub use between::Between;
pub use case::{Case, CaseCheck};
pub use compare::{Compare, CompareType};
pub use conjunction::{Conjunction, ConjunctionOp};
pub use contains::Contains;
pub use date_part::{DatePart, DatePartKind};
pub use date_trunc::DateTrunc;
pub use divide::Divide;
pub use function::Function;
pub use in_list::InList;
pub use length::Length;
pub use not::Not;
pub use reference::Ref;
pub use regexp::RegexpReplace;

use crate::compile::{self, ExprFn, ExprResult, stateless_expr};
use crate::types::{self, Type, build_scalar_value};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, Datum, RecordBatch, Scalar};
use duckdb_planner::duckdb_bridge::duckdb_types::ExpressionType;
use duckdb_planner::expression as duckdb_expression;
use std::fmt::{self, Display};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    TypeError(#[from] types::Error),
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

/// Extract a constant string argument (e.g. a regex pattern or a `date_trunc`
/// unit) from a converted expression, erroring with `context` when the
/// argument is not a string constant.
pub(crate) fn constant_string(e: Expression, context: &str) -> Result<String, Error> {
    match e {
        Expression::Constant(scalar) => {
            let (arr, _) = scalar.get();
            Ok(arr
                .as_string_view_opt()
                .ok_or_else(|| {
                    Error::UnsupportedScalarFunction(format!("{context} must be a string"))
                })?
                .value(0)
                .to_string())
        }
        _ => Err(Error::UnsupportedScalarFunction(format!(
            "{context} must be a constant"
        ))),
    }
}

/// Format an arrow `Scalar<ArrayRef>` constant as `value:Type` for plan
/// display. Falls back to `?:DataType` for unsupported arrow types.
fn format_constant(s: &Scalar<ArrayRef>) -> String {
    let (arr, _is_scalar) = s.get();
    let formatter = ArrayFormatter::try_new(arr, &FormatOptions::default());
    let value = match formatter {
        Ok(f) => f.value(0).to_string(),
        Err(_) => "?".to_string(),
    };
    format!("{value}:{}", arr.data_type())
}

/// An expression in the logical plan. Discriminated by DuckDB's [`ExpressionType`].
#[derive(Debug, Clone)]
pub enum Expression {
    Ref(Ref),
    Compare(Compare),
    Between(Between),
    Constant(Scalar<ArrayRef>),
    AggregateFunc(AggregateFunc),
    Function(Function),
    InList(InList),
    Conjunction(Conjunction),
    Case(Case),
    Not(Not),
}

impl Expression {
    /// Best-effort static result type of a computed expression, used to pick a
    /// group-key extractor when grouping on it (e.g. `GROUP BY CASE …`). Returns
    /// `None` when the type can't be determined cheaply, in which case callers
    /// fall back to their default.
    pub fn result_type(&self) -> Option<Type> {
        match self {
            Expression::Ref(r) => Some(r.return_type.clone()),
            Expression::Constant(s) => match s.get().0.data_type() {
                arrow_schema::DataType::Utf8
                | arrow_schema::DataType::LargeUtf8
                | arrow_schema::DataType::Utf8View => Some(Type::Utf8),
                arrow_schema::DataType::Int8 => Some(Type::Int8),
                arrow_schema::DataType::Int16 => Some(Type::Int16),
                arrow_schema::DataType::Int32 => Some(Type::Int32),
                arrow_schema::DataType::Int64 => Some(Type::Int64),
                _ => None,
            },
            // A CASE's branches are unified to one type by DuckDB, so the ELSE
            // branch's type is the whole expression's type.
            Expression::Case(c) => c.else_expr.result_type(),
            Expression::Function(Function::DateTrunc(_)) => Some(Type::Timestamp),
            _ => None,
        }
    }

    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        match self {
            Expression::Ref(r) => r.compile(),
            Expression::Compare(c) => c.compile(),
            Expression::Constant(c) => {
                let scalar = c.clone();
                Ok(stateless_expr(move |_batch: &RecordBatch| {
                    ExprResult::Scalar(scalar.clone())
                }))
            }
            Expression::Function(f) => f.compile(),
            Expression::Between(b) => b.compile(),
            Expression::InList(i) => i.compile(),
            Expression::Conjunction(c) => c.compile(),
            Expression::Case(c) => c.compile(),
            Expression::Not(n) => n.compile(),
            _ => Err(compile::Error::UnsupportedExpression(self.clone())),
        }
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Ref(r) => write!(f, "{r}"),
            Expression::Compare(c) => write!(f, "{c}"),
            Expression::Between(b) => write!(f, "{b}"),
            Expression::Constant(c) => f.write_str(&format_constant(c)),
            Expression::AggregateFunc(a) => write!(f, "{a}"),
            Expression::Function(fun) => write!(f, "{fun}"),
            Expression::InList(i) => write!(f, "{i}"),
            Expression::Conjunction(c) => write!(f, "{c}"),
            Expression::Case(c) => write!(f, "{c}"),
            Expression::Not(n) => write!(f, "{n}"),
        }
    }
}

impl TryFrom<duckdb_expression::Expression> for Expression {
    type Error = Error;
    fn try_from(e: duckdb_expression::Expression) -> Result<Self, Self::Error> {
        Ok(match e {
            duckdb_expression::Expression::Ref(r) => Expression::Ref(r.try_into()?),
            duckdb_expression::Expression::Compare(c) => Expression::Compare(c.try_into()?),
            duckdb_expression::Expression::Between(b) => Expression::Between(b.try_into()?),
            duckdb_expression::Expression::Constant(c) => {
                Expression::Constant(build_scalar_value(c)?)
            }
            duckdb_expression::Expression::AggregateFunc(a) => {
                Expression::AggregateFunc(a.try_into()?)
            }
            duckdb_expression::Expression::Function(f) => Expression::Function(f.try_into()?),
            duckdb_expression::Expression::InList(i) => Expression::InList(i.try_into()?),
            duckdb_expression::Expression::Conjunction(c) => Expression::Conjunction(c.try_into()?),
            duckdb_expression::Expression::Case(c) => Expression::Case(c.try_into()?),
            duckdb_expression::Expression::Not(n) => Expression::Not(n.try_into()?),
        })
    }
}

impl TryFrom<Box<duckdb_planner::expression::Expression>> for Box<Expression> {
    type Error = Error;
    fn try_from(e: Box<duckdb_planner::expression::Expression>) -> Result<Self, Self::Error> {
        Ok(Box::new((*e).try_into()?))
    }
}

/// A constant comparison against a single column pushed into a table scan.
#[derive(Debug)]
pub struct ConstantComparison {
    pub column_ref: Box<Expression>,
    pub compare_type: CompareType,
    pub constant: Scalar<ArrayRef>,
}

impl TryFrom<duckdb_expression::ConstantComparison> for ConstantComparison {
    type Error = Error;
    fn try_from(c: duckdb_expression::ConstantComparison) -> Result<Self, Self::Error> {
        Ok(ConstantComparison {
            column_ref: Box::<Expression>::try_from(c.column_ref)?,
            compare_type: c.compare_type.try_into()?,
            constant: build_scalar_value(c.constant)?,
        })
    }
}

/// A filter that was pushed down into a table scan.
#[derive(Debug)]
pub enum TableFilter {
    Expression(Box<Expression>),
    ConstantComparison(ConstantComparison),
}

impl TryFrom<duckdb_expression::TableFilter> for TableFilter {
    type Error = Error;
    fn try_from(f: duckdb_expression::TableFilter) -> Result<Self, Self::Error> {
        Ok(match f {
            duckdb_expression::TableFilter::Expression(e) => TableFilter::Expression(e.try_into()?),
            duckdb_expression::TableFilter::ConstantComparison(c) => {
                TableFilter::ConstantComparison(c.try_into()?)
            }
        })
    }
}

impl Display for TableFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TableFilter::Expression(e) => write!(f, "{e}"),
            TableFilter::ConstantComparison(c) => write!(
                f,
                "{} {} {}",
                c.column_ref,
                c.compare_type,
                format_constant(&c.constant),
            ),
        }
    }
}
