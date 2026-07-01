//! Expressions used inside [`Operator`](crate::operator::Operator)s — column
//! references, comparisons, aggregates, scalar functions, and constants.
//!
//! Each expression *kind* lives in its own submodule co-locating the AST type,
//! its `TryFrom` from the DuckDB expression, its `Display`, and (for the
//! evaluable kinds) its `compile` impl producing an
//! [`ExprFn`]. This module holds the cross-cutting
//! pieces: the [`Expression`] / [`Function`] enums that tie the kinds together,
//! the conversion [`enum@Error`], the pushed-down [`TableFilter`], and the small
//! `Display`/constant helpers shared across kinds.

mod aggregate;
mod arithmetic;
mod between;
mod case;
mod cast;
mod compare;
mod conjunction;
mod contains;
mod convert;
mod date_part;
mod date_trunc;
mod divide;
mod function;
mod in_list;
mod interval;
mod length;
mod not;
mod reference;
mod regexp;
mod regexp_jit;

pub use aggregate::{AggregateFunc, CountStar, NumericAggregate};
pub use arithmetic::{Arithmetic, ArithmeticOp};
pub use between::Between;
pub use case::{Case, CaseCheck};
pub use cast::Cast;
pub use compare::{Compare, CompareType};
pub use conjunction::{Conjunction, ConjunctionOp};
pub use contains::Contains;
pub use convert::TemporalConvert;
pub use date_part::{DatePart, DatePartKind};
pub use date_trunc::DateTrunc;
pub use divide::Divide;
pub use function::{Function, ScalarFunctionSignature, builtin_scalar_function};
pub use in_list::InList;
pub use interval::IntervalArithmetic;
pub use length::Length;
pub use not::Not;
pub use reference::Ref;
pub use regexp::RegexpReplace;
pub use regexp_jit::RegexpJitReplace;

use crate::compile::{self, ExprFn, ExprResult, stateless_expr};
use crate::types::{self, Type};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use arrow_array::{ArrayRef, Datum, RecordBatch, Scalar};
use duckdb_planner::duckdb_bridge::duckdb_types::ExpressionType;
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
    #[error("Unsupported interval arithmetic: {0}")]
    UnsupportedInterval(String),
    #[error("Unsupported expression type: {0:?}")]
    UnsupportedExpressionType(ExpressionType),
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
    Cast(Cast),
}

impl Expression {
    /// Static result type of a computed expression, used to pick a group-key
    /// extractor when grouping on it (e.g. `GROUP BY CASE …`). Errors when the
    /// type can't be determined statically, so an unclassified key is rejected
    /// rather than silently mistyped.
    pub fn result_type(&self) -> Result<Type, compile::Error> {
        match self {
            Expression::Ref(r) => Ok(r.return_type.clone()),
            Expression::Constant(s) => match s.get().0.data_type() {
                arrow_schema::DataType::Utf8
                | arrow_schema::DataType::LargeUtf8
                | arrow_schema::DataType::Utf8View => Ok(Type::Utf8),
                arrow_schema::DataType::Int8 => Ok(Type::Int8),
                arrow_schema::DataType::Int16 => Ok(Type::Int16),
                arrow_schema::DataType::Int32 => Ok(Type::Int32),
                arrow_schema::DataType::Int64 => Ok(Type::Int64),
                arrow_schema::DataType::Float32 => Ok(Type::Float32),
                arrow_schema::DataType::Float64 => Ok(Type::Float64),
                _ => Err(compile::Error::IndeterminateResultType(self.clone())),
            },
            // A CASE's branches are unified to one type by DuckDB, so the ELSE
            // branch's type is the whole expression's type.
            Expression::Case(c) => c.else_expr.result_type(),
            // A cast yields its target type.
            Expression::Cast(c) => Ok(c.target.clone()),
            // `date_trunc` and `now()` yield a timestamp; the regex replacers a string.
            Expression::Function(Function::DateTrunc(_) | Function::Now) => Ok(Type::Timestamp),
            // `date`/`timestamp` ± interval keeps the temporal operand's type.
            Expression::Function(Function::IntervalArithmetic(i)) => Ok(i.result.clone()),
            // `make_date`/`make_timestamp` produce the temporal type they convert to.
            Expression::Function(Function::TemporalConvert(c)) => Ok(c.result.clone()),
            Expression::Function(Function::RegexpReplace(_) | Function::RegexpJitReplace(_)) => {
                Ok(Type::Utf8)
            }
            // A date part (`extract(minute …)`) and a byte length each carry the
            // type they evaluate to.
            Expression::Function(Function::DatePart(d)) => Ok(d.return_type.clone()),
            Expression::Function(Function::Length(l)) => Ok(l.return_type.clone()),
            // Arithmetic carries DuckDB's bound result type, so a float/decimal
            // operand surfaces as `DOUBLE`/`DECIMAL` here (which the grouping and
            // aggregation paths then reject).
            Expression::Function(Function::Arithmetic(a)) => Ok(a.return_type.clone()),
            // Everything else (comparisons, `contains`, `Divide`'s Float64
            // quotient, …) has no type we use for grouping.
            _ => Err(compile::Error::IndeterminateResultType(self.clone())),
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
            Expression::Cast(c) => c.compile(),
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
            Expression::Cast(c) => write!(f, "{c}"),
        }
    }
}

/// A constant comparison against a single column pushed into a table scan.
#[derive(Debug)]
pub struct ConstantComparison {
    pub column_ref: Box<Expression>,
    pub compare_type: CompareType,
    pub constant: Scalar<ArrayRef>,
}

/// A filter that was pushed down into a table scan.
#[derive(Debug)]
pub enum TableFilter {
    Expression(Box<Expression>),
    ConstantComparison(ConstantComparison),
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
