//! Typed expressions deserialized from DuckDB's JSON plan.
//!
//! Each node in the logical plan can carry [`Expression`]s that
//! represent column references, constants, comparisons, function calls,
//! aggregates, etc...
//!
//! This file implements and represents those.

use crate::duckdb_bridge::duckdb_types::{ExpressionType, LogicalTypeId, TableFilterType};
use crate::types::ScalarValue;
use custom_deserializer::CustomDeserializer;
use std::fmt;
use std::fmt::{Debug, Display};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Failed to find type key in serve json: {0}")]
    DeserializationMissingType(serde_json::Value),
    #[error("Failed to find value key in serve json: {0}")]
    DeserializationMissingValue(serde_json::Value),
    #[error("Failed to deserialize scalar value: {0}")]
    ScalarDeserializationInvalidValue(serde_json::Value),
    #[error("Unsupported scalar type of duckdb type: {0}")]
    UnsupportedScalar(u8),
    #[error("Json serde deserialization error: {0}")]
    SerdeDeserialize(#[from] serde_json::Error),
    #[error("Unsupported type id: {0}")]
    UnsupportedTypeId(u8),
    #[error("Failed to parse integer value: {0}")]
    IntegerParse(#[from] std::num::ParseIntError),
}

/// A bound column reference — points at a column by its positional index
/// in the child operator's output.
#[derive(CustomDeserializer, Debug)]
pub struct Ref {
    /// Zero-based index into the child operator's output columns.
    pub column_idx: usize,
    /// The column's logical type (e.g. `INTEGER`, `VARCHAR`).
    pub return_type: LogicalTypeId,
}

/// A binary comparison expression (e.g. `<>`, `=`).
#[derive(CustomDeserializer, Debug)]
pub struct Compare {
    pub left: Box<Expression>,
    pub right: Box<Expression>,
    pub compare_type: ExpressionType,
    pub return_type: LogicalTypeId,
}

/// A `BETWEEN` expression (`input BETWEEN lower AND upper`).
#[derive(CustomDeserializer, Debug)]
pub struct Between {
    pub input: Box<Expression>,
    pub lower: Box<Expression>,
    pub upper: Box<Expression>,
    pub lower_inclusive: bool,
    pub upper_inclusive: bool,
}

/// An aggregate function call (e.g. `SUM`, `COUNT`).
#[derive(CustomDeserializer, Debug)]
pub struct AggregateFunc {
    pub aggregate_function: String,
    pub params: Vec<Expression>,
    pub return_type: LogicalTypeId,
}

/// A scalar function call (e.g. `year`, `substring`).
#[derive(CustomDeserializer, Debug)]
pub struct Function {
    pub function: String,
    pub params: Vec<Expression>,
    pub return_type: LogicalTypeId,
}

/// An expression in the logical plan. Discriminated by DuckDB's [`ExpressionType`].
#[derive(CustomDeserializer, Debug)]
pub enum Expression {
    #[type_tag(ExpressionType::BOUND_REF)]
    Ref(Ref),
    #[type_tag(ExpressionType::COMPARE_EQUAL)]
    #[type_tag(ExpressionType::COMPARE_NOTEQUAL)]
    #[type_tag(ExpressionType::COMPARE_LESSTHAN)]
    #[type_tag(ExpressionType::COMPARE_GREATERTHAN)]
    #[type_tag(ExpressionType::COMPARE_LESSTHANOREQUALTO)]
    #[type_tag(ExpressionType::COMPARE_GREATERTHANOREQUALTO)]
    Compare(Compare),
    #[type_tag(ExpressionType::COMPARE_BETWEEN)]
    Between(Between),
    #[type_tag(ExpressionType::VALUE_CONSTANT)]
    Constant(ScalarValue),
    #[type_tag(ExpressionType::BOUND_AGGREGATE)]
    AggregateFunc(AggregateFunc),
    #[type_tag(ExpressionType::BOUND_FUNCTION)]
    Function(Function),
}

/// A constant comparison against a single column (e.g. `col <> 42`).
#[derive(CustomDeserializer, Debug)]
pub struct ConstantComparison {
    pub column_ref: Box<Expression>,
    pub compare_type: ExpressionType,
    pub constant: ScalarValue,
}

/// A filter that was pushed down into a table scan.
#[derive(CustomDeserializer, Debug)]
pub enum TableFilter {
    #[type_tag(TableFilterType::EXPRESSION_FILTER)]
    Expression(Box<Expression>),
    #[type_tag(TableFilterType::CONSTANT_COMPARISON)]
    ConstantComparison(ConstantComparison),
}

impl fmt::Display for TableFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TableFilter::Expression(e) => write!(f, "{e}"),
            TableFilter::ConstantComparison(c) => write!(
                f,
                "{} {} {}:{}",
                c.column_ref,
                compare_symbol(&c.compare_type),
                c.constant.raw_value,
                type_name(&c.constant.logical_type),
            ),
        }
    }
}

/// Map a [`LogicalTypeId`] to its DuckDB type name (e.g. `INTEGER`).
pub fn type_name(t: &LogicalTypeId) -> &'static str {
    let n = t.clone() as u8;
    if n == LogicalTypeId::INTEGER as u8 {
        "INTEGER"
    } else if n == LogicalTypeId::VARCHAR as u8 {
        "VARCHAR"
    } else if n == LogicalTypeId::BOOLEAN as u8 {
        "BOOLEAN"
    } else if n == LogicalTypeId::BIGINT as u8 {
        "BIGINT"
    } else if n == LogicalTypeId::SMALLINT as u8 {
        "SMALLINT"
    } else if n == LogicalTypeId::TINYINT as u8 {
        "TINYINT"
    } else if n == LogicalTypeId::FLOAT as u8 {
        "FLOAT"
    } else if n == LogicalTypeId::DOUBLE as u8 {
        "DOUBLE"
    } else {
        "UNKNOWN"
    }
}

/// Map a comparison [`ExpressionType`] to its SQL operator symbol.
pub fn compare_symbol(t: &ExpressionType) -> &'static str {
    let n = t.clone() as u8;
    if n == ExpressionType::COMPARE_EQUAL as u8 {
        "="
    } else if n == ExpressionType::COMPARE_NOTEQUAL as u8 {
        "<>"
    } else if n == ExpressionType::COMPARE_LESSTHAN as u8 {
        "<"
    } else if n == ExpressionType::COMPARE_GREATERTHAN as u8 {
        ">"
    } else if n == ExpressionType::COMPARE_LESSTHANOREQUALTO as u8 {
        "<="
    } else if n == ExpressionType::COMPARE_GREATERTHANOREQUALTO as u8 {
        ">="
    } else {
        "?"
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Ref(r) => write!(f, "#{}:{}", r.column_idx, type_name(&r.return_type)),
            Expression::Compare(c) => write!(
                f,
                "{} {} {} -> {}",
                c.left,
                compare_symbol(&c.compare_type),
                c.right,
                type_name(&c.return_type)
            ),
            Expression::Between(b) => write!(f, "{} BETWEEN {} AND {}", b.input, b.lower, b.upper),
            Expression::Constant(c) => write!(f, "{}:{}", c.raw_value, type_name(&c.logical_type)),
            Expression::AggregateFunc(a) => {
                let params: Vec<String> = a.params.iter().map(|p| p.to_string()).collect();
                write!(
                    f,
                    "{}({}) -> {}",
                    a.aggregate_function,
                    params.join(", "),
                    type_name(&a.return_type)
                )
            }
            Expression::Function(func) => {
                let params: Vec<String> = func.params.iter().map(|p| p.to_string()).collect();
                write!(
                    f,
                    "{}({}) -> {}",
                    func.function,
                    params.join(", "),
                    type_name(&func.return_type)
                )
            }
        }
    }
}

impl<'de> serde::Deserialize<'de> for ExpressionType {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = <u8 as serde::Deserialize>::deserialize(deserializer)?;
        Ok(unsafe { std::mem::transmute::<u8, ExpressionType>(value) })
    }
}

impl Debug for ExpressionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.clone() as u8)
    }
}

impl Display for ExpressionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.clone() as u8)
    }
}
