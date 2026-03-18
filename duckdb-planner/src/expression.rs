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
#[derive(CustomDeserializer)]
pub struct Ref {
    /// Zero-based index into the child operator's output columns.
    pub column_idx: usize,
    /// The column's logical type (e.g. `INTEGER`, `VARCHAR`).
    pub return_type: LogicalTypeId,
}

/// A binary comparison expression (e.g. `<>`, `=`).
#[derive(CustomDeserializer)]
pub struct Compare {
    pub left: Box<Expression>,
    pub right: Box<Expression>,
    pub compare_type: ExpressionType,
    pub return_type: LogicalTypeId,
}

/// An aggregate function call (e.g. `SUM`, `COUNT`).
#[derive(CustomDeserializer)]
pub struct AggregateFunc {
    pub aggregate_function: String,
    pub params: Vec<Expression>,
    pub return_type: LogicalTypeId,
}

/// A scalar function call (e.g. `year`, `substring`).
#[derive(CustomDeserializer)]
pub struct Function {
    pub function: String,
    pub params: Vec<Expression>,
    pub return_type: LogicalTypeId,
}

/// An expression in the logical plan. Discriminated by DuckDB's [`ExpressionType`].
#[derive(CustomDeserializer)]
pub enum Expression {
    #[type_tag(ExpressionType::BOUND_REF)]
    Ref(Ref),
    #[type_tag(ExpressionType::COMPARE_NOTEQUAL)]
    Compare(Compare),
    #[type_tag(ExpressionType::VALUE_CONSTANT)]
    Constant(ScalarValue),
    #[type_tag(ExpressionType::BOUND_AGGREGATE)]
    AggregateFunc(AggregateFunc),
    #[type_tag(ExpressionType::BOUND_FUNCTION)]
    Function(Function),
}

/// A filter that was pushed down into a table scan.
#[derive(CustomDeserializer)]
pub enum TableFilter {
    #[type_tag(TableFilterType::EXPRESSION_FILTER)]
    Expression(Box<Expression>),
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Ref(r) => write!(f, "#{}", r.column_idx),
            Expression::Compare(c) => write!(f, "{} <> {}", c.left, c.right),
            Expression::Constant(c) => write!(f, "{}", c.logical_type.clone() as u8),
            Expression::AggregateFunc(a) => {
                let params: Vec<String> = a.params.iter().map(|p| p.to_string()).collect();
                write!(f, "{}({})", a.aggregate_function, params.join(", "))
            }
            Expression::Function(func) => {
                let params: Vec<String> = func.params.iter().map(|p| p.to_string()).collect();
                write!(f, "{}({})", func.function, params.join(", "))
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
