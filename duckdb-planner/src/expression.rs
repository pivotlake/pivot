//! Typed expressions built from DuckDB's logical plan.
//!
//! Each node in the logical plan can carry [`Expression`]s that
//! represent column references, constants, comparisons, function calls,
//! aggregates, etc...
//!
//! This file implements and represents those. The trees are built directly from
//! the C++ bridge by [`crate::plan_build`]; there is no intermediate serialized
//! form.

use crate::duckdb_bridge::duckdb_types::{ExpressionType, LogicalTypeId};
use crate::types::ScalarValue;
use std::fmt;
use std::fmt::{Debug, Display};

/// A bound column reference — points at a column by its positional index
/// in the child operator's output.
#[derive(Debug)]
pub struct Ref {
    /// Zero-based index into the child operator's output columns.
    pub column_idx: usize,
    /// The column's logical type (e.g. `INTEGER`, `VARCHAR`).
    pub return_type: LogicalTypeId,
    /// The column's source name carried over from DuckDB's binding, or `None`
    /// when the reference has no alias. Display-only.
    pub name: Option<String>,
}

/// A binary comparison expression (e.g. `<>`, `=`).
#[derive(Debug)]
pub struct Compare {
    pub left: Box<Expression>,
    pub right: Box<Expression>,
    pub compare_type: ExpressionType,
    pub return_type: LogicalTypeId,
}

/// A `BETWEEN` expression (`input BETWEEN lower AND upper`).
#[derive(Debug)]
pub struct Between {
    pub input: Box<Expression>,
    pub lower: Box<Expression>,
    pub upper: Box<Expression>,
    pub lower_inclusive: bool,
    pub upper_inclusive: bool,
}

/// An aggregate function call (e.g. `SUM`, `COUNT`).
#[derive(Debug)]
pub struct AggregateFunc {
    pub aggregate_function: String,
    pub params: Vec<Expression>,
    /// `true` when the call is `COUNT(DISTINCT …)` / `SUM(DISTINCT …)` etc.
    /// (DuckDB's `AggregateType::DISTINCT`).
    pub distinct: bool,
    pub return_type: LogicalTypeId,
}

/// A scalar function call (e.g. `year`, `substring`).
#[derive(Debug)]
pub struct Function {
    pub function: String,
    pub params: Vec<Expression>,
    pub return_type: LogicalTypeId,
}

/// An `input IN (v0, v1, …)` membership test. DuckDB keeps a small constant
/// list as a `BoundOperatorExpression`; the bridge records its first child as
/// `input` and the remaining children as `values`.
#[derive(Debug)]
pub struct InList {
    pub input: Box<Expression>,
    pub values: Vec<Expression>,
}

/// A boolean `AND`/`OR` of two or more child predicates
/// (`BoundConjunctionExpression`). `conjunction_type` is DuckDB's
/// `CONJUNCTION_AND` (50) or `CONJUNCTION_OR` (51); the optimizer rewrites a
/// small `x IN (a, b)` into the `OR` form, so this is how most pushed-down `IN`
/// membership tests arrive.
#[derive(Debug)]
pub struct Conjunction {
    pub conjunction_type: ExpressionType,
    pub children: Vec<Expression>,
}

/// One `WHEN when THEN then` arm of a [`Case`].
#[derive(Debug)]
pub struct CaseCheck {
    pub when: Box<Expression>,
    pub then: Box<Expression>,
}

/// A `CASE WHEN … THEN … [WHEN …] ELSE … END` expression
/// (`BoundCaseExpression`). DuckDB always materializes an `else_expr` —
/// a `CASE` without an explicit `ELSE` carries a `NULL` constant there.
#[derive(Debug)]
pub struct Case {
    pub checks: Vec<CaseCheck>,
    pub else_expr: Box<Expression>,
}

/// Logical negation (`NOT expr`). DuckDB lowers it as a
/// `BoundOperatorExpression` of type [`ExpressionType::OPERATOR_NOT`] with a
/// single child.
#[derive(Debug)]
pub struct Not {
    pub input: Box<Expression>,
}

/// An expression in the logical plan. Discriminated by DuckDB's [`ExpressionType`].
#[derive(Debug)]
pub enum Expression {
    Ref(Ref),
    Compare(Compare),
    Between(Between),
    Constant(ScalarValue),
    AggregateFunc(AggregateFunc),
    Function(Function),
    InList(InList),
    Conjunction(Conjunction),
    Case(Case),
    Not(Not),
}

/// A constant comparison against a single column (e.g. `col <> 42`).
#[derive(Debug)]
pub struct ConstantComparison {
    pub column_ref: Box<Expression>,
    pub compare_type: ExpressionType,
    pub constant: ScalarValue,
}

/// A filter that was pushed down into a table scan.
#[derive(Debug)]
pub enum TableFilter {
    Expression(Box<Expression>),
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

impl ExpressionType {
    /// Reconstruct an [`ExpressionType`] from the `u8` discriminant the bridge
    /// reports. DuckDB defines `ExpressionType` as `enum class : uint8_t`.
    pub(crate) fn from_u8(value: u8) -> Self {
        unsafe { std::mem::transmute::<u8, ExpressionType>(value) }
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
            Expression::InList(in_list) => {
                let values: Vec<String> = in_list.values.iter().map(|v| v.to_string()).collect();
                write!(f, "{} IN ({})", in_list.input, values.join(", "))
            }
            Expression::Conjunction(conj) => {
                let op = if conj.conjunction_type.clone() as u8
                    == ExpressionType::CONJUNCTION_OR as u8
                {
                    "OR"
                } else {
                    "AND"
                };
                let parts: Vec<String> = conj.children.iter().map(|c| c.to_string()).collect();
                write!(f, "({})", parts.join(&format!(" {op} ")))
            }
            Expression::Case(case) => {
                write!(f, "CASE")?;
                for check in &case.checks {
                    write!(f, " WHEN {} THEN {}", check.when, check.then)?;
                }
                write!(f, " ELSE {} END", case.else_expr)
            }
            Expression::Not(n) => write!(f, "NOT({})", n.input),
        }
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
