//! Logical operators that make up the nodes of a query plan.
//!
//! Each variant of [`Operator`] corresponds to a DuckDB [`LogicalOperatorType`]
//! and carries the operator-specific payload (column lists, expressions, etc.).

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::duckdb_bridge::duckdb_types::{LogicalOperatorType, OrderType};
use crate::expression::{Expression, TableFilter};
use custom_deserializer::CustomDeserializer;
use serde_repr::Deserialize_repr;

/// Deserialized form of a table scan — carries a `table_id` index into the
/// tables vector returned alongside the JSON plan from the C++ bridge.
/// Converted into [`Input`] (which holds the resolved
/// `Arc<dyn GetDuckDBTypedColumns>`) by `PlanNode::resolve_tables` during the
/// post-processing pass.
#[derive(CustomDeserializer)]
pub struct RawInput {
    pub table_id: usize,
    pub columns: Vec<Expression>,
    pub filters: Vec<TableFilter>,
}

/// Resolved table scan with the `GetDuckDBTypedColumns` trait object attached.
/// Produced from [`RawInput`] after the planning phase resolves `table_id`
/// to an `Arc<dyn GetDuckDBTypedColumns>` provided by the [`DuckDBBind`](crate::DuckDBBind).
pub struct Input {
    pub table: Arc<dyn crate::catalog_provider::GetDuckDBTypedColumns>,
    pub columns: Vec<Expression>,
    pub filters: Vec<TableFilter>,
}

/// Computes a list of output expressions from its child's columns.
#[derive(CustomDeserializer)]
pub struct Projection {
    pub projections: Vec<Expression>,
}

/// Sort direction for an ORDER BY clause.
///
/// `Default` is what DuckDB emits when no explicit `ASC`/`DESC` is given
/// (equivalent to `Asc` in practice).
#[derive(Deserialize_repr)]
#[repr(u8)]
pub enum OrderByDirection {
    Default = OrderType::ORDER_DEFAULT as u8,
    Asc = OrderType::ASCENDING as u8,
    Desc = OrderType::DESCENDING as u8,
}

/// A single sort key within an ORDER BY or TopN operator.
#[derive(CustomDeserializer)]
pub struct OrderByNode {
    pub direction: OrderByDirection,
    pub expression: Expression,
}

/// Sorts its input by one or more keys.
#[derive(CustomDeserializer)]
pub struct OrderBy {
    pub order_bys: Vec<OrderByNode>,
}

/// GROUP BY + aggregate functions.
#[derive(CustomDeserializer)]
pub struct Aggregate {
    pub groups: Vec<Expression>,
    pub expressions: Vec<Expression>,
}

/// Filters rows by one or more boolean conditions (implicitly ANDed).
#[derive(CustomDeserializer)]
pub struct Filter {
    pub conditions: Vec<Expression>,
}

/// Combined ORDER BY + LIMIT (returns the top N rows).
#[derive(CustomDeserializer)]
pub struct TopN {
    pub order_bys: Vec<OrderByNode>,
    pub limit: usize,
}

/// A single column definition inside a CREATE TABLE statement.
#[derive(CustomDeserializer)]
pub struct CreateTableColumn {
    pub name: String,
    pub col_type: crate::duckdb_bridge::duckdb_types::LogicalTypeId,
}

/// CREATE TABLE with an explicit column list.
#[derive(CustomDeserializer)]
pub struct CreateTable {
    pub name: String,
    pub columns: Vec<CreateTableColumn>,
    pub options: HashMap<String, String>,
    pub if_not_exists: bool,
    pub or_replace: bool,
    pub temporary: bool,
    pub has_query: bool,
    pub constraint_count: usize,
}

/// A logical operator in the query plan. Discriminated by DuckDB's
/// [`LogicalOperatorType`].
#[derive(CustomDeserializer)]
pub enum Operator {
    #[type_tag(LogicalOperatorType::LOGICAL_GET)]
    #[doc(hidden)]
    RawInput(RawInput),
    #[skip_deserialize]
    // LogicalGet is deserialized as RawInput, then converted to Input in a
    // post-processing step to attach the GetDuckDBTypedColumns trait object.
    Input(Input),
    #[type_tag(LogicalOperatorType::LOGICAL_PROJECTION)]
    Projection(Projection),
    #[type_tag(LogicalOperatorType::LOGICAL_ORDER_BY)]
    OrderBy(OrderBy),
    #[type_tag(LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY)]
    Aggregate(Aggregate),
    #[type_tag(LogicalOperatorType::LOGICAL_FILTER)]
    Filter(Filter),
    #[type_tag(LogicalOperatorType::LOGICAL_TOP_N)]
    TopN(TopN),
    #[type_tag(LogicalOperatorType::LOGICAL_CREATE_TABLE)]
    CreateTable(CreateTable),
}

impl Operator {
    /// Returns the expressions that define this operator's output columns, or
    /// `None` for pass-through operators where the output schema is
    /// inherited from the child (Filter, OrderBy, TopN).
    pub fn get_output_expressions(&self) -> Option<Vec<&Expression>> {
        match self {
            Operator::Projection(p) => Some(p.projections.iter().collect()),
            Operator::Aggregate(a) => Some(a.groups.iter().chain(a.expressions.iter()).collect()),
            Operator::Input(ss) => Some(ss.columns.iter().collect()),
            Operator::Filter(_)
            | Operator::OrderBy(_)
            | Operator::TopN(_)
            | Operator::CreateTable(_)
            | Operator::RawInput(_) => None,
        }
    }
}

impl fmt::Display for OrderByNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dir = match self.direction {
            OrderByDirection::Asc | OrderByDirection::Default => "ASC",
            OrderByDirection::Desc => "DESC",
        };
        write!(f, "{} {}", self.expression, dir)
    }
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::RawInput(s) => {
                write!(
                    f,
                    "RawInput({}, [{}])",
                    s.table_id,
                    s.columns
                        .iter()
                        .map(|c| c.to_string())
                        .collect::<Vec<String>>()
                        .join(", ")
                )
            }
            Operator::Input(i) => {
                write!(
                    f,
                    "Input([{}])",
                    i.columns
                        .iter()
                        .map(|c| c.to_string())
                        .collect::<Vec<String>>()
                        .join(", ")
                )
            }
            Operator::Projection(p) => {
                let exprs: Vec<String> = p.projections.iter().map(|e| e.to_string()).collect();
                write!(f, "Projection({})", exprs.join(", "))
            }
            Operator::OrderBy(o) => {
                let orders: Vec<String> = o.order_bys.iter().map(|o| o.to_string()).collect();
                write!(f, "OrderBy({})", orders.join(", "))
            }
            Operator::Aggregate(a) => {
                let groups: Vec<String> = a.groups.iter().map(|e| e.to_string()).collect();
                let exprs: Vec<String> = a.expressions.iter().map(|e| e.to_string()).collect();
                write!(
                    f,
                    "Aggregate(groups: [{}], exprs: [{}])",
                    groups.join(", "),
                    exprs.join(", ")
                )
            }
            Operator::Filter(fl) => {
                let conds: Vec<String> = fl.conditions.iter().map(|e| e.to_string()).collect();
                write!(f, "Filter({})", conds.join(" AND "))
            }
            Operator::TopN(t) => {
                let orders: Vec<String> = t.order_bys.iter().map(|o| o.to_string()).collect();
                write!(f, "TopN(limit: {}, order: {})", t.limit, orders.join(", "))
            }
            Operator::CreateTable(c) => {
                let columns: Vec<String> = c
                    .columns
                    .iter()
                    .map(|col| format!("{}:{:?}", col.name, col.col_type))
                    .collect();
                write!(
                    f,
                    "CreateTable({}, [{}], options: {:?})",
                    c.name,
                    columns.join(", "),
                    c.options
                )
            }
        }
    }
}
