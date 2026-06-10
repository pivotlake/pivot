//! Logical operators that make up the nodes of a query plan.
//!
//! Each variant of [`Operator`] corresponds to a DuckDB [`LogicalOperatorType`]
//! and carries the operator-specific payload (column lists, expressions, etc.).

use std::collections::HashMap;
use std::fmt;

use crate::duckdb_bridge::duckdb_types::{LogicalOperatorType, OrderType};
use crate::dynamic_filter::DynamicFilter;
use crate::expression::{Expression, type_name};
use custom_deserializer::CustomDeserializer;
use serde_repr::Deserialize_repr;

/// Raw, pre-resolution form of a table scan as it comes off the JSON plan.
///
/// `PlanNode::resolve_inputs` turns each `RawInput` into an [`Input`] by looking
/// up the table by `table_id` and binding each [`DynamicFilter`] to its slot.
#[derive(CustomDeserializer, Debug)]
pub struct RawInput {
    pub table_id: usize,
    pub columns: Vec<Expression>,
    pub dynamic_filters: Vec<DynamicFilter>,
    /// Set by the bridge on a late-materialized query's narrow scan so it tags
    /// each row with row-group metadata for the downstream [`Materialize`].
    pub emit_row_group_metadata: bool,
}

/// Resolved table scan with the `DuckDBTable` trait object attached.
/// Produced from [`RawInput`] after the planning phase resolves `table_id`
/// to a `Box<dyn DuckDBTable>` provided by the [`DuckDBBind`](crate::DuckDBBind).
pub struct Input {
    pub table: Box<dyn crate::catalog_provider::DuckDBTable>,
    pub columns: Vec<Expression>,
    pub dynamic_filters: Vec<DynamicFilter>,
    pub emit_row_group_metadata: bool,
}

impl fmt::Debug for Input {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Input")
            .field("columns", &self.columns)
            .finish()
    }
}

/// Computes a list of output expressions from its child's columns.
#[derive(CustomDeserializer, Debug)]
pub struct Projection {
    pub projections: Vec<Expression>,
}

/// Sort direction for an ORDER BY clause.
///
/// `Default` is what DuckDB emits when no explicit `ASC`/`DESC` is given
/// (equivalent to `Asc` in practice).
#[derive(Deserialize_repr, Debug)]
#[repr(u8)]
pub enum OrderByDirection {
    Default = OrderType::ORDER_DEFAULT as u8,
    Asc = OrderType::ASCENDING as u8,
    Desc = OrderType::DESCENDING as u8,
}

/// A single sort key within an ORDER BY or TopN operator.
#[derive(CustomDeserializer, Debug)]
pub struct OrderByNode {
    pub direction: OrderByDirection,
    pub expression: Expression,
}

/// Sorts its input by one or more keys.
#[derive(CustomDeserializer, Debug)]
pub struct OrderBy {
    pub order_bys: Vec<OrderByNode>,
}

/// GROUP BY + aggregate functions.
#[derive(CustomDeserializer, Debug)]
pub struct Aggregate {
    pub groups: Vec<Expression>,
    pub expressions: Vec<Expression>,
}

/// Filters rows by one or more boolean conditions (implicitly ANDed).
#[derive(CustomDeserializer, Debug)]
pub struct Filter {
    pub conditions: Vec<Expression>,
}

/// Combined ORDER BY + LIMIT (returns the top N rows).
#[derive(CustomDeserializer, Debug)]
pub struct TopN {
    pub order_bys: Vec<OrderByNode>,
    pub limit: usize,
    pub offset: usize,
    /// Set when DuckDB's Top-N optimizer installed a dynamic-filter producer on
    /// this node: at runtime the operator publishes its current boundary into
    /// the shared slot so consumer scans elsewhere in the plan can prune.
    pub produces_dynamic_filter: Option<DynamicFilter>,
}

/// Plain `LIMIT n [OFFSET m]` with no ordering attached.
///
/// DuckDB lowers `ORDER BY … LIMIT` into [`TopN`] when its Top-N optimizer
/// fires; a `LogicalLimit` survives to the optimized plan only when there is
/// no ORDER BY beneath it (any rows are valid output) or when the limit+offset
/// window is large enough that DuckDB prefers a full sort + limit over Top-N.
/// `limit` is `None` for a bare `OFFSET m` (unbounded); only constant
/// limits/offsets reach here — percentage and expression forms are rejected by
/// the bridge.
#[derive(CustomDeserializer, Debug)]
pub struct Limit {
    pub limit: Option<usize>,
    pub offset: usize,
}

/// A single column definition inside a CREATE TABLE statement.
#[derive(CustomDeserializer, Debug)]
pub struct CreateTableColumn {
    pub name: String,
    pub col_type: crate::duckdb_bridge::duckdb_types::LogicalTypeId,
}

/// CREATE TABLE with an explicit column list.
#[derive(CustomDeserializer, Debug)]
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

/// Late-materialization fetch, synthesized by the bridge from DuckDB's
/// `late_materialization` optimizer output.
///
/// DuckDB rewrites `SELECT <wide> ... ORDER BY ... LIMIT n` into a row-id SEMI
/// join (full-column scan SEMI-joined against a narrow Top-N pipeline). The
/// bridge collapses that join — which pivot can't execute — into this node: its
/// single child is the narrow pipeline (which carries DuckDB's row-id column,
/// stripped later), and `columns` are the table-schema column indices the
/// full-column side fetched, which pivot re-reads for the surviving rows via its
/// own row-group materializer. Tagged with `LOGICAL_COMPARISON_JOIN` because a
/// raw comparison join never otherwise reaches the Rust deserializer.
///
/// `table_id` is the *same* index as the narrow scan beneath it (both sides of
/// DuckDB's join are the one table), so `resolve_inputs` hands this node a clone
/// of that scan's resolved table — same instance, same filter-pushdown state,
/// which the materializer needs to index row groups by the scan's global ids.
#[derive(CustomDeserializer, Debug)]
pub struct RawMaterialize {
    pub table_id: usize,
    pub columns: Vec<usize>,
}

/// Resolved late-materialization fetch, with its table attached. Produced from
/// [`RawMaterialize`] during `resolve_inputs`.
pub struct Materialize {
    pub table: Box<dyn crate::catalog_provider::DuckDBTable>,
    pub columns: Vec<usize>,
}

impl fmt::Debug for Materialize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Materialize")
            .field("columns", &self.columns)
            .finish()
    }
}

/// The single-row source DuckDB places under a `FROM`-less `SELECT` (e.g.
/// `SELECT drop_cache()` or `SELECT 1`). Carries no payload; compiles to a
/// source that emits one empty row, over which the parent projection evaluates
/// its expressions exactly once.
#[derive(CustomDeserializer, Debug)]
pub struct DummyScan {}

/// A logical operator in the query plan. Discriminated by DuckDB's
/// [`LogicalOperatorType`].
#[derive(CustomDeserializer, Debug)]
pub enum Operator {
    #[type_tag(LogicalOperatorType::LOGICAL_GET)]
    #[doc(hidden)]
    RawInput(RawInput),
    #[skip_deserialize]
    // LogicalGet is deserialized as RawInput, then converted to Input in a
    // post-processing step to attach the DuckDBTable trait object.
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
    #[type_tag(LogicalOperatorType::LOGICAL_LIMIT)]
    Limit(Limit),
    #[type_tag(LogicalOperatorType::LOGICAL_CREATE_TABLE)]
    CreateTable(CreateTable),
    #[type_tag(LogicalOperatorType::LOGICAL_DUMMY_SCAN)]
    DummyScan(DummyScan),
    // The bridge collapses DuckDB's late-materialization SEMI join into this
    // node and emits it tagged with LOGICAL_COMPARISON_JOIN (a raw join never
    // otherwise reaches Rust). Resolved to `Materialize` (table attached) the
    // same way `RawInput` becomes `Input`.
    #[type_tag(LogicalOperatorType::LOGICAL_COMPARISON_JOIN)]
    #[doc(hidden)]
    RawMaterialize(RawMaterialize),
    #[skip_deserialize]
    Materialize(Materialize),
}

impl Operator {
    /// Returns the expressions that define this operator's output columns, or
    /// `None` for pass-through operators where the output schema is
    /// inherited from the child (Filter, OrderBy, TopN, Limit).
    pub fn get_output_expressions(&self) -> Option<Vec<&Expression>> {
        match self {
            Operator::Projection(p) => Some(p.projections.iter().collect()),
            Operator::Aggregate(a) => Some(a.groups.iter().chain(a.expressions.iter()).collect()),
            Operator::Input(ss) => Some(ss.columns.iter().collect()),
            Operator::Filter(_)
            | Operator::OrderBy(_)
            | Operator::TopN(_)
            | Operator::Limit(_)
            | Operator::CreateTable(_)
            | Operator::DummyScan(_)
            | Operator::Materialize(_)
            | Operator::RawMaterialize(_)
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
                let cols = i
                    .columns
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<String>>()
                    .join(", ");
                write!(f, "Input([{cols}])")
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
                write!(
                    f,
                    "TopN(limit: {}, offset: {}, order: {})",
                    t.limit,
                    t.offset,
                    orders.join(", ")
                )
            }
            Operator::Limit(l) => match l.limit {
                Some(limit) => write!(f, "Limit(limit: {}, offset: {})", limit, l.offset),
                None => write!(f, "Limit(limit: unbounded, offset: {})", l.offset),
            },
            Operator::CreateTable(c) => {
                let columns: Vec<String> = c
                    .columns
                    .iter()
                    .map(|col| format!("{}:{}", col.name, type_name(&col.col_type)))
                    .collect();
                let mut options: Vec<(&String, &String)> = c.options.iter().collect();
                options.sort_by(|a, b| a.0.cmp(b.0));
                let options_str: Vec<String> = options
                    .iter()
                    .map(|(k, v)| format!("{k:?}: {v:?}"))
                    .collect();
                write!(
                    f,
                    "CreateTable({}, [{}], options: {{{}}})",
                    c.name,
                    columns.join(", "),
                    options_str.join(", "),
                )
            }
            Operator::DummyScan(_) => write!(f, "DummyScan"),
            Operator::RawMaterialize(m) => {
                let cols: Vec<String> = m.columns.iter().map(|c| format!("#{c}")).collect();
                write!(f, "Materialize([{}])", cols.join(", "))
            }
            Operator::Materialize(m) => {
                let cols: Vec<String> = m.columns.iter().map(|c| format!("#{c}")).collect();
                write!(f, "Materialize([{}])", cols.join(", "))
            }
        }
    }
}
