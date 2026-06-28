//! Logical operators that make up the nodes of a query plan.
//!
//! Each variant of [`Operator`] corresponds to a DuckDB
//! [`LogicalOperatorType`](crate::duckdb_bridge::duckdb_types::LogicalOperatorType)
//! and carries the operator-specific payload (column lists, expressions, etc.).

use std::collections::HashMap;
use std::fmt;

use crate::duckdb_bridge::duckdb_types::{LimitNodeType, LogicalOperatorType, OrderType};
use crate::dynamic_filter::DynamicFilter;
use crate::expression::{Expression, type_name};

impl LogicalOperatorType {
    /// Reconstruct a [`LogicalOperatorType`] from the `u8` discriminant the
    /// bridge reports. DuckDB defines it as `enum class : uint8_t`, so the
    /// discriminant round-trips exactly.
    pub(crate) fn from_u8(value: u8) -> Self {
        unsafe { std::mem::transmute::<u8, LogicalOperatorType>(value) }
    }
}

impl LimitNodeType {
    /// Reconstruct a [`LimitNodeType`] from the `u8` discriminant the bridge
    /// reports. DuckDB defines it as `enum class : uint8_t`.
    pub(crate) fn from_u8(value: u8) -> Self {
        unsafe { std::mem::transmute::<u8, LimitNodeType>(value) }
    }
}

/// Raw, pre-resolution form of a table scan as it comes off the bridge plan.
///
/// `PlanNode::resolve_inputs` turns each `RawInput` into an [`Input`] by looking
/// up the table by `table_id` and binding each [`DynamicFilter`] to its slot.
#[derive(Debug)]
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

/// A scan over a table-valued function (e.g. `generate_series(1, 10)`) rather
/// than a base table. Carries the function name, its bound constant arguments,
/// and the projected output columns (positional `BOUND_REF`s, like
/// [`RawInput`]). The rows are regenerated on the Rust side from the name and
/// args; there is no `table_id` because nothing is read off disk.
#[derive(Debug)]
pub struct TableFunctionScan {
    pub function_name: String,
    pub args: Vec<crate::types::ScalarValue>,
    pub columns: Vec<Expression>,
}

/// Computes a list of output expressions from its child's columns.
#[derive(Debug)]
pub struct Projection {
    pub projections: Vec<Expression>,
}

/// Sort direction for an ORDER BY clause.
///
/// `Default` is what DuckDB emits when no explicit `ASC`/`DESC` is given
/// (equivalent to `Asc` in practice).
#[derive(Debug)]
#[repr(u8)]
pub enum OrderByDirection {
    Default = OrderType::ORDER_DEFAULT as u8,
    Asc = OrderType::ASCENDING as u8,
    Desc = OrderType::DESCENDING as u8,
}

impl OrderByDirection {
    /// Map the `u8` direction the bridge reports (a DuckDB `OrderType`) to a
    /// variant, treating any unrecognised value as [`OrderByDirection::Default`].
    pub(crate) fn from_u8(value: u8) -> Self {
        if value == OrderType::ASCENDING as u8 {
            OrderByDirection::Asc
        } else if value == OrderType::DESCENDING as u8 {
            OrderByDirection::Desc
        } else {
            OrderByDirection::Default
        }
    }
}

/// A single sort key within an ORDER BY or TopN operator.
#[derive(Debug)]
pub struct OrderByNode {
    pub direction: OrderByDirection,
    pub expression: Expression,
}

/// Sorts its input by one or more keys.
#[derive(Debug)]
pub struct OrderBy {
    pub order_bys: Vec<OrderByNode>,
}

/// GROUP BY + aggregate functions.
#[derive(Debug)]
pub struct Aggregate {
    pub groups: Vec<Expression>,
    pub expressions: Vec<Expression>,
}

/// Filters rows by one or more boolean conditions (implicitly ANDed).
#[derive(Debug)]
pub struct Filter {
    pub conditions: Vec<Expression>,
}

/// Combined ORDER BY + LIMIT (returns the top N rows).
#[derive(Debug)]
pub struct TopN {
    pub order_bys: Vec<OrderByNode>,
    pub limit: usize,
    pub offset: usize,
    /// Set when DuckDB's Top-N optimizer installed a dynamic-filter producer on
    /// this node: at runtime the operator publishes its current boundary into
    /// the shared slot so consumer scans elsewhere in the plan can prune.
    pub produces_dynamic_filter: Option<DynamicFilter>,
}

/// A bare `LIMIT … OFFSET …` with no ORDER BY (an ORDER BY + LIMIT is fused into
/// [`TopN`] by the optimizer). `limit` is `None` for an offset-only query (no
/// upper bound); a non-constant (percentage/expression) limit is rejected by the
/// bridge before it reaches here.
#[derive(Debug)]
pub struct Limit {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// A single column definition inside a CREATE TABLE statement.
#[derive(Debug)]
pub struct CreateTableColumn {
    pub name: String,
    pub col_type: crate::duckdb_bridge::duckdb_types::LogicalTypeId,
}

/// CREATE TABLE with an explicit column list.
#[derive(Debug)]
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
/// raw comparison join never otherwise reaches the Rust plan builder.
///
/// `table_id` is the *same* index as the narrow scan beneath it (both sides of
/// DuckDB's join are the one table), so `resolve_inputs` hands this node a clone
/// of that scan's resolved table — same instance, same filter-pushdown state,
/// which the materializer needs to index row groups by the scan's global ids.
#[derive(Debug)]
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
#[derive(Debug)]
pub struct DummyScan {}

/// `EXPLAIN <query>`. DuckDB wraps the optimized plan in a `LOGICAL_EXPLAIN`
/// whose single child is the plan being explained; the bridge emits that child
/// as this node's input. Carries no payload: the consumer renders the child
/// plan as text instead of running it.
#[derive(Debug)]
pub struct Explain {}

/// `SET <name> = <value>` (and, relabelled by the bridge, `RESET <name>`, which
/// arrives with `value = None`).
///
/// DuckDB binds a `SET` of *any* name without validating that the setting
/// exists — that check only fires at execution, which pivot never runs — so
/// `name` is whatever the user typed and it's up to the consumer to decide which
/// names it honours. `value` is the bound constant in string form as DuckDB
/// rendered it (so a boolean reads back as `"true"`/`"false"`); `None` for a
/// `RESET`.
#[derive(Debug)]
pub struct SetVariable {
    pub name: String,
    pub value: Option<String>,
}

/// A logical operator in the query plan. Discriminated by DuckDB's
/// [`LogicalOperatorType`](crate::duckdb_bridge::duckdb_types::LogicalOperatorType).
#[derive(Debug)]
pub enum Operator {
    // `LOGICAL_GET` over a base table is built as `RawInput`, then converted to
    // `Input` by `resolve_inputs` to attach the `DuckDBTable` trait object.
    #[doc(hidden)]
    RawInput(RawInput),
    Input(Input),
    // A table-function get carries no base table, so the bridge re-tags it as
    // `LOGICAL_CHUNK_GET` to distinguish it from a base-table `RawInput`.
    TableFunctionScan(TableFunctionScan),
    Projection(Projection),
    OrderBy(OrderBy),
    Aggregate(Aggregate),
    Filter(Filter),
    TopN(TopN),
    Limit(Limit),
    CreateTable(CreateTable),
    DummyScan(DummyScan),
    Explain(Explain),
    // The bridge tags both SET and RESET as `LOGICAL_SET` (RESET carries no value).
    Set(SetVariable),
    // The bridge collapses DuckDB's late-materialization SEMI join into this
    // node and tags it with `LOGICAL_COMPARISON_JOIN` (a raw join never
    // otherwise reaches Rust). Resolved to `Materialize` (table attached) the
    // same way `RawInput` becomes `Input`.
    #[doc(hidden)]
    RawMaterialize(RawMaterialize),
    Materialize(Materialize),
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
            Operator::TableFunctionScan(t) => Some(t.columns.iter().collect()),
            Operator::Filter(_)
            | Operator::OrderBy(_)
            | Operator::TopN(_)
            | Operator::Limit(_)
            | Operator::CreateTable(_)
            | Operator::DummyScan(_)
            | Operator::Explain(_)
            | Operator::Set(_)
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
            Operator::TableFunctionScan(t) => {
                let args: Vec<String> = t.args.iter().map(|a| a.raw_value.clone()).collect();
                write!(
                    f,
                    "TableFunctionScan({}({}))",
                    t.function_name,
                    args.join(", ")
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
                write!(
                    f,
                    "TopN(limit: {}, offset: {}, order: {})",
                    t.limit,
                    t.offset,
                    orders.join(", ")
                )
            }
            Operator::Limit(l) => {
                let limit = l
                    .limit
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "ALL".to_string());
                write!(
                    f,
                    "Limit(limit: {limit}, offset: {})",
                    l.offset.unwrap_or(0)
                )
            }
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
            Operator::Explain(_) => write!(f, "Explain"),
            Operator::Set(s) => match &s.value {
                Some(v) => write!(f, "Set({} = {v})", s.name),
                None => write!(f, "Reset({})", s.name),
            },
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
