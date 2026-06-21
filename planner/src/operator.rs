//! Operators in a Pivot [`Plan`](crate::Plan).
//!
//! Each variant of [`Operator`] corresponds to one stage of a plan
//! (scan a table, project columns, filter rows, aggregate, sort, top-N,
//! create a table). Operators are produced by convertion from
//! a [`duckdb_operator::Operator`].
//!
//! Operators only describe *what* to do, not *how*; the lowering to a
//! [`RecordBatchOperatorSpec`](dispatch::RecordBatchOperatorSpec) happens in
//! [`crate::compile`].

use crate::catalog::{CreateTableRequest, DuckDBTableAdapter, Table};
use crate::dynamic_filter::DynamicFilter;
use crate::expression::{self, Expression};
use crate::types::{self, type_from_logical};
use duckdb_planner::operator as duckdb_operator;
use std::any::Any;
use std::fmt;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    Expression(#[from] expression::Error),
    #[error("{0}")]
    Type(#[from] types::Error),
}

/// Scans a [`Table`] from the catalog. `columns` lists the requested output
/// columns (each as a [`Ref`](crate::expression::Ref) into the table schema)
/// and `filters` are predicates pushed down into the scan.
#[derive(Debug)]
pub struct Input {
    pub table: Box<dyn Table>,
    pub columns: Vec<Expression>,
    /// Runtime-populated predicates the scan reads from shared slots — installed
    /// by a Top-N (or, in future, a hash join) elsewhere in the plan. Each
    /// prunes row groups against the producer's live boundary value.
    pub dynamic_filters: Vec<DynamicFilter>,
    /// When `true`, the scan tags each emitted row with its global row-group ID
    /// and per-row index (extra metadata columns appended after the data
    /// columns). Set on the narrow scan of a late-materialized query so a
    /// downstream [`Materialize`] can fetch the remaining columns for only the
    /// surviving rows. Always `false` for ordinary scans.
    pub emit_row_group_metadata: bool,
}

impl TryFrom<duckdb_operator::Input> for Input {
    type Error = Error;
    fn try_from(s: duckdb_operator::Input) -> Result<Self, Self::Error> {
        let any: Box<dyn Any> = s.table;
        let wrapper: Box<DuckDBTableAdapter> = any
            .downcast::<DuckDBTableAdapter>()
            .expect("Input.table should be a DuckDBTableAdapter");
        Ok(Input {
            table: wrapper.table,
            columns: s
                .columns
                .into_iter()
                .map(Expression::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            dynamic_filters: s
                .dynamic_filters
                .into_iter()
                .map(DynamicFilter::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            emit_row_group_metadata: s.emit_row_group_metadata,
        })
    }
}

impl TryFrom<duckdb_operator::Materialize> for Materialize {
    type Error = Error;
    fn try_from(m: duckdb_operator::Materialize) -> Result<Self, Self::Error> {
        // Mirror `Input`: the resolved DuckDB table is a `DuckDBTableAdapter`
        // wrapping the pivot `Table`; downcast and take it.
        let any: Box<dyn Any> = m.table;
        let wrapper: Box<DuckDBTableAdapter> = any
            .downcast::<DuckDBTableAdapter>()
            .expect("Materialize.table should be a DuckDBTableAdapter");
        Ok(Materialize {
            table: wrapper.table,
            columns: m.columns,
        })
    }
}

impl fmt::Display for Input {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cols = self
            .columns
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Input([{cols}])")
    }
}

/// Late-materialization fetch: re-reads `columns` from the table for the rows
/// that survived the narrow pipeline below it.
///
/// The bridge synthesizes this by collapsing DuckDB's `late_materialization`
/// row-id SEMI join (see the duckdb-planner `Materialize` operator). Its single
/// input is the narrow pipeline whose scan is tagged with row-group metadata
/// (see [`Input::emit_row_group_metadata`]); this node fetches the requested
/// columns for only the surviving rows, emitting them in `columns` order — which
/// matches the order DuckDB's full-column Get produced, so the projection kept
/// above it lines up positionally without remapping.
#[derive(Debug)]
pub struct Materialize {
    pub table: Box<dyn Table>,
    /// Table-schema (storage) column indices to fetch, in output order.
    pub columns: Vec<usize>,
}

impl fmt::Display for Materialize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cols = self
            .columns
            .iter()
            .map(|c| format!("#{c}"))
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Materialize([{cols}])")
    }
}

/// Computes a list of output expressions from its child's columns.
#[derive(Debug)]
pub struct Projection {
    pub projections: Vec<Expression>,
}

impl TryFrom<duckdb_operator::Projection> for Projection {
    type Error = Error;
    fn try_from(p: duckdb_operator::Projection) -> Result<Self, Self::Error> {
        Ok(Projection {
            projections: p
                .projections
                .into_iter()
                .map(Expression::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl fmt::Display for Projection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let exprs = self
            .projections
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Projection({exprs})")
    }
}

/// Sort direction specification for an ORDER BY clause.
#[derive(Debug)]
pub enum OrderByDirection {
    Default,
    Asc,
    Desc,
}

impl From<duckdb_operator::OrderByDirection> for OrderByDirection {
    fn from(d: duckdb_operator::OrderByDirection) -> Self {
        match d {
            duckdb_operator::OrderByDirection::Default => OrderByDirection::Default,
            duckdb_operator::OrderByDirection::Asc => OrderByDirection::Asc,
            duckdb_operator::OrderByDirection::Desc => OrderByDirection::Desc,
        }
    }
}

impl fmt::Display for OrderByDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            OrderByDirection::Asc | OrderByDirection::Default => "ASC",
            OrderByDirection::Desc => "DESC",
        };
        f.write_str(s)
    }
}

/// A single sort key within an ORDER BY or TopN operator.
#[derive(Debug)]
pub struct OrderByNode {
    pub direction: OrderByDirection,
    pub expression: Expression,
}

impl TryFrom<duckdb_operator::OrderByNode> for OrderByNode {
    type Error = Error;
    fn try_from(n: duckdb_operator::OrderByNode) -> Result<Self, Self::Error> {
        Ok(OrderByNode {
            direction: n.direction.into(),
            expression: n.expression.try_into()?,
        })
    }
}

impl fmt::Display for OrderByNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.expression, self.direction)
    }
}

/// Sorts its input by one or more keys.
#[derive(Debug)]
pub struct OrderBy {
    pub order_bys: Vec<OrderByNode>,
}

impl TryFrom<duckdb_operator::OrderBy> for OrderBy {
    type Error = Error;
    fn try_from(o: duckdb_operator::OrderBy) -> Result<Self, Self::Error> {
        Ok(OrderBy {
            order_bys: o
                .order_bys
                .into_iter()
                .map(OrderByNode::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl fmt::Display for OrderBy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let orders = self
            .order_bys
            .iter()
            .map(|o| o.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "OrderBy({orders})")
    }
}

/// GROUP BY + aggregate functions.
#[derive(Debug)]
pub struct Aggregate {
    pub groups: Vec<Expression>,
    pub expressions: Vec<Expression>,
    /// Set by the `group → TopN` detection pass to `Some((value_slot, limit))`
    /// when this grouped aggregate feeds an `ORDER BY <slot> DESC LIMIT limit`,
    /// so the group operator emits only each partition's top-`limit` rows.
    pub top_k: Option<(usize, usize)>,
}

impl TryFrom<duckdb_operator::Aggregate> for Aggregate {
    type Error = Error;
    fn try_from(a: duckdb_operator::Aggregate) -> Result<Self, Self::Error> {
        Ok(Aggregate {
            top_k: None,
            groups: a
                .groups
                .into_iter()
                .map(Expression::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            expressions: a
                .expressions
                .into_iter()
                .map(Expression::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl fmt::Display for Aggregate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let groups = self
            .groups
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let exprs = self
            .expressions
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Aggregate(groups: [{groups}], exprs: [{exprs}])")
    }
}

/// Filters rows by one or more boolean conditions (implicitly ANDed).
#[derive(Debug)]
pub struct Filter {
    pub conditions: Vec<Expression>,
}

impl TryFrom<duckdb_operator::Filter> for Filter {
    type Error = Error;
    fn try_from(f: duckdb_operator::Filter) -> Result<Self, Self::Error> {
        Ok(Filter {
            conditions: f
                .conditions
                .into_iter()
                .map(Expression::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl fmt::Display for Filter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let conds = self
            .conditions
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" AND ");
        write!(f, "Filter({conds})")
    }
}

/// Combined ORDER BY + LIMIT (returns the top N rows).
#[derive(Debug)]
pub struct TopN {
    pub order_bys: Vec<OrderByNode>,
    pub limit: usize,
    pub offset: usize,
    /// When set, this Top-N is a dynamic-filter producer: at runtime it
    /// publishes its current boundary value into the shared slot so consumer
    /// scans elsewhere in the plan can prune row groups against it.
    pub produces_dynamic_filter: Option<DynamicFilter>,
}

impl TryFrom<duckdb_operator::TopN> for TopN {
    type Error = Error;
    fn try_from(t: duckdb_operator::TopN) -> Result<Self, Self::Error> {
        Ok(TopN {
            order_bys: t
                .order_bys
                .into_iter()
                .map(OrderByNode::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            limit: t.limit,
            offset: t.offset,
            produces_dynamic_filter: t
                .produces_dynamic_filter
                .map(DynamicFilter::try_from)
                .transpose()?,
        })
    }
}

impl fmt::Display for TopN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let orders = self
            .order_bys
            .iter()
            .map(|o| o.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "TopN(limit: {}, offset: {}, order: {orders})",
            self.limit, self.offset
        )
    }
}

/// CREATE TABLE with an explicit column list.
#[derive(Debug)]
pub struct CreateTable {
    pub request: CreateTableRequest,
    pub or_replace: bool,
    pub temporary: bool,
    pub has_query: bool,
    pub constraint_count: usize,
}

impl TryFrom<duckdb_operator::CreateTable> for CreateTable {
    type Error = Error;

    fn try_from(create_table: duckdb_operator::CreateTable) -> Result<Self, Self::Error> {
        Ok(Self {
            request: CreateTableRequest {
                name: create_table.name,
                columns: create_table
                    .columns
                    .into_iter()
                    .map(|column| {
                        Ok(crate::catalog::Column {
                            name: column.name,
                            col_type: type_from_logical(column.col_type)?,
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?,
                options: create_table.options,
                if_not_exists: create_table.if_not_exists,
            },
            or_replace: create_table.or_replace,
            temporary: create_table.temporary,
            has_query: create_table.has_query,
            constraint_count: create_table.constraint_count,
        })
    }
}

impl fmt::Display for CreateTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let columns = self
            .request
            .columns
            .iter()
            .map(|c| format!("{}:{}", c.name, c.col_type))
            .collect::<Vec<_>>()
            .join(", ");
        // Sort by key so the rendered output is deterministic — `HashMap`
        // iteration order is randomized per process and would otherwise flake
        // any snapshot/equality test that includes options.
        let mut options: Vec<(&String, &String)> = self.request.options.iter().collect();
        options.sort_by(|a, b| a.0.cmp(b.0));
        let options_str = options
            .iter()
            .map(|(k, v)| format!("{k:?}: {v:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "CreateTable({}, [{columns}], options: {{{options_str}}})",
            self.request.name
        )
    }
}

/// The single-row source under a `FROM`-less `SELECT` (see
/// [`duckdb_operator::DummyScan`]).
#[derive(Debug)]
pub struct DummyScan;

impl TryFrom<duckdb_operator::DummyScan> for DummyScan {
    type Error = Error;

    fn try_from(_: duckdb_operator::DummyScan) -> Result<Self, Self::Error> {
        Ok(DummyScan)
    }
}

impl fmt::Display for DummyScan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DummyScan")
    }
}

/// `SET <name> = <value>` / `RESET <name>` (the latter arrives with no value).
///
/// A session knob, not a query: it produces no rows and isn't compiled into a
/// dataflow. The server inspects it after planning (see
/// [`Plan::as_set_variable`](crate::Plan::as_set_variable)) and acts on the names
/// it recognises. `value` is DuckDB's serialized constant (a boolean reads back
/// as `"true"`/`"false"`); `None` is a `RESET`.
#[derive(Debug)]
pub struct SetVariable {
    pub name: String,
    pub value: Option<String>,
}

impl TryFrom<duckdb_operator::SetVariable> for SetVariable {
    type Error = Error;

    fn try_from(set: duckdb_operator::SetVariable) -> Result<Self, Self::Error> {
        Ok(SetVariable {
            name: set.name,
            value: set.value.map(|v| v.raw_value),
        })
    }
}

impl fmt::Display for SetVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.value {
            Some(v) => write!(f, "Set({} = {v})", self.name),
            None => write!(f, "Reset({})", self.name),
        }
    }
}

/// An operator in the query plan.
#[derive(Debug)]
pub enum Operator {
    Input(Input),
    Projection(Projection),
    OrderBy(OrderBy),
    Aggregate(Aggregate),
    Filter(Filter),
    TopN(TopN),
    CreateTable(CreateTable),
    DummyScan(DummyScan),
    /// `SET`/`RESET` of a session variable — handled by the server, not compiled.
    SetVariable(SetVariable),
    /// Late-materialization fetch (synthesized by the rewrite, see [`Materialize`]).
    Materialize(Materialize),
}

impl TryFrom<duckdb_operator::Operator> for Operator {
    type Error = Error;

    fn try_from(op: duckdb_operator::Operator) -> Result<Self, Self::Error> {
        Ok(match op {
            duckdb_operator::Operator::Input(s) => Operator::Input(s.try_into()?),
            duckdb_operator::Operator::Projection(p) => Operator::Projection(p.try_into()?),
            duckdb_operator::Operator::OrderBy(o) => Operator::OrderBy(o.try_into()?),
            duckdb_operator::Operator::Aggregate(a) => Operator::Aggregate(a.try_into()?),
            duckdb_operator::Operator::Filter(f) => Operator::Filter(f.try_into()?),
            duckdb_operator::Operator::TopN(t) => Operator::TopN(t.try_into()?),
            duckdb_operator::Operator::CreateTable(c) => Operator::CreateTable(c.try_into()?),
            duckdb_operator::Operator::DummyScan(d) => Operator::DummyScan(d.try_into()?),
            duckdb_operator::Operator::Set(s) => Operator::SetVariable(s.try_into()?),
            duckdb_operator::Operator::Materialize(m) => Operator::Materialize(m.try_into()?),
            duckdb_operator::Operator::RawInput(_) => {
                unreachable!("RawInput should be resolved to Input before reaching the planner")
            }
            duckdb_operator::Operator::RawMaterialize(_) => {
                unreachable!("RawMaterialize should be resolved to Materialize before the planner")
            }
        })
    }
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::Input(i) => write!(f, "{i}"),
            Operator::Projection(p) => write!(f, "{p}"),
            Operator::OrderBy(o) => write!(f, "{o}"),
            Operator::Aggregate(a) => write!(f, "{a}"),
            Operator::Filter(fl) => write!(f, "{fl}"),
            Operator::TopN(t) => write!(f, "{t}"),
            Operator::CreateTable(c) => write!(f, "{c}"),
            Operator::DummyScan(d) => write!(f, "{d}"),
            Operator::SetVariable(s) => write!(f, "{s}"),
            Operator::Materialize(m) => write!(f, "{m}"),
        }
    }
}
