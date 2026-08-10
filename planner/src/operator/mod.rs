//! Operators in a Pivot [`Plan`](crate::Plan).
//!
//! Each variant of [`Operator`] corresponds to one stage of a plan
//! (scan a table, project columns, filter rows, aggregate, sort, top-N,
//! create a table). Operators are produced by convertion from
//! a [`duckdb_planner::handle::Operator`].
//!
//! Each operator *kind* lives in its own submodule co-locating the AST type,
//! its `TryFrom` from the DuckDB operator, its `Display`, and its `compile`
//! impl lowering it into a
//! [`RecordBatchOperatorSpec`](dispatch::RecordBatchOperatorSpec). This module
//! holds the cross-cutting pieces: the [`Operator`] enum that ties the kinds
//! together, the conversion [`enum@Error`], the shared ORDER BY key types, and
//! the dynamic-filter slot helper used by more than one operator.

mod aggregate;
mod compact;
mod create_schema;
mod create_table;
mod cte;
mod dummy_scan;
mod empty_result;
mod explain;
mod filter;
mod input;
mod insert;
mod join;
mod limit;
mod materialize;
mod order_by;
mod projection;
mod set_variable;
mod table_function;
mod top_n;
mod values;

pub use aggregate::Aggregate;
pub use compact::Compact;
pub use create_schema::CreateSchema;
pub use create_table::CreateTable;
pub use cte::{Cte, CteScan};
pub use dummy_scan::DummyScan;
pub use empty_result::EmptyResult;
pub use explain::Explain;
pub use filter::Filter;
pub use input::Input;
pub use insert::Insert;
pub use join::{Join, JoinKind};
pub use limit::Limit;
pub use materialize::Materialize;
pub use order_by::{OrderBy, OrderByDirection, OrderByNode};
pub use projection::Projection;
pub use set_variable::SetVariable;
pub use table_function::{TableFunction, TableFunctionScan, TableFunctionSignature};
pub use top_n::TopN;
pub use values::Values;

use crate::compile::{self, DynamicFilterSlots};
use crate::expression::{self, Expression};
use crate::types::Type;
use dispatch::DynamicFilterSlot;
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    Expression(#[from] expression::Error),
    #[error("{0}")]
    Type(#[from] crate::types::Error),
    /// A plan shape pivot doesn't support yet (an unmapped operator, a
    /// non-constant LIMIT, a table function with named parameters, ...).
    #[error("{0}")]
    Unsupported(String),
    /// A subtree's output columns couldn't be typed while shaping the plan
    /// (the join projection-map replay needs each side's width and types).
    #[error("{0}")]
    Typing(#[from] compile::Error),
    /// A DuckDB exception surfaced while reading the plan across the bridge.
    #[error("{0}")]
    Bridge(#[from] duckdb_planner::BridgeError),
}

/// Get-or-create the shared [`DynamicFilterSlot`] for `slot_id` within this
/// compile. A producer ([`TopN`]) and the consumer scans ([`Input`])
/// referencing the same id resolve to one `Arc`; a later compile of the same
/// (cached) plan mints fresh, empty slots — so no stale boundary or pooled scan
/// memory is reused.
pub(super) fn slot_for(slots: &mut DynamicFilterSlots, slot_id: usize) -> Arc<DynamicFilterSlot> {
    Arc::clone(
        slots
            .entry(slot_id)
            .or_insert_with(|| Arc::new(DynamicFilterSlot::new())),
    )
}

/// An operator in the query plan.
#[derive(Debug)]
pub enum Operator {
    Input(Input),
    Values(Values),
    Insert(Insert),
    /// A scan over a table-valued function (e.g. `generate_series`).
    TableFunctionScan(TableFunctionScan),
    Projection(Projection),
    OrderBy(OrderBy),
    Aggregate(Aggregate),
    Filter(Filter),
    TopN(TopN),
    Limit(Limit),
    /// Inner hash equi-join: probe (first input) against build (second input).
    Join(Join),
    CreateTable(CreateTable),
    CreateSchema(CreateSchema),
    DummyScan(DummyScan),
    /// A source that emits no rows (an optimized-away subtree), keeping the
    /// subtree's output types.
    EmptyResult(EmptyResult),
    /// `SET`/`RESET` of a session variable — handled by the server, not compiled.
    SetVariable(SetVariable),
    /// `COMPACT <table> [FINAL]` — handled by the server, not compiled.
    Compact(Compact),
    /// Late-materialization fetch (synthesized by the rewrite, see [`Materialize`]).
    Materialize(Materialize),
    /// `EXPLAIN <query>`: renders its child plan as text (see [`Explain`]).
    Explain(Explain),
    /// A CTE: its definition (first input) feeding the query reading it (second).
    Cte(Cte),
    /// One place a CTE's rows are read (see [`CteScan`]).
    CteScan(CteScan),
}

impl Operator {
    /// The pivot [`Type`] of each column this operator emits, in output order,
    /// given its inputs' column types (one `Vec` per child, in child order).
    /// This is what lets a pass over the plan ask what any subtree produces
    /// (e.g. rendering variant output columns as JSON text) without knowing
    /// operator shapes.
    pub fn output_types(&self, inputs: &[Vec<Type>]) -> Result<Vec<Type>, compile::Error> {
        match self {
            // Sources: a scan's outputs are its column expressions. (The extra
            // row-group metadata columns an `emit_row_group_metadata` scan
            // appends are plumbing between that scan and its Materialize, not
            // part of the logical schema.)
            Operator::Input(input) => expression_types(&input.columns),
            Operator::TableFunctionScan(scan) => scan.output_types(),
            // A materialize re-fetches table columns by storage index.
            Operator::Materialize(materialize) => {
                let columns = materialize.table.columns();
                Ok(materialize
                    .columns
                    .iter()
                    .map(|&i| columns[i].col_type.clone())
                    .collect())
            }
            // A projection reshapes its input into its expressions.
            Operator::Projection(projection) => expression_types(&projection.projections),
            // An aggregate emits its group keys, then one column per aggregate.
            Operator::Aggregate(aggregate) => {
                let mut types = expression_types(&aggregate.groups)?;
                types.extend(expression_types(&aggregate.expressions)?);
                Ok(types)
            }
            // These only reorder or trim rows; the columns pass through.
            Operator::Filter(_) | Operator::OrderBy(_) | Operator::TopN(_) | Operator::Limit(_) => {
                Ok(inputs[0].clone())
            }
            // A VALUES source emits its expression columns; every row shares
            // the same types, so the first row speaks for all.
            Operator::Values(values) => {
                expression_types(values.rows.first().map_or(&[][..], Vec::as_slice))
            }
            // An INSERT emits one row: the inserted-row count.
            Operator::Insert(_) => Ok(vec![Type::Int64]),
            // A join emits its listed probe columns followed by its listed
            // build columns.
            Operator::Join(join) => Ok(join
                .probe_output
                .iter()
                .map(|&i| inputs[0][i].clone())
                .chain(join.build_output.iter().map(|&i| inputs[1][i].clone()))
                .collect()),
            // A FROM-less SELECT's one-row source has no columns of its own.
            Operator::DummyScan(_) => Ok(Vec::new()),
            // An optimized-away subtree: no rows, but the types survive.
            Operator::EmptyResult(empty) => Ok(empty.types.clone()),
            // EXPLAIN renders its child plan as text, one line per row.
            Operator::Explain(_) => Ok(vec![Type::Utf8]),
            // A CTE emits what the query reading it emits; the definition
            // under its first input is only a source for the scans.
            Operator::Cte(_) => Ok(inputs[1].clone()),
            // A scan of a CTE emits what that CTE's definition produces,
            // recorded when the definition was walked.
            Operator::CteScan(scan) => Ok(scan.types.clone()),
            // Statements, not queries: no result columns.
            Operator::CreateTable(_)
            | Operator::CreateSchema(_)
            | Operator::SetVariable(_)
            | Operator::Compact(_) => Ok(Vec::new()),
        }
    }

    /// Whether each column this operator emits can hold SQL NULLs, in output
    /// order: the nullability companion to [`output_types`](Self::output_types),
    /// resolved per expression via [`Expression::nullability`]. The planner uses
    /// it to route between the branch-free and the null-aware execution paths,
    /// so `false` must be sound while `true` merely costs the fast path.
    pub fn output_nullability(&self, inputs: &[Vec<bool>]) -> Vec<bool> {
        match self {
            // A scan's nullability comes from the table binding, which reports
            // whether each column's data can actually hold NULLs.
            Operator::Input(input) => {
                let table = input.table.nullability();
                input
                    .columns
                    .iter()
                    .map(|e| e.nullability(&table))
                    .collect()
            }
            // Table functions do not report nullability; stay conservative.
            Operator::TableFunctionScan(scan) => {
                vec![true; scan.output_types().map_or(0, |t| t.len())]
            }
            Operator::Materialize(materialize) => {
                let nullability = materialize.table.nullability();
                materialize
                    .columns
                    .iter()
                    .map(|&i| nullability[i])
                    .collect()
            }
            Operator::Projection(projection) => projection
                .projections
                .iter()
                .map(|e| e.nullability(&inputs[0]))
                .collect(),
            Operator::Aggregate(aggregate) => aggregate
                .groups
                .iter()
                .chain(aggregate.expressions.iter())
                .map(|e| e.nullability(&inputs[0]))
                .collect(),
            Operator::Filter(_) | Operator::OrderBy(_) | Operator::TopN(_) | Operator::Limit(_) => {
                inputs[0].clone()
            }
            Operator::DummyScan(_) => Vec::new(),
            Operator::EmptyResult(empty) => vec![true; empty.types.len()],
            Operator::Explain(_) => vec![false],
            // A VALUES row may hold NULL literals; stay conservative.
            Operator::Values(values) => {
                vec![true; values.rows.first().map_or(0, Vec::len)]
            }
            // An INSERT emits one row: the inserted-row count, never NULL.
            Operator::Insert(_) => vec![false],
            // An inner join only emits rows built from both inputs, so each
            // output column keeps its own side's nullability; so does a semi
            // join, whose output is probe columns the build side matched. A
            // build-side outer join also emits unmatched build rows, filling
            // their probe columns with NULL regardless of the probe input's
            // declared nullability.
            Operator::Join(join) => {
                let probe_nullable = matches!(join.kind, JoinKind::BuildOuter);
                join.probe_output
                    .iter()
                    .map(|&i| probe_nullable || inputs[0][i])
                    .chain(join.build_output.iter().map(|&i| inputs[1][i]))
                    .collect()
            }
            Operator::Cte(_) => inputs[1].clone(),
            Operator::CteScan(scan) => scan.nullable.clone(),
            Operator::CreateTable(_)
            | Operator::CreateSchema(_)
            | Operator::SetVariable(_)
            | Operator::Compact(_) => Vec::new(),
        }
    }
}

/// The result type of each expression, in order.
fn expression_types(expressions: &[Expression]) -> Result<Vec<Type>, compile::Error> {
    expressions.iter().map(Expression::result_type).collect()
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::Input(i) => write!(f, "{i}"),
            Operator::Values(v) => write!(f, "{v}"),
            Operator::Insert(i) => write!(f, "{i}"),
            Operator::TableFunctionScan(t) => write!(f, "{t}"),
            Operator::Projection(p) => write!(f, "{p}"),
            Operator::OrderBy(o) => write!(f, "{o}"),
            Operator::Aggregate(a) => write!(f, "{a}"),
            Operator::Filter(fl) => write!(f, "{fl}"),
            Operator::TopN(t) => write!(f, "{t}"),
            Operator::Limit(l) => write!(f, "{l}"),
            Operator::Join(j) => write!(f, "{j}"),
            Operator::CreateTable(c) => write!(f, "{c}"),
            Operator::CreateSchema(c) => write!(f, "{c}"),
            Operator::DummyScan(d) => write!(f, "{d}"),
            Operator::EmptyResult(e) => write!(f, "{e}"),
            Operator::SetVariable(s) => write!(f, "{s}"),
            Operator::Compact(c) => write!(f, "{c}"),
            Operator::Materialize(m) => write!(f, "{m}"),
            Operator::Explain(e) => write!(f, "{e}"),
            Operator::Cte(c) => write!(f, "{c}"),
            Operator::CteScan(c) => write!(f, "{c}"),
        }
    }
}
