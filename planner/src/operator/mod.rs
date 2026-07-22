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
mod create_table;
mod dummy_scan;
mod explain;
mod filter;
mod input;
mod insert;
mod limit;
mod materialize;
mod order_by;
mod projection;
mod set_variable;
mod table_function;
mod top_n;
pub(crate) mod values;

pub use aggregate::Aggregate;
pub use create_table::CreateTable;
pub use dummy_scan::DummyScan;
pub use explain::Explain;
pub use filter::Filter;
pub use input::Input;
pub use insert::Insert;
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
#[derive(Debug, Clone)]
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
    CreateTable(CreateTable),
    DummyScan(DummyScan),
    /// `SET`/`RESET` of a session variable — handled by the server, not compiled.
    SetVariable(SetVariable),
    /// Late-materialization fetch (synthesized by the rewrite, see [`Materialize`]).
    Materialize(Materialize),
    /// `EXPLAIN <query>`: renders its child plan as text (see [`Explain`]).
    Explain(Explain),
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
            // A FROM-less SELECT's one-row source has no columns of its own.
            Operator::DummyScan(_) => Ok(Vec::new()),
            // EXPLAIN renders its child plan as text, one line per row.
            Operator::Explain(_) => Ok(vec![Type::Utf8]),
            // Statements, not queries: no result columns.
            Operator::CreateTable(_) | Operator::SetVariable(_) => Ok(Vec::new()),
        }
    }

    /// Whether this operator, as a plan root, streams result rows to the
    /// client. Statements that only report a command tag (INSERT's count, DDL,
    /// a session SET) do not; the wire protocol answers Describe with "no
    /// data" for them and ignores result formats. Exhaustive on purpose so a
    /// new operator must decide.
    pub fn returns_rows(&self) -> bool {
        match self {
            Operator::Insert(_) | Operator::CreateTable(_) | Operator::SetVariable(_) => false,
            Operator::Input(_)
            | Operator::Values(_)
            | Operator::TableFunctionScan(_)
            | Operator::Projection(_)
            | Operator::OrderBy(_)
            | Operator::Aggregate(_)
            | Operator::Filter(_)
            | Operator::TopN(_)
            | Operator::Limit(_)
            | Operator::DummyScan(_)
            | Operator::Materialize(_)
            | Operator::Explain(_) => true,
        }
    }
}

/// Generates [`Operator::expressions`] and [`Operator::expressions_mut`] from
/// one body: the top-level expression slots each operator owns, listed
/// exhaustively (no wildcard) so a new expression-bearing operator fails to
/// compile until it declares its slots.
macro_rules! operator_expressions {
    ($self:expr, $iter:ident $(, $mut_:tt)?) => {
        match $self {
            Operator::Input(input) => input.columns.$iter().collect(),
            Operator::TableFunctionScan(scan) => scan.columns.$iter().collect(),
            Operator::Projection(projection) => projection.projections.$iter().collect(),
            Operator::Filter(filter) => filter.conditions.$iter().collect(),
            Operator::Aggregate(aggregate) => aggregate
                .groups
                .$iter()
                .chain(aggregate.expressions.$iter())
                .collect(),
            Operator::OrderBy(order_by) => order_by
                .order_bys
                .$iter()
                .map(|key| & $($mut_)? key.expression)
                .collect(),
            Operator::TopN(top_n) => top_n
                .order_bys
                .$iter()
                .map(|key| & $($mut_)? key.expression)
                .collect(),
            Operator::Values(values) => values.rows.$iter().flatten().collect(),
            Operator::Insert(_)
            | Operator::CreateTable(_)
            | Operator::DummyScan(_)
            | Operator::SetVariable(_)
            | Operator::Materialize(_)
            | Operator::Explain(_)
            | Operator::Limit(_) => Vec::new(),
        }
    };
}

impl Operator {
    /// The operator's top-level expressions, in declaration order.
    pub fn expressions(&self) -> Vec<&Expression> {
        operator_expressions!(self, iter)
    }

    /// [`expressions`](Self::expressions), mutably.
    pub fn expressions_mut(&mut self) -> Vec<&mut Expression> {
        operator_expressions!(self, iter_mut, mut)
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
            Operator::CreateTable(c) => write!(f, "{c}"),
            Operator::DummyScan(d) => write!(f, "{d}"),
            Operator::SetVariable(s) => write!(f, "{s}"),
            Operator::Materialize(m) => write!(f, "{m}"),
            Operator::Explain(e) => write!(f, "{e}"),
        }
    }
}
