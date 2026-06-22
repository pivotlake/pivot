//! Operators in a Pivot [`Plan`](crate::Plan).
//!
//! Each variant of [`Operator`] corresponds to one stage of a plan
//! (scan a table, project columns, filter rows, aggregate, sort, top-N,
//! create a table). Operators are produced by convertion from
//! a [`duckdb_operator::Operator`].
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
mod filter;
mod input;
mod limit;
mod materialize;
mod order_by;
mod projection;
mod set_variable;
mod table_function;
mod top_n;

pub use aggregate::Aggregate;
pub use create_table::CreateTable;
pub use dummy_scan::DummyScan;
pub use filter::Filter;
pub use input::Input;
pub use limit::Limit;
pub use materialize::Materialize;
pub use order_by::{OrderBy, OrderByDirection, OrderByNode};
pub use projection::Projection;
pub use set_variable::SetVariable;
pub use table_function::{TableFunction, TableFunctionScan, TableFunctionSignature};
pub use top_n::TopN;

use crate::compile::DynamicFilterSlots;
use crate::expression::{self};
use dispatch::DynamicFilterSlot;
use duckdb_planner::operator as duckdb_operator;
use std::fmt;
use std::sync::{Arc, RwLock};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    Expression(#[from] expression::Error),
    #[error("{0}")]
    Type(#[from] crate::types::Error),
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
            .or_insert_with(|| Arc::new(RwLock::new(None))),
    )
}

/// An operator in the query plan.
#[derive(Debug)]
pub enum Operator {
    Input(Input),
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
}

impl TryFrom<duckdb_operator::Operator> for Operator {
    type Error = Error;

    fn try_from(op: duckdb_operator::Operator) -> Result<Self, Self::Error> {
        Ok(match op {
            duckdb_operator::Operator::Input(s) => Operator::Input(s.try_into()?),
            duckdb_operator::Operator::TableFunctionScan(t) => {
                Operator::TableFunctionScan(t.try_into()?)
            }
            duckdb_operator::Operator::Projection(p) => Operator::Projection(p.try_into()?),
            duckdb_operator::Operator::OrderBy(o) => Operator::OrderBy(o.try_into()?),
            duckdb_operator::Operator::Aggregate(a) => Operator::Aggregate(a.try_into()?),
            duckdb_operator::Operator::Filter(f) => Operator::Filter(f.try_into()?),
            duckdb_operator::Operator::TopN(t) => Operator::TopN(t.try_into()?),
            duckdb_operator::Operator::Limit(l) => Operator::Limit(l.try_into()?),
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
        }
    }
}
