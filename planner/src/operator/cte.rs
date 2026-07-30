//! [`Cte`] and [`CteScan`] — a subquery computed once and read in several places.

use crate::compile::Error;
use crate::types::Type;
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use std::fmt;

/// A CTE: its definition, then the query reading it.
///
/// The first input produces the rows; the second is the query that reads them
/// through [`CteScan`]s, and is what this node emits. DuckDB materializes a CTE
/// (rather than inlining a copy of it per reference) when computing it once is
/// the cheaper bet, e.g. when it ends in an aggregate.
#[derive(Debug)]
pub struct Cte {
    /// DuckDB's index for this CTE, which its [`CteScan`]s carry.
    pub cte_index: usize,
    /// How many places in the body read it.
    pub sites: usize,
}

impl fmt::Display for Cte {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cte(#{}, sites: {})", self.cte_index, self.sites)
    }
}

impl Cte {
    pub(crate) fn compile(
        &self,
        definition: RecordBatchOperatorSpec,
        body: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        Ok(body.with_cte(definition, self.cte_index, self.sites))
    }
}

/// One place a CTE's rows are read.
///
/// A leaf of the plan: the rows arrive from the CTE's definition, which lives
/// under the [`Cte`] node above this one, rather than from a table.
#[derive(Debug)]
pub struct CteScan {
    /// Which CTE these rows come from.
    pub cte_index: usize,
    /// What the CTE's definition produces, in output order.
    pub types: Vec<Type>,
    /// Whether each of those columns can hold NULLs.
    pub nullable: Vec<bool>,
}

impl fmt::Display for CteScan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CteScan(#{})", self.cte_index)
    }
}

impl CteScan {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        Ok(RecordBatchOperatorSpec::cte_scan(
            dispatcher,
            self.cte_index,
        ))
    }
}
