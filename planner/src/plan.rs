//! The [`Plan`] tree: Pivot's plan IR.
//!
//! A [`Plan`] is the result of translating a [`duckdb_planner::PlanNode`] into
//! Pivot's own structs (the `TryFrom` impl below recurses through the
//! tree). Each [`PlanNode`] holds an [`Operator`] and its child plan nodes.
//!
//! The [`Plan`] carries the [`Catalog`] it was bound against (operators like
//! `CREATE TABLE` need it at execution time). The dispatcher, on the other
//! hand, is passed in at [`compile`](crate::compile) time — it represents
//! "what worker pool runs this plan" and isn't a property of the plan itself.

use crate::catalog::Catalog;
use crate::operator::{self, Operator};
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    Operator(#[from] operator::Error),
}

/// One node in a [`Plan`] tree: an [`Operator`] plus its (already-translated)
/// child nodes. `name` carries the DuckDB-side label, useful for debugging.
#[derive(Debug)]
pub struct PlanNode {
    pub name: String,
    pub inputs: Vec<PlanNode>,
    pub operator: Operator,
}

impl TryFrom<duckdb_planner::PlanNode> for PlanNode {
    type Error = Error;

    fn try_from(p: duckdb_planner::PlanNode) -> Result<Self, Self::Error> {
        Ok(PlanNode {
            name: p.name,
            inputs: p
                .inputs
                .into_iter()
                .map(PlanNode::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            operator: p.operator.try_into()?,
        })
    }
}

impl PlanNode {
    fn fmt_indented(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let prefix = "  ".repeat(indent);
        writeln!(f, "{prefix}{}", self.operator)?;
        for child in &self.inputs {
            child.fmt_indented(f, indent + 1)?;
        }
        Ok(())
    }
}

impl fmt::Display for PlanNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indented(f, 0)
    }
}

/// A fully-translated plan ready for compilation.
///
/// Produced by [`Planner::plan`](crate::Planner::plan); converted into an
/// executable [`RecordBatchOperatorSpec`](dispatch::RecordBatchOperatorSpec)
/// via [`Plan::compile`](crate::compile), which takes the dispatcher.
#[derive(Debug)]
pub struct Plan {
    pub catalog: Arc<dyn Catalog>,
    pub root: PlanNode,
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.root)
    }
}
