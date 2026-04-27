//! The [`Plan`] tree: Pivot's plan IR.
//!
//! A [`Plan`] is the result of translating a [`duckdb_planner::PlanNode`] into
//! Pivot's own structs (the `TryFrom` impl below recurses through the
//! tree). Each [`PlanNode`] holds an [`Operator`] and its child plan nodes.
//!
//! [`PlanContext`] is threaded through the lowering step in
//! [`compile`](crate::compile) so operators that need to act on the
//! catalog / read settings at execution time can use it.

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

/// State shared across the whole compile pass for one [`Plan`].
///
/// Currently just the [`Catalog`], handed to operators that need it
/// (e.g. for `CREATE TABLE` to mutate the catalog at execution time).
#[derive(Debug, Clone)]
pub struct PlanContext {
    pub catalog: Arc<dyn Catalog>,
}

/// A fully-translated plan ready for compilation.
///
/// Produced by [`Planner::plan`](crate::Planner::plan); converted into an
/// executable [`RecordBatchOperatorSpec`](dispatch::RecordBatchOperatorSpec)
/// via [`Plan::compile`](crate::compile).
#[derive(Debug)]
pub struct Plan {
    pub plan_context: PlanContext,
    pub root: PlanNode,
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.root)
    }
}
