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
use crate::expression::Expression;
use crate::operator::{self, Operator, OrderByDirection};
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

impl PlanNode {
    /// Detect `grouped Aggregate → (pass-through projections) → TopN(ORDER BY
    /// <agg col> DESC LIMIT k)` and annotate the aggregate with `top_k`, so the
    /// group operator emits only each partition's top-k rows instead of every
    /// group (top-k is decomposable across partitions). Only handles DESC with
    /// a single ref order key traced through column-ref-only projections.
    pub(crate) fn annotate_group_topn(&mut self) {
        for child in &mut self.inputs {
            child.annotate_group_topn();
        }

        let (mut col, limit) = match &self.operator {
            Operator::TopN(t) if t.order_bys.len() == 1 => {
                let ob = &t.order_bys[0];
                match (&ob.direction, &ob.expression) {
                    (OrderByDirection::Desc, Expression::Ref(r)) => (r.column_idx, t.limit),
                    _ => return,
                }
            }
            _ => return,
        };

        // Walk down single-input projections, mapping the order column, until
        // we reach the aggregate.
        let mut node = match self.inputs.first_mut() {
            Some(n) => n,
            None => return,
        };
        loop {
            enum Step {
                Proj(usize),
                SetAgg(usize),
                Stop,
            }
            let step = match &node.operator {
                Operator::Projection(p) => match p.projections.get(col) {
                    Some(Expression::Ref(r)) => Step::Proj(r.column_idx),
                    _ => Step::Stop,
                },
                Operator::Aggregate(a) if !a.groups.is_empty() && col >= a.groups.len() => {
                    Step::SetAgg(col - a.groups.len())
                }
                _ => Step::Stop,
            };
            match step {
                Step::Proj(next) => {
                    col = next;
                    node = match node.inputs.first_mut() {
                        Some(n) => n,
                        None => return,
                    };
                }
                Step::SetAgg(slot) => {
                    if let Operator::Aggregate(a) = &mut node.operator {
                        a.top_k = Some((slot, limit));
                    }
                    return;
                }
                Step::Stop => return,
            }
        }
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
