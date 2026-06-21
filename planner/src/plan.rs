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
use crate::operator::{self, Operator, OrderByDirection, SetVariable};
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
    /// <agg col> DESC [, …] LIMIT k)` and annotate the aggregate with `top_k`, so
    /// the group operator emits only each partition's top-k rows instead of every
    /// group (top-k is decomposable across partitions). Without it, a
    /// high-cardinality grouped `ORDER BY <agg> DESC LIMIT k` materialises,
    /// decodes, and fully sorts every group only to keep `k` — the dominant cost
    /// on such queries.
    ///
    /// Driven by the *primary* (first) order key, which must be a DESC ref to an
    /// aggregate column traced through column-ref-only projections. Secondary
    /// order keys (tiebreakers) are left to the surviving `TopN`, which re-sorts
    /// the per-partition candidates under the full order. Exact when the primary
    /// key has no ties at the limit boundary; with such ties it is tie-approximate
    /// — the same class of approximation a single-key DESC LIMIT pushdown already
    /// makes, additionally dropping secondary-key disambiguation among the tied
    /// boundary group.
    pub(crate) fn annotate_group_topn(&mut self) {
        for child in &mut self.inputs {
            child.annotate_group_topn();
        }

        let (mut col, limit) = match &self.operator {
            Operator::TopN(t) if !t.order_bys.is_empty() => {
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

impl Plan {
    /// If this plan is a bare `SET`/`RESET`, return it. Such a statement is a
    /// session command, not a query — it compiles to nothing — so the server
    /// checks this first and acts on the variables it recognises instead of
    /// running a dataflow.
    ///
    /// Requires no child operators: a plain `SET x = <const>`/`RESET x` has none,
    /// whereas DuckDB's `SET VARIABLE x = <expr>` (a user variable, which pivot
    /// doesn't support) carries its value as a child subtree — exclude that so it
    /// falls through to a clean "not compilable" error rather than being applied
    /// with a bogus value.
    pub fn as_set_variable(&self) -> Option<&SetVariable> {
        match &self.root.operator {
            Operator::SetVariable(set) if self.root.inputs.is_empty() => Some(set),
            _ => None,
        }
    }
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.root)
    }
}
