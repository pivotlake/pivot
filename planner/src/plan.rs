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
use crate::expression::{AggregateFunc, Expression, Function, NumericAggregate, Ref};
use crate::operator::{self, Operator};
use crate::types::Type;
use std::collections::{HashMap, HashSet};
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
    /// Collapse DuckDB's `AVG` lowering back into a single `Avg` aggregate.
    ///
    /// DuckDB rewrites `AVG(c)` into a `sum(c)` + `count(c)` aggregate pair with
    /// a parent `sum / count` divide projection. Executed verbatim that forces
    /// the exact `i128` sum through pivot's `Int64` aggregate output column,
    /// truncating large sums (e.g. `AVG(UserID)`). Recognising the pattern and
    /// folding it into one `Avg(c)` aggregate lets the operator divide the full
    /// `i128` sum by the count in `f64`, matching DuckDB's `(double)sum/count`.
    pub(crate) fn collapse_avg(&mut self) {
        for child in &mut self.inputs {
            child.collapse_avg();
        }

        // Shape must be: Projection whose only child is a groupless Aggregate.
        if !matches!(self.operator, Operator::Projection(_)) || self.inputs.len() != 1 {
            return;
        }
        let agg_exprs = match &self.inputs[0].operator {
            Operator::Aggregate(a) if a.groups.is_empty() => a.expressions.clone(),
            _ => return,
        };
        let projections = match &self.operator {
            Operator::Projection(p) => p.projections.clone(),
            _ => return,
        };

        // Collect `divide(Ref(i), Ref(j))` projections where the aggregate has
        // `Sum(c)` at i and `Count(c)` at j for the same column c.
        // Each tuple: (projection position, sum index, count index, column).
        let mut pairs: Vec<(usize, usize, usize, Ref)> = Vec::new();
        for (pos, p) in projections.iter().enumerate() {
            let Expression::Function(Function::Divide(d)) = p else {
                continue;
            };
            let (Expression::Ref(li), Expression::Ref(rj)) = (d.left.as_ref(), d.right.as_ref())
            else {
                continue;
            };
            if let (
                Some(Expression::AggregateFunc(AggregateFunc::Sum(s))),
                Some(Expression::AggregateFunc(AggregateFunc::Count(c))),
            ) = (agg_exprs.get(li.column_idx), agg_exprs.get(rj.column_idx))
            {
                if s.column.column_idx == c.column.column_idx {
                    pairs.push((pos, li.column_idx, rj.column_idx, s.column.clone()));
                }
            }
        }
        if pairs.is_empty() {
            return;
        }

        // Rebuild the aggregate's expression list: drop each consumed count,
        // turn each consumed sum into an Avg, keep everything else. Track the
        // old-index -> new-index remapping for projection rewriting.
        let removed: HashSet<usize> = pairs.iter().map(|&(_, _, j, _)| j).collect();
        let to_avg: HashMap<usize, Ref> =
            pairs.iter().map(|(_, i, _, col)| (*i, col.clone())).collect();

        let mut new_exprs: Vec<Expression> = Vec::new();
        let mut old_to_new: Vec<Option<usize>> = vec![None; agg_exprs.len()];
        for (k, e) in agg_exprs.iter().enumerate() {
            if removed.contains(&k) {
                continue;
            }
            old_to_new[k] = Some(new_exprs.len());
            match to_avg.get(&k) {
                Some(col) => new_exprs.push(Expression::AggregateFunc(AggregateFunc::Avg(
                    NumericAggregate { column: col.clone() },
                ))),
                None => new_exprs.push(e.clone()),
            }
        }

        // Rewrite projections: each divide becomes a ref to its new Avg column;
        // each surviving column ref is renumbered.
        let pair_pos: HashMap<usize, usize> = pairs.iter().map(|&(pos, i, _, _)| (pos, i)).collect();
        let new_projs: Vec<Expression> = projections
            .iter()
            .enumerate()
            .map(|(pos, p)| {
                if let Some(&i) = pair_pos.get(&pos) {
                    Expression::Ref(Ref {
                        column_idx: old_to_new[i].unwrap(),
                        return_type: Type::Float64,
                    })
                } else if let Expression::Ref(r) = p {
                    Expression::Ref(Ref {
                        column_idx: old_to_new[r.column_idx]
                            .expect("projection references a collapsed aggregate column"),
                        return_type: r.return_type.clone(),
                    })
                } else {
                    p.clone()
                }
            })
            .collect();

        if let Operator::Aggregate(a) = &mut self.inputs[0].operator {
            a.expressions = new_exprs;
        }
        if let Operator::Projection(p) = &mut self.operator {
            p.projections = new_projs;
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
