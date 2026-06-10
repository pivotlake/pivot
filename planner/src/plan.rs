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
    /// Fuse `Limit → OrderBy` into a single [`TopN`](Operator::TopN).
    ///
    /// DuckDB only emits `LogicalTopN` when its Top-N optimizer decides the
    /// window is small enough; a large `LIMIT … OFFSET` over an ORDER BY is
    /// planned as a full sort with a separate `LogicalLimit` above it. Pivot's
    /// Top-N operator handles arbitrary windows (workers keep `limit + offset`
    /// candidates each), so re-fusing recovers the partial-sort path — and the
    /// `group → TopN` top-k annotation below — instead of a fully materialized
    /// sort followed by truncation.
    ///
    /// Unbounded limits (`OFFSET` without `LIMIT`) are left unfused: `TopN`
    /// needs a finite window, and the plain `OrderBy → Limit` pipeline is
    /// correct for them (the sort emits one globally sorted batch, which the
    /// limit stage merely slices).
    pub(crate) fn fuse_limit_order_by(&mut self) {
        for child in &mut self.inputs {
            child.fuse_limit_order_by();
        }

        let Operator::Limit(limit) = &self.operator else {
            return;
        };
        let Some(limit_rows) = limit.limit else {
            return;
        };
        if !matches!(
            self.inputs.first().map(|n| &n.operator),
            Some(Operator::OrderBy(_))
        ) {
            return;
        }

        let order_by_node = self.inputs.remove(0);
        let Operator::OrderBy(order_by) = order_by_node.operator else {
            unreachable!("matched OrderBy above");
        };
        self.operator = Operator::TopN(crate::operator::TopN {
            order_bys: order_by.order_bys,
            limit: limit_rows,
            offset: limit.offset,
            // DuckDB never installs a dynamic-filter producer on a plan it
            // declined to Top-N-optimize, so there is no slot to wire up.
            produces_dynamic_filter: None,
        });
        self.inputs = order_by_node.inputs;
    }

    /// Detect `grouped Aggregate → (projections) → Limit` with no ordering and
    /// annotate the aggregate with `output_limit = limit + offset`: any that
    /// many complete groups answer the query, so the group operator can stop
    /// merging partitions early. Projections preserve row count, so walking
    /// through them is safe; the `Limit` stays for the exact global window.
    pub(crate) fn annotate_group_limit(&mut self) {
        for child in &mut self.inputs {
            child.annotate_group_limit();
        }

        let Operator::Limit(limit) = &self.operator else {
            return;
        };
        let Some(rows) = limit.limit else {
            return;
        };
        let needed = rows + limit.offset;

        let mut node = match self.inputs.first_mut() {
            Some(n) => n,
            None => return,
        };
        loop {
            match &mut node.operator {
                Operator::Projection(_) => {
                    node = match node.inputs.first_mut() {
                        Some(n) => n,
                        None => return,
                    };
                }
                Operator::Aggregate(a) if !a.groups.is_empty() => {
                    a.output_limit = Some(needed);
                    return;
                }
                _ => return,
            }
        }
    }

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
                    // The per-partition window must cover the offset too: the
                    // global rows `offset..offset+limit` are only guaranteed to
                    // be among each partition's top `limit + offset`.
                    (OrderByDirection::Desc, Expression::Ref(r)) => {
                        (r.column_idx, t.limit + t.offset)
                    }
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
