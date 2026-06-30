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
use dispatch::GroupLimit;
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
                    // Keep `limit + offset` rows per partition: the downstream
                    // `LIMIT k OFFSET m` discards the first `m`, so a group pruned
                    // to only `k` per partition would leave nothing past the
                    // offset (e.g. `LIMIT 10 OFFSET 1000`).
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
                        // Don't push top-k onto an aggregate whose `sort_key` isn't
                        // ordered: a string MIN/MAX (raw `ArenaKey`/StringView header
                        // bits, not lexicographic order) or any float SUM/MIN/MAX
                        // (raw `f64` bits, not numeric order). A per-partition top-k
                        // would keep the wrong rows, so leave it to the full `TopN`
                        // sort above (only the pushdown is skipped).
                        use crate::expression::AggregateFunc;
                        use crate::operator::is_float_type;
                        use crate::types::Type;
                        let unordered_sort_key = match a.expressions.get(slot) {
                            Some(Expression::AggregateFunc(
                                AggregateFunc::Min(x) | AggregateFunc::Max(x),
                            )) => {
                                x.column.return_type == Type::Utf8
                                    || is_float_type(&x.column.return_type)
                            }
                            Some(Expression::AggregateFunc(AggregateFunc::Sum(x))) => {
                                is_float_type(&x.column.return_type)
                            }
                            _ => false,
                        };
                        if !unordered_sort_key {
                            a.output_limit = Some(GroupLimit::TopK { slot, limit });
                        }
                    }
                    return;
                }
                Step::Stop => return,
            }
        }
    }

    /// Detect `grouped Aggregate → (pass-through projections) → Limit k` (a plain
    /// `LIMIT` with no ORDER BY) and annotate the aggregate with
    /// [`GroupLimit::First`], so each partition emits at most `k + offset` groups
    /// instead of every group. With no ordering, SQL leaves which rows the LIMIT
    /// keeps undefined, so any `k + offset` groups per partition are a valid
    /// candidate set — the surviving `Limit` trims to the final window.
    ///
    /// Mirrors [`annotate_group_topn`](Self::annotate_group_topn) but without a
    /// sort key: it doesn't need to trace an order column, only to confirm the
    /// chain `Limit → projections → grouped Aggregate`.
    pub(crate) fn annotate_group_limit(&mut self) {
        for child in &mut self.inputs {
            child.annotate_group_limit();
        }

        // Keep `limit + offset` per partition: a downstream `OFFSET m` discards
        // the first `m`, so pruning to only `k` would leave nothing past it.
        let window = match &self.operator {
            Operator::Limit(l) => match l.limit {
                Some(limit) => limit + l.offset,
                // An offset-only `LIMIT ALL` keeps every group; nothing to prune.
                None => return,
            },
            _ => return,
        };

        // Walk down single-input pass-through projections until the aggregate.
        let mut node = match self.inputs.first_mut() {
            Some(n) => n,
            None => return,
        };
        loop {
            let is_passthrough_projection = match &node.operator {
                Operator::Projection(p) => p
                    .projections
                    .iter()
                    .all(|e| matches!(e, Expression::Ref(_))),
                _ => false,
            };
            if is_passthrough_projection {
                node = match node.inputs.first_mut() {
                    Some(n) => n,
                    None => return,
                };
                continue;
            }
            if let Operator::Aggregate(a) = &mut node.operator {
                // A `COUNT(DISTINCT)` aggregate lowers to a two-level group-by
                // whose output isn't this operator's table, so the pushdown
                // wouldn't be a simple per-partition cap. Leave those alone; the
                // common multi-aggregate / count grouped path takes the cap.
                if !a.groups.is_empty() && a.output_limit.is_none() {
                    a.output_limit = Some(GroupLimit::First { limit: window });
                }
            }
            return;
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
    /// The result column names DuckDB resolved for the client, in output order
    /// (e.g. `["hour", "count_star()"]`). Stamped onto the compiled output's
    /// schema. Empty when unavailable, in which case the operators' own field
    /// names stand.
    pub output_names: Vec<String>,
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
