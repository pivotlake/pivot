//! The [`Plan`] tree: Pivot's plan IR.
//!
//! A [`Plan`] is the result of walking DuckDB's plan handles (see
//! `build`) into Pivot's own structs. Each [`PlanNode`] holds an
//! [`Operator`] and its child plan nodes.
//!
//! A read plan carries the snapshot-bound [`BoundTable`](crate::catalog::BoundTable)
//! objects DuckDB resolved into its scans. Each scan records that table's
//! identity and version, so a later transaction can reuse the plan only when a
//! walk of the plan finds that every revision still matches.
use crate::catalog::CatalogTransaction;
use crate::compile;
use crate::expression::Expression;
use crate::operator::{
    self, Compact, CopyFromStdin, Operator, OrderByDirection, SetVariable, TransactionStatement,
};
use crate::types::Type;
use dispatch::{GroupLimit, RowDelivery};
use std::fmt;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    Operator(#[from] operator::Error),
    #[error("{0}")]
    OutputType(#[from] compile::Error),
}

/// One node in a [`Plan`] tree: an [`Operator`] plus its child nodes. `name`
/// carries the DuckDB-side label, useful for debugging.
#[derive(Debug)]
pub struct PlanNode {
    pub name: String,
    pub inputs: Vec<PlanNode>,
    pub operator: Operator,
}

impl PlanNode {
    /// The pivot [`Type`] of each column this node emits, in output order,
    /// derived from the operators themselves (a projection from its
    /// expressions, a scan from its table, pass-through operators from their
    /// input). Lets a pass over the plan ask what any subtree produces.
    pub fn output_types(&self) -> Result<Vec<Type>, compile::Error> {
        let inputs = self
            .inputs
            .iter()
            .map(PlanNode::output_types)
            .collect::<Result<Vec<_>, _>>()?;
        self.operator.output_types(&inputs)
    }

    /// Whether each column this node emits can hold SQL NULLs, in output order:
    /// the nullability companion to [`output_types`](Self::output_types).
    pub fn output_nullability(&self) -> Vec<bool> {
        let inputs: Vec<Vec<bool>> = self
            .inputs
            .iter()
            .map(PlanNode::output_nullability)
            .collect();
        self.operator.output_nullability(&inputs)
    }

    /// Whether this node and all of its inputs are safe to reuse after their
    /// table revisions have been validated. Mutating and session statements
    /// retain state tied to the transaction that planned them. Table functions
    /// can capture invocation-specific data without exposing revision
    /// dependencies, so their plans are not reused.
    pub(crate) fn is_cacheable(&self) -> bool {
        let operator_is_cacheable = match &self.operator {
            Operator::Input(input) => input.table.is_plan_cacheable(),
            Operator::Insert(_)
            | Operator::CreateTable(_)
            | Operator::CreateSchema(_)
            | Operator::DropTable(_)
            | Operator::DropSchema(_)
            | Operator::CreateUser(_)
            | Operator::DropUser(_)
            | Operator::SetVariable(_)
            | Operator::Compact(_)
            | Operator::CopyFromStdin(_)
            | Operator::Transaction(_)
            | Operator::TableFunctionScan(_) => false,
            _ => true,
        };
        operator_is_cacheable && self.inputs.iter().all(PlanNode::is_cacheable)
    }

    /// Whether every table scan still has the same identity and version in
    /// `transaction`.
    pub(crate) fn has_matching_table_revisions(
        &self,
        transaction: &dyn CatalogTransaction,
    ) -> bool {
        let operator_matches = match &self.operator {
            Operator::Input(input) => {
                let table_reference = input.table.table_reference();
                let table_revision = input.table.table_revision();
                transaction.table_revision(&table_reference).as_ref() == Some(&table_revision)
            }
            _ => true,
        };
        operator_matches
            && self
                .inputs
                .iter()
                .all(|input| input.has_matching_table_revisions(transaction))
    }

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
    /// Only fires for a *single* order key (a DESC ref to an aggregate column,
    /// traced through column-ref-only projections). A multi-key sort falls back to
    /// the full `TopN`: a per-partition prune by the primary key alone can't honour
    /// the secondary tiebreakers, so among groups tied on the primary key at the
    /// limit boundary it would keep arbitrary ones and drop the rows the secondary
    /// keys actually select, returning the wrong rows. A single key has no such hazard:
    /// ties under one DESC key are order-ambiguous in SQL, so keeping any of the
    /// tied boundary groups is a valid answer.
    pub(crate) fn annotate_group_topn(&mut self) {
        for child in &mut self.inputs {
            child.annotate_group_topn();
        }

        let (mut col, limit) = match &self.operator {
            // Single order key only (see the doc comment); multi-key sorts fall
            // back to the full TopN above.
            Operator::TopN(t) if t.order_bys.len() == 1 => {
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
                        // Don't push top-k onto a slot whose `sort_key` isn't ordered:
                        // a string extreme (raw `ArenaKey`/StringView header bits) or a
                        // float aggregate (the cell holds `f64` bits, so the widened
                        // sort key is not numerically ordered). A per-partition top-k
                        // would keep the wrong rows; leave it to the full `TopN` sort
                        // above (only the pushdown is skipped).
                        use crate::expression::AggregateFunc;
                        let unordered_sort_key = match a.expressions.get(slot) {
                            Some(Expression::AggregateFunc(
                                AggregateFunc::Min(x) | AggregateFunc::Max(x),
                            )) => matches!(
                                x.argument.result_type().ok(),
                                Some(Type::Utf8 | Type::Float32 | Type::Float64)
                            ),
                            Some(Expression::AggregateFunc(AggregateFunc::Sum(x))) => matches!(
                                x.argument.result_type().ok(),
                                Some(Type::Float32 | Type::Float64)
                            ),
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

    /// Tell every filter whether to hand its surviving rows down as full
    /// batches or as soon as they are selected, based on the operator that
    /// consumes them.
    ///
    /// A consumer that reads its whole input before producing anything (a
    /// group-by, a join, a sort) wants full batches: it pays its per-batch
    /// costs once per batch, and nothing it does can happen earlier. A LIMIT
    /// or a Top-N is the opposite, because both act on rows as they arrive: a
    /// LIMIT cancels its input once it has enough rows, and a Top-N publishes
    /// the boundary that prunes row groups from the scan. Batching rows up for
    /// those defers the decision until a filter has selected a whole batch's
    /// worth, and under a selective filter that is long enough to read most of
    /// the table before anything downstream can stop it.
    ///
    /// The nearest consumer decides, so each operator that cares imposes its
    /// choice on the subtree beneath it: under `Limit → Aggregate → Filter`
    /// the aggregate is what the filter feeds, and it still wants batches.
    pub(crate) fn annotate_filter_delivery(&mut self) {
        self.annotate_filter_delivery_below(RowDelivery::Coalesced);
    }

    fn annotate_filter_delivery_below(&mut self, consumer: RowDelivery) {
        if let Operator::Filter(filter) = &mut self.operator {
            filter.delivery = consumer;
        }
        // What this node's own inputs feed, which is this node unless it is
        // transparent (a projection, a materialize) and passes the choice on.
        let below = match &self.operator {
            Operator::Limit(_) | Operator::TopN(_) => RowDelivery::Immediate,
            Operator::Aggregate(_) | Operator::Join(_) | Operator::OrderBy(_) => {
                RowDelivery::Coalesced
            }
            _ => consumer,
        };
        for child in &mut self.inputs {
            child.annotate_filter_delivery_below(below);
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
    pub root: PlanNode,
    /// The result column names DuckDB resolved for the client, in output order
    /// (e.g. `["hour", "count_star()"]`). Stamped onto the compiled output's
    /// schema. Empty when unavailable, in which case the operators' own field
    /// names stand.
    pub output_names: Vec<String>,
    /// DuckDB embedded query-dependent state while optimizing this statement,
    /// so a later execution must plan it again.
    pub requires_rebind: bool,
}

impl Plan {
    pub fn is_cacheable(&self) -> bool {
        !self.requires_rebind && self.root.is_cacheable()
    }

    pub fn has_matching_table_revisions(&self, transaction: &dyn CatalogTransaction) -> bool {
        self.root.has_matching_table_revisions(transaction)
    }

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

    /// This plan as a `COMPACT` statement, if that's what it is. Like a `SET`,
    /// the server checks this before compiling: compaction runs
    /// coordinator-side, off the worker pool, because it drives dataflows of
    /// its own.
    pub fn as_compact(&self) -> Option<&Compact> {
        match &self.root.operator {
            Operator::Compact(compact) => Some(compact),
            _ => None,
        }
    }

    /// This plan as a `COPY ... FROM STDIN` statement, if that's what it is.
    /// Like a `SET`, the server checks this before compiling: the row data
    /// arrives over the connection's copy-in sub-protocol, so the server
    /// drives the ingest itself.
    pub fn as_copy_from_stdin(&self) -> Option<&CopyFromStdin> {
        match &self.root.operator {
            Operator::CopyFromStdin(copy) => Some(copy),
            _ => None,
        }
    }

    /// This plan as a `BEGIN`/`COMMIT`/`ROLLBACK` statement, if that's what it
    /// is. Like a `SET`, the server checks this before compiling. Pivot
    /// commits every statement individually, so the server answers these
    /// without doing anything: they exist for PostgreSQL drivers that wrap
    /// statements in a transaction by default.
    pub fn as_transaction_stmt(&self) -> Option<TransactionStatement> {
        match &self.root.operator {
            Operator::Transaction(statement) => Some(*statement),
            _ => None,
        }
    }
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.root)
    }
}
