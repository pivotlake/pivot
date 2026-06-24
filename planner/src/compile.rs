//! Compil a Pivot [`Plan`] into an executable
//! [`RecordBatchOperatorSpec`].
//!
//! Compilation is a recursive walk: each [`PlanNode`] compiles its inputs
//! first, then dispatches to the per-operator `compile` impl (e.g. a
//! [`Filter`](crate::operator::Filter) compiles into
//! [`RecordBatchOperatorSpec::filter`]). Expressions compile into **builder
//! closures** ([`ExprFn`]) — `Fn()` returning a per-worker
//! `FnMut(&RecordBatch) -> ExprResult` — matching dispatch's two-level
//! closure pattern so each worker gets its own state without synchronization.
//!
//! # Organization
//!
//! This module holds the cross-cutting compile infrastructure: the compile
//! [`enum@Error`], the [`ExprResult`]/[`ExprFn`] closure types, the
//! `DynamicFilterSlots` registry, and the recursive [`Plan`]/[`PlanNode`]
//! walk. The per-operator `compile` impls live alongside their AST types in the
//! [`operator`](crate::operator) submodules, and the per-expression `compile`
//! impls producing [`ExprFn`]s live in the [`expression`](crate::expression)
//! submodules, not here.
//!

use crate::catalog::{Catalog, QueryContext};
use crate::expression::Expression;
use crate::types::Type;
use crate::{Plan, PlanNode};
use arrow_array::{ArrayRef, Datum, RecordBatch, Scalar};
use dispatch::{DataFlowDispatcher, DynamicFilterSlot, RecordBatchOperatorSpec};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Per-`compile` registry of dynamic-filter slots, keyed by `slot_id`.
///
/// Each [`Plan::compile`] starts with an empty registry; the producer (`TopN`)
/// and consumer (`Input`) for a given `slot_id` lazily get-or-create one shared
/// [`DynamicFilterSlot`] from it, so they end up holding the same `Arc`. The
/// registry is intentionally *not* part of the (cacheable) `Plan`: a fresh set
/// of empty slots is minted on every compile, so a reused plan never carries a
/// stale boundary — or a slice of a previous run's pooled scan buffers — across
/// executions.
pub(crate) type DynamicFilterSlots = HashMap<usize, Arc<DynamicFilterSlot>>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Unexpected input Expression {0}")]
    UnexpectedInputExpression(Expression),
    #[error("Unsupported projection expression: {0:?}")]
    UnsupportedProjectionExpression(Expression),
    #[error("Unsupported order by expression: {0:?}")]
    UnsupportedOrderByExpression(Expression),
    #[error("Unsupported top k expression: {0:?}")]
    UnsupportedTopKExpression(Expression),
    #[error("Unsupported aggregate expression: {0:?}")]
    UnsupportedAggregateExpression(Expression),
    #[error("Unsupported aggregate expression amount: {0}")]
    UnsupportedAggregateExpressionAmount(usize),
    #[error("Unsupported expression: {0:?}")]
    UnsupportedExpression(Expression),
    #[error("Unsupported type for group by: {0:?}")]
    DataTypeNotSupportedForGroupBy(Type),
    #[error("Cannot statically determine the result type of expression: {0:?}")]
    IndeterminateResultType(Expression),
    #[error("Unsupported expression for contains: {0:?}")]
    UnsupportedExpressionForContainsNeedle(Expression),
    #[error("Unsupported haystack expression for contains: {0:?}")]
    UnsupportedExpressionForContainsHaystack(Expression),
    #[error("Failed to downcast scalar into string: {0:?}")]
    FailedToDowncastScalarIntoString(Scalar<ArrayRef>),
    #[error("Invalid regexp_replace pattern '{pattern}': {source}")]
    InvalidRegexPattern {
        pattern: String,
        #[source]
        source: regex::Error,
    },
    #[error("CREATE TABLE does not support OR REPLACE yet")]
    UnsupportedCreateTableOrReplace,
    #[error("CREATE TEMPORARY TABLE is not supported yet")]
    UnsupportedTemporaryCreateTable,
    #[error("CREATE TABLE AS SELECT is not supported yet")]
    UnsupportedCreateTableAs,
    #[error("CREATE TABLE with constraints is not supported yet ({0} constraint(s))")]
    UnsupportedCreateTableConstraints(usize),
    #[error("CREATE TABLE nodes should not have input operators")]
    UnexpectedCreateTableInputs,
    #[error("compiling table scan: {0}")]
    TableScan(#[source] crate::catalog::Error),
    #[error("creating table: {0}")]
    CreateTable(#[source] crate::catalog::Error),
    #[error("SET/RESET is a session command, not a compilable query")]
    SetVariableNotCompilable,
    #[error("Unsupported table function: {0}")]
    UnsupportedTableFunction(String),
    #[error("Invalid argument to table function {function}: {message}")]
    InvalidTableFunctionArgument { function: String, message: String },
}

impl Plan {
    /// Lower this plan into an executable
    /// [`dispatch::RecordBatchOperatorSpec`] on the
    /// given dispatcher. The catalog the plan was bound against is read from
    /// [`Plan::catalog`] for operators that need it at runtime
    /// (e.g. `CREATE TABLE`).
    pub fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // A cached plan is re-run through here, so this is where it must pick up
        // data committed since it was planned. The query context reloads each
        // table a scan touches to its latest version the first time it compiles —
        // lazily, and once per table, so a table feeding both a scan and a late
        // materialize reloads a single time and both read one snapshot.
        let ctx = self.catalog.query_context();
        let mut slots = DynamicFilterSlots::new();
        self.root
            .compile(dispatcher, &self.catalog, ctx.as_ref(), &mut slots, false)
    }
}

impl PlanNode {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        catalog: &Arc<dyn Catalog>,
        ctx: &dyn QueryContext,
        slots: &mut DynamicFilterSlots,
        // Whether this node's parent is a `Filter`. Used so the condition cache
        // fires only at the *outermost* filter of a stacked-filter chain.
        parent_is_filter: bool,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // Peephole: an unfiltered global MIN/MAX or COUNT(*) over a bare scan is
        // fully determined by table metadata (e.g. parquet row-group
        // statistics). It must run before the child scan is compiled, since
        // succeeding means no scan happens at all. `try_compile_from_stats`
        // checks the rest of the shape (single bare-scan child, no predicates).
        if let crate::Operator::Aggregate(agg) = &self.operator
            && let Some(spec) = agg.try_compile_from_stats(&self.inputs, dispatcher, ctx)?
        {
            return Ok(spec);
        }

        // Condition cache: the *outermost* `Filter` of a chain of one or more
        // `Filter`s sitting on a table scan (DuckDB may emit the `WHERE` as
        // several stacked filters). Firing at the outermost — `!parent_is_filter`
        // — captures the rows the *whole* filter keeps, not a sub-filter's. On the
        // first run the scan emits row-group metadata so the survivors can be
        // recorded; on a repeat the scan replays only those rows.
        if ctx.condition_cache_enabled()
            && !parent_is_filter
            && matches!(self.operator, crate::Operator::Filter(_))
            && let Some((scan_node, filters)) = collect_filter_chain(self)
            && let crate::Operator::Input(scan) = &scan_node.operator
            && scan.has_data_columns()
        {
            let filter_id = condition_filter_id(self);
            // A filter judged non-selective on a previous run is left out of the
            // cache: fall through to the plain scan path (no populate, no replay).
            if !ctx.is_condition_skipped(filter_id) {
                let replay = ctx.is_condition_populated(filter_id);
                // Replaying injects the cached survivors (no metadata needed). First
                // sight forces row-group metadata so the populator above the chain
                // can attribute survivors, then strips it back off.
                let force_metadata = !replay && !scan.emit_row_group_metadata;
                let scan_spec = scan.compile(
                    dispatcher,
                    ctx,
                    slots,
                    force_metadata,
                    replay.then_some(filter_id),
                )?;
                // Rebuild the chain innermost-first (filters are outermost-first).
                let mut spec = scan_spec;
                for filter in filters.iter().rev() {
                    spec = filter.compile(spec)?;
                }
                return Ok(if replay {
                    // The filters above re-run over the already-surviving rows (a
                    // no-op), so replay stays a pure optimization.
                    spec
                } else {
                    ctx.populate_condition_cache(spec, filter_id, force_metadata)
                });
            }
        }

        let child_parent_is_filter = matches!(self.operator, crate::Operator::Filter(_));
        let mut inputs = Vec::with_capacity(self.inputs.len());
        for input in &self.inputs {
            inputs.push(input.compile(dispatcher, catalog, ctx, slots, child_parent_is_filter)?);
        }

        match &self.operator {
            crate::Operator::Input(o) => o.compile(dispatcher, ctx, slots, false, None),
            crate::Operator::TableFunctionScan(o) => o.compile(dispatcher, catalog.as_ref(), ctx),
            crate::Operator::Projection(o) => o.compile(inputs.remove(0)),
            crate::Operator::Filter(o) => o.compile(inputs.remove(0)),
            crate::Operator::Aggregate(o) => o.compile(inputs.remove(0)),
            crate::Operator::OrderBy(o) => o.compile(inputs.remove(0)),
            crate::Operator::TopN(o) => o.compile(inputs.remove(0), slots),
            crate::Operator::Limit(o) => o.compile(inputs.remove(0)),
            crate::Operator::Materialize(o) => o.compile(inputs.remove(0), ctx),
            crate::Operator::CreateTable(o) => {
                if !inputs.is_empty() {
                    return Err(Error::UnexpectedCreateTableInputs);
                }
                o.compile(dispatcher, catalog)
            }
            crate::Operator::DummyScan(o) => o.compile(dispatcher),
            // SET/RESET is intercepted by the server after planning (it toggles
            // session state, not data), so it should never reach compilation.
            crate::Operator::SetVariable(_) => Err(Error::SetVariableNotCompilable),
        }
    }
}

/// A stable identity for a `Filter`-over-scan subtree, used as the condition
/// cache key. Hashes the node's textual plan (its `Display`, which recurses
/// through the filter conditions and the scan's columns), so two runs of the same
/// query agree and different filters don't collide. Table *version* is not folded
/// in: the cache is cleared whenever a table's files change (see
/// `QueryConditionCache::clear`), so every live entry already matches the current
/// data.
fn condition_filter_id(node: &PlanNode) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    node.to_string().hash(&mut hasher);
    hasher.finish()
}

/// If `node` is a chain of one or more `Filter`s ending in an `Input`, return the
/// `Input` node and the filters (outermost first). `None` for any other shape
/// (e.g. a projection between filters), so the condition cache only engages when
/// the filters sit directly on the scan and together form the whole `WHERE`.
fn collect_filter_chain(node: &PlanNode) -> Option<(&PlanNode, Vec<&crate::operator::Filter>)> {
    let mut filters = Vec::new();
    let mut current = node;
    loop {
        match &current.operator {
            crate::Operator::Filter(filter) => {
                let [child] = current.inputs.as_slice() else {
                    return None;
                };
                filters.push(filter);
                current = child;
            }
            crate::Operator::Input(_) if !filters.is_empty() => return Some((current, filters)),
            _ => return None,
        }
    }
}

/// What an evaluated expression produces for one input batch: either a
/// per-row [`Array`](ExprResult::Array) (column refs, comparisons, scalar
/// functions) or a single [`Scalar`](ExprResult::Scalar) (constants). Both
/// variants implement arrow's [`Datum`] so consumers can pass them straight
/// into kernels.
pub enum ExprResult {
    Array(ArrayRef),
    Scalar(Scalar<ArrayRef>),
}

impl ExprResult {
    pub fn as_datum(&self) -> &dyn Datum {
        match self {
            ExprResult::Array(a) => a,
            ExprResult::Scalar(s) => s,
        }
    }
}

/// A per-batch evaluator: takes a [`RecordBatch`] and produces an
/// [`ExprResult`]. `FnMut` so the closure can hold mutable per-worker state
/// (e.g. a [`dispatch::Contains`] cache).
pub type ExprEvalFn = Box<dyn FnMut(&RecordBatch) -> ExprResult + Send>;

/// Builder for [`ExprEvalFn`]: called once per worker thread to produce that
/// worker's private evaluator. Mirrors the two-level closure pattern used by
/// dispatch's [`filter`](dispatch::RecordBatchOperatorSpec::filter) /
/// [`project`](dispatch::RecordBatchOperatorSpec::project) APIs.
pub type ExprFn = Box<dyn Fn() -> ExprEvalFn + Send + Sync>;

/// Wraps a stateless expression closure into the builder pattern (closure returning closure).
pub(crate) fn stateless_expr<F>(f: F) -> ExprFn
where
    F: Fn(&RecordBatch) -> ExprResult + Send + Sync + Clone + 'static,
{
    Box::new(move || {
        let f = f.clone();
        Box::new(move |batch: &RecordBatch| f(batch)) as ExprEvalFn
    })
}
