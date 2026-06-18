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
//! - `operator` (private) — per-operator `compile` impls (one `impl` block
//!   per [`Operator`](crate::operator::Operator) variant).
//!
//! The per-expression `compile` impls producing [`ExprFn`]s live alongside
//! their AST types in the [`expression`](crate::expression) submodules, not
//! here.
//!

mod dummy_scan;
mod operator;

use crate::catalog::Catalog;
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
    #[error("Unexpected agg expression: {0:?}")]
    UnexpectedAggExpression(Expression),
    #[error("Unsupported projection expression: {0:?}")]
    UnsupportedProjectionExpression(Expression),
    #[error("Unsupported order by expression: {0:?}")]
    UnsupportedOrderByExpression(Expression),
    #[error("Unsupported top k expression: {0:?}")]
    UnsupportedTopKExpression(Expression),
    #[error("Unsupported aggregate functions with {0:?} params")]
    UnsupportedAggregateGroupAmount(usize),
    #[error("Unsupported aggregate expression: {0:?}")]
    UnsupportedAggregateExpression(Expression),
    #[error("Unsupported aggregate expression amount: {0}")]
    UnsupportedAggregateExpressionAmount(usize),
    #[error("Unsupported expression: {0:?}")]
    UnsupportedExpression(Expression),
    #[error("Unsupported type for group by: {0:?}")]
    DataTypeNotSupportedForGroupBy(Type),
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
        let mut slots = DynamicFilterSlots::new();
        let mut memo = crate::compile::operator::RefreshMemo::new();
        self.root.compile(dispatcher, &self.catalog, &mut slots, &mut memo)
    }
}

impl PlanNode {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        catalog: &Arc<dyn Catalog>,
        slots: &mut DynamicFilterSlots,
        memo: &mut crate::compile::operator::RefreshMemo,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let mut inputs = Vec::with_capacity(self.inputs.len());
        for input in &self.inputs {
            inputs.push(input.compile(dispatcher, catalog, slots, memo)?);
        }

        match &self.operator {
            crate::Operator::Input(o) => o.compile(dispatcher, catalog.as_ref(), slots, memo),
            crate::Operator::Projection(o) => o.compile(inputs.remove(0)),
            crate::Operator::Filter(o) => o.compile(inputs.remove(0)),
            crate::Operator::Aggregate(o) => o.compile(inputs.remove(0)),
            crate::Operator::OrderBy(o) => o.compile(inputs.remove(0)),
            crate::Operator::TopN(o) => o.compile(inputs.remove(0), slots),
            crate::Operator::Materialize(o) => o.compile(inputs.remove(0), catalog.as_ref(), memo),
            crate::Operator::CreateTable(o) => {
                if !inputs.is_empty() {
                    return Err(Error::UnexpectedCreateTableInputs);
                }
                o.compile(dispatcher, catalog)
            }
            crate::Operator::DummyScan(o) => o.compile(dispatcher),
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
