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
use crate::condition_key::{ConditionKeyOutcome, build_condition_key};
use crate::expression::Expression;
use crate::types::Type;
use crate::{Plan, PlanNode};
use arrow_array::{ArrayRef, Datum, RecordBatch, Scalar, UInt32Array};
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
    #[error("Invalid regexp_jit_replace pattern '{pattern}': {source}")]
    InvalidJitRegexPattern {
        pattern: String,
        #[source]
        source: pcre2::Error,
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
        let compiled = self
            .root
            .compile(dispatcher, &self.catalog, ctx.as_ref(), &mut slots)?;
        Ok(stamp_output_names(compiled, &self.output_names))
    }
}

/// Stamp DuckDB's client-facing result column names onto the compiled plan's
/// output. Naming happens once here, at the root, rather than inside each
/// operator: the operators are free to use whatever intermediate field names
/// are convenient, and the query's actual output schema is fixed up last.
///
/// The names are DuckDB's select-list order, which the output columns follow.
/// Only the leading `names.len()` fields are renamed, so a wider output (an
/// operator that appends trailing bookkeeping columns) keeps its extra fields;
/// a narrower one than DuckDB reported is left untouched rather than risk
/// mislabeling.
fn stamp_output_names(spec: RecordBatchOperatorSpec, names: &[String]) -> RecordBatchOperatorSpec {
    if names.is_empty() {
        return spec;
    }
    let names = Arc::new(names.to_vec());
    spec.project(move || {
        let names = names.clone();
        move |batch: RecordBatch| {
            let schema = batch.schema();
            let fields = schema.fields();
            // Fewer columns than DuckDB reported names (e.g. EXPLAIN emits one
            // column where DuckDB resolves two): leave the batch untouched.
            if fields.len() < names.len() {
                return batch;
            }
            // Already correctly named (a plain `SELECT a, b` or `SELECT *`):
            // skip the rebuild and pass the batch through zero-copy.
            if names
                .iter()
                .zip(fields.iter())
                .all(|(name, field)| field.name() == name)
            {
                return batch;
            }
            // `with_name` preserves each field's data type, nullability, and
            // metadata (e.g. an extension-type annotation); only the leading
            // `names.len()` fields are renamed, so trailing fields are reused
            // by their `Arc` rather than rebuilt.
            let renamed: Vec<arrow_schema::FieldRef> = fields
                .iter()
                .enumerate()
                .map(|(i, field)| match names.get(i) {
                    Some(name) => Arc::new(field.as_ref().clone().with_name(name.clone())),
                    None => field.clone(),
                })
                .collect();
            let new_schema = Arc::new(arrow_schema::Schema::new_with_metadata(
                renamed,
                schema.metadata().clone(),
            ));
            // We own `batch`, so move its column arrays out (there is at least
            // one, since `fields.len() >= names.len() >= 1`) instead of cloning.
            let (_, columns, _) = batch.into_parts();
            RecordBatch::try_new(new_schema, columns).unwrap()
        }
    })
}

impl PlanNode {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        catalog: &Arc<dyn Catalog>,
        ctx: &dyn QueryContext,
        slots: &mut DynamicFilterSlots,
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

        // Peephole: a stack of Filters sitting directly on a table scan hands
        // its whole conjunction to the table, which may answer row groups from
        // a condition cache (injecting known surviving positions) and provide
        // an observer to populate it. Fires at the topmost filter of the stack
        // (recursion reaches it first); any non-cacheable shape falls through
        // to the generic path below unchanged.
        if let crate::Operator::Filter(_) = &self.operator
            && let Some(spec) = try_compile_condition_cached_scan(self, dispatcher, ctx, slots)?
        {
            return Ok(spec);
        }

        // EXPLAIN renders its (already-optimized) child plan as text and emits
        // that. The child is formatted here, not compiled, so the explained
        // query never runs. This must happen before the input-compile loop below.
        if let crate::Operator::Explain(explain) = &self.operator {
            let plan_text = self
                .inputs
                .first()
                .map(PlanNode::to_string)
                .unwrap_or_default();
            return explain.compile(dispatcher, plan_text);
        }

        let mut inputs = Vec::with_capacity(self.inputs.len());
        for input in &self.inputs {
            inputs.push(input.compile(dispatcher, catalog, ctx, slots)?);
        }

        match &self.operator {
            crate::Operator::Input(o) => o.compile(dispatcher, ctx, slots),
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
            // EXPLAIN is handled above, before inputs are compiled.
            crate::Operator::Explain(_) => unreachable!("Explain is compiled before its inputs"),
            // SET/RESET is intercepted by the server after planning (it toggles
            // session state, not data), so it should never reach compilation.
            crate::Operator::SetVariable(_) => Err(Error::SetVariableNotCompilable),
        }
    }
}

/// Compile a `Filter[-> Filter] -> Input` chain through the table's condition
/// cache. `Some(spec)` when the chain matched and compiled; `Ok(None)` when the
/// shape or the conjunction is not cacheable, sending the caller down the
/// generic path with zero behavior change.
fn try_compile_condition_cached_scan(
    node: &PlanNode,
    dispatcher: &DataFlowDispatcher,
    ctx: &dyn QueryContext,
    slots: &mut DynamicFilterSlots,
) -> Result<Option<RecordBatchOperatorSpec>, Error> {
    // Walk the adjacent Filter nodes down to the scan, collecting the stack
    // top-down. Anything else in between (a projection, a limit) breaks the
    // chain; the peephole then fires again lower if a filter sits under it.
    let mut filter_stack: Vec<&crate::operator::Filter> = Vec::new();
    let mut current = node;
    let input = loop {
        match &current.operator {
            crate::Operator::Filter(filter) if current.inputs.len() == 1 => {
                filter_stack.push(filter);
                current = &current.inputs[0];
            }
            crate::Operator::Input(input) => break input,
            _ => return Ok(None),
        }
    };

    let Some(scan_output_names) = input.build_scan_output_names() else {
        return Ok(None);
    };
    // Bottom-up (pushdown predicates first), matching the order the generic
    // path would evaluate the stack in.
    let conditions: Vec<&Expression> = filter_stack
        .iter()
        .rev()
        .flat_map(|filter| filter.conditions.iter())
        .collect();
    let condition = match build_condition_key(&conditions, &scan_output_names) {
        ConditionKeyOutcome::Cacheable(condition) => condition,
        ConditionKeyOutcome::NotCacheable => return Ok(None),
    };

    let compiled = input.compile_with_condition(dispatcher, ctx, slots, &condition)?;
    match compiled.observer {
        // Nothing to observe (cache disabled, backend without a cache, or every
        // row group already answered): compile the stack per node exactly as
        // the generic path would. Injected row groups re-evaluate to all-true.
        None => {
            let mut spec = compiled.spec;
            for filter in filter_stack.iter().rev() {
                spec = filter.compile(spec)?;
            }
            Ok(Some(spec))
        }
        // Observing: the stack must run as ONE fused filter, because the
        // observer's completeness accounting counts every scanned row; a lower
        // filter dropping rows first would leave row groups forever
        // incomplete. The metadata columns ride behind the data columns, so
        // the conditions' positional references are unaffected.
        Some(observer) => {
            let mut spec = crate::operator::Filter::compile_observed(
                conditions.into_iter(),
                compiled.spec,
                observer,
            )?;
            if compiled.strip_metadata {
                spec = strip_trailing_metadata_columns(spec);
            }
            Ok(Some(spec))
        }
    }
}

/// Drop the trailing row-group-metadata columns a scan appended for
/// observation, restoring the schema everything above the filter expects.
fn strip_trailing_metadata_columns(spec: RecordBatchOperatorSpec) -> RecordBatchOperatorSpec {
    spec.project(|| {
        |batch: RecordBatch| {
            let metadata_columns = dispatch::trailing_metadata_columns(&batch.schema());
            if metadata_columns == 0 {
                return batch;
            }
            let kept: Vec<usize> = (0..batch.num_columns() - metadata_columns).collect();
            batch.project(&kept).unwrap()
        }
    })
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

    /// The result as a full column of `num_rows` rows: an [`Array`](Self::Array)
    /// passes through; a [`Scalar`](Self::Scalar) (e.g. `SELECT 1`) is broadcast
    /// to that length so it can sit beside the per-row columns.
    pub fn into_array(self, num_rows: usize) -> ArrayRef {
        match self {
            ExprResult::Array(a) => a,
            ExprResult::Scalar(s) => {
                let indices = UInt32Array::from(vec![0u32; num_rows]);
                arrow::compute::take(&s.into_inner(), &indices, None).unwrap()
            }
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
