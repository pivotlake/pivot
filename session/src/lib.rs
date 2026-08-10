//! The embedded engine session: plan, compile, and run SQL statements
//! in-process, with no wire protocol involved.
//!
//! This is the neutral execution core every frontend builds on. The server's
//! pgwire handlers and HTTP console drive these primitives per connection with
//! their own response shaping; the `pivot-cli` shell runs statements through
//! [`Session`] directly. Whatever the frontend, a statement takes the same
//! path: [`plan_query`] (the thread-local DuckDB planner plus the
//! [`PlanCache`]), compilation and execution on the dispatch workers, and one
//! catalog transaction committed on success and rolled back on failure.
//!
//! Sizing an instance from the machine lives in [`bootstrap`], and
//! [`raise_open_file_limit`] takes the descriptor headroom every embedding
//! wants at startup.

pub mod bootstrap;
mod limits;
mod plan_cache;

use std::cell::RefCell;
use std::sync::Arc;

use arrow_array::{Array, Int64Array, RecordBatch};
use dispatch::DataFlowHandle;
use tokio::task::JoinError;

pub use limits::raise_open_file_limit;
pub use plan_cache::PlanCache;

thread_local! {
    /// One [`planner::Planner`] (and its non-`Send` DuckDB context) per
    /// tokio blocking-pool thread, lazily initialised on first use. The
    /// catalog is process-global so it's safe to share the same planner
    /// across connections that land on the same thread.
    static PLANNER: RefCell<Option<planner::Planner>> = const { RefCell::new(None) };
}

fn with_planner<R>(
    catalog: &Arc<catalog::PivotCatalog>,
    f: impl FnOnce(&mut planner::Planner) -> R,
) -> Result<R, planner::Error> {
    PLANNER.with_borrow_mut(|opt| {
        let planner = match opt {
            Some(planner) => planner,
            None => {
                // Attach every datastore as its own database (so a query can name
                // it). The planner holds no catalog; each query's transaction
                // (from `catalog.begin_transaction()`) does all table/DDL resolution.
                let names = catalog
                    .iter_datastores()
                    .map(|(name, _)| name.clone())
                    .collect();
                opt.insert(planner::Planner::from_datastore_names(
                    names,
                    catalog.default_datastore_name().to_string(),
                )?)
            }
        };
        Ok(f(planner))
    })
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Plan(#[from] planner::Error),
    #[error("compile error: {0}")]
    Compile(#[from] planner::compile::Error),
    #[error(transparent)]
    DataFlow(#[from] dispatch::DataFlowError),
    #[error(transparent)]
    Catalog(#[from] planner::catalog::Error),
    #[error("waiter thread panicked: {0}")]
    WorkerPanic(JoinError),
    #[error("waiter thread panicked: {0}")]
    PlannerPanic(JoinError),
    #[error("invalid INSERT row-count result: {0}")]
    InvalidInsertResult(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A statement's result through the embedded path.
pub enum StatementOutcome {
    /// Query results. `column_names` come from the plan, so they are present
    /// even when the query produced no batches at all.
    Rows {
        column_names: Vec<String>,
        batches: Vec<RecordBatch>,
    },
    /// An INSERT's affected-row count.
    Affected(u64),
    /// A statement that completes with only its tag: `SET`, `RESET`,
    /// `COMPACT`, DDL.
    Command(&'static str),
}

/// An embedded SQL session over an open catalog: the same planning, caching,
/// and execution as a wire connection, minus the wire.
pub struct Session {
    catalog: Arc<catalog::PivotCatalog>,
    dispatcher: dispatch::DataFlowDispatcher,
    plan_cache: Arc<PlanCache>,
}

impl Session {
    pub fn new(
        catalog: Arc<catalog::PivotCatalog>,
        dispatcher: dispatch::DataFlowDispatcher,
    ) -> Self {
        Self::with_plan_cache(catalog, dispatcher, Arc::new(PlanCache::default()))
    }

    /// As [`new`](Self::new), sharing an existing plan cache: a process
    /// serving several paths (wire connections, console) keeps one cache warm
    /// across them.
    pub fn with_plan_cache(
        catalog: Arc<catalog::PivotCatalog>,
        dispatcher: dispatch::DataFlowDispatcher,
        plan_cache: Arc<PlanCache>,
    ) -> Self {
        Self {
            catalog,
            dispatcher,
            plan_cache,
        }
    }

    /// Run one statement to completion. Dropping the returned future cancels
    /// the running dataflow, so a caller's Ctrl-C handling is just a drop.
    ///
    /// Uses [`RecordBatchOperatorSpec::collect`](dispatch::RecordBatchOperatorSpec::collect), which appends a `CopyOut` stage:
    /// each batch's ring-backed buffers are deep-copied to plain heap allocations on
    /// the worker, so the returned batches are safe to hold and drop on this
    /// (non-worker) thread. Calling `execute().collect()` instead would return
    /// ring-backed batches whose `Drop` reaches `memory_ctx()` off-worker and aborts.
    pub async fn run(&self, sql: &str) -> Result<StatementOutcome, String> {
        // One transaction per statement: the query binds and compiles against
        // this snapshot of the catalog. Committed on success, rolled back on
        // failure (the async block scopes the `?` early-returns so both paths
        // land below).
        let transaction = self.catalog.begin_transaction();
        let result = async {
            let plan = plan_query(
                &self.catalog,
                transaction.clone(),
                self.plan_cache.as_ref(),
                sql,
            )
            .await
            .map_err(|e| e.to_string())?;
            // A SET/RESET is a per-connection concept; the embedded session
            // accepts it as a no-op, like the wire path does for unknown names.
            if let Some(set) = plan.as_set_variable() {
                return Ok(StatementOutcome::Command(if set.value.is_none() {
                    "RESET"
                } else {
                    "SET"
                }));
            }
            // A COMPACT runs its sweeps here on the coordinator and returns no rows.
            if let Some(request) = plan.as_compact() {
                execute_compact(&self.catalog, request)
                    .await
                    .map_err(|e| e.to_string())?;
                return Ok(StatementOutcome::Command("COMPACT"));
            }
            let is_insert = matches!(&plan.root.operator, planner::Operator::Insert(_));
            // DDL executes as a dataflow like everything else, but its result
            // is its completion, not its (empty, internal) output rows.
            let ddl_tag = match &plan.root.operator {
                planner::Operator::CreateTable(_) => Some("CREATE TABLE"),
                planner::Operator::CreateSchema(_) => Some("CREATE SCHEMA"),
                _ => None,
            };
            let column_names = plan.output_names.clone();
            // Compile and launch the dataflow (with the CopyOut cap) on the blocking
            // pool; `execute_copying` returns the running handle without collecting.
            // The closure gets its own clone of the transaction Arc only because
            // spawn_blocking moves its captures to another thread.
            let dispatcher = self.dispatcher.clone();
            let handle = tokio::task::spawn_blocking({
                let transaction = transaction.clone();
                move || -> Result<DataFlowHandle<RecordBatch>, String> {
                    let spec = plan
                        .compile(&dispatcher, transaction.as_ref())
                        .map_err(|e| e.to_string())?;
                    Ok(spec.execute_copying())
                }
            })
            .await
            .map_err(|e| e.to_string())??;

            // If this future is dropped before collection finishes, cancel the
            // running dataflow so its workers stop instead of finishing a
            // doomed query.
            let guard = CancelOnDrop::new(handle.cancel_token());
            let batches = tokio::task::spawn_blocking(move || handle.collect())
                .await
                .map_err(|e| e.to_string())?
                .map_err(|e| e.to_string())?;
            guard.defuse();
            if is_insert {
                let [batch] = batches.as_slice() else {
                    return Err(format!(
                        "invalid INSERT result: expected one batch, got {}",
                        batches.len()
                    ));
                };
                let count = parse_affected_rows(batch)
                    .map_err(|message| format!("invalid INSERT row-count result: {message}"))?;
                return Ok(StatementOutcome::Affected(count as u64));
            }
            if let Some(tag) = ddl_tag {
                return Ok(StatementOutcome::Command(tag));
            }
            Ok(StatementOutcome::Rows {
                column_names,
                batches,
            })
        }
        .await;
        match result {
            Ok(outcome) => {
                transaction.commit().await.map_err(|e| e.to_string())?;
                Ok(outcome)
            }
            Err(error) => {
                transaction.rollback();
                Err(error)
            }
        }
    }
}

/// Run a `COMPACT` statement: resolve the datastore it names (default when
/// unqualified) and sweep the table synchronously. The sweeps commit through
/// the datastore's own log CAS, independent of the statement's transaction.
pub async fn execute_compact(
    catalog: &Arc<catalog::PivotCatalog>,
    request: &planner::Compact,
) -> Result<u64> {
    let datastore_name = request
        .datastore
        .as_deref()
        .unwrap_or_else(|| catalog.default_datastore_name());
    let datastore = catalog
        .iter_datastores()
        .find(|(name, _)| name.as_str() == datastore_name)
        .map(|(_, datastore)| Arc::clone(datastore))
        .ok_or_else(|| {
            planner::catalog::Error::Other(
                format!("COMPACT: no datastore named `{datastore_name}`").into(),
            )
        })?;
    let table = planner::catalog::SchemaQualifiedTableName::new(
        request
            .schema
            .as_deref()
            .unwrap_or(planner::DEFAULT_SCHEMA_NAME),
        request.table.as_str(),
    );
    Ok(datastore.compact(&table, request.final_sweep).await?)
}

/// Reuse the exact SQL's plan when its complete table-revision map matches this
/// transaction; otherwise plan inside the same transaction (on the blocking
/// pool, where the thread-local DuckDB planner lives) and cache the result
/// only when the planner marked it safe.
pub async fn plan_query(
    catalog: &Arc<catalog::PivotCatalog>,
    transaction: Arc<dyn planner::catalog::CatalogTransaction>,
    plan_cache: &PlanCache,
    query: &str,
) -> Result<Arc<planner::Plan>> {
    if let Some(plan) = plan_cache.get(query, transaction.as_ref()) {
        return Ok(plan);
    }

    let catalog = catalog.clone();
    let cache_key = query.to_string();
    let query = cache_key.clone();
    let planning_transaction = transaction;
    let plan = tokio::task::spawn_blocking(move || -> Result<Arc<planner::Plan>> {
        with_planner(&catalog, |planner| {
            Ok(Arc::new(planner.plan(&query, planning_transaction)?))
        })?
    })
    .await
    .map_err(Error::PlannerPanic)??;
    if plan.is_cacheable() {
        plan_cache.insert(cache_key, plan.clone());
    }
    Ok(plan)
}

/// Decode an INSERT dataflow's internal result: one row of one Int64 column,
/// the affected-row count.
pub fn parse_affected_rows(batch: &RecordBatch) -> std::result::Result<usize, String> {
    if batch.num_rows() != 1 || batch.num_columns() != 1 {
        return Err(format!(
            "expected one row and one column, got {} rows and {} columns",
            batch.num_rows(),
            batch.num_columns()
        ));
    }
    let counts = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| format!("expected Int64, got {}", batch.column(0).data_type()))?;
    if counts.is_null(0) {
        return Err("count is null".to_string());
    }
    usize::try_from(counts.value(0)).map_err(|_| "count is negative or too large".to_string())
}

/// If the statement's driving future is dropped (cancel request, client
/// disconnect, Ctrl-C) before the dataflow finishes, this guard fires
/// `cancel()` so the workers stop instead of running the rest of a doomed
/// query.
pub struct CancelOnDrop {
    token: Option<dispatch::CancelToken>,
}

impl CancelOnDrop {
    pub fn new(token: dispatch::CancelToken) -> Self {
        Self { token: Some(token) }
    }

    /// Disarm the guard once the query has finished naturally.
    pub fn defuse(mut self) {
        self.token.take();
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(t) = self.token.take() {
            t.cancel();
        }
    }
}
