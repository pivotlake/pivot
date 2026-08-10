//! pgwire handler glue.
//!
//! The connection lifecycle is fully async: there is no per-connection planner
//! thread. Instead each [`do_query`](PivotQueryHandler::do_query) call hops to
//! `tokio::task::spawn_blocking` to drive the [`planner::Planner`] (which
//! wraps a non-`Send` DuckDB context). The planner is cached in a
//! thread-local on each blocking-pool thread, so subsequent queries that land
//! on the same blocking thread reuse the same DuckDB context — there is one
//! global catalog (see `Server::new`) so it is safe for the planner to be
//! shared across connections that happen to land on the same thread.
//!
//! Cancellation has two sources:
//! - **psql Ctrl-C** sends a `CancelRequest` on a *new* TCP connection. pgwire
//!   routes it through [`ConnectionManager`] to the in-flight query's
//!   `oneshot::Sender`, which trips its `_on_query` `select!` and drops the
//!   `do_query` future.
//! - **client disconnect** drops the whole connection task, which drops the
//!   `do_query` future too.
//!
//! In either case the [`CancelOnDrop`] guard fires
//! [`dispatch::CancelToken::cancel`] on the running dataflow.

use std::cell::RefCell;
use std::fmt::Debug;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::arrow_to_pgwire::PGRowBatch;
use crate::auth::Authenticator;
use arrow_array::{Array, Int64Array, RecordBatch};
use async_trait::async_trait;
use dispatch::{CancelToken, DataFlowHandle, DataFlowStats};
use futures::{Sink, SinkExt, stream};
use lru::LruCache;
use metastore::Metastore;
use pgwire::api::auth::StartupHandler;
use pgwire::api::cancel::{CancelHandler, DefaultCancelHandler};
use pgwire::api::portal::Format;
use pgwire::api::query::SimpleQueryHandler;
use pgwire::api::results::{QueryResponse, Response, Tag};
use pgwire::api::store::PortalStore;
use pgwire::api::{
    ClientInfo, ClientPortalStore, ConnectionManager, NoopHandler, PgWireServerHandlers,
};
use pgwire::error::PgWireResult;
use pgwire::error::{ErrorInfo, PgWireError};
use pgwire::messages::PgWireBackendMessage;
use pgwire::messages::response::NoticeResponse;
use thiserror::Error;
use tokio::task::JoinError;
use tracing::{info, warn};

/// Per-connection flag (a GUC-style name) toggled with `SET pivot_stats = true`.
/// (DuckDB's parser rejects the bare Postgres `= on` keyword, so use `= true`,
/// `= 1`, or quoted `= 'on'`.)
const STATS_FLAG: &str = "pivot_stats";

/// Per-connection flag toggled with `SET perf = 1`: while on, each query the
/// session runs is profiled with its own `perf record` (see [`crate::perf`]),
/// provided the server was started with `PIVOT_PERF_DIR`. Only recognised with
/// the `perf` feature; otherwise `SET perf = …` is an accepted no-op.
#[cfg(feature = "perf")]
const PERF_FLAG: &str = "perf";

thread_local! {
    /// One [`planner::Planner`] (and its non-`Send` DuckDB context) per
    /// tokio blocking-pool thread, lazily initialised on first use. The
    /// catalog is process-global so it's safe to share the same planner
    /// across connections that land on the same thread.
    static PLANNER: RefCell<Option<planner::Planner>> = const { RefCell::new(None) };
}

/// Maximum distinct SQL strings retained.
const PLAN_CACHE_QUERY_CAPACITY: usize = 128;

/// A bounded, shared cache of planned read queries.
///
/// The map is an LRU over exact SQL strings and retains one plan per query.
pub(crate) struct PlanCache {
    inner: Mutex<LruCache<String, Arc<planner::Plan>>>,
}

impl Default for PlanCache {
    fn default() -> Self {
        Self::new(PLAN_CACHE_QUERY_CAPACITY)
    }
}

impl PlanCache {
    fn new(query_capacity: usize) -> Self {
        Self {
            inner: Mutex::new(LruCache::new(
                NonZeroUsize::new(query_capacity).expect("plan cache capacity must be non-zero"),
            )),
        }
    }

    /// Find the cached plan if its revisions match `transaction`'s frozen
    /// snapshots. A revision mismatch discards the entry.
    fn get(
        &self,
        query: &str,
        transaction: &dyn planner::catalog::CatalogTransaction,
    ) -> Option<Arc<planner::Plan>> {
        let mut inner = self.inner.lock().unwrap();
        let plan = inner.get(query)?.clone();
        if plan.has_matching_table_revisions(transaction) {
            Some(plan)
        } else {
            let _ = inner.pop(query);
            None
        }
    }

    /// Insert one cacheable plan, replacing any plan for the same SQL.
    fn insert(&self, query: String, plan: Arc<planner::Plan>) {
        debug_assert!(plan.is_cacheable());
        self.inner.lock().unwrap().put(query, plan);
    }
}

pub(crate) fn with_planner<R>(
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

/// Run `sql` to completion in-process and return its result batches - the path
/// the HTTP dashboard uses instead of the Postgres wire. Plans on the blocking
/// pool (the thread-local DuckDB planner), then compiles and runs on the
/// dispatch workers. A `SET`/`RESET` is a session no-op here and yields nothing.
///
/// Uses [`RecordBatchOperatorSpec::collect`](dispatch::RecordBatchOperatorSpec::collect), which appends a `CopyOut` stage:
/// each batch's ring-backed buffers are deep-copied to plain heap allocations on
/// the worker, so the returned batches are safe to hold and drop on this
/// (non-worker) thread. Calling `execute().collect()` instead would return
/// ring-backed batches whose `Drop` reaches `memory_ctx()` off-worker and aborts.
pub(crate) async fn execute_sql(
    catalog: Arc<catalog::PivotCatalog>,
    dispatcher: dispatch::DataFlowDispatcher,
    plan_cache: Arc<PlanCache>,
    sql: String,
) -> Result<Vec<arrow_array::RecordBatch>, String> {
    // One transaction per statement: the query binds and compiles against this
    // snapshot of the catalog. Committed on success, rolled back on failure
    // (the async block scopes the `?` early-returns so both paths land below).
    let transaction = catalog.begin_transaction();
    let result = async {
        let plan = plan_query(&catalog, transaction.clone(), plan_cache.as_ref(), &sql)
            .await
            .map_err(|e| e.to_string())?;
        // A SET/RESET compiles to no dataflow; nothing to return.
        if plan.as_set_variable().is_some() {
            return Ok(Vec::new());
        }
        // A COMPACT runs its sweeps here on the coordinator and returns no rows.
        if let Some(request) = plan.as_compact() {
            execute_compact(&catalog, request)
                .await
                .map_err(|e| e.to_string())?;
            return Ok(Vec::new());
        }
        // Compile and launch the dataflow (with the CopyOut cap) on the blocking
        // pool; `execute_copying` returns the running handle without collecting.
        // The closure gets its own clone of the transaction Arc only because
        // spawn_blocking moves its captures to another thread.
        let handle = tokio::task::spawn_blocking({
            let transaction = transaction.clone();
            move || -> Result<DataFlowHandle<arrow_array::RecordBatch>, String> {
                let spec = plan
                    .compile(&dispatcher, transaction.as_ref())
                    .map_err(|e| e.to_string())?;
                Ok(spec.execute_copying())
            }
        })
        .await
        .map_err(|e| e.to_string())??;

        // If this future is dropped before collection finishes - e.g. the HTTP
        // client aborted the request via the dashboard's Stop button - cancel the
        // running dataflow so its workers stop instead of finishing a doomed query.
        let guard = CancelOnDrop::new(handle.cancel_token());
        let batches = tokio::task::spawn_blocking(move || handle.collect())
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        guard.defuse();
        Ok(batches)
    }
    .await;
    match result {
        Ok(batches) => {
            transaction.commit().await.map_err(|e| e.to_string())?;
            Ok(batches)
        }
        Err(error) => {
            transaction.rollback();
            Err(error)
        }
    }
}

#[derive(Debug, Error)]
pub(crate) enum Error {
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
    #[error("failed to encode result row: {0}")]
    Encode(String),
}

impl Error {
    /// Convert into a `PgWireError::UserError` so it serialises as a normal
    /// error response on the wire (severity `ERROR`, populated SQLSTATE).
    pub fn into_pgwire(self) -> PgWireError {
        const INTERNAL_ERROR: &str = "XX000";

        let info = ErrorInfo::new(
            "ERROR".to_string(),
            INTERNAL_ERROR.to_string(),
            self.to_string(),
        );
        PgWireError::UserError(Box::new(info))
    }
}

type Result<T, E = Error> = std::result::Result<T, E>;

/// Reuse the exact SQL's plan when its complete table-revision map matches this
/// transaction; otherwise plan inside the same transaction and cache the result
/// only when the planner marked it safe.
/// Run a `COMPACT` statement: resolve the datastore it names (default when
/// unqualified) and sweep the table synchronously. The sweeps commit through
/// the datastore's own log CAS, independent of the statement's transaction.
async fn execute_compact(
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

async fn plan_query(
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

/// Plan a prepared statement's execute. The parameter values are substituted
/// into the plan as constants, so the result is specific to these values and
/// deliberately never touches the SQL-keyed plan cache.
async fn plan_prepared_query(
    catalog: &Arc<catalog::PivotCatalog>,
    transaction: Arc<dyn planner::catalog::CatalogTransaction>,
    query: &str,
    parameters: Vec<planner::ParameterValue>,
) -> Result<Arc<planner::Plan>> {
    let catalog = catalog.clone();
    let query = query.to_string();
    tokio::task::spawn_blocking(move || -> Result<Arc<planner::Plan>> {
        with_planner(&catalog, |planner| {
            Ok(Arc::new(planner.plan_with_parameters(
                &query,
                transaction,
                &parameters,
            )?))
        })?
    })
    .await
    .map_err(Error::PlannerPanic)?
}

/// Output converted before it leaves the dispatch worker. Query
/// batches become pgwire rows; INSERT's internal one-row result becomes an
/// owned count that the coordinator turns into an `INSERT 0 n` command tag.
enum WorkerOutput {
    QueryRows(std::result::Result<PGRowBatch, String>),
    AffectedRows(std::result::Result<usize, String>),
}

fn build_worker_output(
    batch: RecordBatch,
    is_insert: bool,
    result_format: &Format,
) -> WorkerOutput {
    if !is_insert {
        return WorkerOutput::QueryRows(PGRowBatch::new(batch, result_format));
    }
    let count = (|| {
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
    })();
    WorkerOutput::AffectedRows(count)
}

fn build_pgwire_response(outputs: Vec<WorkerOutput>, is_insert: bool) -> Result<Response> {
    if is_insert {
        if outputs.len() != 1 {
            return Err(Error::InvalidInsertResult(format!(
                "expected one affected-row output, got {}",
                outputs.len()
            )));
        }
        let count = match outputs.into_iter().next().unwrap() {
            WorkerOutput::AffectedRows(count) => count.map_err(Error::InvalidInsertResult)?,
            WorkerOutput::QueryRows(_) => {
                return Err(Error::InvalidInsertResult(
                    "received a query batch".to_string(),
                ));
            }
        };
        return Ok(Response::Execution(
            Tag::new("INSERT").with_oid(0).with_rows(count),
        ));
    }

    let batches = outputs
        .into_iter()
        .map(|output| match output {
            WorkerOutput::QueryRows(batch) => batch.map_err(Error::Encode),
            WorkerOutput::AffectedRows(_) => Err(Error::InvalidInsertResult(
                "received an INSERT count for a query".to_string(),
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    let fields = batches
        .first()
        .map_or(Arc::new(vec![]), |batch| batch.fields.clone());
    Ok(Response::Query(QueryResponse::new(
        fields,
        stream::iter(batches.into_iter().flat_map(|batch| batch.rows).map(Ok)),
    )))
}

/// If `do_query` is dropped (cancel request, client disconnect)
/// before the dataflow finishes, this guard fires `handle.cancel()` so the
/// workers stop instead of running the rest of a doomed query.
struct CancelOnDrop {
    token: Option<CancelToken>,
}

impl CancelOnDrop {
    fn new(token: CancelToken) -> Self {
        Self { token: Some(token) }
    }

    /// Disarm the guard once the query has finished naturally.
    fn defuse(mut self) {
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

/// pgwire `SimpleQueryHandler` and `ExtendedQueryHandler` (the latter in
/// [`crate::extended_query`]): plans, compiles, and runs each query on the
/// blocking pool, surfacing errors and supporting cancellation.
pub struct PivotQueryHandler {
    pub(crate) catalog: Arc<catalog::PivotCatalog>,
    dispatcher: dispatch::DataFlowDispatcher,
    plan_cache: Arc<PlanCache>,
    /// The extended protocol's Parse-time planner, shared with pgwire's
    /// describe defaults through `ExtendedQueryHandler::query_parser`.
    pub(crate) query_parser: Arc<crate::extended_query::PivotQueryParser>,
}

impl PivotQueryHandler {
    pub fn new(
        catalog: Arc<catalog::PivotCatalog>,
        dispatcher: dispatch::DataFlowDispatcher,
        plan_cache: Arc<PlanCache>,
    ) -> Self {
        Self {
            query_parser: Arc::new(crate::extended_query::PivotQueryParser::new(
                catalog.clone(),
            )),
            catalog,
            dispatcher,
            plan_cache,
        }
    }

    /// Plan `query`, then either run it (timing each phase, tallying the
    /// dataflow's IO/CPU work when `collect_stats`) or — if it turned out to be a
    /// `SET`/`RESET` — return that for the caller to apply to the connection.
    ///
    /// `parameters` are a prepared statement's bound values (empty for the
    /// simple protocol and unparameterized statements), and `result_format`
    /// the client-requested per-column result encoding (always text for the
    /// simple protocol).
    ///
    /// The whole statement runs inside one catalog transaction: everything it
    /// binds and compiles reads that snapshot, and the transaction is committed
    /// on success and rolled back on failure (or cancellation).
    pub(crate) async fn run_query(
        &self,
        query: &str,
        parameters: Vec<planner::ParameterValue>,
        result_format: Format,
        collect_stats: bool,
        with_perf: bool,
    ) -> Result<Outcome> {
        let transaction = self.catalog.begin_transaction();
        // The async block scopes the body's `?` early-returns so success and
        // failure both land on the commit/rollback at the end.
        let result: Result<Outcome> = async {
            let dispatcher = self.dispatcher.clone();
            let query = query.to_string();

            let started = Instant::now();
            let plan = if parameters.is_empty() {
                plan_query(
                    &self.catalog,
                    transaction.clone(),
                    self.plan_cache.as_ref(),
                    &query,
                )
                .await?
            } else {
                plan_prepared_query(&self.catalog, transaction.clone(), &query, parameters).await?
            };
            let plan_time = started.elapsed();

            // A `SET`/`RESET` is a session command, not a query — DuckDB parsed and
            // typed it for us, so there's no string-munging here. It compiles to no
            // dataflow; hand it back for `do_query` to apply to the connection.
            if let Some(set) = plan.as_set_variable() {
                return Ok(Outcome::Set {
                    name: set.name.clone(),
                    value: set.value.clone(),
                });
            }
            // A COMPACT also compiles to no dataflow: the sweep drives
            // dataflows of its own, so it runs here on the coordinator, never
            // on a worker.
            if let Some(request) = plan.as_compact() {
                execute_compact(&self.catalog, request).await?;
                return Ok(Outcome::Compact);
            }
            let is_insert = matches!(&plan.root.operator, planner::Operator::Insert(_));

            // When the session ran `SET perf = 1`, start a `perf record` scoped to the
            // worker threads and mark this query's dataflows profiled. Marking them
            // makes the workers run *only* this dataflow for its duration (other
            // queries on the pool pause), so the capture is just this
            // dataflow, and exclusive mode lifts as the dataflows finish. `start`
            // runs on a blocking thread (it sleeps waiting for perf to attach); the
            // guard is stopped off the reactor after drain (its `child.wait()` blocks
            // until the report flushes). On error/cancel the guard instead drops in
            // this async frame, which is rare. Compiled out without the feature.
            #[cfg(feature = "perf")]
            let (dispatcher, mut perf_guard) = if with_perf {
                let sql = query.clone();
                let guard = tokio::task::spawn_blocking(move || crate::perf::start(&sql))
                    .await
                    .ok()
                    .flatten();
                match guard {
                    Some(guard) => (dispatcher.with_profiling(true), Some(guard)),
                    None => (dispatcher, None),
                }
            } else {
                (dispatcher, None)
            };
            #[cfg(not(feature = "perf"))]
            let _ = with_perf;

            // Compile the plan into a fresh dataflow and launch it. `compile` is pure
            // pivot work (no DuckDB), so it runs on any blocking thread without the
            // planner thread-local. `execute_with_stats` turns on the dataflow's
            // IO/CPU tally only when the client asked for it.
            let started = Instant::now();
            let compile_transaction = transaction.clone();
            let handle = tokio::task::spawn_blocking(move || -> Result<DataFlowHandle<_>> {
                let outputs = plan
                    .compile(&dispatcher, compile_transaction.as_ref())?
                    .map(move || {
                        let result_format = result_format.clone();
                        move |batch| build_worker_output(batch, is_insert, &result_format)
                    });
                Ok(if collect_stats {
                    outputs.execute_with_stats()
                } else {
                    outputs.execute()
                })
            })
            .await
            .map_err(Error::PlannerPanic)??;
            let compile_time = started.elapsed();

            // Cancel the dataflow if our future is dropped before drain finishes —
            // covers both psql Ctrl-C (pgwire's `_on_query` select drops us) and
            // raw disconnects (whole connection task dropped).
            let guard = CancelOnDrop::new(handle.cancel_token());
            let started = Instant::now();
            let (outputs, flow) = tokio::task::spawn_blocking(move || handle.collect_with_stats())
                .await
                .map_err(Error::WorkerPanic)??;
            let exec_time = started.elapsed();
            guard.defuse();

            // Stop perf off the reactor: `child.wait()` blocks until the report is
            // flushed (seconds for a large capture), which would otherwise stall this
            // tokio worker thread.
            #[cfg(feature = "perf")]
            if let Some(perf) = perf_guard.take() {
                let _ = tokio::task::spawn_blocking(move || drop(perf)).await;
            }

            let response = build_pgwire_response(outputs, is_insert)?;
            Ok(Outcome::Response(
                response,
                QueryStats {
                    plan: plan_time,
                    compile: compile_time,
                    exec: exec_time,
                    flow,
                },
            ))
        }
        .await;
        match result {
            Ok(outcome) => {
                transaction.commit().await?;
                Ok(outcome)
            }
            Err(error) => {
                transaction.rollback();
                Err(error)
            }
        }
    }
}

/// What [`run_query`](PivotQueryHandler::run_query) resolved a statement to.
pub(crate) enum Outcome {
    /// A completed statement response plus where its time went.
    Response(Response, QueryStats),
    /// A `SET`/`RESET` of a session variable (DuckDB-parsed). `value` is `None`
    /// for `RESET`; the server decides which names actually mean anything.
    Set { name: String, value: Option<String> },
    /// A completed `COMPACT` statement; the sweeps already ran.
    Compact,
}

/// Where a query's time went — phase wall-clocks plus the dataflow's IO/CPU
/// tally — formatted into a one-line client `NOTICE` when stats are on.
pub(crate) struct QueryStats {
    plan: Duration,
    compile: Duration,
    exec: Duration,
    flow: DataFlowStats,
}

impl QueryStats {
    fn summary(&self) -> String {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
        // The IO/cpu figures are sums over workers and in-flight operations, so
        // the totals can top exec time.
        format!(
            "stats: plan={:.1}ms compile={:.1}ms exec={:.1}ms | \
             disk={} ops/{:.1}MiB/read={:.1}ms/write={:.1}ms  \
             http={} ops/{:.1}MiB/get={:.1}ms/upload={:.1}ms  \
             http-disk-cache={} ops/{:.1}MiB/{:.1}ms  cpu={:.1}ms",
            ms(self.plan),
            ms(self.compile),
            ms(self.exec),
            self.flow.disk_requests,
            mib(self.flow.disk_bytes),
            ms(self.flow.disk_read_time),
            ms(self.flow.disk_write_time),
            self.flow.http_requests,
            mib(self.flow.http_bytes),
            ms(self.flow.http_get_time),
            ms(self.flow.http_upload_time),
            self.flow.disk_cache_requests,
            mib(self.flow.disk_cache_bytes),
            ms(self.flow.disk_cache_time),
            ms(self.flow.cpu),
        )
    }
}

/// Apply a `SET`/`RESET` to the connection. pivot only knows [`STATS_FLAG`];
/// every other name is accepted as a no-op, since clients routinely set GUCs
/// (`client_encoding`, `application_name`, …) we don't model. Acks with the verb
/// the client used (`SET`/`RESET`).
fn apply_set<C: ClientInfo>(client: &mut C, name: &str, value: Option<&str>) -> Response {
    if name.eq_ignore_ascii_case(STATS_FLAG) {
        // RESET (no value) or any non-truthy value turns it off.
        if value.is_some_and(is_truthy) {
            client
                .metadata_mut()
                .insert(STATS_FLAG.to_string(), "on".to_string());
        } else {
            client.metadata_mut().remove(STATS_FLAG);
        }
    }
    #[cfg(feature = "perf")]
    if name.eq_ignore_ascii_case(PERF_FLAG) {
        if value.is_some_and(is_truthy) {
            client
                .metadata_mut()
                .insert(PERF_FLAG.to_string(), "on".to_string());
        } else {
            client.metadata_mut().remove(PERF_FLAG);
        }
    }
    Response::Execution(Tag::new(if value.is_none() { "RESET" } else { "SET" }))
}

/// Whether a serialized `SET` value reads as on. Covers the spellings DuckDB may
/// produce — `true`/`1` for a boolean/int literal, `on`/`yes` for a quoted string.
fn is_truthy(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "true" | "t" | "1" | "on" | "yes"
    )
}

/// Whether this connection has `pivot_stats` on.
pub(crate) fn stats_on<C: ClientInfo>(client: &C) -> bool {
    client.metadata().get(STATS_FLAG).is_some_and(|v| v == "on")
}

/// Whether this connection has `perf` on (set via `SET perf = 1`).
#[cfg(feature = "perf")]
pub(crate) fn perf_on<C: ClientInfo>(client: &C) -> bool {
    client.metadata().get(PERF_FLAG).is_some_and(|v| v == "on")
}

/// Turn a completed statement's [`Outcome`] into its wire [`Response`]:
/// apply a `SET`/`RESET` to the connection, and send the stats breakdown as
/// an INFO notice before the rows when the session asked for it. Shared by
/// the simple and extended query handlers.
pub(crate) async fn build_outcome_response<C>(
    client: &mut C,
    outcome: Outcome,
    with_stats: bool,
) -> PgWireResult<Response>
where
    C: ClientInfo + Sink<PgWireBackendMessage> + Unpin + Send,
    PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
{
    Ok(match outcome {
        Outcome::Set { name, value } => apply_set(client, &name, value.as_deref()),
        Outcome::Compact => Response::Execution(Tag::new("COMPACT")),
        Outcome::Response(res, stats) => {
            if with_stats {
                let notice = NoticeResponse::from(ErrorInfo::new(
                    "INFO".to_string(),
                    "00000".to_string(),
                    stats.summary(),
                ));
                client
                    .send(PgWireBackendMessage::NoticeResponse(notice))
                    .await?;
            }
            res
        }
    })
}

#[async_trait]
impl SimpleQueryHandler for PivotQueryHandler {
    async fn do_query<C>(&self, client: &mut C, query: &str) -> PgWireResult<Vec<Response>>
    where
        C: ClientInfo + ClientPortalStore + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::PortalStore: PortalStore,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        let with_stats = stats_on(client);
        #[cfg(feature = "perf")]
        let with_perf = perf_on(client);
        #[cfg(not(feature = "perf"))]
        let with_perf = false;

        info!(sql = %query, "query received");
        let outcome = self
            .run_query(
                query,
                Vec::new(),
                Format::UnifiedText,
                with_stats,
                with_perf,
            )
            .await
            .map_err(|e| {
                warn!(error = %e, sql = %query, "query failed");
                e.into_pgwire()
            })?;

        let res = build_outcome_response(client, outcome, with_stats).await?;

        info!(sql = %query, "query succeeded");
        Ok(vec![res])
    }
}

/// Bundle handed to `pgwire::tokio::process_socket` for each connection. Holds
/// the query handler instance reused across the process.
///
/// The default cancel handler is enough: pgwire routes each `CancelRequest`
/// packet through the shared `ConnectionManager` (populated by
/// [`crate::auth::UserStartupHandler`]) to the in-flight query's `do_query`
/// future, which
/// is then dropped. The `CancelOnDrop` guard inside [`PivotQueryHandler::run_query`]
/// fires the dataflow's cancel token from that drop, so we never need to
/// reach into the dispatch layer from a cancel handler.
pub struct PivotHandlers {
    query_handler: Arc<PivotQueryHandler>,
    cancel_handler: Arc<DefaultCancelHandler>,
    /// Shared authentication configuration. Every new connection reads its
    /// current method from the metastore using its startup user name.
    authenticator: Authenticator,
}

impl PivotHandlers {
    pub fn new(
        catalog: Arc<catalog::PivotCatalog>,
        dispatcher: dispatch::DataFlowDispatcher,
        plan_cache: Arc<PlanCache>,
        metastore: Arc<dyn Metastore>,
    ) -> Self {
        let manager = Arc::new(ConnectionManager::new());
        Self {
            query_handler: Arc::new(PivotQueryHandler::new(catalog, dispatcher, plan_cache)),
            cancel_handler: Arc::new(DefaultCancelHandler::new(manager.clone())),
            authenticator: Authenticator::new(metastore, manager),
        }
    }
}

impl PgWireServerHandlers for PivotHandlers {
    fn simple_query_handler(&self) -> Arc<impl SimpleQueryHandler> {
        self.query_handler.clone()
    }

    /// pgwire calls this once per connection, which is what per-user routing and
    /// a SCRAM handshake need: the selected method and handshake state span
    /// several messages.
    fn startup_handler(&self) -> Arc<impl StartupHandler> {
        self.authenticator.startup_handler()
    }

    fn cancel_handler(&self) -> Arc<impl CancelHandler> {
        self.cancel_handler.clone()
    }

    fn extended_query_handler(&self) -> Arc<impl pgwire::api::query::ExtendedQueryHandler> {
        self.query_handler.clone()
    }

    fn copy_handler(&self) -> Arc<impl pgwire::api::copy::CopyHandler> {
        Arc::new(NoopHandler)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use planner::DEFAULT_DATASTORE_NAME;
    use planner::catalog::{BoundTable, Column, TableReference, TableRevision};
    use std::collections::HashMap;

    #[derive(Clone, Debug)]
    struct CacheTestTable {
        reference: TableReference,
        revision: TableRevision,
    }

    impl BoundTable for CacheTestTable {
        fn table_reference(&self) -> TableReference {
            self.reference.clone()
        }

        fn table_revision(&self) -> TableRevision {
            self.revision.clone()
        }

        fn compile_scan(
            &self,
            _dispatcher: &dispatch::DataFlowDispatcher,
            _projection: dispatch::Projection,
            _dynamic_filters: Vec<planner::catalog::DynamicScanPredicate>,
            _emit_row_group_metadata: bool,
        ) -> planner::catalog::Result<dispatch::RecordBatchOperatorSpec> {
            unreachable!("cache unit test does not compile its plans")
        }

        fn columns(&self) -> Vec<Column> {
            Vec::new()
        }

        fn clone_box(&self) -> Box<dyn BoundTable> {
            Box::new(self.clone())
        }
    }

    #[derive(Debug)]
    struct RevisionTransaction {
        revisions: HashMap<TableReference, TableRevision>,
    }

    #[async_trait]
    impl planner::catalog::CatalogTransaction for RevisionTransaction {
        fn does_schema_exist(&self, _datastore: &str, schema: &str) -> bool {
            schema == planner::DEFAULT_SCHEMA_NAME
        }

        fn bind_table(&self, _reference: &TableReference) -> Option<Box<dyn BoundTable>> {
            None
        }

        fn table_revision(&self, reference: &TableReference) -> Option<TableRevision> {
            self.revisions.get(reference).cloned()
        }
    }

    fn table() -> TableReference {
        table_in_schema(planner::DEFAULT_SCHEMA_NAME)
    }

    /// The same table name in `schema`. Two schemas can each hold an `events`,
    /// so these are references to different tables.
    fn table_in_schema(schema: &str) -> TableReference {
        TableReference {
            datastore: DEFAULT_DATASTORE_NAME.to_string(),
            schema: schema.to_string(),
            table: "events".to_string(),
        }
    }

    fn revision(identity: &str, version: u64) -> TableRevision {
        TableRevision {
            identity: identity.to_string(),
            version,
        }
    }

    fn transaction(identity: &str, version: u64) -> RevisionTransaction {
        transaction_over(table(), identity, version)
    }

    /// A transaction whose only known table is `reference`, at `identity` and
    /// `version`.
    fn transaction_over(
        reference: TableReference,
        identity: &str,
        version: u64,
    ) -> RevisionTransaction {
        RevisionTransaction {
            revisions: HashMap::from([(reference, revision(identity, version))]),
        }
    }

    fn plan(identity: &str, version: u64) -> Arc<planner::Plan> {
        plan_over(table(), identity, version)
    }

    /// A one-input plan that reads `reference` at `identity` and `version`.
    fn plan_over(reference: TableReference, identity: &str, version: u64) -> Arc<planner::Plan> {
        Arc::new(planner::Plan {
            root: planner::PlanNode {
                name: "input".to_string(),
                inputs: Vec::new(),
                operator: planner::Operator::Input(planner::operator::Input {
                    table: Box::new(CacheTestTable {
                        reference,
                        revision: revision(identity, version),
                    }),
                    columns: Vec::new(),
                    dynamic_filters: Vec::new(),
                    emit_row_group_metadata: false,
                }),
            },
            output_names: Vec::new(),
        })
    }

    #[test]
    fn cache_hit_requires_the_same_identity_and_version() {
        let cache = PlanCache::new(2);
        let cached = plan("table-id", 7);
        cache.insert("SELECT * FROM events".to_string(), cached.clone());

        let hit = cache
            .get("SELECT * FROM events", &transaction("table-id", 7))
            .unwrap();
        let changed_version = cache.get("SELECT * FROM events", &transaction("table-id", 8));
        cache.insert("SELECT * FROM events".to_string(), cached.clone());
        let changed_identity = cache.get("SELECT * FROM events", &transaction("new-table-id", 7));

        assert!(Arc::ptr_eq(&hit, &cached));
        assert!(changed_version.is_none());
        assert!(changed_identity.is_none());
        assert!(cache.inner.lock().unwrap().is_empty());
    }

    /// A cached plan stays valid only while the tables it bound are unchanged,
    /// and a table is identified by its schema as much as by its name: two
    /// schemas can each hold an `events`, and they are different tables. A
    /// revision recorded for one of them must therefore not vouch for a plan
    /// built over the other, however alike the two revisions look.
    #[test]
    fn a_revision_from_another_schema_does_not_validate_a_cached_plan() {
        const QUERY: &str = "SELECT * FROM analytics.events";

        let cache = PlanCache::new(2);
        let cached = plan_over(table_in_schema("analytics"), "table-id", 7);
        cache.insert(QUERY.to_string(), cached.clone());

        let hit = cache
            .get(
                QUERY,
                &transaction_over(table_in_schema("analytics"), "table-id", 7),
            )
            .unwrap();
        let other_schema = cache.get(QUERY, &transaction_over(table(), "table-id", 7));

        assert!(Arc::ptr_eq(&hit, &cached));
        assert!(other_schema.is_none());
    }

    #[test]
    fn inserting_a_new_revision_replaces_the_previous_plan() {
        let cache = PlanCache::new(2);
        let first = plan("table-id", 7);
        let second = plan("table-id", 8);
        cache.insert("SELECT * FROM events".to_string(), first);
        cache.insert("SELECT * FROM events".to_string(), second.clone());

        let second_hit = cache
            .get("SELECT * FROM events", &transaction("table-id", 8))
            .unwrap();

        assert!(Arc::ptr_eq(&second_hit, &second));
        assert_eq!(cache.inner.lock().unwrap().len(), 1);
    }

    #[test]
    fn inserting_over_capacity_evicts_the_least_recently_used_query() {
        let cache = PlanCache::new(2);
        let first = plan("first", 1);
        let second = plan("second", 1);
        let third = plan("third", 1);
        cache.insert("first query".to_string(), first.clone());
        cache.insert("second query".to_string(), second);
        cache.get("first query", &transaction("first", 1)).unwrap();

        cache.insert("third query".to_string(), third.clone());

        assert!(
            cache
                .get("second query", &transaction("second", 1))
                .is_none()
        );
        assert!(Arc::ptr_eq(
            &cache.get("first query", &transaction("first", 1)).unwrap(),
            &first
        ));
        assert!(Arc::ptr_eq(
            &cache.get("third query", &transaction("third", 1)).unwrap(),
            &third
        ));
    }
}
