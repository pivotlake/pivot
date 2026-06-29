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
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::arrow_to_pgwire::PGRowBatch;
use async_trait::async_trait;
use dispatch::{CancelToken, DataFlowHandle, DataFlowStats};
use futures::{Sink, SinkExt, stream};
use pgwire::api::auth::StartupHandler;
use pgwire::api::auth::noop::NoopStartupHandler;
use pgwire::api::cancel::{CancelHandler, DefaultCancelHandler};
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

/// Whether a planned query is a read-only (SELECT) plan whose `Plan` is safe to
/// cache. Identified *positively* from the plan's root operator: only the
/// read-only operators are cacheable, so any statement we don't explicitly
/// allow — `CreateTable` today, or any operator variant added later — defaults
/// to not-cached (and flushes the cache, since it may change the schema cached
/// plans were built against).
fn plan_is_cacheable(plan: &planner::Plan) -> bool {
    use planner::Operator;
    matches!(
        plan.root.operator,
        Operator::Input(_)
            | Operator::Projection(_)
            | Operator::Filter(_)
            | Operator::Aggregate(_)
            | Operator::OrderBy(_)
            | Operator::TopN(_)
            | Operator::Explain(_)
    )
}

fn with_planner<R>(
    catalog: &Arc<dyn planner::catalog::Catalog>,
    f: impl FnOnce(&mut planner::Planner) -> R,
) -> R {
    PLANNER.with_borrow_mut(|opt| {
        let planner = opt.get_or_insert_with(|| planner::Planner::new(catalog.clone()));
        f(planner)
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
    catalog: Arc<dyn planner::catalog::Catalog>,
    dispatcher: dispatch::DataFlowDispatcher,
    sql: String,
) -> Result<Vec<arrow_array::RecordBatch>, String> {
    let plan = {
        let catalog = catalog.clone();
        tokio::task::spawn_blocking(move || with_planner(&catalog, |p| p.plan(&sql)))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?
    };
    // A SET/RESET compiles to no dataflow; nothing to return.
    if plan.as_set_variable().is_some() {
        return Ok(Vec::new());
    }
    let plan = Arc::new(plan);
    // Compile and launch the dataflow (with the CopyOut cap) on the blocking
    // pool; `execute_copying` returns the running handle without collecting.
    let handle = tokio::task::spawn_blocking(
        move || -> Result<DataFlowHandle<arrow_array::RecordBatch>, String> {
            let spec = plan.compile(&dispatcher).map_err(|e| e.to_string())?;
            Ok(spec.execute_copying())
        },
    )
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

#[derive(Debug, Error)]
enum Error {
    #[error(transparent)]
    Plan(#[from] planner::Error),
    #[error("compile error: {0}")]
    Compile(#[from] planner::compile::Error),
    #[error(transparent)]
    DataFlow(#[from] dispatch::DataFlowError),
    #[error("waiter thread panicked: {0}")]
    WorkerPanic(JoinError),
    #[error("waiter thread panicked: {0}")]
    PlannerPanic(JoinError),
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

/// pgwire `SimpleQueryHandler`: plans, compiles, and runs each query on the
/// blocking pool, surfacing errors and supporting cancellation.
pub struct PivotQueryHandler {
    catalog: Arc<dyn planner::catalog::Catalog>,
    dispatcher: dispatch::DataFlowDispatcher,
    /// Cache of planned (but not yet compiled) query plans, keyed by SQL text.
    /// Planning a statement (DuckDB optimize + bridge round-trip + plan
    /// translation) is a fixed few-millisecond cost paid on every query — a
    /// large fraction of a small query's latency. Repeated SELECTs (the common
    /// case for dashboards/benchmarks) reuse the cached `Plan` and only re-run
    /// the cheap `compile` + execute. Shared across connections; only SELECTs
    /// are cached and any non-SELECT statement flushes it (see `run_query`).
    plan_cache: Arc<Mutex<HashMap<String, Arc<planner::Plan>>>>,
}

impl PivotQueryHandler {
    pub fn new(
        catalog: Arc<dyn planner::catalog::Catalog>,
        dispatcher: dispatch::DataFlowDispatcher,
    ) -> Self {
        Self {
            catalog,
            dispatcher,
            plan_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Plan `query`, then either run it (timing each phase, tallying the
    /// dataflow's IO/CPU work when `collect_stats`) or — if it turned out to be a
    /// `SET`/`RESET` — return that for the caller to apply to the connection.
    async fn run_query(
        &self,
        query: &str,
        collect_stats: bool,
        with_perf: bool,
    ) -> Result<Outcome> {
        let dispatcher = self.dispatcher.clone();
        let query = query.to_string();

        // Reuse a cached plan if we've planned this exact SQL before. Only
        // read-only SELECT plans are ever inserted, so a cache hit is always a
        // SELECT regardless of what `query` is. Planning is a fixed few-ms cost;
        // skipping it on repeated SELECTs shaves that off every query after the
        // first — which the `plan` phase time below makes visible (near-zero on a
        // hit, the full planner round-trip on a miss).
        let started = Instant::now();
        let cached = self.plan_cache.lock().unwrap().get(&query).cloned();
        let plan = match cached {
            Some(plan) => plan,
            None => {
                let catalog = self.catalog.clone();
                let q = query.clone();
                let plan = tokio::task::spawn_blocking(move || -> Result<Arc<planner::Plan>> {
                    with_planner(&catalog, |planner| Ok(Arc::new(planner.plan(&q)?)))
                })
                .await
                .map_err(Error::PlannerPanic)??;
                // Cache SELECTs; treat anything else (DDL/DML/…) as a cache
                // flush — it may invalidate the schema cached plans were built
                // against — and don't cache it.
                let mut cache = self.plan_cache.lock().unwrap();
                if plan_is_cacheable(&plan) {
                    cache.insert(query.clone(), plan.clone());
                } else {
                    cache.clear();
                }
                plan
            }
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

        // When the session ran `SET perf = 1`, start a `perf record` scoped to the
        // worker threads and mark this query's dataflows profiled. Marking them
        // makes the workers run *only* this dataflow for its duration (other
        // queries / ingest on the pool pause), so the capture is just this
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
        let handle = tokio::task::spawn_blocking(move || -> Result<DataFlowHandle<_>> {
            let rows = plan.compile(&dispatcher)?.map(|| |b| PGRowBatch::from(b));
            Ok(if collect_stats {
                rows.execute_with_stats()
            } else {
                rows.execute()
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
        let (batches, flow) = tokio::task::spawn_blocking(move || handle.collect_with_stats())
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

        let fields = batches
            .first()
            .map_or(Arc::new(vec![]), |b| b.fields.clone());
        let response = Response::Query(QueryResponse::new(
            fields,
            stream::iter(batches.into_iter().flat_map(|b| b.rows).map(Ok)),
        ));
        Ok(Outcome::Query(
            response,
            QueryStats {
                plan: plan_time,
                compile: compile_time,
                exec: exec_time,
                flow,
            },
        ))
    }
}

/// What [`run_query`](PivotQueryHandler::run_query) resolved a statement to.
enum Outcome {
    /// A normal query: its rows plus where its time went.
    Query(Response, QueryStats),
    /// A `SET`/`RESET` of a session variable (DuckDB-parsed). `value` is `None`
    /// for `RESET`; the server decides which names actually mean anything.
    Set { name: String, value: Option<String> },
}

/// Where a query's time went — phase wall-clocks plus the dataflow's IO/CPU
/// tally — formatted into a one-line client `NOTICE` when stats are on.
struct QueryStats {
    plan: Duration,
    compile: Duration,
    exec: Duration,
    flow: DataFlowStats,
}

impl QueryStats {
    fn summary(&self) -> String {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
        // The IO/cpu figures are sums (over workers and in-flight reads), so
        // `time/reads` is the average read latency and the totals can top exec.
        format!(
            "stats: plan={:.1}ms compile={:.1}ms exec={:.1}ms | \
             disk={} reads/{:.1}MiB/{:.1}ms  http={} reads/{:.1}MiB/{:.1}ms  \
             http-disk-cache={} reads/{:.1}MiB/{:.1}ms  cpu={:.1}ms",
            ms(self.plan),
            ms(self.compile),
            ms(self.exec),
            self.flow.disk_requests,
            mib(self.flow.disk_bytes),
            ms(self.flow.disk_time),
            self.flow.http_requests,
            mib(self.flow.http_bytes),
            ms(self.flow.http_time),
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
fn stats_on<C: ClientInfo>(client: &C) -> bool {
    client.metadata().get(STATS_FLAG).is_some_and(|v| v == "on")
}

/// Whether this connection has `perf` on (set via `SET perf = 1`).
#[cfg(feature = "perf")]
fn perf_on<C: ClientInfo>(client: &C) -> bool {
    client.metadata().get(PERF_FLAG).is_some_and(|v| v == "on")
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
            .run_query(query, with_stats, with_perf)
            .await
            .map_err(|e| {
                warn!(error = %e, sql = %query, "query failed");
                e.into_pgwire()
            })?;

        let res = match outcome {
            Outcome::Set { name, value } => apply_set(client, &name, value.as_deref()),
            Outcome::Query(res, stats) => {
                // Send the breakdown as an INFO notice before the rows.
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
        };

        info!(sql = %query, "query succeeded");
        Ok(vec![res])
    }
}

/// Startup handler that registers each new connection with the shared
/// [`ConnectionManager`] so that subsequent `CancelRequest` packets can be
/// routed back to the running query. Otherwise behaves as a noop (no auth).
pub struct PivotStartupHandler {
    manager: Arc<ConnectionManager>,
}

impl PivotStartupHandler {
    pub fn new(manager: Arc<ConnectionManager>) -> Self {
        Self { manager }
    }
}

impl NoopStartupHandler for PivotStartupHandler {
    fn connection_manager(&self) -> Option<Arc<ConnectionManager>> {
        Some(self.manager.clone())
    }
}

/// Bundle handed to `pgwire::tokio::process_socket` for each connection. Holds
/// the query handler instance reused across the process.
///
/// The default cancel handler is enough: pgwire routes each `CancelRequest`
/// packet through the shared `ConnectionManager` (populated by
/// [`PivotStartupHandler`]) to the in-flight query's `do_query` future, which
/// is then dropped. The `CancelOnDrop` guard inside [`PivotQueryHandler::run_query`]
/// fires the dataflow's cancel token from that drop, so we never need to
/// reach into the dispatch layer from a cancel handler.
pub struct PivotHandlers {
    query_handler: Arc<PivotQueryHandler>,
    startup_handler: Arc<PivotStartupHandler>,
    cancel_handler: Arc<DefaultCancelHandler>,
}

impl PivotHandlers {
    pub fn new(
        catalog: Arc<dyn planner::catalog::Catalog>,
        dispatcher: dispatch::DataFlowDispatcher,
    ) -> Self {
        let manager = Arc::new(ConnectionManager::new());
        Self {
            query_handler: Arc::new(PivotQueryHandler::new(catalog, dispatcher)),
            startup_handler: Arc::new(PivotStartupHandler::new(manager.clone())),
            cancel_handler: Arc::new(DefaultCancelHandler::new(manager)),
        }
    }
}

impl PgWireServerHandlers for PivotHandlers {
    fn simple_query_handler(&self) -> Arc<impl SimpleQueryHandler> {
        self.query_handler.clone()
    }

    fn startup_handler(&self) -> Arc<impl StartupHandler> {
        self.startup_handler.clone()
    }

    fn cancel_handler(&self) -> Arc<impl CancelHandler> {
        self.cancel_handler.clone()
    }

    fn extended_query_handler(&self) -> Arc<impl pgwire::api::query::ExtendedQueryHandler> {
        Arc::new(NoopHandler)
    }

    fn copy_handler(&self) -> Arc<impl pgwire::api::copy::CopyHandler> {
        Arc::new(NoopHandler)
    }
}
