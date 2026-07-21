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
use crate::pg_param;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, Scalar};
use arrow_schema::DataType;
use async_trait::async_trait;
use dispatch::{CancelToken, DataFlowHandle, DataFlowStats, RecordBatchOperatorSpec};
use futures::{Sink, SinkExt, stream};
use pgwire::api::Type;
use pgwire::api::auth::StartupHandler;
use pgwire::api::auth::noop::NoopStartupHandler;
use pgwire::api::cancel::{CancelHandler, DefaultCancelHandler};
use pgwire::api::portal::{Format, Portal};
use pgwire::api::query::{ExtendedQueryHandler, SimpleQueryHandler};
use pgwire::api::results::{FieldInfo, QueryResponse, Response, Tag};
use pgwire::api::stmt::QueryParser;
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
    // One transaction per statement: the query binds and compiles against this
    // snapshot of the catalog. Committed on success, rolled back on failure
    // (the async block scopes the `?` early-returns so both paths land below).
    let transaction = catalog.begin_transaction();
    let result = async {
        let plan = {
            let catalog = catalog.clone();
            let transaction = transaction.clone();
            tokio::task::spawn_blocking(move || {
                with_planner(&catalog, |p| p.plan(&sql, transaction))
            })
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
            tokio::task::spawn_blocking(move || catalog.commit_transaction(transaction))
                .await
                .map_err(|e| e.to_string())?
                .map_err(|e| e.to_string())?;
            Ok(batches)
        }
        Err(error) => {
            catalog.rollback_transaction(transaction);
            Err(error)
        }
    }
}

#[derive(Debug, Error)]
enum Error {
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
    #[error("transaction commit thread panicked: {0}")]
    CommitPanic(JoinError),
    #[error("invalid INSERT row-count result: {0}")]
    InvalidInsertResult(String),
    #[error("binding parameters failed: {0}")]
    ParamBind(String),
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

/// Output converted before it leaves the dispatch worker. Query
/// batches become pgwire rows; INSERT's internal one-row result becomes an
/// owned count that the coordinator turns into an `INSERT 0 n` command tag.
enum WorkerOutput {
    QueryRows(PGRowBatch),
    AffectedRows(std::result::Result<usize, String>),
}

fn build_worker_output(batch: RecordBatch, is_insert: bool, format: &Format) -> WorkerOutput {
    if !is_insert {
        return WorkerOutput::QueryRows(PGRowBatch::from_batch(batch, format));
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
            WorkerOutput::QueryRows(batch) => Ok(batch),
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

/// Build a dataflow spec (`build_spec`, run on the blocking pool since it may
/// compile a plan), launch it, and drain it to per-worker [`WorkerOutput`]s plus
/// the IO/CPU stats. Each output batch is turned into a `WorkerOutput` on the
/// worker (an INSERT count or query rows, per `is_insert`). A [`CancelOnDrop`]
/// guard cancels the running flow if this future is dropped before the drain
/// finishes. Shared by the simple and extended query handlers.
async fn run_dataflow<F>(
    build_spec: F,
    is_insert: bool,
    collect_stats: bool,
    result_format: Arc<Format>,
) -> Result<(Vec<WorkerOutput>, DataFlowStats)>
where
    F: FnOnce() -> Result<RecordBatchOperatorSpec> + Send + 'static,
{
    let handle = tokio::task::spawn_blocking(move || -> Result<DataFlowHandle<WorkerOutput>> {
        let outputs = build_spec()?.map(move || {
            let format = result_format.clone();
            move |batch| build_worker_output(batch, is_insert, &format)
        });
        Ok(if collect_stats {
            outputs.execute_with_stats()
        } else {
            outputs.execute()
        })
    })
    .await
    .map_err(Error::PlannerPanic)??;
    let guard = CancelOnDrop::new(handle.cancel_token());
    let (outputs, flow) = tokio::task::spawn_blocking(move || handle.collect_with_stats())
        .await
        .map_err(Error::WorkerPanic)??;
    guard.defuse();
    Ok((outputs, flow))
}

/// pgwire `SimpleQueryHandler`: plans, compiles, and runs each query on the
/// blocking pool, surfacing errors and supporting cancellation.
pub struct PivotQueryHandler {
    catalog: Arc<dyn planner::catalog::Catalog>,
    dispatcher: dispatch::DataFlowDispatcher,
    /// Cache of planned (but not yet compiled) query plans, keyed by SQL text.
    /// Planning a statement (DuckDB optimize + bridge round-trip + plan
    /// translation) is a fixed few-millisecond cost paid on every query, a
    /// large fraction of a small query's latency. Repeated SELECTs (the common
    /// case for dashboards/benchmarks) reuse the cached `Plan` and only re-run
    /// the cheap `compile` + execute. Shared across connections; only SELECTs
    /// are cached and any non-SELECT statement flushes it (see `run_query`).
    ///
    /// Plans carry no catalog state, so a cache hit compiled under the current
    /// query's transaction reads that transaction's view. DDL flushes the
    /// cache (it may change the schema cached plans were bound against).
    plan_cache: Arc<Mutex<HashMap<String, Arc<planner::Plan>>>>,
    /// The extended-protocol query parser (plans a prepared statement once at
    /// Parse time). Shared with the `ExtendedQueryHandler` impl on this struct.
    query_parser: Arc<PivotQueryParser>,
}

impl PivotQueryHandler {
    pub fn new(
        catalog: Arc<dyn planner::catalog::Catalog>,
        dispatcher: dispatch::DataFlowDispatcher,
    ) -> Self {
        Self {
            query_parser: Arc::new(PivotQueryParser {
                catalog: catalog.clone(),
            }),
            catalog,
            dispatcher,
            plan_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Plan `query`, then either run it (timing each phase, tallying the
    /// dataflow's IO/CPU work when `collect_stats`) or — if it turned out to be a
    /// `SET`/`RESET` — return that for the caller to apply to the connection.
    ///
    /// The whole statement runs inside one catalog transaction: everything it
    /// binds and compiles reads that snapshot, and the transaction is committed
    /// on success and rolled back on failure (or cancellation).
    async fn run_query(
        &self,
        query: &str,
        collect_stats: bool,
        with_perf: bool,
    ) -> Result<Outcome> {
        let transaction = self.catalog.begin_transaction();
        // The async block scopes the body's `?` early-returns so success and
        // failure both land on the commit/rollback at the end.
        let result: Result<Outcome> = async {
            let dispatcher = self.dispatcher.clone();
            let query = query.to_string();

            // Reuse a cached plan if we've planned this exact SQL before. Only
            // read-only SELECT plans are ever inserted, so a cache hit is always a
            // SELECT regardless of what `query` is. Plans carry no snapshot, so a
            // hit still reads this transaction's view at compile. Planning is a
            // fixed few-ms cost; skipping it on repeated SELECTs shaves that off
            // every query after the first, which the `plan` phase time below makes
            // visible (near-zero on a hit, the full planner round-trip on a miss).
            let started = Instant::now();
            let cached = self.plan_cache.lock().unwrap().get(&query).cloned();
            let plan = match cached {
                Some(plan) => plan,
                None => {
                    let catalog = self.catalog.clone();
                    let q = query.clone();
                    let planning_transaction = transaction.clone();
                    let plan =
                        tokio::task::spawn_blocking(move || -> Result<Arc<planner::Plan>> {
                            with_planner(&catalog, |planner| {
                                Ok(Arc::new(planner.plan(&q, planning_transaction)?))
                            })
                        })
                        .await
                        .map_err(Error::PlannerPanic)??;
                    // Cache SELECTs; treat anything else (DDL/DML/…) as a cache
                    // flush (it may invalidate the schema cached plans were built
                    // against) and don't cache it.
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

            // Compile the plan into a fresh dataflow, launch it, and drain it.
            // `compile` is pure pivot work (no DuckDB), so it runs on the blocking
            // pool without the planner thread-local. `run_dataflow` cancels the
            // flow if our future is dropped before the drain finishes (psql
            // Ctrl-C, a raw disconnect).
            let started = Instant::now();
            let compile_transaction = transaction.clone();
            let (outputs, flow) = run_dataflow(
                move || {
                    plan.compile(&dispatcher, compile_transaction.as_ref())
                        .map_err(Error::from)
                },
                is_insert,
                collect_stats,
                // The simple query protocol always emits text.
                Arc::new(Format::UnifiedText),
            )
            .await?;
            let compile_time = Duration::ZERO;
            let exec_time = started.elapsed();

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
                let catalog = self.catalog.clone();
                tokio::task::spawn_blocking(move || catalog.commit_transaction(transaction))
                    .await
                    .map_err(Error::CommitPanic)??;
                Ok(outcome)
            }
            Err(error) => {
                self.catalog.rollback_transaction(transaction);
                Err(error)
            }
        }
    }
}

/// What [`run_query`](PivotQueryHandler::run_query) resolved a statement to.
enum Outcome {
    /// A completed statement response plus where its time went.
    Response(Response, QueryStats),
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
            Outcome::Response(res, stats) => {
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

// ---------------------------------------------------------------------------
// Extended query protocol (prepared statements).
//
// The JDBC drivers Kafka Connect uses drive everything through Parse/Bind/
// Execute. A statement is planned once at Parse (the expensive DuckDB round-trip
// is cached on the `Prepared`); each Execute binds fresh values and runs. Only
// Prepared `VALUES` parameters are decoded straight into Arrow columns and
// handed through the normal plan to the table's insert sink. Scalar parameters
// are bound into expression holes during compile. Statements without parameters
// use the same compile/execute core.
// ---------------------------------------------------------------------------

/// A statement parsed (and planned) once at Parse time, reused across its Bind/
/// Execute cycles, plus the metadata the extended protocol's Describe needs.
#[derive(Clone)]
pub struct Prepared {
    plan: Arc<planner::Plan>,
    /// Ordered parameter types (Postgres OIDs), for the statement's
    /// `ParameterDescription`.
    param_types: Vec<Type>,
    /// The Arrow type to decode each bound parameter value at, or `None` if the
    /// parameter's type is unknown (the plan didn't resolve it and the client
    /// didn't declare it). Decoding such a parameter is an error.
    param_arrow: Vec<Option<DataType>>,
    /// Result columns for `RowDescription`; empty for a statement that produces
    /// no rows (INSERT / DDL / SET).
    result_fields: Vec<FieldInfo>,
}

/// Plans each prepared statement once at Parse time. A parameter's type comes
/// from the plan where DuckDB resolves it (an INSERT target column) and
/// otherwise from the client's declared Parse types. The resulting [`Prepared`]
/// is stored by pgwire and handed back at Bind/Describe/Execute.
pub struct PivotQueryParser {
    catalog: Arc<dyn planner::catalog::Catalog>,
}

#[async_trait]
impl QueryParser for PivotQueryParser {
    type Statement = Prepared;

    async fn parse_sql<C>(
        &self,
        _client: &C,
        sql: &str,
        types: &[Option<Type>],
    ) -> PgWireResult<Prepared>
    where
        C: ClientInfo + Unpin + Send + Sync,
    {
        let catalog = self.catalog.clone();
        let sql = sql.to_string();
        let plan = match tokio::task::spawn_blocking(move || -> Result<Arc<planner::Plan>> {
            // Planning reads a snapshot but writes nothing; roll the transaction
            // back either way (a prepared plan carries no snapshot state, so it
            // recompiles cleanly under each Execute's own transaction).
            let transaction = catalog.begin_transaction();
            let planned = with_planner(&catalog, |planner| planner.plan(&sql, transaction.clone()));
            catalog.rollback_transaction(transaction);
            Ok(Arc::new(planned?))
        })
        .await
        {
            Ok(Ok(plan)) => plan,
            Ok(Err(e)) => return Err(e.into_pgwire()),
            Err(e) => return Err(Error::PlannerPanic(e).into_pgwire()),
        };
        let (param_types, param_arrow) = resolve_param_types(&plan, types);
        let result_fields = result_fields(&plan).map_err(Error::into_pgwire)?;
        Ok(Prepared {
            plan,
            param_types,
            param_arrow,
            result_fields,
        })
    }

    fn get_parameter_types(&self, stmt: &Prepared) -> PgWireResult<Vec<Type>> {
        Ok(stmt.param_types.clone())
    }

    fn get_result_schema(
        &self,
        stmt: &Prepared,
        column_format: Option<&Format>,
    ) -> PgWireResult<Vec<FieldInfo>> {
        crate::arrow_to_pgwire::format_result_fields(
            &stmt.result_fields,
            column_format.unwrap_or(&Format::UnifiedText),
        )
        .map_err(|error| Error::ParamBind(error).into_pgwire())
    }
}

/// Resolve each parameter's type by combining the plan with the client's Parse
/// declarations: the plan types an INSERT-target parameter, and the client's
/// declared type covers a parameter used only in a comparison (which DuckDB
/// leaves untyped). Returns the Postgres OIDs (for `ParameterDescription`) and
/// the Arrow type to decode each bound value at. It is `None` where neither source
/// gives a type, which makes decoding that parameter an error.
fn resolve_param_types(
    plan: &planner::Plan,
    client_types: &[Option<Type>],
) -> (Vec<Type>, Vec<Option<DataType>>) {
    let plan_types = plan.parameter_types();
    let count = plan_types.len().max(client_types.len());
    let mut pg_types = Vec::with_capacity(count);
    let mut arrow_types = Vec::with_capacity(count);
    for i in 0..count {
        let client_type = client_types.get(i).and_then(Option::as_ref);
        let arrow = match client_type {
            Some(client_type) => crate::arrow_to_pgwire::arrow_type_for_pg(client_type),
            None => plan_types
                .get(i)
                .and_then(Option::as_ref)
                .map(planner::types::physical_arrow_type),
        };
        let pg = arrow
            .as_ref()
            .map(pg_param::pg_type_for_param)
            .unwrap_or(Type::UNKNOWN);
        pg_types.push(pg);
        arrow_types.push(arrow);
    }
    (pg_types, arrow_types)
}

/// A statement produces no result rows (so Describe returns no `RowDescription`
/// columns): an INSERT, a `CREATE TABLE`, or a `SET`/`RESET`.
fn produces_no_rows(plan: &planner::Plan) -> bool {
    plan.as_set_variable().is_some()
        || matches!(
            plan.root.operator,
            planner::Operator::Insert(_) | planner::Operator::CreateTable(_)
        )
}

/// The result columns (`RowDescription`) a statement produces, derived from its
/// plan without executing. Empty for a statement that yields no rows.
fn result_fields(plan: &planner::Plan) -> Result<Vec<FieldInfo>> {
    if produces_no_rows(plan) {
        return Ok(Vec::new());
    }
    let types = plan.root.output_types().map_err(Error::Compile)?;
    let arrow: Vec<DataType> = types
        .iter()
        .map(planner::types::physical_arrow_type)
        .collect();
    Ok((*pg_param::result_fields(&plan.output_names, &arrow)).clone())
}

/// Decode a portal's bound parameters into scalar values, one per placeholder,
/// each at the parameter's resolved type. These are the values the plan's
/// `Parameter` holes and prepared `VALUES` bind to at compile time.
fn decode_params<S: Clone>(
    portal: &Portal<S>,
    param_arrow: &[Option<DataType>],
) -> Result<Vec<Scalar<ArrayRef>>> {
    if portal.parameter_len() != param_arrow.len() {
        return Err(Error::ParamBind(format!(
            "expected {} parameters, got {}",
            param_arrow.len(),
            portal.parameter_len()
        )));
    }
    if let Format::Individual(formats) = &portal.parameter_format
        && formats.len() != param_arrow.len()
    {
        return Err(Error::ParamBind(format!(
            "expected {} parameter format codes, got {}",
            param_arrow.len(),
            formats.len()
        )));
    }
    param_arrow
        .iter()
        .enumerate()
        .map(|(index, arrow_type)| {
            let arrow_type = arrow_type.as_ref().ok_or_else(|| {
                Error::ParamBind(format!(
                    "the type of parameter ${} is unknown; declare it or add a cast",
                    index + 1
                ))
            })?;
            pg_param::bind_param(portal, index, arrow_type)
                .map_err(|e| Error::ParamBind(e.to_string()))
        })
        .collect()
}

impl PivotQueryHandler {
    /// Run a prepared statement's portal to completion inside one catalog
    /// transaction: decode its bound parameters, then compile the (cached) plan
    /// with those values and execute. Binding the parameters into the plan is the
    /// whole of it. There is no special INSERT path. A `SET`/`RESET` is handed
    /// back for the caller to apply to the connection.
    async fn run_prepared<S: Clone>(
        &self,
        prepared: &Prepared,
        portal: &Portal<S>,
        collect_stats: bool,
    ) -> Result<Outcome> {
        let transaction = self.catalog.begin_transaction();
        let clears_plan_cache = !plan_is_cacheable(&prepared.plan);
        let result: Result<Outcome> = async {
            // A `SET`/`RESET` over the extended protocol is still a session
            // command; hand it back for the caller to apply to the connection.
            if let Some(set) = prepared.plan.as_set_variable() {
                return Ok(Outcome::Set {
                    name: set.name.clone(),
                    value: set.value.clone(),
                });
            }

            // Honor the client's requested result-column formats; reject a binary
            // request for a column type we can only encode in text.
            crate::arrow_to_pgwire::validate_result_format(
                &prepared.result_fields,
                &portal.result_column_format,
            )
            .map_err(Error::ParamBind)?;
            let result_format = Arc::new(portal.result_column_format.clone());

            let params = decode_params(portal, &prepared.param_arrow)?;
            let is_insert = matches!(prepared.plan.root.operator, planner::Operator::Insert(_));

            let started = Instant::now();
            let plan = prepared.plan.clone();
            let dispatcher = self.dispatcher.clone();
            let compile_transaction = transaction.clone();
            let (outputs, flow) = run_dataflow(
                move || {
                    plan.compile_with_params(&dispatcher, compile_transaction.as_ref(), &params)
                        .map_err(Error::from)
                },
                is_insert,
                collect_stats,
                result_format,
            )
            .await?;
            let exec = started.elapsed();

            let response = build_pgwire_response(outputs, is_insert)?;
            Ok(Outcome::Response(
                response,
                QueryStats {
                    plan: Duration::ZERO,
                    compile: Duration::ZERO,
                    exec,
                    flow,
                },
            ))
        }
        .await;
        match result {
            Ok(outcome) => {
                let catalog = self.catalog.clone();
                tokio::task::spawn_blocking(move || catalog.commit_transaction(transaction))
                    .await
                    .map_err(Error::CommitPanic)??;
                if clears_plan_cache {
                    self.plan_cache.lock().unwrap().clear();
                }
                Ok(outcome)
            }
            Err(error) => {
                self.catalog.rollback_transaction(transaction);
                Err(error)
            }
        }
    }
}

#[async_trait]
impl ExtendedQueryHandler for PivotQueryHandler {
    type Statement = Prepared;
    type QueryParser = PivotQueryParser;

    fn query_parser(&self) -> Arc<Self::QueryParser> {
        self.query_parser.clone()
    }

    async fn do_query<C>(
        &self,
        client: &mut C,
        portal: &Portal<Self::Statement>,
        _max_rows: usize,
    ) -> PgWireResult<Response>
    where
        C: ClientInfo + ClientPortalStore + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::PortalStore: PortalStore<Statement = Self::Statement>,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        let with_stats = stats_on(client);
        let prepared = &portal.statement.statement;
        let outcome = self
            .run_prepared(prepared, portal, with_stats)
            .await
            .map_err(|e| {
                warn!(error = %e, "prepared query failed");
                e.into_pgwire()
            })?;

        let res = match outcome {
            Outcome::Set { name, value } => apply_set(client, &name, value.as_deref()),
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
        };
        Ok(res)
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
        self.query_handler.clone()
    }

    fn copy_handler(&self) -> Arc<impl pgwire::api::copy::CopyHandler> {
        Arc::new(NoopHandler)
    }
}
