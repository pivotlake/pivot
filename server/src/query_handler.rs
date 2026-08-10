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

use std::fmt::Debug;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::arrow_to_pgwire::PGRowBatch;
use crate::auth::Authenticator;
use arrow_array::RecordBatch;
use async_trait::async_trait;
use dispatch::{DataFlowHandle, DataFlowStats};
use futures::{Sink, SinkExt, stream};
use metastore::Metastore;
use pgwire::api::auth::StartupHandler;
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
use session::{CancelOnDrop, Error, PlanCache, Session, StatementOutcome};
use session::{execute_compact, parse_affected_rows, plan_query};
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

/// Run `sql` to completion in-process and return its result batches - the path
/// the HTTP dashboard uses instead of the Postgres wire. Statements without
/// result rows (SET, COMPACT, INSERT) yield no batches.
pub(crate) async fn execute_sql(
    catalog: Arc<catalog::PivotCatalog>,
    dispatcher: dispatch::DataFlowDispatcher,
    plan_cache: Arc<PlanCache>,
    sql: String,
) -> Result<Vec<arrow_array::RecordBatch>, String> {
    let session = Session::with_plan_cache(catalog, dispatcher, plan_cache);
    Ok(match session.run(&sql).await? {
        StatementOutcome::Rows { batches, .. } => batches,
        StatementOutcome::Affected(_) | StatementOutcome::Command(_) => Vec::new(),
    })
}

/// Convert an execution error into a `PgWireError::UserError` so it serialises
/// as a normal error response on the wire (severity `ERROR`, populated
/// SQLSTATE).
fn into_pgwire(error: Error) -> PgWireError {
    const INTERNAL_ERROR: &str = "XX000";

    let info = ErrorInfo::new(
        "ERROR".to_string(),
        INTERNAL_ERROR.to_string(),
        error.to_string(),
    );
    PgWireError::UserError(Box::new(info))
}

type Result<T, E = Error> = std::result::Result<T, E>;

/// Output converted before it leaves the dispatch worker. Query
/// batches become pgwire rows; INSERT's internal one-row result becomes an
/// owned count that the coordinator turns into an `INSERT 0 n` command tag.
enum WorkerOutput {
    QueryRows(PGRowBatch),
    AffectedRows(std::result::Result<usize, String>),
}

fn build_worker_output(batch: RecordBatch, is_insert: bool) -> WorkerOutput {
    if !is_insert {
        return WorkerOutput::QueryRows(PGRowBatch::from(batch));
    }
    WorkerOutput::AffectedRows(parse_affected_rows(&batch))
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

/// pgwire `SimpleQueryHandler`: plans, compiles, and runs each query on the
/// blocking pool, surfacing errors and supporting cancellation.
pub struct PivotQueryHandler {
    catalog: Arc<catalog::PivotCatalog>,
    dispatcher: dispatch::DataFlowDispatcher,
    plan_cache: Arc<PlanCache>,
}

impl PivotQueryHandler {
    pub fn new(
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

            let started = Instant::now();
            let plan = plan_query(
                &self.catalog,
                transaction.clone(),
                self.plan_cache.as_ref(),
                &query,
            )
            .await?;
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
                    .map(move || move |batch| build_worker_output(batch, is_insert));
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
enum Outcome {
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
                into_pgwire(e)
            })?;

        let res = match outcome {
            Outcome::Set { name, value } => apply_set(client, &name, value.as_deref()),
            Outcome::Compact => Response::Execution(Tag::new("COMPACT")),
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
        Arc::new(NoopHandler)
    }

    fn copy_handler(&self) -> Arc<impl pgwire::api::copy::CopyHandler> {
        Arc::new(NoopHandler)
    }
}
