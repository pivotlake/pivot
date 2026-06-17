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

use crate::arrow_to_pgwire::PGRowBatch;
use async_trait::async_trait;
use dispatch::{CancelToken, DataFlowHandle};
use futures::{Sink, stream};
use pgwire::api::auth::StartupHandler;
use pgwire::api::auth::noop::NoopStartupHandler;
use pgwire::api::cancel::{CancelHandler, DefaultCancelHandler};
use pgwire::api::query::SimpleQueryHandler;
use pgwire::api::results::{QueryResponse, Response};
use pgwire::api::store::PortalStore;
use pgwire::api::{
    ClientInfo, ClientPortalStore, ConnectionManager, NoopHandler, PgWireServerHandlers,
};
use pgwire::error::PgWireResult;
use pgwire::error::{ErrorInfo, PgWireError};
use pgwire::messages::PgWireBackendMessage;
use thiserror::Error;
use tokio::task::JoinError;
use tracing::{info, warn};

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

    async fn run_query(&self, query: &str) -> Result<Response> {
        let dispatcher = self.dispatcher.clone();
        let query = query.to_string();

        // Reuse a cached plan if we've planned this exact SQL before. Only
        // read-only SELECT plans are ever inserted, so a cache hit is always a
        // SELECT regardless of what `query` is. Planning is a fixed few-ms cost;
        // skipping it on repeated SELECTs shaves that off every query after the
        // first.
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

        // Compile the (possibly cached) plan into a fresh dataflow and launch
        // it. `compile` is pure pivot work (no DuckDB), so it runs on any
        // blocking thread without the planner thread-local.
        let handle = tokio::task::spawn_blocking(move || -> Result<DataFlowHandle<_>> {
            let spec = plan.compile(&dispatcher)?;
            Ok(spec.map(|| |b| PGRowBatch::from(b)).execute())
        })
        .await
        .map_err(Error::PlannerPanic)??;

        // Cancel the dataflow if our future is dropped before drain finishes —
        // covers both psql Ctrl-C (pgwire's `_on_query` select drops us) and
        // raw disconnects (whole connection task dropped).
        let guard = CancelOnDrop::new(handle.cancel_token());
        let batches: Vec<_> = tokio::task::spawn_blocking(move || handle.collect())
            .await
            .map_err(Error::WorkerPanic)??;
        guard.defuse();

        let fields = batches
            .first()
            .map_or(Arc::new(vec![]), |b| b.fields.clone());
        Ok(Response::Query(QueryResponse::new(
            fields,
            stream::iter(batches.into_iter().flat_map(|b| b.rows).map(Ok)),
        )))
    }
}

#[async_trait]
impl SimpleQueryHandler for PivotQueryHandler {
    async fn do_query<C>(&self, _client: &mut C, query: &str) -> PgWireResult<Vec<Response>>
    where
        C: ClientInfo + ClientPortalStore + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::PortalStore: PortalStore,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        info!(sql = %query, "query received");
        let res = self.run_query(query).await.map_err(|e| {
            warn!(error = %e, sql = %query, "query failed");
            e.into_pgwire()
        })?;
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
