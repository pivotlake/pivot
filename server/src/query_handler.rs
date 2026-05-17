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
use std::sync::Arc;

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

thread_local! {
    /// One [`planner::Planner`] (and its non-`Send` DuckDB context) per
    /// tokio blocking-pool thread, lazily initialised on first use. The
    /// catalog is process-global so it's safe to share the same planner
    /// across connections that land on the same thread.
    static PLANNER: RefCell<Option<planner::Planner>> = const { RefCell::new(None) };
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
}

impl PivotQueryHandler {
    pub fn new(
        catalog: Arc<dyn planner::catalog::Catalog>,
        dispatcher: dispatch::DataFlowDispatcher,
    ) -> Self {
        Self {
            catalog,
            dispatcher,
        }
    }

    async fn run_query(&self, query: &str) -> Result<Response> {
        let catalog = self.catalog.clone();
        let dispatcher = self.dispatcher.clone();
        let query = query.to_string();
        let handle = tokio::task::spawn_blocking(move || -> Result<DataFlowHandle<_>> {
            with_planner(&catalog, |planner| {
                let plan = planner.plan(&query)?;
                let spec = plan.compile(&dispatcher)?;
                Ok(spec.map(|| |b| PGRowBatch::from(b)).execute())
            })
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

        let fields = batches.first().map_or(Arc::new(vec![]), |b| b.fields.clone());
        Ok(Response::Query(QueryResponse::new(fields, stream::iter(batches.into_iter().map(|b| b.rows).flatten().map(Ok)))))
    }
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
        let res = self.run_query(query).await.map_err(|e| e.into_pgwire())?;
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
/// The default cancel handler is used as it
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
