//! The bundled web dashboard, served in-process (enabled with `--http-bind`).
//!
//! Because it runs inside the server, it reads the engine's live state
//! directly - the catalog and the process's own CPU/memory - neither of which
//! the Postgres wire exposes. It also
//! runs the query console's SQL on the same planner + dispatch pool as every
//! other query (no second hop), and serves the built React frontend (embedded
//! in the binary).
//!
//! Read-only except `/api/query`; meant to sit on a trusted network, not face
//! the public internet. Endpoints:
//!
//! - `GET  /api/health`   → `{ "status": "ok" }`
//! - `GET  /api/overview` → catalog + system
//! - `POST /api/query`    → run SQL in-process, return JSON rows
//! - everything else      → the frontend (SPA fallback to `index.html`)
//!
//! Each endpoint group lives in its own submodule; this module owns the shared
//! [`IntrospectState`], the router that wires the handlers together, and the
//! [`serve`] entry point the server calls.

mod frontend;
mod json;
mod overview;
mod query;
mod system;
mod tables;

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::Json;
use axum::Router;
use axum::routing::{get, post};
use catalog::PivotCatalog;
use sysinfo::{Pid, System};
use tokio::net::TcpListener;

use frontend::ui;
use overview::overview;
use query::query;
use tables::{default_files_page, files_page, rowgroups_page};

/// Shared state for the dashboard handlers. The handlers live in the sibling
/// submodules and reach the engine through these fields.
#[derive(Clone)]
pub(crate) struct IntrospectState {
    pub(super) catalog: Arc<PivotCatalog>,
    pub(super) dispatcher: dispatch::DataFlowDispatcher,
    /// One persistent `System` so CPU usage is measured across polls.
    pub(super) system: Arc<Mutex<System>>,
    pub(super) pid: Option<Pid>,
}

impl IntrospectState {
    pub(crate) fn new(
        catalog: Arc<PivotCatalog>,
        dispatcher: dispatch::DataFlowDispatcher,
    ) -> Self {
        Self {
            catalog,
            dispatcher,
            system: Arc::new(Mutex::new(System::new())),
            pid: sysinfo::get_current_pid().ok(),
        }
    }

    fn router(self) -> Router {
        Router::new()
            .route("/api/health", get(health))
            .route("/api/overview", get(overview))
            .route("/api/query", post(query))
            .route("/api/tables/{name}/files", get(default_files_page))
            .route(
                "/api/datastores/{datastore}/tables/{name}/files",
                get(files_page),
            )
            .route("/api/tables/{name}/rowgroups", get(rowgroups_page))
            .fallback(ui)
            .with_state(self)
    }
}

/// Serve the dashboard on `bind` until `shutdown` resolves.
pub(crate) async fn serve(
    bind: SocketAddr,
    state: IntrospectState,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    let listener = TcpListener::bind(bind).await?;
    tracing::info!(addr = %bind, "serving web dashboard");
    axum::serve(listener, state.router())
        .with_graceful_shutdown(shutdown)
        .await
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}
