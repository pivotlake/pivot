//! The bundled web dashboard, served in-process (enabled with `--http-bind`).
//!
//! Because it runs inside the server, it reads the engine's live state
//! directly - the catalog, the compacter's counters, the
//! process's own CPU/memory - none of which the Postgres wire exposes. It also
//! runs the query console's SQL on the same planner + dispatch pool as every
//! other query (no second hop), and serves the built React frontend (embedded
//! in the binary).
//!
//! Read-only except `/api/query`; meant to sit on a trusted network, not face
//! the public internet. Endpoints:
//!
//! - `GET  /api/health`   → `{ "status": "ok" }`
//! - `GET  /api/overview` → catalog + live compaction stats + system
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

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::Json;
use axum::Router;
use axum::routing::{get, post};
use catalog::ParquetCatalog;
use compact::Compacter;
use sysinfo::{Pid, System};
use tokio::net::TcpListener;

use frontend::ui;
use overview::overview;
use query::query;
use tables::{files_page, rowgroups_page};

/// Shared state for the dashboard handlers. The handlers live in the sibling
/// submodules and reach the engine through these fields.
#[derive(Clone)]
pub(crate) struct IntrospectState {
    pub(super) catalog: Arc<ParquetCatalog>,
    pub(super) catalog_dyn: Arc<dyn planner::catalog::Catalog>,
    pub(super) dispatcher: dispatch::DataFlowDispatcher,
    /// The bundled compacter's counters, when one runs in this process.
    pub(super) compacter: Option<Arc<Compacter>>,
    /// One persistent `System` so CPU usage is measured across polls.
    pub(super) system: Arc<Mutex<System>>,
    pub(super) pid: Option<Pid>,
    /// Last `(cumulative rows, sampled_at)` per table, for insert-rate deltas.
    samples: Arc<Mutex<HashMap<String, (u64, Instant)>>>,
}

impl IntrospectState {
    pub(crate) fn new(
        catalog: Arc<ParquetCatalog>,
        dispatcher: dispatch::DataFlowDispatcher,
        compacter: Option<Arc<Compacter>>,
    ) -> Self {
        Self {
            catalog_dyn: catalog.clone(),
            catalog,
            dispatcher,
            compacter,
            system: Arc::new(Mutex::new(System::new())),
            pid: sysinfo::get_current_pid().ok(),
            samples: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn router(self) -> Router {
        Router::new()
            .route("/api/health", get(health))
            .route("/api/overview", get(overview))
            .route("/api/query", post(query))
            .route("/api/tables/{name}/files", get(files_page))
            .route("/api/tables/{name}/rowgroups", get(rowgroups_page))
            .fallback(ui)
            .with_state(self)
    }

    /// Record a fresh cumulative row count for `table` and return rows/sec since
    /// the previous sample (`None` on the first sample).
    pub(super) fn record_rate(&self, table: &str, rows: u64) -> Option<f64> {
        let now = Instant::now();
        let mut samples = self.samples.lock().unwrap();
        let rate = samples.get(table).and_then(|(prev_rows, prev_at)| {
            let secs = now.duration_since(*prev_at).as_secs_f64();
            (secs > 0.0 && rows >= *prev_rows).then(|| (rows - *prev_rows) as f64 / secs)
        });
        samples.insert(table.to_string(), (rows, now));
        rate
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
