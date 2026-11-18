//! The pivot web page's backend.
//!
//! The page cannot speak the PostgreSQL wire protocol, so this does it for
//! it: each request reads one of the pivot server's system tables and answers
//! with its rows as JSON.
//!
//! - `/api/connections`: `system.connections`, the server's clients.
//! - `/api/memory`: `system.memory_blocks`, grouped by NUMA node and state.
//! - `/api/tables`: `system.tables`, with each table's file count.
//! - `/api/tables/{id}/files`: `system.table_files` for one table, with the
//!   compaction sizes this backend was told the server runs with.
//! - `/api/columns`: `system.columns`, each column with its table's name,
//!   which the page's query editor completes names from.
//! - `POST /api/query`: runs the statement in the request's body and answers
//!   with its result, for the page's query editor.
//!
//! A read that fails is answered with the reason rather than with stale rows,
//! so the page never draws what the server no longer has.
//!
//! Told where the built page is (`--static-dir`), it serves that too, so one
//! process is the whole of the web page: what the container image runs.
//! Every setting can also come from the environment, which is how a
//! container is given them.

mod pivot;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use pivot::{
    ColumnsReading, CompactionSettings, ConnectionsReading, MemoryReading, PivotServer,
    StatementResult, TableFilesReading, TablesReading,
};
use serde::{Deserialize, Serialize};
use tower_http::services::ServeDir;
use tracing::{info, warn};

#[derive(Debug, Parser)]
#[command(
    about = "Serves a pivot server's connections, memory use and tables to the pivot web page"
)]
struct Options {
    /// Host of the pivot server to describe.
    #[arg(long, env = "PIVOT_HOST", default_value = "127.0.0.1")]
    pivot_host: String,
    /// Port of the pivot server to describe.
    #[arg(long, env = "PIVOT_PORT", default_value_t = 5432)]
    pivot_port: u16,
    /// User to log in to the pivot server as.
    #[arg(long, env = "PIVOT_USER", default_value = "pivot")]
    user: String,
    /// That user's password, for a user that has one.
    #[arg(long, env = "PIVOT_PASSWORD")]
    password: Option<String>,
    /// Address to serve the page's requests on.
    #[arg(long, env = "PIVOT_WEB_LISTEN", default_value = "127.0.0.1:5175")]
    listen: SocketAddr,
    /// Directory holding the built page (`npm run build` writes it to
    /// `web/dist`), to serve beside the requests. Without it only the
    /// requests are served, and the page comes from Vite.
    #[arg(long, env = "PIVOT_WEB_STATIC_DIR")]
    static_dir: Option<PathBuf>,
    /// The pivot server's `compact_bytes`, in bytes: the size it cuts merged
    /// files at. The page tells which files a compaction would merge by it.
    #[arg(long, env = "PIVOT_COMPACT_BYTES", default_value_t = 1024 * 1024 * 1024)]
    compact_bytes: u64,
    /// The pivot server's `compact_min_files`.
    #[arg(long, env = "PIVOT_COMPACT_MIN_FILES", default_value_t = 10)]
    compact_min_files: u64,
}

/// The pivot server's default `compact_merge_bytes`, as a multiple of its
/// `compact_bytes`.
const MERGE_TARGET_MULTIPLIER: f64 = 1.3;

/// How many mutually overlapping files one of the pivot server's layout
/// merges rewrites.
const LAYOUT_CLIQUE_SIZE: u64 = 6;

/// What a failed read is answered with.
#[derive(Debug, Serialize)]
struct Failure {
    error: String,
}

impl Failure {
    /// The answer to a request whose read of the pivot server failed.
    fn describe(pivot: &PivotServer, error: pivot::Error) -> (StatusCode, Json<Self>) {
        warn!(%error, pivot = %pivot.address(), "could not read the pivot server");
        let failure = Self {
            error: format!("pivot server at {}: {error}", pivot.address()),
        };
        (StatusCode::BAD_GATEWAY, Json(failure))
    }
}

async fn read_connections(
    State(pivot): State<Arc<PivotServer>>,
) -> Result<Json<ConnectionsReading>, (StatusCode, Json<Failure>)> {
    pivot
        .read_connections()
        .await
        .map(Json)
        .map_err(|error| Failure::describe(&pivot, error))
}

async fn read_memory(
    State(pivot): State<Arc<PivotServer>>,
) -> Result<Json<MemoryReading>, (StatusCode, Json<Failure>)> {
    pivot
        .read_memory()
        .await
        .map(Json)
        .map_err(|error| Failure::describe(&pivot, error))
}

async fn read_tables(
    State(pivot): State<Arc<PivotServer>>,
) -> Result<Json<TablesReading>, (StatusCode, Json<Failure>)> {
    pivot
        .read_tables()
        .await
        .map(Json)
        .map_err(|error| Failure::describe(&pivot, error))
}

async fn read_table_files(
    State(pivot): State<Arc<PivotServer>>,
    Path(table_id): Path<String>,
) -> Result<Json<TableFilesReading>, (StatusCode, Json<Failure>)> {
    pivot
        .read_table_files(&table_id)
        .await
        .map(Json)
        .map_err(|error| Failure::describe(&pivot, error))
}

async fn read_columns(
    State(pivot): State<Arc<PivotServer>>,
) -> Result<Json<ColumnsReading>, (StatusCode, Json<Failure>)> {
    pivot
        .read_columns()
        .await
        .map(Json)
        .map_err(|error| Failure::describe(&pivot, error))
}

/// A statement the page asks to have run.
#[derive(Debug, Deserialize)]
struct StatementRequest {
    sql: String,
}

/// A statement the pivot server refuses is answered with what the server
/// said of it, as the user's to fix; any other failure is the read's.
async fn run_statement(
    State(pivot): State<Arc<PivotServer>>,
    Json(request): Json<StatementRequest>,
) -> Result<Json<StatementResult>, (StatusCode, Json<Failure>)> {
    pivot
        .run_statement(&request.sql)
        .await
        .map(Json)
        .map_err(|error| match error.read_statement_error() {
            Some(refusal) => {
                let failure = Failure {
                    error: refusal.to_string(),
                };
                (StatusCode::UNPROCESSABLE_ENTITY, Json(failure))
            }
            None => Failure::describe(&pivot, error),
        })
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt::init();
    let options = Options::parse();

    let pivot = Arc::new(PivotServer::new(
        options.pivot_host,
        options.pivot_port,
        options.user,
        options.password,
        CompactionSettings {
            target_bytes: options.compact_bytes,
            merge_target_bytes: (options.compact_bytes as f64 * MERGE_TARGET_MULTIPLIER) as u64,
            min_files: options.compact_min_files,
            layout_clique_size: LAYOUT_CLIQUE_SIZE,
        },
    ));
    let router = Router::new()
        .route("/api/connections", get(read_connections))
        .route("/api/memory", get(read_memory))
        .route("/api/tables", get(read_tables))
        .route("/api/tables/:table_id/files", get(read_table_files))
        .route("/api/columns", get(read_columns))
        .route("/api/query", post(run_statement))
        .with_state(pivot.clone());
    // Whatever is not one of the requests is a file of the page.
    let router = match &options.static_dir {
        Some(static_dir) => router.fallback_service(ServeDir::new(static_dir)),
        None => router,
    };

    let listener = tokio::net::TcpListener::bind(options.listen).await?;
    info!(
        listen = %options.listen,
        pivot = %pivot.address(),
        page = ?options.static_dir,
        "serving the pivot web page"
    );
    axum::serve(listener, router).await
}
