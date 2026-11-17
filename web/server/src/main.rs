//! The pivot web page's backend.
//!
//! The page cannot speak the PostgreSQL wire protocol, so this does it for
//! it: each request reads one of the pivot server's system tables and answers
//! with its rows as JSON.
//!
//! - `/api/connections`: `system.connections`, the server's clients.
//! - `/api/memory`: `system.memory_blocks`, grouped by NUMA node and state.
//!
//! A read that fails is answered with the reason rather than with stale rows,
//! so the page never draws what the server no longer has.

mod pivot;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::get;
use axum::{Json, Router};
use clap::Parser;
use pivot::{ConnectionsReading, MemoryReading, PivotServer};
use serde::Serialize;
use tracing::{info, warn};

#[derive(Debug, Parser)]
#[command(about = "Serves a pivot server's connections and memory use to the pivot web page")]
struct Options {
    /// Host of the pivot server to describe.
    #[arg(long, default_value = "127.0.0.1")]
    pivot_host: String,
    /// Port of the pivot server to describe.
    #[arg(long, default_value_t = 5432)]
    pivot_port: u16,
    /// User to log in to the pivot server as.
    #[arg(long, default_value = "pivot")]
    user: String,
    /// That user's password, for a user that has one.
    #[arg(long, env = "PIVOT_PASSWORD")]
    password: Option<String>,
    /// Address to serve the page's requests on.
    #[arg(long, default_value = "127.0.0.1:5175")]
    listen: SocketAddr,
}

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

#[tokio::main]
async fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt::init();
    let options = Options::parse();

    let pivot = Arc::new(PivotServer::new(
        options.pivot_host,
        options.pivot_port,
        options.user,
        options.password,
    ));
    let router = Router::new()
        .route("/api/connections", get(read_connections))
        .route("/api/memory", get(read_memory))
        .with_state(pivot.clone());

    let listener = tokio::net::TcpListener::bind(options.listen).await?;
    info!(listen = %options.listen, pivot = %pivot.address(), "serving the pivot web page's requests");
    axum::serve(listener, router).await
}
