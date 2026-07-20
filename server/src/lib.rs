//! `server` exposes pivotdb over the PostgreSQL v3 wire protocol so any
//! Postgres client (`psql`, `tokio-postgres`, JDBC, etc.) can connect and
//! issue queries.
//!
//! The crate is a thin glue layer: [`pgwire`] drives the wire protocol,
//! [`planner`] turns each SQL string into an executable plan against a
//! caller-supplied [`Catalog`](planner::catalog::Catalog), and [`dispatch`]
//! runs the resulting dataflow on its thread-per-core worker pool. Each query
//! hops to `tokio::task::spawn_blocking` to drive the (non-`Send`) DuckDB
//! planner; the planner is cached in a thread-local on each blocking-pool
//! thread and reused across queries. When run as a binary, the default is to run with the default
//! `catalog::ParquetCatalog`.
//!
//! The public interface: hand a bind address to [`Server::new`] together
//! with a [`Dispatch`](dispatch::Dispatch) (from
//! [`Dispatch::spin_up`](dispatch::Dispatch::spin_up)), your catalog, and the
//! compaction settings; then call [`Server::serve`] with a shutdown future. The
//! returned future runs the accept loop until shutdown is signalled or a worker
//! dies. The bundled compacter (from the `compact` crate) runs alongside the
//! query path and re-encodes its Parquet on the same dispatch workers; it is
//! stopped before the workers stop.
//!
//! # Example
//!
//! ```no_run
//! use std::net::SocketAddr;
//! use std::sync::Arc;
//!
//! use catalog::ParquetCatalog;
//! use dispatch::Dispatch;
//! use server::Server;
//!
//! # async fn run() -> Result<(), server::Error> {
//! let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
//! let dispatch = Dispatch::spin_up(workers, 32, None);
//! let catalog = Arc::new(ParquetCatalog::new(dispatch.dispatcher().clone()));
//! let bind: SocketAddr = "127.0.0.1:5433".parse().unwrap();
//!
//! let server = Server::new(bind, dispatch, catalog, 0, 4, server::DEFAULT_CATALOG_REFRESH);
//! // Returns when ctrl_c fires, or earlier if a dispatch worker dies.
//! server.serve(Box::pin(async {
//!     let _ = tokio::signal::ctrl_c().await;
//! })).await
//! # }
//! ```

mod arrow_to_pgwire;
mod http;
#[cfg(feature = "perf")]
mod perf;
mod query_handler;
mod server;

pub use server::{DEFAULT_CATALOG_REFRESH, Error, Server};
