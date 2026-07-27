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
//! `datastore_delta::DeltaDatastore`.
//!
//! The public interface: hand a bind address to [`Server::new`] together
//! with a [`Dispatch`](dispatch::Dispatch) (from
//! [`Dispatch::spin_up`](dispatch::Dispatch::spin_up)) and your catalog; then
//! call [`Server::serve`] with a shutdown future. The returned future runs the
//! accept loop until shutdown is signalled or a worker dies. Each datastore
//! self-manages its own refresh and (optional) compaction, re-encoding Parquet
//! on the same dispatch workers; those tasks watch the shared exit flag the
//! server flips on shutdown and stop before the workers do.
//!
//! # Example
//!
//! ```no_run
//! use std::collections::HashMap;
//! use std::net::SocketAddr;
//! use std::sync::Arc;
//!
//! use catalog::{DEFAULT_DATASTORE_NAME, Datastore, PivotCatalog};
//! use datastore_delta::DeltaDatastore;
//! use dispatch::Dispatch;
//! use server::Server;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
//! let dispatch = Dispatch::spin_up(workers, 32, None);
//! let datastore: Arc<dyn Datastore> =
//!     DeltaDatastore::open_local("/var/lib/pivot/default", dispatch.dispatcher())?;
//! let catalog = Arc::new(PivotCatalog::new(
//!     HashMap::from([(DEFAULT_DATASTORE_NAME.to_string(), datastore)]),
//!     DEFAULT_DATASTORE_NAME.to_string(),
//! )?);
//! let bind: SocketAddr = "127.0.0.1:5433".parse().unwrap();
//!
//! let server = Server::new(bind, dispatch, catalog);
//! // Returns when ctrl_c fires, or earlier if a dispatch worker dies.
//! server.serve(Box::pin(async {
//!     let _ = tokio::signal::ctrl_c().await;
//! })).await?;
//! # Ok(())
//! # }
//! ```

mod arrow_to_pgwire;
mod http;
#[cfg(feature = "perf")]
mod perf;
mod query_handler;
mod server;

pub use server::{Error, Server};
