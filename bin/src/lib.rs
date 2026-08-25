//! Everything the `pivot` executable is built from.
//!
//! - [`execution`]: statement transactions, planning, the plan cache, dispatch,
//!   cancellation, and command classification, independent of any transport.
//! - [`server`]: the PostgreSQL-wire and HTTP adapters around the executor, and
//!   the configured foreground process behind `pivot server`.
//! - [`shell`]: the embedded single-datastore instance and the interactive
//!   `pivot open` shell, which renders result batches straight to the terminal.
//!
//! The library exists so the integration tests can drive a [`server::Server`]
//! or a [`shell::ShellInstance`] in-process; `src/main.rs` is the only binary.

pub mod execution;
pub mod server;
pub mod shell;
