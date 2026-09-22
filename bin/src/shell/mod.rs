//! Embedded single-datastore support and the interactive `pivot open` shell.

mod instance;
mod parser;
mod progress;
mod render;
mod repl;
mod target;

pub use instance::{ShellInstance, ShellLimits};
pub use target::{
    DatastoreKind, ICEBERG_CREDENTIAL_VARIABLE, ICEBERG_OAUTH2_SERVER_URI_VARIABLE,
    ICEBERG_SCOPE_VARIABLE, ICEBERG_TOKEN_VARIABLE, OpenOptions, ShellTarget,
};

/// Open an interactive SQL shell over one target: a Pivot datastore named by a
/// local directory or an object-store URI, or an Iceberg REST catalog.
pub fn run(target: ShellTarget) -> Result<(), Box<dyn std::error::Error>> {
    run_with_limits(target, ShellLimits::default())
}

/// Open an interactive SQL shell with optional resource limits.
pub fn run_with_limits(
    target: ShellTarget,
    limits: ShellLimits,
) -> Result<(), Box<dyn std::error::Error>> {
    crate::server::raise_open_file_limit();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(repl::run_shell(target, limits))
}
