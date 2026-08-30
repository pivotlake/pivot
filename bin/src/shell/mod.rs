//! Embedded single-datastore support and the interactive `pivot open` shell.

mod instance;
mod parser;
mod render;
mod repl;

pub use instance::{ShellInstance, ShellLimits};

/// Open an interactive SQL shell over one datastore, named by a local directory
/// or an object-store URI.
pub fn run(datastore_location: String) -> Result<(), Box<dyn std::error::Error>> {
    run_with_limits(datastore_location, ShellLimits::default())
}

/// Open an interactive SQL shell with optional resource limits.
pub fn run_with_limits(
    datastore_location: String,
    limits: ShellLimits,
) -> Result<(), Box<dyn std::error::Error>> {
    crate::server::raise_open_file_limit();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(repl::run_shell(datastore_location, limits))
}
