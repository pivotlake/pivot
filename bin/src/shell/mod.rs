//! Embedded single-datastore support and the interactive `pivot open` shell.

mod copy;
mod instance;
mod parser;
mod progress;
mod render;
mod repl;
mod target;

pub use instance::{ShellInstance, ShellLimits};
pub use target::ShellTarget;

/// Open an interactive SQL shell over one Pivot datastore, named by a local
/// directory or an object-store URI.
pub fn run(datastore_location: String) -> Result<(), Box<dyn std::error::Error>> {
    run_with_limits(
        ShellTarget::pivot(datastore_location),
        ShellLimits::default(),
    )
}

/// Open an interactive SQL shell over `target` with optional resource limits.
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
