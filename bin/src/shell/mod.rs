//! Embedded single-datastore support and the interactive `pivot open` shell.

mod instance;
mod parser;
mod render;
mod repl;

pub use instance::{DiskCacheOptions, OpenOptions, ShellInstance};

/// Open an interactive SQL shell over one datastore, named by a local directory
/// or an object-store URI, with the given resource budgets.
pub fn run(
    datastore_location: String,
    options: OpenOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(repl::run_shell(datastore_location, options))
}
