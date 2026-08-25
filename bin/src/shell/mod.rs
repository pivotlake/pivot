//! Embedded single-datastore support and the interactive `pivot open` shell.

mod instance;
mod parser;
mod render;
mod repl;

pub use instance::ShellInstance;

/// Open an interactive SQL shell over one datastore, named by a local directory
/// or an object-store URI.
pub fn run(datastore_location: String) -> Result<(), Box<dyn std::error::Error>> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(repl::run_shell(datastore_location))
}
