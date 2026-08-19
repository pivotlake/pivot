//! Embedded single-datastore support and the interactive `pivot open` shell.

use std::path::PathBuf;

mod instance;
mod parser;
mod render;
mod repl;

pub use instance::ShellInstance;

/// Open an interactive SQL shell over one local datastore.
pub fn run(datastore_directory: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(repl::run_shell(datastore_directory))
}
