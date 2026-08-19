//! Internal server executable used by the benchmark harness.
//!
//! The public installation exposes the same runtime as `pivot server`. Keeping
//! this thin target in the `server` package lets performance tests profile the
//! server directly without routing process startup through the CLI package.

use std::process::ExitCode;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "pivotdb-server",
    about = "Run the Pivot database server in the foreground",
    version
)]
struct Args {
    #[command(flatten)]
    options: server::ServerOptions,
}

fn main() -> ExitCode {
    match server::run(Args::parse().options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("pivotdb-server: {error}");
            ExitCode::FAILURE
        }
    }
}
