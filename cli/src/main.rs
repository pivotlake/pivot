//! Pivot command-line entry point.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "pivot", about = "Pivot command-line tools", version)]
struct Args {
    #[command(subcommand)]
    command: PivotCommand,
}

#[derive(Subcommand, Debug)]
enum PivotCommand {
    /// Open a local datastore in the Pivot SQL shell.
    Open {
        /// Directory containing the datastore. It is created if it does not exist.
        #[arg(value_name = "DATASTORE_DIRECTORY")]
        datastore_directory: PathBuf,
    },
    /// Run the Pivot database server in the foreground.
    Server(server::ServerOptions),
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    match Args::parse().command {
        PivotCommand::Open {
            datastore_directory,
        } => shell::run(datastore_directory),
        PivotCommand::Server(options) => server::run(options).map_err(Into::into),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("pivot: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use clap::Parser;

    use super::{Args, PivotCommand};

    #[test]
    fn requires_a_command_and_an_open_datastore_directory() {
        assert!(Args::try_parse_from(["pivot"]).is_err());
        assert!(Args::try_parse_from(["pivot", "open"]).is_err());
        assert!(Args::try_parse_from(["pivot", "shell", "/var/lib/pivot"]).is_err());

        let args = Args::try_parse_from(["pivot", "open", "/var/lib/pivot"]).unwrap();
        let PivotCommand::Open {
            datastore_directory,
        } = args.command
        else {
            panic!("open did not parse as the open command");
        };
        assert_eq!(datastore_directory, PathBuf::from("/var/lib/pivot"));
    }

    #[test]
    fn server_requires_a_config_file() {
        assert!(Args::try_parse_from(["pivot", "server"]).is_err());
        assert!(Args::try_parse_from(["pivot", "server", "install"]).is_err());

        let args = Args::try_parse_from(["pivot", "server", "--config", "/etc/pivot/config.yaml"])
            .unwrap();
        assert!(matches!(args.command, PivotCommand::Server(_)));
    }
}
