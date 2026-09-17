//! Pivot command-line entry point.

use std::num::NonZeroUsize;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use metastore_disk::ByteSize;

#[derive(Parser, Debug)]
#[command(name = "pivot", about = "Pivot command-line tools", version)]
struct Args {
    #[command(subcommand)]
    command: PivotCommand,
}

#[derive(Subcommand, Debug)]
enum PivotCommand {
    /// Open a datastore in the Pivot SQL shell.
    ///
    /// The datastore is named either by a local directory, which is created if
    /// it does not exist, or by an object-store URI: s3://bucket/prefix (s3://
    /// also spelled s3a://), gs://bucket/prefix, or file:///path.
    ///
    /// Object-store credentials are read from the environment. S3 takes
    /// AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY when both are set, or uses
    /// anonymous access when both are absent. Its region comes from AWS_REGION
    /// or AWS_DEFAULT_REGION when set, and is discovered with HeadBucket
    /// otherwise. An optional AWS_ENDPOINT_URL names a path-style S3-compatible
    /// endpoint such as MinIO. GCS follows Application Default Credentials:
    /// GOOGLE_APPLICATION_CREDENTIALS, then the gcloud login file, then the
    /// instance metadata server.
    Open {
        /// Local directory or object-store URI holding the datastore.
        #[arg(value_name = "DATASTORE_LOCATION")]
        datastore_location: String,

        /// Buffer-pool memory budget (suffixes k/m/g/t, base-1024).
        /// Defaults to 80% of the machine's physical memory (PIVOT_MEMORY_PCT)
        /// minus a 4 GiB reserve for allocations outside the pool.
        #[arg(long, value_name = "SIZE")]
        memory: Option<ByteSize>,

        /// Number of dispatch worker threads. Defaults to all available cores.
        #[arg(long, value_name = "COUNT")]
        workers: Option<NonZeroUsize>,
    },
    /// Run the Pivot database server in the foreground.
    Server(bin::server::ServerOptions),
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    match Args::parse().command {
        PivotCommand::Open {
            datastore_location,
            memory,
            workers,
        } => bin::shell::run_with_limits(
            datastore_location,
            bin::shell::ShellLimits {
                memory_bytes: memory.map(ByteSize::as_bytes),
                workers: workers.map(NonZeroUsize::get),
            },
        ),
        PivotCommand::Server(options) => bin::server::run(options).map_err(Into::into),
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
    use clap::Parser;

    use super::{Args, PivotCommand};

    fn open_location(argument: &str) -> String {
        let args = Args::try_parse_from(["pivot", "open", argument]).unwrap();
        let PivotCommand::Open {
            datastore_location, ..
        } = args.command
        else {
            panic!("open did not parse as the open command");
        };
        datastore_location
    }

    #[test]
    fn requires_a_command_and_an_open_datastore_location() {
        assert!(Args::try_parse_from(["pivot"]).is_err());
        assert!(Args::try_parse_from(["pivot", "open"]).is_err());
        assert!(Args::try_parse_from(["pivot", "shell", "/var/lib/pivot"]).is_err());

        assert_eq!(open_location("/var/lib/pivot"), "/var/lib/pivot");
    }

    #[test]
    fn open_takes_an_object_store_uri_verbatim() {
        assert_eq!(open_location("s3://bucket/prefix"), "s3://bucket/prefix");
        assert_eq!(open_location("gs://bucket/prefix"), "gs://bucket/prefix");
    }

    #[test]
    fn server_requires_a_config_file() {
        assert!(Args::try_parse_from(["pivot", "server"]).is_err());
        assert!(Args::try_parse_from(["pivot", "server", "install"]).is_err());

        let args =
            Args::try_parse_from(["pivot", "server", "--config", "/etc/pivot/pivot.yaml"]).unwrap();
        assert!(matches!(args.command, PivotCommand::Server(_)));
    }
}
