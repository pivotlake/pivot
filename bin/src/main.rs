//! Pivot command-line entry point.

use std::path::PathBuf;
use std::process::ExitCode;

use bin::shell::{DiskCacheOptions, OpenOptions};
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
    /// AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, its region from AWS_REGION
    /// or AWS_DEFAULT_REGION (us-east-1 when neither is set), and an optional
    /// AWS_ENDPOINT_URL naming a path-style S3-compatible endpoint such as
    /// MinIO. GCS follows Application Default Credentials:
    /// GOOGLE_APPLICATION_CREDENTIALS, then the gcloud login file, then the
    /// instance metadata server.
    Open {
        /// Local directory or object-store URI holding the datastore.
        #[arg(value_name = "DATASTORE_LOCATION")]
        datastore_location: String,

        /// Buffer pool memory budget, such as `4g` or `512m`. Defaults to half
        /// of the machine's physical memory. The budget must fit in the memory
        /// available right now.
        #[arg(long, value_name = "SIZE")]
        memory: Option<ByteSize>,

        /// Number of dispatch worker threads. Defaults to the machine's core
        /// count.
        #[arg(long, value_name = "COUNT", value_parser = clap::value_parser!(u64).range(1..))]
        workers: Option<u64>,

        /// Cache remote (object store) reads on local disk in this directory.
        /// Local datastores are never cached; they are read from the
        /// filesystem directly.
        #[arg(long, value_name = "DIR")]
        disk_cache: Option<PathBuf>,

        /// Size budget for the disk cache, such as `64g` (the default).
        #[arg(long, value_name = "SIZE", requires = "disk_cache")]
        disk_cache_size: Option<ByteSize>,
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
            disk_cache,
            disk_cache_size,
        } => bin::shell::run(
            datastore_location,
            OpenOptions {
                memory_bytes: memory.map(ByteSize::as_bytes),
                workers: workers.map(|count| count as usize),
                disk_cache: disk_cache.map(|dir| DiskCacheOptions {
                    dir,
                    size_bytes: disk_cache_size.map(ByteSize::as_bytes),
                }),
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
    fn open_parses_resource_flags_with_size_suffixes() {
        let args = Args::try_parse_from([
            "pivot",
            "open",
            "/var/lib/pivot",
            "--memory",
            "4g",
            "--workers",
            "2",
            "--disk-cache",
            "/var/cache/pivot",
            "--disk-cache-size",
            "512m",
        ])
        .unwrap();

        let PivotCommand::Open {
            memory,
            workers,
            disk_cache,
            disk_cache_size,
            ..
        } = args.command
        else {
            panic!("open did not parse as the open command");
        };
        assert_eq!(memory.unwrap().as_bytes(), 4 * 1024 * 1024 * 1024);
        assert_eq!(workers, Some(2));
        assert_eq!(
            disk_cache,
            Some(std::path::PathBuf::from("/var/cache/pivot"))
        );
        assert_eq!(disk_cache_size.unwrap().as_bytes(), 512 * 1024 * 1024);
    }

    #[test]
    fn open_rejects_malformed_resource_flags() {
        assert!(Args::try_parse_from(["pivot", "open", "/data", "--memory", "4x"]).is_err());
        assert!(Args::try_parse_from(["pivot", "open", "/data", "--workers", "0"]).is_err());
        assert!(
            Args::try_parse_from(["pivot", "open", "/data", "--disk-cache-size", "1g"]).is_err(),
            "--disk-cache-size without --disk-cache names a cache that does not exist"
        );
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
