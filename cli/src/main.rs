//! Pivot command-line entry point.

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
    },
    /// Run the Pivot database server in the foreground.
    Server(server::ServerOptions),
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    match Args::parse().command {
        PivotCommand::Open { datastore_location } => shell::run(datastore_location),
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
    use clap::{CommandFactory, Parser};

    use super::{Args, PivotCommand};

    fn open_location(argument: &str) -> String {
        let args = Args::try_parse_from(["pivot", "open", argument]).unwrap();
        let PivotCommand::Open { datastore_location } = args.command else {
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
    fn server_accepts_config_or_direct_datastores() {
        assert!(Args::try_parse_from(["pivot", "server"]).is_err());
        assert!(Args::try_parse_from(["pivot", "server", "install"]).is_err());

        let args = Args::try_parse_from(["pivot", "server", "--config", "/etc/pivot/config.yaml"])
            .unwrap();
        assert!(matches!(args.command, PivotCommand::Server(_)));
        let args = Args::try_parse_from([
            "pivot",
            "server",
            "--config",
            "/etc/pivot/config.yaml",
            "--metastore-file",
            "/var/lib/pivot/metastore.yaml",
        ])
        .unwrap();
        assert!(matches!(args.command, PivotCommand::Server(_)));

        for location in [
            "/var/lib/pivot",
            "file:///var/lib/pivot",
            "s3://analytics/warm",
            "gs://analytics/cold",
        ] {
            let args = Args::try_parse_from(["pivot", "server", "--datastore", location]).unwrap();
            assert!(matches!(args.command, PivotCommand::Server(_)));
        }

        let args =
            Args::try_parse_from(["pivot", "server", "--datastore", "hot=/var/lib/pivot/hot"])
                .unwrap();
        assert!(matches!(args.command, PivotCommand::Server(_)));
        let args = Args::try_parse_from([
            "pivot",
            "server",
            "--datastore",
            "hot=/var/lib/pivot/hot",
            "--datastore",
            "warm=s3://analytics/warm",
            "--default-datastore",
            "hot",
        ])
        .unwrap();
        assert!(matches!(args.command, PivotCommand::Server(_)));
    }

    #[test]
    fn server_rejects_removed_or_mixed_datastore_options() {
        assert!(Args::try_parse_from(["pivot", "server", "--datastore-type", "s3"]).is_err());
        assert!(
            Args::try_parse_from(["pivot", "server", "--datastore-address", "bucket/prefix"])
                .is_err()
        );
        assert!(
            Args::try_parse_from([
                "pivot",
                "server",
                "--config",
                "pivot.yaml",
                "--datastore",
                "s3://bucket/prefix",
            ])
            .is_err()
        );
        assert!(
            Args::try_parse_from([
                "pivot",
                "server",
                "--datastore",
                "hot=/tmp/hot",
                "--metastore-file",
                "metastore.yaml",
            ])
            .is_err()
        );
        assert!(Args::try_parse_from(["pivot", "server", "--default-datastore", "hot",]).is_err());
        assert!(
            Args::try_parse_from(["pivot", "server", "--datastore", "https://example.com/db"])
                .is_err()
        );
        assert!(
            Args::try_parse_from(["pivot", "server", "--datastore", "system=/tmp/system",])
                .is_err()
        );
    }

    #[test]
    fn server_help_describes_all_startup_forms_and_direct_s3_credentials() {
        let mut command = Args::command();
        let server = command.find_subcommand_mut("server").unwrap();
        let help = server.render_long_help().to_string();

        for text in [
            "--config <FILE>",
            "--datastore <[NAME=]LOCATION>",
            "--default-datastore <NAME>",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "./.pivot/metastore.yaml",
        ] {
            assert!(help.contains(text), "server help omitted {text:?}:\n{help}");
        }
        for text in ["--datastore-type", "--datastore-address"] {
            assert!(
                !help.contains(text),
                "server help retained {text:?}:\n{help}"
            );
        }
    }
}
