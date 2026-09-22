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
    /// The datastore is a Pivot datastore (--kind pivot, the default) named by
    /// a local directory, which is created if it does not exist, or by an
    /// object-store URI: s3://bucket/prefix (s3:// also spelled s3a://),
    /// gs://bucket/prefix, or file:///path. Or it is the read-only tables of an
    /// Iceberg REST catalog (--kind iceberg) named by the catalog's base URI,
    /// whose namespaces are the shell's schemas.
    ///
    /// Object-store credentials are read from the environment, for a Pivot
    /// datastore's store and for the files of an Iceberg table whose catalog
    /// vends no credentials of its own. S3 takes AWS_ACCESS_KEY_ID and
    /// AWS_SECRET_ACCESS_KEY when both are set, or uses anonymous access when
    /// both are absent. Its region comes from AWS_REGION or AWS_DEFAULT_REGION
    /// when set, and is discovered with HeadBucket otherwise. An optional
    /// AWS_ENDPOINT_URL names a path-style S3-compatible endpoint such as
    /// MinIO. GCS follows Application Default Credentials:
    /// GOOGLE_APPLICATION_CREDENTIALS, then the gcloud login file, then the
    /// instance metadata server.
    ///
    /// An Iceberg catalog's credentials are read from the environment too:
    /// PIVOT_ICEBERG_TOKEN is a bearer token sent on every request, or
    /// PIVOT_ICEBERG_CREDENTIAL an OAuth2 client credential
    /// (client_id:client_secret) exchanged for a token at the catalog's token
    /// endpoint, or at PIVOT_ICEBERG_OAUTH2_SERVER_URI when set, for the scope
    /// PIVOT_ICEBERG_SCOPE when set. A catalog that needs neither is opened
    /// with both unset.
    Open(bin::shell::OpenOptions),
    /// Run the Pivot database server in the foreground.
    Server(bin::server::ServerOptions),
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    match Args::parse().command {
        PivotCommand::Open(options) => {
            let (target, limits) = options.into_target_and_limits()?;
            bin::shell::run_with_limits(target, limits)
        }
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
    use bin::shell::ShellTarget;
    use clap::Parser;

    use super::{Args, PivotCommand};

    fn open_target(arguments: &[&str]) -> ShellTarget {
        let args = Args::try_parse_from(
            ["pivot", "open"]
                .into_iter()
                .chain(arguments.iter().copied()),
        )
        .unwrap();
        let PivotCommand::Open(options) = args.command else {
            panic!("open did not parse as the open command");
        };
        options.into_target_and_limits().unwrap().0
    }

    #[test]
    fn requires_a_command_and_an_open_location() {
        assert!(Args::try_parse_from(["pivot"]).is_err());
        assert!(Args::try_parse_from(["pivot", "open"]).is_err());
        assert!(Args::try_parse_from(["pivot", "shell", "/var/lib/pivot"]).is_err());

        assert_eq!(
            open_target(&["/var/lib/pivot"]).location(),
            "/var/lib/pivot"
        );
    }

    #[test]
    fn open_takes_an_object_store_uri_verbatim() {
        assert_eq!(
            open_target(&["s3://bucket/prefix"]).location(),
            "s3://bucket/prefix"
        );
        assert_eq!(
            open_target(&["gs://bucket/prefix"]).location(),
            "gs://bucket/prefix"
        );
    }

    #[test]
    fn open_takes_an_iceberg_catalog_by_kind() {
        let target = open_target(&["--kind", "iceberg", "https://catalog.example.com/api"]);

        assert!(matches!(target, ShellTarget::Iceberg(_)));
        assert_eq!(target.location(), "https://catalog.example.com/api");
    }

    #[test]
    fn open_rejects_an_unknown_kind() {
        assert!(
            Args::try_parse_from(["pivot", "open", "--kind", "delta", "/var/lib/pivot"]).is_err()
        );
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
