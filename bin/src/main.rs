//! Pivot command-line entry point.

use std::num::NonZeroUsize;
use std::process::ExitCode;

use bin::memory::MemoryBudget;
use bin::shell::ShellTarget;
use clap::{Parser, Subcommand, ValueEnum};
use datastore_pivotlake::TableFormat;

#[derive(Parser, Debug)]
#[command(name = "pivot", about = "Pivot command-line tools", version)]
struct Args {
    #[command(subcommand)]
    command: PivotCommand,
}

/// Which datastore implementation `pivot open` opens.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DatastoreKind {
    /// A pivotlake datastore in a local directory or at an object-store URI.
    /// Also accepted as `pivot`, the kind's name before it was renamed.
    #[value(alias = "pivot")]
    Pivotlake,
    /// The read-only tables of an Iceberg REST catalog at an http(s) URI.
    Iceberg,
}

#[derive(Subcommand, Debug)]
enum PivotCommand {
    /// Open a datastore in the Pivot SQL shell.
    ///
    /// A pivotlake datastore (the default kind) is named either by a local
    /// directory, which is created if it does not exist, or by an object-store
    /// URI: s3://bucket/prefix (s3:// also spelled s3a://), gs://bucket/prefix,
    /// or file:///path.
    ///
    /// An Iceberg datastore (--kind iceberg) is a REST catalog at an http(s)
    /// URI, whose tables are served read-only; each namespace is a schema.
    /// The catalog is authenticated to with PIVOT_ICEBERG_TOKEN (a bearer
    /// token) or PIVOT_ICEBERG_CREDENTIAL (an OAuth2 client_id:client_secret,
    /// refined by the optional PIVOT_ICEBERG_OAUTH2_SERVER_URI and
    /// PIVOT_ICEBERG_OAUTH2_SCOPE), or with nothing when neither is set. The
    /// tables' files are read from wherever the catalog says they are, with the
    /// credentials the catalog vends or those of the environment below.
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
        /// Local directory or object-store URI holding the datastore, or an
        /// Iceberg REST catalog's http(s) URI with --kind iceberg.
        #[arg(value_name = "DATASTORE_LOCATION")]
        datastore_location: String,

        /// Which datastore implementation the location names.
        #[arg(long, value_enum, default_value_t = DatastoreKind::Pivotlake)]
        kind: DatastoreKind,

        /// The table format a pivotlake database that does not exist yet is
        /// created in: iceberg (the default) or delta (also spelled deltalake).
        #[arg(long, value_name = "FORMAT")]
        table_format: Option<TableFormat>,

        /// The warehouse to serve, for an Iceberg catalog that serves several.
        #[arg(long, value_name = "WAREHOUSE")]
        warehouse: Option<String>,

        /// Buffer-pool memory budget: a size (suffixes k/m/g/t, base-1024), or
        /// a share of the machine's physical memory such as 80%, the default,
        /// minus a 4 GiB reserve for allocations outside the pool.
        #[arg(long, value_name = "SIZE")]
        memory: Option<MemoryBudget>,

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
            kind,
            table_format,
            warehouse,
            memory,
            workers,
        } => bin::shell::run_with_limits(
            build_shell_target(kind, datastore_location, table_format, warehouse)?,
            bin::shell::ShellLimits {
                memory,
                workers: workers.map(NonZeroUsize::get),
            },
        ),
        PivotCommand::Server(options) => bin::server::run(options).map_err(Into::into),
    }
}

/// What `pivot open` opens, from its arguments. Only a REST catalog has a
/// warehouse to name; only a pivotlake database has a table format.
fn build_shell_target(
    kind: DatastoreKind,
    datastore_location: String,
    table_format: Option<TableFormat>,
    warehouse: Option<String>,
) -> Result<ShellTarget, Box<dyn std::error::Error>> {
    match kind {
        DatastoreKind::Pivotlake => {
            if warehouse.is_some() {
                return Err("--warehouse applies only to --kind iceberg".into());
            }
            Ok(ShellTarget::pivot_with_table_format(
                datastore_location,
                table_format,
            ))
        }
        DatastoreKind::Iceberg => {
            if table_format.is_some() {
                return Err("--table-format applies only to --kind pivotlake".into());
            }
            Ok(ShellTarget::iceberg(datastore_location, warehouse)?)
        }
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

    use super::{Args, DatastoreKind, PivotCommand, TableFormat};

    fn parse_open(arguments: &[&str]) -> (String, DatastoreKind, Option<String>) {
        let args = Args::try_parse_from([&["pivot", "open"], arguments].concat()).unwrap();
        let PivotCommand::Open {
            datastore_location,
            kind,
            warehouse,
            ..
        } = args.command
        else {
            panic!("open did not parse as the open command");
        };
        (datastore_location, kind, warehouse)
    }

    fn parse_format(arguments: &[&str]) -> Option<TableFormat> {
        let args = Args::try_parse_from([&["pivot", "open"], arguments].concat()).unwrap();
        let PivotCommand::Open { table_format, .. } = args.command else {
            panic!("open did not parse as the open command");
        };
        table_format
    }

    fn open_location(argument: &str) -> String {
        parse_open(&[argument]).0
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
    fn open_defaults_to_the_pivotlake_kind() {
        let (_, kind, warehouse) = parse_open(&["/var/lib/pivot"]);

        assert_eq!(kind, DatastoreKind::Pivotlake);
        assert!(warehouse.is_none());
    }

    #[test]
    fn open_takes_pivot_as_the_pivotlake_kind() {
        let (_, kind, _) = parse_open(&["--kind", "pivot", "/var/lib/pivot"]);

        assert_eq!(kind, DatastoreKind::Pivotlake);
    }

    #[test]
    fn open_takes_an_iceberg_catalog_and_its_warehouse() {
        let (location, kind, warehouse) = parse_open(&[
            "--kind",
            "iceberg",
            "https://catalog.example.com/api",
            "--warehouse",
            "s3://lake/warehouse",
        ]);

        assert_eq!(location, "https://catalog.example.com/api");
        assert_eq!(kind, DatastoreKind::Iceberg);
        assert_eq!(warehouse.as_deref(), Some("s3://lake/warehouse"));
    }

    #[test]
    fn open_takes_the_table_format_of_a_new_database_with_its_alias() {
        assert_eq!(parse_format(&["/var/lib/pivot"]), None);
        assert_eq!(
            parse_format(&["--table-format", "delta", "/var/lib/pivot"]),
            Some(TableFormat::Delta)
        );
        assert_eq!(
            parse_format(&["--table-format", "deltalake", "/var/lib/pivot"]),
            Some(TableFormat::Delta)
        );
        assert_eq!(
            parse_format(&["--table-format", "iceberg", "/var/lib/pivot"]),
            Some(TableFormat::Iceberg)
        );
        assert!(
            Args::try_parse_from(["pivot", "open", "--table-format", "hudi", "/var/lib/pivot"])
                .is_err()
        );
    }

    #[test]
    fn a_format_reaches_the_pivotlake_target_and_is_refused_for_a_catalog() {
        let target = super::build_shell_target(
            DatastoreKind::Pivotlake,
            "/var/lib/lake".to_string(),
            Some(TableFormat::Delta),
            None,
        )
        .unwrap();
        let error = super::build_shell_target(
            DatastoreKind::Iceberg,
            "https://catalog.example.com/api".to_string(),
            Some(TableFormat::Iceberg),
            None,
        )
        .expect_err("a catalog's tables have the format the catalog gave them");

        assert!(
            matches!(&target, bin::shell::ShellTarget::Pivotlake { location, table_format: Some(TableFormat::Delta) } if location == "/var/lib/lake")
        );
        assert!(error.to_string().contains("--table-format"), "{error}");
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
