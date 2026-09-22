//! What `pivot open` opens: the command's options, and the target they name.
//!
//! The target is one of the datastore kinds a server config's `kind` selects,
//! spelled on the command line instead of in a file. A `pivot` datastore is
//! named by its location, as the server names it; an `iceberg` datastore by
//! its catalog's URI, with the config's other fields as flags. The catalog's
//! secret, a `token` or an OAuth2 `credential` in the config's `secrets`
//! section, is read from the environment here, as an object store's keys are,
//! so it never sits in a process listing or a shell history.

use std::num::NonZeroUsize;

use clap::{Args, ValueEnum};
use datastore_iceberg::{IcebergCatalogAuth, IcebergCatalogConfig};
use metastore_disk::ByteSize;

use crate::shell::ShellLimits;

/// A bearer token sent on every request to the Iceberg catalog.
pub const ICEBERG_TOKEN_VARIABLE: &str = "PIVOT_ICEBERG_TOKEN";
/// An OAuth2 client credential (`client_id:client_secret`) exchanged for a
/// token at the Iceberg catalog's token endpoint.
pub const ICEBERG_CREDENTIAL_VARIABLE: &str = "PIVOT_ICEBERG_CREDENTIAL";
/// The OAuth2 token endpoint the credential is exchanged at, when it is not
/// the catalog's own.
pub const ICEBERG_OAUTH2_SERVER_URI_VARIABLE: &str = "PIVOT_ICEBERG_OAUTH2_SERVER_URI";
/// The OAuth2 scope requested with the credential.
pub const ICEBERG_SCOPE_VARIABLE: &str = "PIVOT_ICEBERG_SCOPE";

/// Options accepted by `pivot open`.
#[derive(Args, Debug)]
pub struct OpenOptions {
    /// What to open, read according to --kind. For `pivot`, a local directory,
    /// created if it does not exist, or an object-store URI: s3://bucket/prefix
    /// (s3:// also spelled s3a://), gs://bucket/prefix, or file:///path. For
    /// `iceberg`, the catalog's base URI, such as https://catalog.example.com/api.
    #[arg(value_name = "LOCATION")]
    location: String,

    /// The kind of datastore LOCATION names, as `kind` names it in a server
    /// config.
    #[arg(long, value_enum, default_value_t = DatastoreKind::Pivot)]
    kind: DatastoreKind,

    /// The warehouse to serve, when the Iceberg catalog serves several.
    /// Applies to --kind iceberg only.
    #[arg(long, value_name = "NAME")]
    warehouse: Option<String>,

    /// A further Iceberg REST client property, passed through as written (a
    /// `prefix`, a `header.*`). Repeatable. Applies to --kind iceberg only.
    #[arg(long = "property", value_name = "KEY=VALUE", value_parser = parse_property)]
    properties: Vec<(String, String)>,

    /// Buffer-pool memory budget (suffixes k/m/g/t, base-1024).
    /// Defaults to 80% of the machine's physical memory (PIVOT_MEMORY_PCT)
    /// minus a 4 GiB reserve for allocations outside the pool.
    #[arg(long, value_name = "SIZE")]
    memory: Option<ByteSize>,

    /// Number of dispatch worker threads. Defaults to all available cores.
    #[arg(long, value_name = "COUNT")]
    workers: Option<NonZeroUsize>,
}

/// The datastore kinds the shell opens, named as a server config's `kind`
/// names them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum DatastoreKind {
    /// A Pivot datastore at a local directory or an object-store URI.
    Pivot,
    /// The read-only tables of an Iceberg REST catalog.
    Iceberg,
}

/// What the shell opens.
#[derive(Clone, Debug)]
pub enum ShellTarget {
    /// A Pivot datastore at a local directory or an object-store URI
    /// (`s3://bucket/prefix`, `gs://bucket/prefix`), whose credentials come
    /// from the environment.
    Pivot { location: String },
    /// The read-only tables of an Iceberg REST catalog.
    Iceberg(IcebergCatalogConfig),
}

impl ShellTarget {
    /// The Pivot datastore at `location`.
    pub fn pivot(location: &str) -> Self {
        Self::Pivot {
            location: location.to_string(),
        }
    }

    /// Where the target is, as it was spelled: the datastore's location, or
    /// the catalog's URI.
    pub fn location(&self) -> &str {
        match self {
            Self::Pivot { location } => location,
            Self::Iceberg(config) => &config.uri,
        }
    }
}

impl OpenOptions {
    /// The target the options name, with the catalog secret the environment
    /// carries, and the resource limits they set.
    pub fn into_target_and_limits(self) -> Result<(ShellTarget, ShellLimits)> {
        let limits = ShellLimits {
            memory_bytes: self.memory.map(ByteSize::as_bytes),
            workers: self.workers.map(NonZeroUsize::get),
        };
        let target = self.into_target(|name| std::env::var(name).ok())?;
        Ok((target, limits))
    }

    /// The target the options name; `read_variable` is the environment the
    /// catalog secret is read from.
    fn into_target(self, read_variable: impl Fn(&str) -> Option<String>) -> Result<ShellTarget> {
        match self.kind {
            DatastoreKind::Pivot => {
                if self.warehouse.is_some() {
                    return Err(Error::IcebergOnlyFlag {
                        flag: "--warehouse",
                    });
                }
                if !self.properties.is_empty() {
                    return Err(Error::IcebergOnlyFlag { flag: "--property" });
                }
                if is_web_address(&self.location) {
                    return Err(Error::CatalogUriAsLocation {
                        location: self.location,
                    });
                }
                Ok(ShellTarget::Pivot {
                    location: self.location,
                })
            }
            DatastoreKind::Iceberg => Ok(ShellTarget::Iceberg(IcebergCatalogConfig {
                uri: self.location,
                warehouse: self.warehouse,
                auth: read_iceberg_auth(read_variable)?,
                properties: self.properties.into_iter().collect(),
            })),
        }
    }
}

/// Whether `location` is spelled as a web address, which no storage backend
/// is: a catalog's URI given without the kind that reads it.
fn is_web_address(location: &str) -> bool {
    matches!(location.split_once("://"), Some(("http" | "https", _)))
}

/// What the catalog is authenticated to with, from the `PIVOT_ICEBERG_*`
/// variables: at most one of the token and the credential, and the OAuth2
/// options only alongside the credential they apply to.
fn read_iceberg_auth(
    read_variable: impl Fn(&str) -> Option<String>,
) -> Result<Option<IcebergCatalogAuth>> {
    let token = read_variable(ICEBERG_TOKEN_VARIABLE);
    let credential = read_variable(ICEBERG_CREDENTIAL_VARIABLE);
    let server_uri = read_variable(ICEBERG_OAUTH2_SERVER_URI_VARIABLE);
    let scope = read_variable(ICEBERG_SCOPE_VARIABLE);

    if credential.is_none() {
        if server_uri.is_some() {
            return Err(Error::OAuth2OptionWithoutCredential {
                variable: ICEBERG_OAUTH2_SERVER_URI_VARIABLE,
            });
        }
        if scope.is_some() {
            return Err(Error::OAuth2OptionWithoutCredential {
                variable: ICEBERG_SCOPE_VARIABLE,
            });
        }
    }
    match (token, credential) {
        (None, None) => Ok(None),
        (Some(token), None) => Ok(Some(IcebergCatalogAuth::Token(token))),
        (None, Some(credential)) => Ok(Some(IcebergCatalogAuth::OAuth2 {
            credential,
            server_uri,
            scope,
        })),
        (Some(_), Some(_)) => Err(Error::BothTokenAndCredential),
    }
}

/// Split a `KEY=VALUE` property at its first `=`.
fn parse_property(written: &str) -> std::result::Result<(String, String), String> {
    match written.split_once('=') {
        Some((key, value)) if !key.is_empty() => Ok((key.to_string(), value.to_string())),
        _ => Err(format!("`{written}` is not of the form KEY=VALUE")),
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("`{flag}` applies to `--kind iceberg` only")]
    IcebergOnlyFlag { flag: &'static str },
    #[error(
        "`{location}` is not a storage location; an Iceberg REST catalog is opened with `--kind iceberg`"
    )]
    CatalogUriAsLocation { location: String },
    #[error(
        "{ICEBERG_TOKEN_VARIABLE} and {ICEBERG_CREDENTIAL_VARIABLE} are both set; an Iceberg catalog is authenticated to with one of them"
    )]
    BothTokenAndCredential,
    #[error("{variable} is set without the {ICEBERG_CREDENTIAL_VARIABLE} it applies to")]
    OAuth2OptionWithoutCredential { variable: &'static str },
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use clap::Parser;

    use super::*;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        open: OpenOptions,
    }

    fn parse(arguments: &[&str]) -> OpenOptions {
        Cli::try_parse_from(std::iter::once("pivot").chain(arguments.iter().copied()))
            .unwrap()
            .open
    }

    fn environment(variables: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let variables: HashMap<String, String> = variables
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        move |name| variables.get(name).cloned()
    }

    fn iceberg_config(options: OpenOptions, variables: &[(&str, &str)]) -> IcebergCatalogConfig {
        match options.into_target(environment(variables)).unwrap() {
            ShellTarget::Iceberg(config) => config,
            ShellTarget::Pivot { location } => panic!("opened a Pivot datastore at {location}"),
        }
    }

    #[test]
    fn a_location_without_a_kind_is_a_pivot_datastore() {
        let options = parse(&["s3://bucket/prefix"]);

        let target = options.into_target(environment(&[])).unwrap();

        assert_eq!(target.location(), "s3://bucket/prefix");
        assert!(matches!(target, ShellTarget::Pivot { .. }));
    }

    #[test]
    fn a_web_address_needs_the_iceberg_kind() {
        let options = parse(&["https://catalog.example.com/api"]);

        let error = options.into_target(environment(&[])).unwrap_err();

        assert!(error.to_string().contains("--kind iceberg"), "{error}");
    }

    #[test]
    fn the_iceberg_flags_are_refused_for_a_pivot_datastore() {
        let with_warehouse = parse(&["/var/lib/pivot", "--warehouse", "lake"]);
        let with_property = parse(&["/var/lib/pivot", "--property", "prefix=lake"]);

        let warehouse_error = with_warehouse.into_target(environment(&[])).unwrap_err();
        let property_error = with_property.into_target(environment(&[])).unwrap_err();

        assert_eq!(
            warehouse_error.to_string(),
            "`--warehouse` applies to `--kind iceberg` only"
        );
        assert_eq!(
            property_error.to_string(),
            "`--property` applies to `--kind iceberg` only"
        );
    }

    #[test]
    fn an_iceberg_catalog_takes_its_options_from_the_flags() {
        let options = parse(&[
            "--kind",
            "iceberg",
            "https://catalog.example.com/api",
            "--warehouse",
            "lake",
            "--property",
            "prefix=v1",
            "--property",
            "header.X-Tenant=acme=inc",
        ]);

        let config = iceberg_config(options, &[]);

        assert_eq!(config.uri, "https://catalog.example.com/api");
        assert_eq!(config.warehouse.as_deref(), Some("lake"));
        assert!(config.auth.is_none());
        assert_eq!(config.properties["prefix"], "v1");
        assert_eq!(config.properties["header.X-Tenant"], "acme=inc");
    }

    #[test]
    fn a_property_must_be_a_key_and_a_value() {
        let parsed =
            Cli::try_parse_from(["pivot", "--kind", "iceberg", "http://c", "--property", "=v"]);

        assert!(parsed.is_err());
    }

    #[test]
    fn the_catalog_token_comes_from_the_environment() {
        let options = parse(&["--kind", "iceberg", "http://catalog"]);

        let config = iceberg_config(options, &[(ICEBERG_TOKEN_VARIABLE, "t0k3n")]);

        assert!(matches!(config.auth, Some(IcebergCatalogAuth::Token(token)) if token == "t0k3n"));
    }

    #[test]
    fn the_catalog_credential_carries_its_oauth2_options() {
        let options = parse(&["--kind", "iceberg", "http://catalog"]);

        let config = iceberg_config(
            options,
            &[
                (ICEBERG_CREDENTIAL_VARIABLE, "id:secret"),
                (ICEBERG_OAUTH2_SERVER_URI_VARIABLE, "https://auth/token"),
                (ICEBERG_SCOPE_VARIABLE, "PRINCIPAL_ROLE:ALL"),
            ],
        );

        let Some(IcebergCatalogAuth::OAuth2 {
            credential,
            server_uri,
            scope,
        }) = config.auth
        else {
            panic!("the credential was not read");
        };
        assert_eq!(credential, "id:secret");
        assert_eq!(server_uri.as_deref(), Some("https://auth/token"));
        assert_eq!(scope.as_deref(), Some("PRINCIPAL_ROLE:ALL"));
    }

    #[test]
    fn a_token_and_a_credential_together_are_refused() {
        let options = parse(&["--kind", "iceberg", "http://catalog"]);

        let error = options
            .into_target(environment(&[
                (ICEBERG_TOKEN_VARIABLE, "t"),
                (ICEBERG_CREDENTIAL_VARIABLE, "c"),
            ]))
            .unwrap_err();

        assert!(matches!(error, Error::BothTokenAndCredential));
    }

    #[test]
    fn an_oauth2_option_without_a_credential_is_refused() {
        let options = parse(&["--kind", "iceberg", "http://catalog"]);

        let error = options
            .into_target(environment(&[(ICEBERG_SCOPE_VARIABLE, "s")]))
            .unwrap_err();

        assert!(matches!(
            error,
            Error::OAuth2OptionWithoutCredential {
                variable: ICEBERG_SCOPE_VARIABLE
            }
        ));
    }

    #[test]
    fn the_limits_come_from_the_resource_flags() {
        let options = parse(&["/var/lib/pivot", "--memory", "4g", "--workers", "3"]);

        let (_, limits) = options.into_target_and_limits().unwrap();

        assert_eq!(
            limits,
            ShellLimits {
                memory_bytes: Some(4 * 1024 * 1024 * 1024),
                workers: Some(3),
            }
        );
    }
}
