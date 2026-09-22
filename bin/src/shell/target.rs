//! What `pivot open` opens: which datastore implementation, and where it is.

use datastore_iceberg::env::EnvCredentialError;
use datastore_iceberg::{IcebergCatalogAuth, IcebergCatalogConfig};

/// The datastore a shell instance serves.
#[derive(Clone, Debug)]
pub enum ShellTarget {
    /// A Pivot datastore in a local directory (created if it does not exist)
    /// or at an object-store URI.
    Pivot { location: String },
    /// The read-only tables of an Iceberg REST catalog.
    Iceberg(IcebergCatalogConfig),
}

impl ShellTarget {
    pub fn pivot(location: impl Into<String>) -> Self {
        Self::Pivot {
            location: location.into(),
        }
    }

    /// The catalog at `catalog_uri`, authenticated to with the credentials
    /// the environment carries (see [`IcebergCatalogAuth::from_env`]).
    pub fn iceberg(
        catalog_uri: String,
        warehouse: Option<String>,
    ) -> Result<Self, EnvCredentialError> {
        Ok(Self::Iceberg(IcebergCatalogConfig {
            uri: catalog_uri,
            warehouse,
            auth: IcebergCatalogAuth::from_env()?,
            properties: Default::default(),
        }))
    }

    /// Where the datastore is, as it was spelled: a Pivot datastore's
    /// directory or object-store URI, or an Iceberg catalog's URI.
    pub fn location(&self) -> &str {
        match self {
            Self::Pivot { location } => location,
            Self::Iceberg(config) => &config.uri,
        }
    }
}
