//! What `pivot open` opens: which datastore implementation, and where it is.

use datastore_iceberg::{IcebergCatalogAuth, IcebergCatalogConfig};
use datastore_pivotlake::TableFormat;

/// The datastore a shell instance serves.
#[derive(Clone, Debug)]
pub enum ShellTarget {
    /// A pivotlake datastore in a local directory (created if it does not exist)
    /// or at an object-store URI. `table_format` names the table format a database
    /// that does not exist yet is created in; an existing database keeps the
    /// format it records.
    Pivotlake {
        location: String,
        table_format: Option<TableFormat>,
    },
    /// The read-only tables of an Iceberg REST catalog.
    Iceberg(IcebergCatalogConfig),
}

impl ShellTarget {
    /// The pivotlake datastore at `location`, created in the default table
    /// format if it does not exist yet.
    pub fn pivot(location: impl Into<String>) -> Self {
        Self::pivot_with_table_format(location, None)
    }

    /// The pivotlake datastore at `location`, created in `table_format` if it does
    /// not exist yet.
    pub fn pivot_with_table_format(
        location: impl Into<String>,
        table_format: Option<TableFormat>,
    ) -> Self {
        Self::Pivotlake {
            location: location.into(),
            table_format,
        }
    }

    /// The Iceberg REST catalog at `uri`, authenticated to with the
    /// credentials the environment carries (see
    /// [`IcebergCatalogAuth::from_env`]). A location that is not an http(s)
    /// URI names no catalog: Iceberg tables in a directory or bucket are a
    /// pivotlake datastore's.
    pub fn iceberg(
        uri: String,
        warehouse: Option<String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if !is_rest_catalog_uri(&uri) {
            return Err(format!(
                "`{uri}` is not an Iceberg REST catalog URI (http:// or https://); Iceberg tables in a directory or bucket are a pivotlake datastore, opened without --kind"
            )
            .into());
        }
        Ok(Self::Iceberg(IcebergCatalogConfig {
            uri,
            warehouse,
            auth: IcebergCatalogAuth::from_env()?,
            properties: Default::default(),
        }))
    }

    /// Where the datastore is, as it was spelled: a datastore's directory or
    /// object-store URI, or an Iceberg catalog's URI.
    pub fn location(&self) -> &str {
        match self {
            Self::Pivotlake { location, .. } => location,
            Self::Iceberg(config) => &config.uri,
        }
    }
}

/// Whether `location` names a REST catalog rather than a store: a catalog is
/// reached over http(s), a store is a directory or a bucket.
fn is_rest_catalog_uri(location: &str) -> bool {
    location.starts_with("http://") || location.starts_with("https://")
}
