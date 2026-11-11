//! The metadata document a table version was loaded from: the base every
//! commit from it is written on top of.

use std::sync::Arc;

use iceberg::spec::TableMetadata;
use iceberg::table::Table as CatalogTable;

use crate::{Error, Result};

/// A metadata document the table was loaded from: where it is, as the next
/// version's metadata log records it, and what it says.
#[derive(Clone)]
pub struct CurrentMetadata {
    pub metadata_location: String,
    pub metadata: Arc<TableMetadata>,
}

impl CurrentMetadata {
    /// The metadata `table`, as a catalog handed it over, is at. A table
    /// only staged in the catalog has no metadata location yet and is
    /// refused.
    pub fn from_catalog_table(table: &CatalogTable) -> Result<CurrentMetadata> {
        let metadata_location =
            table
                .metadata_location()
                .ok_or_else(|| Error::TableMetadataMissing {
                    table: table.identifier().to_string(),
                })?;
        Ok(CurrentMetadata {
            metadata_location: metadata_location.to_string(),
            metadata: table.metadata_ref(),
        })
    }
}

#[cfg(test)]
mod tests {
    use iceberg::io::FileIO;
    use iceberg::{NamespaceIdent, Runtime, TableIdent};

    use super::*;

    #[test]
    fn a_catalog_table_without_a_metadata_location_is_refused() {
        let metadata: TableMetadata = serde_json::from_value(serde_json::json!({
            "format-version": 2,
            "table-uuid": "9c12d441-03fe-4693-9a96-a0705ddf69c1",
            "location": "file:///table",
            "last-sequence-number": 0,
            "last-updated-ms": 0,
            "last-column-id": 1,
            "schemas": [{
                "type": "struct",
                "schema-id": 0,
                "fields": [{"id": 1, "name": "id", "required": false, "type": "long"}],
            }],
            "current-schema-id": 0,
            "partition-specs": [{"spec-id": 0, "fields": []}],
            "default-spec-id": 0,
            "last-partition-id": 999,
            "sort-orders": [{"order-id": 0, "fields": []}],
            "default-sort-order-id": 0,
        }))
        .unwrap();
        let staged = crate::block_on(async {
            CatalogTable::builder()
                .metadata(metadata)
                .identifier(TableIdent::new(
                    NamespaceIdent::new("db".to_string()),
                    "t".to_string(),
                ))
                .file_io(FileIO::new_with_memory())
                .runtime(Runtime::try_current().unwrap())
                .build()
                .unwrap()
        });

        let refused = CurrentMetadata::from_catalog_table(&staged);

        assert!(matches!(
            refused,
            Err(Error::TableMetadataMissing { table }) if table == "db.t"
        ));
    }
}
