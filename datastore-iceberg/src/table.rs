//! What indexing a catalog table needs beyond the metadata the catalog hands
//! over: the store its files are read through, and the size of its manifest
//! list.

use iceberg::table::Table;
use iceberg_tables::store::read_vended_s3_credentials;
use iceberg_tables::{CurrentMetadata, TableStore};
use object_storage::ExternalStoreFactory;
use planner::catalog::SchemaQualifiedTableName;

use crate::{Error, Result};

/// The current metadata `table` carries, as the shared crate holds it. A
/// table staged in the catalog has no metadata location yet and cannot be
/// read.
pub(crate) fn current_metadata(
    name: &SchemaQualifiedTableName,
    table: &Table,
) -> iceberg_tables::Result<CurrentMetadata> {
    let metadata_location =
        table
            .metadata_location()
            .ok_or_else(|| iceberg_tables::Error::TableMetadataMissing {
                table: name.to_string(),
                location: table.identifier().to_string(),
            })?;
    Ok(CurrentMetadata {
        metadata_location: metadata_location.to_string(),
        metadata: table.metadata_ref(),
    })
}

/// The store `table`'s files are read and written through: opened with the
/// credentials the catalog vended for the table, or through `store_factory`
/// when it vended none. Opened when the table is indexed rather than when a
/// query loads it, since opening a store may ask the bucket for its region;
/// and opened again at every refresh, so vended credentials never outlive
/// their session.
pub(crate) fn open_table_store(
    name: &SchemaQualifiedTableName,
    table: &Table,
    metadata_location: &str,
    store_factory: &dyn ExternalStoreFactory,
) -> Result<TableStore> {
    let table_name = name.to_string();
    let vended = read_vended_s3_credentials(&table_name, table.file_io().config().props())?;
    Ok(TableStore::open(
        &table_name,
        metadata_location,
        store_factory,
        vended,
    )?)
}

/// The size of `table`'s current manifest list, or `None` when the table has
/// no current snapshot. The snapshot names its manifest list without a
/// length, and a ring read needs one, so the store is asked once, when the
/// table is indexed, rather than on every query over it.
pub(crate) fn fetch_manifest_list_size(
    name: &SchemaQualifiedTableName,
    table: &Table,
    store: &TableStore,
) -> Result<Option<u64>> {
    let Some(snapshot) = table.metadata().current_snapshot() else {
        return Ok(None);
    };
    let path = snapshot.manifest_list();
    let size = store
        .object_size(path)?
        .ok_or_else(|| Error::MissingMetadataObject {
            table: name.to_string(),
            path: path.to_string(),
        })?;
    Ok(Some(size))
}
