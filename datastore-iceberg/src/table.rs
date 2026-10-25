//! What indexing a catalog table needs beyond the metadata the catalog hands
//! over: the store its files are read through, and the size of its manifest
//! list.

use iceberg::table::Table;
use iceberg_tables::TableStore;
use iceberg_tables::store::read_vended_s3_credentials;
use object_storage::ExternalStoreFactory;
use planner::catalog::{SchemaQualifiedTableName, TableRevision};

use crate::{Error, Result};

/// The store `table`'s files are read and written through: signing with the
/// credentials the catalog vended for the table, or opened through
/// `store_factory` when it vended none. Opened when the table is indexed rather
/// than when a query loads it, since opening a store may ask the bucket for its
/// region. At a refresh, `previous` is the store the last index held: keys
/// vended again for it renew it in place, so a statement still reading or
/// writing through it carries on with keys that have not expired.
pub(crate) fn open_table_store(
    name: &SchemaQualifiedTableName,
    table: &Table,
    store_factory: &dyn ExternalStoreFactory,
    previous: Option<&TableStore>,
) -> Result<TableStore> {
    let table_name = name.to_string();
    let metadata_location = table
        .metadata_location()
        .ok_or_else(|| Error::StagedTable {
            table: table_name.clone(),
        })?;
    let vended = read_vended_s3_credentials(&table_name, table.file_io().config().props())?;
    Ok(TableStore::open_or_renew(
        previous,
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

/// The revision a plan over `table` is cached under: the table's identity
/// across versions, and the metadata file this version was loaded from, which
/// every commit replaces.
pub(crate) fn revision(table: &iceberg_tables::Table) -> TableRevision {
    TableRevision {
        identity: table.uuid().to_string(),
        version: table.metadata_location().to_string(),
    }
}
