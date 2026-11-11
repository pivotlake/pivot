//! What indexing a catalog table needs beyond the metadata the catalog hands
//! over: the store its files are read through, and the size of its manifest
//! list.

use std::sync::Arc;

use iceberg::table::Table;
use iceberg_tables::io::store::{open_store_at_root, read_vended_s3_credentials};
use iceberg_tables::metadata::layout::key_under_root;
use iceberg_tables::split_location;
use object_storage::{ExternalStoreFactory, ObjectStore};
use planner::catalog::SchemaQualifiedTableName;

use crate::{Error, Result};

/// The store `table`'s files are read and written through, opened on the
/// root of `metadata_location`: with the credentials the catalog vended for
/// the table, or through `store_factory` when it vended none. Opened when
/// the table is indexed rather than when a query loads it, since opening a
/// store may ask the bucket for its region; and opened again at every
/// refresh, so vended credentials never outlive their session.
pub(crate) fn open_table_store(
    name: &SchemaQualifiedTableName,
    table: &Table,
    metadata_location: &str,
    store_factory: &dyn ExternalStoreFactory,
) -> Result<Arc<dyn ObjectStore>> {
    let vended = read_vended_s3_credentials(&name.to_string(), table.file_io().config().props())?;
    let (root, _) = split_location(metadata_location)?;
    Ok(open_store_at_root(&root, store_factory, vended)?)
}

/// The size of `table`'s current manifest list, or `None` when the table has
/// no current snapshot. The snapshot names its manifest list without a
/// length, and a ring read needs one, so `store`, opened on the root of
/// `metadata_location`, is asked once, when the table is indexed, rather
/// than on every query over it.
pub(crate) fn fetch_manifest_list_size(
    name: &SchemaQualifiedTableName,
    table: &Table,
    metadata_location: &str,
    store: &dyn ObjectStore,
) -> Result<Option<u64>> {
    let Some(snapshot) = table.metadata().current_snapshot() else {
        return Ok(None);
    };
    let path = snapshot.manifest_list();
    let (root, _) = split_location(metadata_location)?;
    let key = key_under_root(&name.to_string(), &root, path)?;
    let size = store
        .object_size(&key)
        .map_err(iceberg_tables::Error::from)?
        .ok_or_else(|| Error::MissingMetadataObject {
            table: name.to_string(),
            path: path.to_string(),
        })?;
    Ok(Some(size))
}
