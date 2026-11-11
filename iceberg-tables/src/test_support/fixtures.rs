//! What this crate's unit tests share: one worker pool, and a table around
//! hand-made metadata.

use std::sync::{Arc, OnceLock};

use dispatch::{DataFlowDispatcher, Dispatch};
use iceberg::spec::TableMetadata;
use object_storage::{LocalStore, ObjectPath, ObjectStore};
use planner::catalog::SchemaQualifiedTableName;

use crate::files::manifests::ManifestList;
use crate::metadata::layout::TableLocation;
use crate::metadata::pivotlake_metadata_store::PivotlakeMetadataStore;
use crate::metadata::state::CurrentMetadata;
use crate::table::Table;

/// One worker pool shared by every test in the binary, for the reads a
/// load makes over it.
pub(crate) fn test_pool() -> DataFlowDispatcher {
    static POOL: OnceLock<Dispatch> = OnceLock::new();
    POOL.get_or_init(|| Dispatch::spin_up(2, 64, None))
        .dispatcher()
        .clone()
}

/// The table `metadata` describes, for a test of what a table answers from
/// its metadata document alone. Nothing of it is in a store: its manifest
/// list is taken to be empty, and its directory is one nothing was written
/// to.
pub(crate) fn build_table_at(metadata: TableMetadata) -> Table {
    let directory = tempfile::tempdir().unwrap();
    let store: Arc<dyn ObjectStore> = Arc::new(LocalStore::new(directory.path()).unwrap());
    let location =
        TableLocation::in_datastore_store("main.t", Arc::clone(&store), ObjectPath::new("t"))
            .unwrap();
    let name = SchemaQualifiedTableName::new("main", "t");
    let manifest_list = ManifestList::empty(&metadata);
    Table::from_manifest_list(
        &test_pool(),
        Arc::new(PivotlakeMetadataStore::new(location.clone())),
        location,
        &name,
        Arc::new(CurrentMetadata {
            metadata_location: String::new(),
            metadata: Arc::new(metadata),
        }),
        manifest_list,
        None,
    )
    .unwrap()
}
