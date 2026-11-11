//! Table-level test helpers layered on the reusable object-store harness:
//! committing to a table through its handle, and seeing or touching an
//! Iceberg table's metadata the way another process would.

pub use object_storage::test_support::*;

use std::sync::Arc;

use dispatch::{DataFlowDispatcher, Dispatch};
use iceberg_tables::commit::COMMIT_RETRIES;
use iceberg_tables::data_files::DataFileEntry;
use iceberg_tables::directory::metadata_location_version;
use iceberg_tables::iceberg::spec::{
    SnapshotReference, SnapshotRetention, StatisticsFile, TableMetadata, TableMetadataBuilder,
};
use iceberg_tables::layout::{VERSION_HINT_FILE, parse_hint, parse_metadata_file_version};
use iceberg_tables::{DirectoryMetadataPointer, MetadataPointer, Table, TableLocation, TableStore};
use object_storage::{FileRef, ObjectPath, ObjectStore, open_store};
use parquet_engine::FileStats;
use planner::catalog::SchemaQualifiedTableName;

use crate::{CatalogTable, Error, FileEntry, PartitionValues, Result};

impl CatalogTable {
    /// Write `bytes` as a new data file and commit it into the table.
    pub fn append_data_file(
        &mut self,
        path: ObjectPath,
        bytes: &[u8],
        partition: Option<PartitionValues>,
    ) -> Result<()> {
        if self.file_refs()?.iter().any(|file| file.path == path) {
            return Ok(());
        }
        self.store()
            .put(&ObjectPath::new(self.location()).resolve(&path), bytes)?;
        let entry = FileEntry {
            file: FileRef {
                path,
                size: bytes.len() as u64,
            },
            partition,
            stats: None,
            sorted_by: Arc::default(),
        };
        self.commit_entries(&[], &[entry], true)?;
        Ok(())
    }

    /// Atomically replace `removed` files with `added` files in one version.
    pub fn replace_data_files(
        &mut self,
        removed: &[ObjectPath],
        added: &[FileEntry],
    ) -> Result<()> {
        self.commit_entries(removed, added, false)
    }
}

/// The Iceberg table at `table_directory` under the database at `root_uri`,
/// as a process that opens it from its path sees it.
/// The worker pool the simulated outside writer reads manifests over, one
/// per test binary.
fn external_writer_pool() -> DataFlowDispatcher {
    static POOL: std::sync::OnceLock<Dispatch> = std::sync::OnceLock::new();
    POOL.get_or_init(|| Dispatch::spin_up(2, 64, None))
        .dispatcher()
        .clone()
}

/// The Iceberg table at `table_directory` under the database at `root_uri`,
/// at its current version, as a process that opens it from its path sees it.
fn open_directory_table(root_uri: &str, table_directory: &str) -> Result<Table> {
    let store: Arc<dyn ObjectStore> = open_store(root_uri)?.into();
    let location = TableLocation::new(Arc::clone(&store), ObjectPath::new(table_directory))?;
    let metadata_pointer: Arc<dyn MetadataPointer> =
        Arc::new(DirectoryMetadataPointer::new(location.clone()));
    let table_store = TableStore::over_datastore_store(table_directory, store)?;
    Ok(Table::load(
        &external_writer_pool(),
        metadata_pointer,
        location,
        table_store,
        &SchemaQualifiedTableName::in_default_schema(table_directory),
    )?)
}

/// Commit `bytes` as a new data file `file_name` of the Iceberg table at
/// `table_directory` under the database at `root_uri`, the way a writer in
/// another process would: load the table's current version from the store
/// and commit one version above it. `rows` is recorded as the file's row
/// count; no column bounds are recorded.
pub fn commit_external_append(
    root_uri: &str,
    table_directory: &str,
    file_name: &str,
    bytes: &[u8],
    rows: i64,
) -> Result<u64> {
    commit_from_outside(root_uri, table_directory, &[], file_name, bytes, rows)
}

/// Commit, as one version of the Iceberg table at `table_directory`, the
/// retirement of the live files named `removed` and the addition of `bytes`
/// as `file_name` holding `rows` rows: what a rewrite of the table's files
/// (compaction, a delete) commits. Returns the committed version.
pub fn commit_external_replace(
    root_uri: &str,
    table_directory: &str,
    removed: &[&str],
    file_name: &str,
    bytes: &[u8],
    rows: i64,
) -> Result<u64> {
    commit_from_outside(root_uri, table_directory, removed, file_name, bytes, rows)
}

/// Commit, from outside the datastore, the retirement of the live files
/// named `removed` and the addition of `bytes` as `file_name` holding
/// `rows` rows, reloading and retrying when another writer commits first, as
/// a writer in another process does. Returns the committed version.
fn commit_from_outside(
    root_uri: &str,
    table_directory: &str,
    removed: &[&str],
    file_name: &str,
    bytes: &[u8],
    rows: i64,
) -> Result<u64> {
    let mut table = open_directory_table(root_uri, table_directory)?;
    let path = ObjectPath::new(file_name);
    table
        .location()
        .store()
        .put(&table.location().data_file_key(&path), bytes)?;
    for _ in 0..COMMIT_RETRIES {
        let live = table.entries()?;
        let removed_entries: Vec<DataFileEntry> = removed
            .iter()
            .map(|name| {
                live.iter()
                    .find(|entry| entry.file.path.as_str() == *name)
                    .cloned()
                    .ok_or_else(|| Error::CommitConflict {
                        location: table_directory.to_string(),
                        file: name.to_string(),
                    })
            })
            .collect::<Result<_>>()?;
        let added = DataFileEntry {
            file: FileRef {
                path: path.clone(),
                size: bytes.len() as u64,
            },
            partition: None,
            stats: Some(Arc::new(FileStats {
                num_records: Some(rows),
                min_values: Default::default(),
                max_values: Default::default(),
                null_counts: Default::default(),
            })),
            sort_order_id: None,
            origin: None,
        };
        if let Some(committed) = table.commit(&removed_entries, vec![added])? {
            return Ok(metadata_location_version(committed.metadata_location())?);
        }
        table = open_directory_table(root_uri, table_directory)?;
    }
    Err(iceberg_tables::Error::CommitRacesExhausted {
        attempts: COMMIT_RETRIES,
    }
    .into())
}

/// Publish, as the next version of the Iceberg table at `table_directory`,
/// its current metadata with `change` applied (handed that metadata and a
/// builder on it): what another tool that alters the table commits.
fn publish_metadata_change(
    root_uri: &str,
    table_directory: &str,
    change: impl FnOnce(&TableMetadata, TableMetadataBuilder) -> TableMetadataBuilder,
) -> Result<()> {
    let table = open_directory_table(root_uri, table_directory)?;
    let builder = TableMetadata::clone(table.metadata())
        .into_builder(Some(table.metadata_location().to_string()));
    let changed = change(table.metadata(), builder)
        .build()
        .map_err(iceberg_tables::Error::spec)?
        .metadata;
    DirectoryMetadataPointer::new(table.location().clone())
        .publish(Some(table.current()), Arc::new(changed))?
        .ok_or(iceberg_tables::Error::CommitRacesExhausted { attempts: 1 })?;
    Ok(())
}

/// Point a tag named `tag` at the current snapshot of the Iceberg table at
/// `table_directory`.
pub fn tag_current_snapshot(root_uri: &str, table_directory: &str, tag: &str) -> Result<()> {
    publish_metadata_change(root_uri, table_directory, |metadata, builder| {
        let snapshot_id = metadata
            .current_snapshot_id()
            .expect("a tag points at a snapshot");
        builder
            .set_ref(
                tag,
                SnapshotReference::new(
                    snapshot_id,
                    SnapshotRetention::Tag {
                        max_ref_age_ms: None,
                    },
                ),
            )
            .expect("a tag on an existing snapshot is valid")
    })
}

/// Set the property `key` of the Iceberg table at `table_directory` to
/// `value`.
pub fn set_table_property(
    root_uri: &str,
    table_directory: &str,
    key: &str,
    value: &str,
) -> Result<()> {
    publish_metadata_change(root_uri, table_directory, |_metadata, builder| {
        builder
            .set_properties([(key.to_string(), value.to_string())].into())
            .expect("the property is not a reserved one")
    })
}

/// Write `file_name` under the metadata directory of the Iceberg table at
/// `table_directory` and name it in the table's metadata as the statistics
/// file of the current snapshot, the way an engine that computes statistics
/// does.
pub fn register_statistics_file(
    root_uri: &str,
    table_directory: &str,
    file_name: &str,
) -> Result<()> {
    let table = open_directory_table(root_uri, table_directory)?;
    let location = table.location();
    let bytes = b"statistics";
    location
        .store()
        .put(&location.metadata_key(file_name), bytes)?;
    let statistics_path = location.metadata_uri(file_name);
    publish_metadata_change(root_uri, table_directory, |metadata, builder| {
        builder.set_statistics(StatisticsFile {
            snapshot_id: metadata
                .current_snapshot_id()
                .expect("statistics describe a snapshot"),
            statistics_path,
            file_size_in_bytes: bytes.len() as i64,
            file_footer_size_in_bytes: 0,
            key_metadata: None,
            blob_metadata: Vec::new(),
        })
    })
}

/// Expire, as of `now_ms`, the snapshots of the Iceberg table at
/// `table_directory` and delete nothing: what a process that died between
/// committing an expiry and deleting for it leaves behind. Returns how many
/// snapshots it dropped.
pub fn expire_snapshots_without_deleting(
    root_uri: &str,
    table_directory: &str,
    now_ms: u64,
) -> Result<usize> {
    let table = open_directory_table(root_uri, table_directory)?;
    let expiry = table.expire_snapshots(now_ms as i64)?;
    Ok(expiry.map_or(0, |expiry| expiry.expired.snapshot_count()))
}

/// How many snapshots the current metadata of the Iceberg table at
/// `table_directory` retains.
pub fn retained_snapshots(root_uri: &str, table_directory: &str) -> Result<usize> {
    let table = open_directory_table(root_uri, table_directory)?;
    Ok(table.metadata().snapshots().len())
}

/// The metadata versions present under the Iceberg table at
/// `table_directory`, in ascending order.
pub fn metadata_versions(root_uri: &str, table_directory: &str) -> Result<Vec<u64>> {
    let table = open_directory_table(root_uri, table_directory)?;
    let mut versions: Vec<u64> = table
        .location()
        .store()
        .list(&table.location().metadata_dir())?
        .objects
        .iter()
        .filter_map(|object| parse_metadata_file_version(object.file.path.as_str()))
        .collect();
    versions.sort_unstable();
    Ok(versions)
}

/// The metadata versions the metadata log of the Iceberg table at
/// `table_directory` names, in the log's order.
pub fn logged_metadata_versions(root_uri: &str, table_directory: &str) -> Result<Vec<u64>> {
    let table = open_directory_table(root_uri, table_directory)?;
    Ok(table
        .metadata()
        .metadata_log()
        .iter()
        .filter_map(|entry| parse_metadata_file_version(object_name(&entry.metadata_file)))
        .collect())
}

/// The file name the URI of a metadata object ends in.
fn object_name(uri: &str) -> &str {
    uri.rsplit('/').next().unwrap_or(uri)
}

/// The version the Iceberg table's hint names.
pub fn hinted_version(root_uri: &str, table_directory: &str) -> Result<Option<u64>> {
    let table = open_directory_table(root_uri, table_directory)?;
    let key = table.location().metadata_key(VERSION_HINT_FILE);
    Ok(table
        .location()
        .store()
        .get(&key)?
        .map(|bytes| parse_hint(&bytes))
        .transpose()?)
}

/// The names of the live data files of the Iceberg table at
/// `table_directory`, as its current version names them, sorted.
pub fn live_data_file_names(root_uri: &str, table_directory: &str) -> Result<Vec<String>> {
    let table = open_directory_table(root_uri, table_directory)?;
    let mut names: Vec<String> = table
        .entries()?
        .iter()
        .map(|entry| entry.file.path.to_string())
        .collect();
    names.sort();
    Ok(names)
}
