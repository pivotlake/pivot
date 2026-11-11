//! Table-level test helpers layered on the reusable object-store harness:
//! opening a database from its URI, listing and scanning a table through
//! its handle, committing to it, and seeing or touching an Iceberg table's
//! metadata the way another process would.

pub use object_storage::test_support::*;

use std::ops::ControlFlow;
use std::sync::Arc;

use dispatch::{DataFlowDispatcher, Dispatch};
use iceberg_tables::commit::COMMIT_RETRIES;
use iceberg_tables::files::data_files::DataFileEntry;
use iceberg_tables::iceberg::spec::{
    SnapshotReference, SnapshotRetention, TableMetadata, TableMetadataBuilder,
};
use iceberg_tables::metadata::layout::{
    VERSION_HINT_FILE, parse_hint, parse_metadata_file_version,
};
use iceberg_tables::metadata::pivotlake_metadata_store::metadata_location_version;
use iceberg_tables::{MetadataStore, PivotlakeMetadataStore, Table, TableLocation};
use object_storage::{FileRef, ObjectPath, ObjectStore, open_store};
use parquet_engine::{FileStats, ParquetTable};
use planner::catalog::SchemaQualifiedTableName;
use planner::expression::Expression;

use crate::{
    CatalogTable, Error, FileEntry, PartitionValues, PivotlakeDatastore, Result, TableFormat,
};

impl PivotlakeDatastore {
    /// Open a persisted database rooted at `uri`, a local directory (or
    /// `file://…`), or a remote `s3://…` object store, as
    /// [`from_store`](Self::from_store) opens it over the store at `uri`,
    /// without background maintenance. A database with no manifest yet is
    /// created in the default table format.
    pub fn open(uri: &str, dispatcher: &DataFlowDispatcher) -> Result<Arc<Self>> {
        Self::open_with_format(uri, dispatcher, None)
    }

    /// [`open`](Self::open), naming the table format a database that does not
    /// exist yet is created in. An existing database keeps the format its
    /// manifest records; naming a different one is an error.
    pub fn open_with_format(
        uri: &str,
        dispatcher: &DataFlowDispatcher,
        format: Option<TableFormat>,
    ) -> Result<Arc<Self>> {
        let store: Arc<dyn ObjectStore> = open_store(uri)?.into();
        Self::from_store(store, dispatcher, None, format)
    }

    /// The current committed files of the table `name`, refreshed to its
    /// latest version first, so a commit by INSERT or compaction (in this
    /// process or another) is reflected. `None` if no such table exists.
    pub fn table_files(&self, name: &SchemaQualifiedTableName) -> Result<Option<Vec<FileRef>>> {
        let Some(mut table) = self.table_handle(name) else {
            return Ok(None);
        };
        let _ = table.refresh();
        Ok(Some(table.file_refs()?))
    }
}

impl CatalogTable {
    /// The table's current files as [`FileRef`]s, in the order a walk visits
    /// them.
    pub fn file_refs(&self) -> Result<Vec<FileRef>> {
        let mut files = Vec::new();
        self.walk_files(&mut |walked| {
            files.push(walked.file.clone());
            ControlFlow::Continue(())
        })?;
        Ok(files)
    }

    /// The table's current files, partition by partition, in the order a
    /// walk visits them, each partition named as the walk names it.
    fn list_files_by_partition(&self) -> Result<Vec<(Vec<u8>, Vec<FileRef>)>> {
        let mut partitions: Vec<(Vec<u8>, Vec<FileRef>)> = Vec::new();
        self.walk_files(&mut |walked| {
            match partitions.last_mut() {
                Some((partition, files)) if partition.as_slice() == walked.partition => {
                    files.push(walked.file.clone());
                }
                _ => partitions.push((walked.partition.to_vec(), vec![walked.file.clone()])),
            }
            ControlFlow::Continue(())
        })?;
        Ok(partitions)
    }

    /// Every live data file at this version with its partition tuple and its
    /// bounds on the sort columns, in the order a walk visits them.
    pub fn live_entries(&self) -> Result<Vec<FileEntry>> {
        let mut entries = Vec::new();
        for (partition, files) in self.list_files_by_partition()? {
            let paths: Vec<ObjectPath> = files.into_iter().map(|file| file.path).collect();
            for entry in self.find_files(&partition, &paths)? {
                entries.push(entry.expect("a walked file is live in the version walked"));
            }
        }
        Ok(entries)
    }

    /// Each committed file paired with the typed partition tuple recorded for
    /// it (or `None`), reflecting the current committed version.
    pub fn file_partitions(&self) -> Result<Vec<(ObjectPath, Option<PartitionValues>)>> {
        Ok(self
            .live_entries()?
            .into_iter()
            .map(|entry| (entry.file.path, entry.partition))
            .collect())
    }

    /// The scan view a query over `filters` reads, without its nullability
    /// flags: see `CatalogTable::load_scan_plan`.
    pub fn plan_scan(&self, filters: &[Expression]) -> Result<Arc<ParquetTable>> {
        Ok(self.load_scan_plan(filters)?.parquet)
    }

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
    let location = TableLocation::in_datastore_store(
        table_directory,
        store,
        ObjectPath::new(table_directory),
    )?;
    let metadata_store: Arc<dyn MetadataStore> =
        Arc::new(PivotlakeMetadataStore::new(location.clone()));
    Ok(Table::load(
        &external_writer_pool(),
        metadata_store,
        location,
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
            first_row_id: None,
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
    PivotlakeMetadataStore::new(table.location().clone())
        .commit_metadata(Some(table.current()), Arc::new(changed))?
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
