//! Table-level test helpers layered on the reusable object-store harness:
//! opening a database from its URI, listing and scanning a table through
//! its handle, committing to it, and seeing or touching an Iceberg table's
//! metadata the way another process would.

pub use object_storage::test_support::*;

use std::collections::BTreeMap;
use std::sync::Arc;

use dispatch::{DataFlowDispatcher, Dispatch};
use iceberg_tables::commit::COMMIT_RETRIES;
use iceberg_tables::files::data_files::AddedDataFile;
use iceberg_tables::iceberg::spec::{
    SnapshotReference, SnapshotRetention, TableMetadata, TableMetadataBuilder,
};
use iceberg_tables::table_metadata_cache::{
    DEFAULT_TABLE_METADATA_CACHE_BYTES, TableCacheSlot, TableMetadataCache,
};
use iceberg_tables::{MetadataStore, Partition, Table, TableLocation};
use object_storage::{FileRef, ObjectPath, ObjectStore, open_store};
use parquet_engine::{FileStats, ParquetTable};
use planner::catalog::SchemaQualifiedTableName;
use planner::expression::Expression;

use crate::compact::sort_key::compare_file_order;
use crate::file_entry::PartitionValues;
use crate::iceberg::directory_metadata_store::{
    DirectoryMetadataStore, VERSION_HINT_FILE, metadata_location_version, parse_hint,
    parse_metadata_file_version,
};
use crate::table::TableFile;
use crate::table_format::{CompactionFile, PartitionHandle, RemovedFile};
use crate::{CatalogTable, Error, FileEntry, PivotlakeDatastore, Result, TableFormat};

/// The files of one partition, beside the partition, each as the record
/// the compacter takes of it.
type PartitionRecords = (PartitionHandle, Vec<Arc<dyn CompactionFile>>);

impl PivotlakeDatastore {
    /// Whether this datastore defines a schema named `schema`.
    pub fn contains_schema(&self, schema: &str) -> bool {
        self.tables_index.read().unwrap().contains_schema(schema)
    }

    /// Whether a table named `name` exists in the datastore.
    pub fn contains_table(&self, name: &SchemaQualifiedTableName) -> bool {
        self.tables_index.read().unwrap().contains_table(name)
    }

    /// Open a persisted database rooted at `uri`, a local directory (or
    /// `file://…`), or a remote `s3://…` object store, as
    /// [`from_store`](Self::from_store) opens it over the store at `uri`,
    /// without background maintenance, with a table metadata cache of its
    /// own. A database with no manifest yet is created in the default
    /// table format.
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
        let table_metadata_cache = TableMetadataCache::new(DEFAULT_TABLE_METADATA_CACHE_BYTES);
        Self::from_store(store, dispatcher, None, format, table_metadata_cache)
    }

    /// A clone of the named table's current state, for a caller that reads it
    /// (a compaction candidate scan, introspection) or refreshes it to the
    /// latest committed version. A copy drifts as soon as anything commits, and
    /// it is never a commit base: writes go through
    /// `commit_to_table`. `None` if no such table
    /// exists.
    pub fn table_handle(&self, name: &SchemaQualifiedTableName) -> Option<CatalogTable> {
        self.tables_index
            .read()
            .unwrap()
            .get_table_by_name(name)
            .cloned()
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

    /// [`table_files`](Self::table_files), each partition's in the order
    /// maintenance holds a partition's files in
    /// ([`CatalogTable::file_refs_in_maintenance_order`]).
    pub fn table_files_in_maintenance_order(
        &self,
        name: &SchemaQualifiedTableName,
    ) -> Result<Option<Vec<FileRef>>> {
        let Some(mut table) = self.table_handle(name) else {
            return Ok(None);
        };
        let _ = table.refresh();
        Ok(Some(table.file_refs_in_maintenance_order()?))
    }
}

impl CatalogTable {
    /// The table's current files as [`FileRef`]s, partition by partition,
    /// each partition's in the order the table's format lists them.
    pub fn file_refs(&self) -> Result<Vec<FileRef>> {
        let partitions = self.list_files_by_partition()?;
        let files = partitions.iter().flat_map(|(_, files)| files);
        Ok(files.map(|file| file.file().clone()).collect())
    }

    /// The table's current files, partition by partition, each partition's
    /// in the order maintenance holds a partition's files in: the records
    /// the compacter takes of them, ordered as it orders them, by where
    /// each file starts on the table's sort key, then by its commit, then
    /// by its path.
    pub fn file_refs_in_maintenance_order(&self) -> Result<Vec<FileRef>> {
        let mut ordered = Vec::new();
        for (_, mut files) in self.list_files_by_partition()? {
            files.sort_by(|left, right| compare_file_order(left.as_ref(), right.as_ref()));
            ordered.extend(files.iter().map(|file| file.file().clone()));
        }
        Ok(ordered)
    }

    /// The table's current files, partition by partition in the order the
    /// table's format gives its partitions, each as the record the
    /// compacter takes of it,
    /// read through the manifests the compacter loads them in: a
    /// partition's files stand manifest after manifest, and within a
    /// manifest as the manifest lists them.
    fn list_files_by_partition(&self) -> Result<Vec<PartitionRecords>> {
        let manifests = self.load_manifest_files(&self.list_manifest_ids())?;
        let mut partitions: BTreeMap<PartitionHandle, Vec<Arc<dyn CompactionFile>>> =
            BTreeMap::new();
        for listed in manifests.into_iter().flatten() {
            let files = partitions.entry(listed.partition).or_default();
            files.extend(listed.files);
        }
        Ok(partitions.into_iter().collect())
    }

    /// The partition the live file at each of `paths` sits in, in the
    /// order given. A path this version has no live file at is a commit
    /// conflict: another writer removed the file first.
    fn find_partitions_holding(&self, paths: &[ObjectPath]) -> Result<Vec<PartitionHandle>> {
        let partitions = self.list_files_by_partition()?;
        let find_partition_holding = |path: &ObjectPath| {
            let holding = partitions.iter().find(|(_, files)| {
                let mut files = files.iter();
                files.any(|file| file.file().path == *path)
            });
            let (partition, _) = holding.ok_or_else(|| self.conflict_at(path))?;
            Ok(partition.clone())
        };
        paths.iter().map(find_partition_holding).collect()
    }

    /// Every live data file at this version with its partition tuple, as a
    /// lookup by its path finds it, and with its row count and its bounds
    /// on the sort columns as the record the compacter takes of it holds
    /// them, a partition at a time, in the order the table's format gives
    /// its partitions.
    pub fn live_entries(&self) -> Result<Vec<FileEntry>> {
        let mut entries = Vec::new();
        for (partition, files) in self.list_files_by_partition()? {
            let paths: Vec<ObjectPath> =
                files.iter().map(|file| file.file().path.clone()).collect();
            let found = self.table.find_file_entries(&partition, &paths)?;
            for (file, entry) in files.iter().zip(found) {
                let mut entry = entry.expect("a listed file is live in the version that lists it");
                entry.stats = file.build_entry().stats;
                entries.push(entry);
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
    /// flags: see `CatalogTable::load_scan_selection`.
    pub fn select_scan(&self, filters: &[Expression]) -> Result<Arc<ParquetTable>> {
        Ok(self.load_scan_selection(filters)?.row_groups)
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

    /// Commit bare `entries` whose footers are not yet in hand: read the
    /// added files' footers, then commit the files with them as
    /// [`commit_files`](Self::commit_files) does, an entry that carries no
    /// stats recording those its footer holds. A writer of the datastore
    /// (INSERT, compaction) already holds the footers and commits with
    /// `commit_files`.
    fn commit_entries(
        &mut self,
        removed: &[ObjectPath],
        added: &[FileEntry],
        data_change: bool,
    ) -> Result<()> {
        let footers = self.fetch_footers(added)?;
        let added: Vec<TableFile> = added
            .iter()
            .zip(footers)
            .map(|(entry, footer)| TableFile::new(entry.clone(), footer.row_groups))
            .collect();
        if removed.is_empty() {
            return self.commit_files(&[], added, data_change);
        }
        let removed_partitions = self.find_partitions_holding(removed)?;
        let removed: Vec<RemovedFile<'_>> = removed_partitions
            .iter()
            .zip(removed)
            .map(|(partition, path)| RemovedFile { partition, path })
            .collect();
        self.commit_files(&removed, added, data_change)
    }

    /// Read the footers of `entries`, files under this table's location that
    /// its metadata does not hold yet, one per entry in order, each reconciled
    /// by name against the declared columns.
    fn fetch_footers(&self, entries: &[FileEntry]) -> Result<Vec<parquet_engine::FileRowGroups>> {
        let store = self.store();
        let location = ObjectPath::new(self.location());
        let to_fetch: Vec<object_storage::DataFile> = entries
            .iter()
            .map(|entry| entry.file.clone().into_data_file(store.as_ref(), &location))
            .collect::<object_storage::Result<_>>()?;
        Ok(parquet_engine::load_file_row_groups(
            &self.dispatcher,
            &to_fetch,
            parquet_engine::TableColumns::by_name(self.columns()),
        )?)
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
/// at its current version, as a process that opens it from its path sees
/// it, with a table metadata cache of its own.
fn open_directory_table(root_uri: &str, table_directory: &str) -> Result<Table> {
    let table_metadata_cache = TableMetadataCache::new(DEFAULT_TABLE_METADATA_CACHE_BYTES);
    open_directory_table_over(root_uri, table_directory, table_metadata_cache)
}

/// [`open_directory_table`], the table keeping what is read of it in
/// `table_metadata_cache`.
fn open_directory_table_over(
    root_uri: &str,
    table_directory: &str,
    table_metadata_cache: Arc<TableMetadataCache>,
) -> Result<Table> {
    let store: Arc<dyn ObjectStore> = open_store(root_uri)?.into();
    let name = SchemaQualifiedTableName::in_default_schema(table_directory);
    let location =
        TableLocation::in_datastore_store(&name, store, ObjectPath::new(table_directory))?;
    let metadata_store: Arc<dyn MetadataStore> =
        Arc::new(DirectoryMetadataStore::new(location.clone()));
    // The database knows a table it created by the UUID the table's
    // metadata carries.
    let metadata_file = metadata_store.load_latest(&name)?;
    let cache_slot = TableCacheSlot {
        cache: table_metadata_cache,
        table_id: metadata_file.metadata.uuid(),
    };
    Ok(Table::at(
        &external_writer_pool(),
        metadata_store,
        location,
        &name,
        cache_slot,
        metadata_file,
        None,
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
        .put(&table.location().resolve_table_path(&path), bytes)?;
    let removed_paths: Vec<ObjectPath> =
        removed.iter().map(|name| ObjectPath::new(*name)).collect();
    let conflict_at = |file: &str| Error::CommitConflict {
        location: table_directory.to_string(),
        file: file.to_string(),
    };
    for _ in 0..COMMIT_RETRIES {
        let removed_partitions = find_partitions_holding(&table, table_directory, &removed_paths)?;
        let removed_files: Vec<iceberg_tables::commit::RemovedFile<'_>> = removed_partitions
            .iter()
            .zip(&removed_paths)
            .map(|(partition, path)| iceberg_tables::commit::RemovedFile { partition, path })
            .collect();
        let added = AddedDataFile {
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
        };
        let committed = table.commit(&removed_files, vec![added]);
        match committed {
            Ok(Some(committed)) => {
                return Ok(metadata_location_version(committed.metadata_location())?);
            }
            Ok(None) => {}
            Err(iceberg_tables::Error::RemovedFileNotLive { file, .. }) => {
                return Err(conflict_at(&file));
            }
            Err(error) => return Err(error.into()),
        }
        table = open_directory_table(root_uri, table_directory)?;
    }
    Err(iceberg_tables::Error::CommitRacesExhausted {
        attempts: COMMIT_RETRIES,
    }
    .into())
}

/// The partition the live file of `table` at each of `paths` sits in, in
/// the order given, with no manifest read when there are none. A path
/// `table` has no live file at is a commit conflict at `table_directory`:
/// another writer removed the file first.
fn find_partitions_holding(
    table: &Table,
    table_directory: &str,
    paths: &[ObjectPath],
) -> Result<Vec<Arc<Partition>>> {
    if paths.is_empty() {
        return Ok(Vec::new());
    }
    let manifests = table.load_manifest_files(&table.list_data_manifest_paths())?;
    let listed_partitions: Vec<_> = manifests.into_iter().flatten().collect();
    let find_partition_holding = |path: &ObjectPath| {
        let holding = listed_partitions.iter().find(|listed| {
            let mut files = listed.files.iter();
            files.any(|file| file.file().path == *path)
        });
        let listed = holding.ok_or_else(|| Error::CommitConflict {
            location: table_directory.to_string(),
            file: path.to_string(),
        })?;
        Ok(Arc::clone(&listed.partition))
    };
    paths.iter().map(find_partition_holding).collect()
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
    DirectoryMetadataStore::new(table.location().clone())
        .commit_metadata(Some(table.metadata_file()), Arc::new(changed))?;
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

/// How many data manifests the current version of the Iceberg table at
/// `table_directory` lists.
pub fn count_data_manifests(root_uri: &str, table_directory: &str) -> Result<usize> {
    let table = open_directory_table(root_uri, table_directory)?;
    Ok(table.manifest_list().data_manifests().count())
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

/// The paths the manifest list of the current version of the Iceberg table
/// at `table_directory` gives its data manifests.
pub fn list_data_manifest_paths(root_uri: &str, table_directory: &str) -> Result<Vec<String>> {
    let table = open_directory_table(root_uri, table_directory)?;
    let manifests = table.manifest_list().data_manifests();
    Ok(manifests
        .map(|manifest| manifest.manifest_path.clone())
        .collect())
}

/// How many parts `table_metadata_cache`, the cache of the datastore that
/// serves the database at `root_uri`, holds, between them, of the manifests
/// at `manifests` of the Iceberg table at `table_directory`, each by the
/// path a manifest list gives it.
pub fn count_cached_manifest_parts(
    table_metadata_cache: &Arc<TableMetadataCache>,
    root_uri: &str,
    table_directory: &str,
    manifests: &[String],
) -> Result<usize> {
    let table =
        open_directory_table_over(root_uri, table_directory, Arc::clone(table_metadata_cache))?;
    let cached = manifests.iter();
    Ok(cached
        .map(|manifest| table.list_cached_manifest_parts(manifest).len())
        .sum())
}

/// The path of every live data file of the Iceberg table at
/// `table_directory` with the partition values its current version records
/// for it, sorted by path.
pub fn live_data_file_partitions(
    root_uri: &str,
    table_directory: &str,
) -> Result<Vec<(ObjectPath, Option<PartitionValues>)>> {
    let table = open_directory_table(root_uri, table_directory)?;
    let mut partitions: Vec<(ObjectPath, Option<PartitionValues>)> = table
        .entries()?
        .into_iter()
        .map(|entry| (entry.file.path, entry.partition))
        .collect();
    partitions.sort_by(|(left, _), (right, _)| left.as_str().cmp(right.as_str()));
    Ok(partitions)
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
