//! The Iceberg format: per table, numbered metadata files under
//! `<directory>/metadata/`, each version landed with a conditional write
//! that admits exactly one writer, so writers in several processes share a
//! table with no catalog service. Readable by any Iceberg reader from the
//! table's path.
//!
//! A table's live files, with their statistics and partition tuples, are
//! held in memory from the current snapshot's manifests; their footers are
//! read when a scan, a merge or a layout measurement first needs them and
//! remembered for the life of the version.

pub(crate) mod commit;
pub(crate) mod data_files;
mod error;
pub(crate) mod layout;
pub(crate) mod load;
pub(crate) mod manifest_writer;
pub(crate) mod retention;
pub(crate) mod schema;
pub(crate) mod state;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use datastore_iceberg::TableStore;
use dispatch::DataFlowDispatcher;
use object_storage::{ObjectPath, ObjectStore};
use parquet_engine::{RowGroupMetadata, TableColumns};
use planner::catalog::{Column, SchemaQualifiedTableName};
use tracing::{info, warn};
use uuid::Uuid;

use self::commit::{COMMIT_RETRIES, TableCreation};
use self::data_files::{DataFileEntry, EntryContext};
pub(crate) use self::error::{Error, Result};
use self::layout::{TableLocation, VERSION_HINT_FILE, parse_metadata_file_version};
use self::manifest_writer::{read_manifest, read_manifest_list};
use self::retention::snapshot_retention;
use self::state::TableState;
use crate::file_entry::{FileEntry, constant_partition};
use crate::format::{Format, FormatTable};
use crate::table::{TableFile, TableStorage};

/// The Iceberg backend of one database.
pub(crate) struct IcebergFormat {
    store: Arc<dyn ObjectStore>,
    dispatcher: DataFlowDispatcher,
}

impl IcebergFormat {
    pub(crate) fn new(store: Arc<dyn ObjectStore>, dispatcher: DataFlowDispatcher) -> Self {
        Self { store, dispatcher }
    }
}

impl Format for IcebergFormat {
    /// Write the table's first metadata version, holding the adopted files as
    /// its first snapshot when there are any. An adopted file of a partitioned
    /// table must sit in one partition, which its statistics prove; an
    /// Iceberg tuple has no "unknown" value, so one that does not is refused.
    fn create_table(
        &self,
        directory: &ObjectPath,
        name: &SchemaQualifiedTableName,
        id: Uuid,
        columns: Vec<Column>,
        partition_by: Vec<String>,
        sort_by: Vec<String>,
        adopted: Vec<TableFile>,
    ) -> crate::Result<Arc<dyn FormatTable>> {
        let location = TableLocation::new(self.store.clone(), directory.clone())?;
        let entries = adopted
            .iter()
            .map(|file| adopted_entry(file, &partition_by))
            .collect::<Result<Vec<_>>>()?;
        let state = commit::initialize_table(
            &location,
            &name.to_string(),
            id,
            TableCreation {
                columns: &columns,
                partition_by: &partition_by,
                sort_by: &sort_by,
                adopted: entries,
            },
            commit::now_unix_ms(),
        )?;
        let mut table =
            IcebergTable::new(name.clone(), state, self.dispatcher.clone(), Footers::new())?;
        table.remember(&adopted);
        Ok(Arc::new(table))
    }

    /// Find the table's current metadata version, read it and decode the
    /// manifests of its current snapshot. No footer is read until a scan
    /// needs one.
    fn load_table(
        &self,
        directory: &ObjectPath,
        name: &SchemaQualifiedTableName,
        _id: Uuid,
    ) -> crate::Result<Arc<dyn FormatTable>> {
        let location = TableLocation::new(self.store.clone(), directory.clone())?;
        let state = load::load_table(&location, &name.to_string())?;
        Ok(Arc::new(IcebergTable::new(
            name.clone(),
            state,
            self.dispatcher.clone(),
            Footers::new(),
        )?))
    }
}

/// The entry a new table adopts `file` under: its statistics (from the entry,
/// else derived from the footers) and, for a partitioned table, the constant
/// partition tuple its statistics prove.
fn adopted_entry(file: &TableFile, partition_by: &[String]) -> Result<DataFileEntry> {
    let stats = file
        .entry
        .stats
        .clone()
        .unwrap_or_else(|| Arc::new(parquet_engine::aggregate_file_stats(&file.row_groups)));
    let partition = if partition_by.is_empty() {
        None
    } else {
        Some(constant_partition(&stats, partition_by).ok_or_else(|| {
            Error::PartitionNotConstant {
                file: file.entry.file.path.to_string(),
            }
        })?)
    };
    Ok(DataFileEntry {
        file: file.entry.file.clone(),
        partition,
        stats: Some(stats),
        origin: None,
    })
}

/// A manifest's entry as the shared datastore holds it: the same identity,
/// tuple and statistics, without where the manifest records it.
fn to_file_entry(entry: &DataFileEntry) -> FileEntry {
    FileEntry {
        file: entry.file.clone(),
        partition: entry.partition.clone(),
        stats: entry.stats.clone(),
    }
}

/// The footers read so far for one version's files, by path.
type Footers = HashMap<ObjectPath, Vec<Arc<RowGroupMetadata>>>;

/// One Iceberg table at one metadata version: the metadata and manifests it
/// was loaded from, its live files, and the footers read for them so far.
struct IcebergTable {
    /// The qualified name the table is indexed under, for messages and the
    /// store's file locating.
    name: SchemaQualifiedTableName,
    state: Arc<TableState>,
    /// `state.entries` as the shared datastore reads them.
    entries: Vec<FileEntry>,
    /// Locates a data file by its full URI under the database's store, for
    /// footer reads.
    table_store: TableStore,
    /// The declared columns keyed by the field ids the files carry.
    table_columns: TableColumns,
    dispatcher: DataFlowDispatcher,
    /// The footers read for this version so far: a scan over many small
    /// files reads each footer once per version, however many queries bind
    /// it.
    footers: Mutex<Footers>,
}

impl IcebergTable {
    fn new(
        name: SchemaQualifiedTableName,
        state: TableState,
        dispatcher: DataFlowDispatcher,
        footers: Footers,
    ) -> Result<Self> {
        let table_name = name.to_string();
        let table_store =
            TableStore::over_datastore_store(&table_name, Arc::clone(state.location.store()))?;
        let metadata = &state.current.metadata;
        let logical_types = datastore_iceberg::read_logical_types(metadata.properties())?;
        let table_columns = datastore_iceberg::table_columns(
            &table_name,
            metadata.current_schema(),
            &logical_types,
        )?;
        let entries = state.entries.iter().map(to_file_entry).collect();
        Ok(Self {
            name,
            state: Arc::new(state),
            entries,
            table_store,
            table_columns,
            dispatcher,
            footers: Mutex::new(footers),
        })
    }

    /// This table at the version `state` describes, keeping the footers read
    /// for the files still live.
    fn at_state(&self, state: TableState) -> Result<Self> {
        let live: HashSet<&ObjectPath> =
            state.entries.iter().map(|entry| &entry.file.path).collect();
        let footers = self
            .footers
            .lock()
            .unwrap()
            .iter()
            .filter(|(path, _)| live.contains(path))
            .map(|(path, row_groups)| (path.clone(), row_groups.clone()))
            .collect();
        Self::new(self.name.clone(), state, self.dispatcher.clone(), footers)
    }

    /// Keep the footers a writer holds for `files`, so a scan after the
    /// commit does not read them back.
    fn remember(&mut self, files: &[TableFile]) {
        let footers = self.footers.get_mut().unwrap();
        for file in files {
            footers.insert(file.entry.file.path.clone(), file.row_groups.clone());
        }
    }

    /// This version's state with `removed` dropped and `added` adopted, at the
    /// same metadata.
    fn state_with_swap(&self, removed: &[ObjectPath], added: &[TableFile]) -> TableState {
        let state = &self.state;
        let mut entries: Vec<DataFileEntry> = state
            .entries
            .iter()
            .filter(|entry| !removed.contains(&entry.file.path))
            .cloned()
            .collect();
        entries.extend(added.iter().map(uncommitted_entry));
        TableState {
            location: state.location.clone(),
            current: Arc::clone(&state.current),
            columns: state.columns.clone(),
            partition_by: state.partition_by.clone(),
            sort_by: state.sort_by.clone(),
            manifest_list: state.manifest_list.clone(),
            manifest_list_size: state.manifest_list_size,
            manifests: state.manifests.clone(),
            entries,
        }
    }

    /// The metadata objects present under the table's metadata directory
    /// that this datastore writes (manifests, manifest lists, metadata
    /// files), each by name with its modification time. Anything else there
    /// (the hint, another tool's files) is not listed and so never swept.
    fn list_metadata_objects(&self) -> Result<Vec<(String, u64)>> {
        Ok(self
            .state
            .location
            .store()
            .list(&self.state.location.metadata_dir())?
            .objects
            .into_iter()
            .filter(|object| {
                let name = object.file.path.as_str();
                name != VERSION_HINT_FILE
                    && (name.ends_with(".avro") || parse_metadata_file_version(name).is_some())
            })
            .map(|object| (object.file.path.to_string(), object.modified_unix_ms))
            .collect())
    }
}

/// The entry a commit records for a file a writer just uploaded.
fn uncommitted_entry(file: &TableFile) -> DataFileEntry {
    DataFileEntry {
        file: file.entry.file.clone(),
        partition: file.entry.partition.clone(),
        stats: Some(
            file.entry.stats.clone().unwrap_or_else(|| {
                Arc::new(parquet_engine::aggregate_file_stats(&file.row_groups))
            }),
        ),
        origin: None,
    }
}

impl FormatTable for IcebergTable {
    fn version(&self) -> u64 {
        self.state.current.version
    }

    fn columns(&self) -> &[Column] {
        &self.state.columns
    }

    fn partition_by(&self) -> &[String] {
        &self.state.partition_by
    }

    fn sort_by(&self) -> &[String] {
        &self.state.sort_by
    }

    fn entries(&self) -> &[FileEntry] {
        &self.entries
    }

    /// Every Iceberg field this datastore declares is optional, so every
    /// column can hold NULLs.
    fn nullability(&self) -> Vec<bool> {
        vec![true; self.state.columns.len()]
    }

    /// The field id the table's schema gives each declared column, in column
    /// order: what the files it writes stamp on their columns.
    fn field_ids(&self) -> Option<Vec<i32>> {
        let schema = self.state.current.metadata.current_schema();
        Some(
            self.state
                .columns
                .iter()
                .map(|column| {
                    schema
                        .field_id_by_name(&column.name)
                        .expect("every declared column is a field of the table's schema")
                })
                .collect(),
        )
    }

    /// Read over the pool the footers not yet held, once; answer the rest
    /// from memory.
    fn row_groups(&self, files: &[&FileEntry]) -> crate::Result<Vec<Vec<Arc<RowGroupMetadata>>>> {
        let mut footers = self.footers.lock().unwrap();
        let missing: Vec<&FileEntry> = files
            .iter()
            .copied()
            .filter(|entry| !footers.contains_key(&entry.file.path))
            .collect();
        if !missing.is_empty() {
            let data_files = missing
                .iter()
                .map(|entry| {
                    let uri = self.state.location.uri_for(&entry.file.path)?;
                    Ok(self.table_store.data_file(&uri, entry.file.size)?)
                })
                .collect::<Result<Vec<_>>>()?;
            let loaded = parquet_engine::load_file_row_groups(
                &self.dispatcher,
                &data_files,
                self.table_columns.clone(),
            )?;
            for (entry, footer) in missing.iter().zip(loaded) {
                footers.insert(entry.file.path.clone(), footer.row_groups);
            }
        }
        Ok(files
            .iter()
            .map(|entry| {
                footers
                    .get(&entry.file.path)
                    .expect("every requested footer was held or just read")
                    .clone()
            })
            .collect())
    }

    /// One short listing from the version held; a newer version is read with
    /// the manifests already decoded carried over.
    fn refresh(&self) -> crate::Result<Option<Arc<dyn FormatTable>>> {
        let Some(state) = load::refresh_table(&self.state, &self.name.to_string())? else {
            return Ok(None);
        };
        Ok(Some(Arc::new(self.at_state(state)?)))
    }

    fn commit(
        &self,
        removed: &[FileEntry],
        added: &[TableFile],
        data_change: bool,
        now_ms: u64,
    ) -> crate::Result<Option<Arc<dyn FormatTable>>> {
        let removed_entries = removed
            .iter()
            .map(|removed| {
                self.state
                    .entries
                    .iter()
                    .find(|entry| entry.file.path == removed.file.path)
                    .cloned()
                    .ok_or_else(|| Error::UncommittedRemoval {
                        file: removed.file.path.to_string(),
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let added_entries: Vec<DataFileEntry> = added.iter().map(uncommitted_entry).collect();
        let now_ms = i64::try_from(now_ms).unwrap_or(i64::MAX);
        let Some(state) = commit::commit_file_changes(
            &self.state,
            &removed_entries,
            added_entries,
            data_change,
            now_ms,
        )?
        else {
            return Ok(None);
        };
        let mut table = self.at_state(state)?;
        table.remember(added);
        Ok(Some(Arc::new(table)))
    }

    fn with_files_swapped(
        &self,
        removed: &[ObjectPath],
        added: Vec<TableFile>,
    ) -> Arc<dyn FormatTable> {
        let state = self.state_with_swap(removed, &added);
        let mut table = self
            .at_state(state)
            .expect("a swap keeps the metadata this version was built from");
        table.remember(&added);
        Arc::new(table)
    }

    /// The table's own retention property, or the default.
    fn deleted_file_retention(&self) -> crate::Result<Duration> {
        Ok(snapshot_retention(&self.state.current.metadata)?)
    }

    /// Expire every snapshot that has outlived the retention window, as one
    /// metadata-only commit. A writer landing first expires them itself, so
    /// a lost race only reloads and looks again.
    fn maintain(&self, now_ms: u64) -> crate::Result<Option<Arc<dyn FormatTable>>> {
        let now_ms = i64::try_from(now_ms).unwrap_or(i64::MAX);
        let mut refreshed: Option<IcebergTable> = None;
        for _ in 0..COMMIT_RETRIES {
            let table: &IcebergTable = refreshed.as_ref().unwrap_or(self);
            if let Some(state) = commit::expire_snapshots(&table.state, now_ms)? {
                return Ok(Some(Arc::new(table.at_state(state)?)));
            }
            let Some(state) = load::refresh_table(&table.state, &table.name.to_string())? else {
                return Ok(None);
            };
            let next = table.at_state(state)?;
            refreshed = Some(next);
        }
        Ok(None)
    }

    /// Bring a lagging hint up to date, then delete every object under the
    /// table that no retained snapshot or metadata-log entry reaches and that
    /// is older than the retention window. The hint is advanced before
    /// anything is deleted, so a reader never follows it to a version this
    /// sweep removes.
    fn sweep(
        &self,
        name: &SchemaQualifiedTableName,
        storage: &TableStorage,
        now_ms: u64,
    ) -> crate::Result<()> {
        let location = &self.state.location;
        if load::read_version_hint(location)?.is_none_or(|hinted| hinted < self.version()) {
            commit::write_version_hint(location, self.version())?;
        }

        let retention_ms =
            u64::try_from(self.deleted_file_retention()?.as_millis()).unwrap_or(u64::MAX);
        let cutoff = now_ms.saturating_sub(retention_ms);
        let reachable = reachable_objects(self)?;

        let mut deleted = 0u64;
        let mut first_error: Option<crate::Error> = None;
        for (path, modified_ms) in storage.list_data_files()? {
            if reachable.data_paths.contains(&path) || modified_ms > cutoff {
                continue;
            }
            match storage.delete_data_file(&path) {
                Ok(()) => deleted += 1,
                Err(error) => {
                    warn!(table = %name, file = %path, error = %error, "vacuum: deleting file failed");
                    first_error.get_or_insert(error);
                }
            }
        }
        for (object, modified_ms) in self.list_metadata_objects()? {
            if reachable.metadata_names.contains(&object) || modified_ms > cutoff {
                continue;
            }
            match location.store().delete(&location.metadata_key(&object)) {
                Ok(()) => deleted += 1,
                Err(error) => {
                    warn!(table = %name, object = %object, error = %error, "vacuum: deleting metadata object failed");
                    first_error.get_or_insert(error.into());
                }
            }
        }
        if deleted > 0 {
            info!(table = %name, objects = deleted, "vacuumed objects");
        }
        first_error.map_or(Ok(()), Err)
    }
}

/// What a table's retained snapshots and metadata log reach: the data files
/// some retained snapshot names, and the metadata objects (manifests,
/// manifest lists, metadata files) that are still part of the table.
struct ReachableObjects {
    data_paths: HashSet<ObjectPath>,
    metadata_names: HashSet<String>,
}

/// Walk every retained snapshot of `table` from its manifest list down to
/// its data files. The current snapshot's manifests are already decoded;
/// older snapshots' are read, each once however many snapshots share it.
fn reachable_objects(table: &IcebergTable) -> Result<ReachableObjects> {
    let state = &table.state;
    let metadata = &state.current.metadata;
    let table_name = table.name.to_string();
    let location = &state.location;
    let ctx = EntryContext {
        schema: metadata.current_schema(),
        partition_spec: metadata.default_partition_spec(),
        columns: &state.columns,
        location,
    };
    let mut data_paths = HashSet::new();
    let mut metadata_names = HashSet::new();
    metadata_names.insert(object_name(&state.current.metadata_location));
    for entry in metadata.metadata_log() {
        metadata_names.insert(object_name(&entry.metadata_file));
    }
    let mut manifests_seen: HashSet<String> = HashSet::new();
    for snapshot in metadata.snapshots() {
        let list_uri = snapshot.manifest_list();
        metadata_names.insert(object_name(list_uri));
        let manifest_files = if Some(snapshot.snapshot_id()) == metadata.current_snapshot_id() {
            state.manifest_list.clone()
        } else {
            let bytes = read_object(location, &table_name, list_uri)?;
            read_manifest_list(&table_name, list_uri, &bytes)?
        };
        for file in manifest_files {
            metadata_names.insert(object_name(&file.manifest_path));
            if !manifests_seen.insert(file.manifest_path.clone()) {
                continue;
            }
            let entries = match state.manifests.get(&file.manifest_path) {
                Some(loaded) => loaded.entries.clone(),
                None => {
                    let bytes = read_object(location, &table_name, &file.manifest_path)?;
                    read_manifest(&table_name, &file, &bytes, &ctx)?.entries
                }
            };
            data_paths.extend(entries.into_iter().map(|entry| entry.file.path));
        }
    }
    Ok(ReachableObjects {
        data_paths,
        metadata_names,
    })
}

fn read_object(location: &TableLocation, table_name: &str, uri: &str) -> Result<Vec<u8>> {
    let path = location.file_path_for_uri(uri)?;
    location
        .store()
        .get(&location.data_file_key(&path))?
        .ok_or_else(|| Error::MissingMetadataObject {
            table: table_name.to_string(),
            path: uri.to_string(),
        })
}

/// The file name a metadata object's URI ends in.
fn object_name(uri: &str) -> String {
    uri.rsplit('/').next().unwrap_or(uri).to_string()
}
