//! The Iceberg format: per table, numbered metadata files under
//! `<directory>/metadata/`, each version landed with a conditional write
//! that admits exactly one writer, so writers in several processes share a
//! table with no catalog service. Readable by any Iceberg reader from the
//! table's path.
//!
//! The table itself is the `iceberg-tables` crate's, with its directory as
//! the catalog that holds its current-metadata pointer; this module adapts it
//! to the format seam. A table's live files, with their statistics and
//! partition tuples, are decoded from the current snapshot's manifests when
//! the version is loaded; their footers are read when a scan, a merge or a
//! layout measurement first needs them and remembered for the life of the
//! version.

use std::sync::Arc;
use std::time::Duration;

use dispatch::DataFlowDispatcher;
use iceberg_tables::commit::{COMMIT_RETRIES, TableCreation, now_unix_ms};
use iceberg_tables::data_files::DataFileEntry;
use iceberg_tables::directory::metadata_location_version;
use iceberg_tables::layout::{VERSION_HINT_FILE, parse_metadata_file_version};
use iceberg_tables::retention::snapshot_retention;
use iceberg_tables::{
    DirectoryCatalog, FileDescriptor, Table, TableCatalog, TableLocation, TableStore,
};
use object_storage::{ObjectPath, ObjectStore};
use parquet_engine::{ParquetTable, RowGroupMetadata};
use planner::catalog::{Column, SchemaQualifiedTableName};
use planner::expression::Expression;
use tracing::{info, warn};
use uuid::Uuid;

use crate::file_entry::{FileEntry, constant_partition};
use crate::format::{Format, FormatTable, TableFormat};
use crate::table::{TableFile, TableStorage};

pub(crate) use iceberg_tables::{Error, Result};

/// The Iceberg backend of one database.
pub(crate) struct IcebergFormat {
    store: Arc<dyn ObjectStore>,
    dispatcher: DataFlowDispatcher,
}

impl IcebergFormat {
    pub(crate) fn new(store: Arc<dyn ObjectStore>, dispatcher: DataFlowDispatcher) -> Self {
        Self { store, dispatcher }
    }

    /// Where the table at `directory` lives: its location under the
    /// database's store, the directory catalog that numbers its versions, and
    /// the store its files are read through.
    fn table_parts(
        &self,
        directory: &ObjectPath,
        name: &SchemaQualifiedTableName,
    ) -> Result<(TableLocation, Arc<dyn TableCatalog>, TableStore)> {
        let location = TableLocation::new(self.store.clone(), directory.clone())?;
        let catalog: Arc<dyn TableCatalog> = Arc::new(DirectoryCatalog::new(location.clone()));
        let store = TableStore::over_datastore_store(&name.to_string(), self.store.clone())?;
        Ok((location, catalog, store))
    }
}

impl Format for IcebergFormat {
    fn name(&self) -> TableFormat {
        TableFormat::Iceberg
    }

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
        let (location, catalog, store) = self.table_parts(directory, name)?;
        let entries = adopted
            .iter()
            .map(|file| adopted_entry(file, &partition_by))
            .collect::<Result<Vec<_>>>()?;
        let table = Table::create(
            &self.dispatcher,
            catalog,
            location,
            store,
            &name.to_string(),
            id,
            TableCreation {
                columns: &columns,
                partition_by: &partition_by,
                sort_by: &sort_by,
                adopted: entries,
            },
            now_unix_ms(),
        )?;
        let table = IcebergTable::new(name.clone(), table)?;
        table.remember(&adopted)?;
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
        let (location, catalog, store) = self.table_parts(directory, name)?;
        let table = Table::load(
            &self.dispatcher,
            catalog,
            location,
            store,
            &name.to_string(),
        )?;
        Ok(Arc::new(IcebergTable::new(name.clone(), table)?))
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

/// One Iceberg table at one metadata version, as the format seam sees it.
struct IcebergTable {
    /// The qualified name the table is indexed under, for messages.
    name: SchemaQualifiedTableName,
    table: Table,
    /// The directory the table's versions are numbered in, for the version
    /// hint a sweep brings up to date.
    directory: DirectoryCatalog,
    /// The table's live files as the shared datastore reads them, decoded
    /// with the version.
    entries: Vec<FileEntry>,
}

impl IcebergTable {
    fn new(name: SchemaQualifiedTableName, table: Table) -> Result<Self> {
        let entries = table.entries()?.iter().map(to_file_entry).collect();
        let directory = DirectoryCatalog::new(table.location().clone());
        Ok(Self {
            name,
            table,
            directory,
            entries,
        })
    }

    /// This table at the version `next`, keeping the footers read for the
    /// files still live.
    fn advanced_to(&self, next: Table) -> Result<Self> {
        next.carry_footers_from(&self.table)?;
        Self::new(self.name.clone(), next)
    }

    /// Keep the footers a writer holds for `files`, so a scan after the
    /// commit does not read them back.
    fn remember(&self, files: &[TableFile]) -> Result<()> {
        for file in files {
            self.table
                .remember_footer(&file.entry.file.path, file.row_groups.clone())?;
        }
        Ok(())
    }

    /// The metadata objects present under the table's metadata directory
    /// that this datastore writes (manifests, manifest lists, metadata
    /// files), each by name with its modification time. Anything else there
    /// (the hint, another tool's files) is not listed and so never swept.
    fn list_metadata_objects(&self) -> Result<Vec<(String, u64)>> {
        let location = self.table.location();
        Ok(location
            .store()
            .list(&location.metadata_dir())?
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

impl FormatTable for IcebergTable {
    /// The number of the metadata file this version was loaded from.
    fn version(&self) -> u64 {
        metadata_location_version(self.table.metadata_location())
            .expect("a directory table is loaded from a numbered metadata file")
    }

    fn columns(&self) -> &[Column] {
        self.table.columns()
    }

    fn partition_by(&self) -> &[String] {
        self.table.partition_by()
    }

    fn sort_by(&self) -> &[String] {
        self.table.sort_by()
    }

    fn entries(&self) -> &[FileEntry] {
        &self.entries
    }

    /// The field id the table's schema gives each declared column, in column
    /// order: what the files it writes stamp on their columns.
    fn field_ids(&self) -> Option<Vec<i32>> {
        let schema = self.table.metadata().current_schema();
        Some(
            self.table
                .columns()
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
        let location = self.table.location();
        let descriptors = files
            .iter()
            .map(|entry| {
                Ok(FileDescriptor {
                    path: location.uri_for(&entry.file.path)?,
                    length: entry.file.size,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(self
            .table
            .read_footers(&descriptors)?
            .into_iter()
            .map(|footer| footer.row_groups)
            .collect())
    }

    fn scan(&self, filters: &[Expression]) -> crate::Result<Arc<ParquetTable>> {
        Ok(Arc::new(self.table.scan(filters)?))
    }

    /// One short listing from the version held; a newer version is read
    /// whole.
    fn refresh(&self) -> crate::Result<Option<Arc<dyn FormatTable>>> {
        let Some(next) = self.table.refresh()? else {
            return Ok(None);
        };
        Ok(Some(Arc::new(self.advanced_to(next)?)))
    }

    fn commit(
        &self,
        removed: &[FileEntry],
        added: &[TableFile],
        data_change: bool,
        now_ms: u64,
    ) -> crate::Result<Option<Arc<dyn FormatTable>>> {
        let live = self.table.entries()?;
        let removed_entries = removed
            .iter()
            .map(|removed| {
                live.iter()
                    .find(|entry| entry.file.path == removed.file.path)
                    .cloned()
                    .ok_or_else(|| Error::UncommittedRemoval {
                        file: removed.file.path.to_string(),
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let added_entries: Vec<DataFileEntry> = added.iter().map(uncommitted_entry).collect();
        let now_ms = i64::try_from(now_ms).unwrap_or(i64::MAX);
        let Some(next) = self
            .table
            .commit(&removed_entries, added_entries, data_change, now_ms)?
        else {
            return Ok(None);
        };
        let table = self.advanced_to(next)?;
        table.remember(added)?;
        Ok(Some(Arc::new(table)))
    }

    fn with_files_swapped(
        &self,
        removed: &[ObjectPath],
        added: Vec<TableFile>,
    ) -> Arc<dyn FormatTable> {
        let swapped = self
            .table
            .with_files_swapped(removed, added.iter().map(uncommitted_entry).collect())
            .expect("a swap keeps the metadata this version was built from");
        let table = Self::new(self.name.clone(), swapped)
            .expect("a swap keeps the manifests this version decoded");
        table
            .remember(&added)
            .expect("a swapped-in file is under the table's store");
        Arc::new(table)
    }

    /// The table's own retention property, or the default.
    fn deleted_file_retention(&self) -> crate::Result<Duration> {
        Ok(snapshot_retention(self.table.metadata())?)
    }

    /// Expire every snapshot that has outlived the retention window, as one
    /// metadata-only commit. A writer landing first expires them itself, so
    /// a lost race only reloads and looks again.
    fn maintain(&self, now_ms: u64) -> crate::Result<Option<Arc<dyn FormatTable>>> {
        let now_ms = i64::try_from(now_ms).unwrap_or(i64::MAX);
        let mut refreshed: Option<IcebergTable> = None;
        for _ in 0..COMMIT_RETRIES {
            let table: &IcebergTable = refreshed.as_ref().unwrap_or(self);
            if let Some(next) = table.table.expire_snapshots(now_ms)? {
                return Ok(Some(Arc::new(table.advanced_to(next)?)));
            }
            let Some(next) = table.table.refresh()? else {
                return Ok(None);
            };
            refreshed = Some(table.advanced_to(next)?);
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
        let location = self.table.location();
        if self
            .directory
            .read_version_hint()?
            .is_none_or(|hinted| hinted < self.version())
        {
            self.directory.write_version_hint(self.version())?;
        }

        let retention_ms =
            u64::try_from(self.deleted_file_retention()?.as_millis()).unwrap_or(u64::MAX);
        let cutoff = now_ms.saturating_sub(retention_ms);
        let reachable = self.table.reachable_objects()?;

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
