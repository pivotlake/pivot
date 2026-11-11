//! The Iceberg format: per table, numbered metadata files under
//! `<directory>/metadata/`, each version committed with a conditional write
//! that admits exactly one writer, so writers in several processes share a
//! table with no catalog service. Readable by any Iceberg reader from the
//! table's path.
//!
//! The table itself is the `iceberg-tables` crate's, with its directory as
//! its current-metadata pointer; this module adapts it to the format seam.
//! A table's live files are held once, in the table's
//! file index: the current snapshot's manifests are decoded into it when
//! the version is loaded, the ones an earlier version already decoded not
//! again. No footer is held: a scan, a merge or a layout measurement reads
//! the footers of the files it works on and lets go of them when it is
//! done, the bytes it read staying in the worker pool's caches.

use std::sync::Arc;
use std::time::Duration;

use dispatch::DataFlowDispatcher;
use iceberg_tables::commit::{COMMIT_RETRIES, TableCreation};
use iceberg_tables::data_files::DataFileEntry;
use iceberg_tables::directory::metadata_location_version;
use iceberg_tables::expired_objects::ExpiredHistory;
use iceberg_tables::{DirectoryMetadataPointer, FileDescriptor, TableLocation, TableStore};
use object_storage::{FileRef, ObjectPath, ObjectStore};
use parquet_engine::{FileStats, ParquetTable, RowGroupMetadata};
use planner::catalog::{Column, SchemaQualifiedTableName};
use planner::expression::Expression;
use tracing::{info, warn};
use uuid::Uuid;

use crate::file_entry::{FileEntry, constant_partition};
use crate::file_layout::{VisitFile, WalkedFile};
use crate::table::{TableFile, TableStorage};
use crate::table_format::{SnapshotExpiry, Table, TableFormat};

pub(crate) use iceberg_tables::{Error, Result};

/// How long an object under a table's directory that the table does not use
/// is left alone before a `VACUUM` may delete it, judged by the object's
/// modification time. A writer uploads its files before it commits them, so
/// for that long a file no metadata names yet is indistinguishable from one
/// that never will be named; the delay has to outlast the longest write in
/// flight. Three days is the delay Iceberg's own orphan-file removal
/// defaults to.
pub const UNUSED_OBJECT_SAFETY_DELAY: Duration = Duration::from_secs(3 * 24 * 60 * 60);

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

impl TableFormat for IcebergFormat {
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
    ) -> crate::Result<Arc<dyn Table>> {
        let location = TableLocation::new(self.store.clone(), directory.clone())?;
        let store = TableStore::over_datastore_store(&name.to_string(), self.store.clone())?;
        let entries = adopted
            .iter()
            .map(|file| adopted_entry(file, &partition_by))
            .collect::<Result<Vec<_>>>()?;
        let table = iceberg_tables::Table::create(
            &self.dispatcher,
            DirectoryMetadataPointer::new(location.clone()),
            location,
            store,
            name,
            id,
            TableCreation {
                columns: &columns,
                partition_by: &partition_by,
                sort_by: &sort_by,
                adopted: entries,
            },
        )?;
        Ok(Arc::new(IcebergTable::new(name.clone(), table)?))
    }

    /// Find the table's current metadata version, read it and decode the
    /// manifests of its current snapshot. No footer is read until a scan
    /// needs one.
    fn load_table(
        &self,
        directory: &ObjectPath,
        name: &SchemaQualifiedTableName,
        _id: Uuid,
    ) -> crate::Result<Arc<dyn Table>> {
        let location = TableLocation::new(self.store.clone(), directory.clone())?;
        let store = TableStore::over_datastore_store(&name.to_string(), self.store.clone())?;
        let table = iceberg_tables::Table::load(
            &self.dispatcher,
            Arc::new(DirectoryMetadataPointer::new(location.clone())),
            location,
            store,
            name,
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
        // Nobody here ordered the file's rows, so its entry names no order.
        sort_order_id: None,
        origin: None,
    })
}

/// One Iceberg table at one metadata version, as the format seam sees it.
struct IcebergTable {
    /// The qualified name the table is indexed under, for messages.
    name: SchemaQualifiedTableName,
    table: iceberg_tables::Table,
    /// The directory the table's versions are numbered in, for the version
    /// hint a sweep brings up to date.
    directory: DirectoryMetadataPointer,
}

impl IcebergTable {
    /// The table at the version `table` is at, with its file index built:
    /// a version held here answers from memory which files it has, whatever
    /// becomes of its manifests in the store afterwards.
    fn new(name: SchemaQualifiedTableName, table: iceberg_tables::Table) -> Result<Self> {
        table.file_index()?;
        let directory = DirectoryMetadataPointer::new(table.location().clone());
        Ok(Self {
            name,
            table,
            directory,
        })
    }

    /// The entry a commit records for `file`, which a writer just uploaded:
    /// its sort order is the table's when the writer reports the rows in
    /// that order, and the unsorted one otherwise.
    fn build_uncommitted_entry(&self, file: &TableFile) -> DataFileEntry {
        DataFileEntry {
            file: file.entry.file.clone(),
            partition: file.entry.partition.clone(),
            stats: Some(file.entry.stats.clone().unwrap_or_else(|| {
                Arc::new(parquet_engine::aggregate_file_stats(&file.row_groups))
            })),
            sort_order_id: Some(
                self.table
                    .find_sort_order_id_of_rows_sorted_by(&file.entry.sorted_by),
            ),
            origin: None,
        }
    }

    /// This table at the version `next`, its file index built on this
    /// version's.
    fn advanced_to(&self, next: iceberg_tables::Table) -> Result<Self> {
        next.carry_over_from(&self.table);
        Self::new(self.name.clone(), next)
    }

    /// Delete, through `storage`, the objects the snapshots `expired` used
    /// that this version, the one their expiry committed, does not, and the
    /// previous metadata files `expired` that it no longer names. An object
    /// outside the table's own directory (a file the table adopted where it
    /// lay) is not the table's to delete and is left. The hint is brought up
    /// to this version before a metadata file is deleted, so a reader never
    /// follows it to a version deleted here. Nothing here fails the
    /// maintenance step: what is not deleted, a later `VACUUM` reclaims.
    fn delete_objects_of_expiry(&self, storage: &TableStorage, expired: &ExpiredHistory) {
        if expired.metadata_file_count() > 0
            && let Err(error) = self.advance_lagging_hint()
        {
            warn!(
                table = %self.name,
                error = %error,
                "bringing the version hint up to date failed; VACUUM reclaims the expiry's objects"
            );
            return;
        }
        let objects = match self.table.collect_expired_objects(expired) {
            Ok(objects) => objects,
            Err(error) => {
                warn!(
                    table = %self.name,
                    error = %error,
                    "finding the objects of an expiry failed; VACUUM reclaims them"
                );
                return;
            }
        };
        let mut deleted = 0u64;
        for path in objects.iter().filter(|path| is_under_table_directory(path)) {
            match storage.delete_data_file(path) {
                Ok(()) => deleted += 1,
                Err(error) => {
                    warn!(table = %self.name, object = %path, error = %error, "deleting an expired object failed; VACUUM reclaims it");
                }
            }
        }
        info!(
            table = %self.name,
            snapshots = expired.snapshot_count(),
            metadata_files = expired.metadata_file_count(),
            objects = deleted,
            "expired snapshots and previous metadata files and deleted the objects only they used"
        );
    }

    /// Rewrite the version hint to name this version when it names an
    /// earlier one or is missing. A hint falls behind when its write after
    /// a commit fails or another tool commits without one.
    fn advance_lagging_hint(&self) -> crate::Result<()> {
        if self
            .directory
            .read_version_hint()?
            .is_none_or(|hinted| hinted < self.version())
        {
            self.directory.write_version_hint(self.version())?;
        }
        Ok(())
    }

    /// How old, by its modification time, an object the table does not use
    /// must be before a sweep deletes it: the safety delay, or the table's
    /// retention window when that is longer. A writer whose base version is
    /// younger than the window relies on the numbered metadata file after
    /// it still being there if it ever was (that is how it learns it lost),
    /// so no metadata file younger than the window may go.
    fn unused_object_delay(&self) -> Result<Duration> {
        Ok(self
            .table
            .snapshot_retention()?
            .max(UNUSED_OBJECT_SAFETY_DELAY))
    }

    /// The directory-name prefixes (`name=`) of the table's partition fields
    /// whose names would otherwise read as hidden: a partition directory of
    /// such a field holds data, so a sweep walks into it.
    fn list_hidden_looking_partition_prefixes(&self) -> Vec<String> {
        self.table
            .metadata()
            .partition_specs_iter()
            .flat_map(|spec| spec.fields())
            .filter(|field| has_hidden_name(&field.name))
            .map(|field| format!("{}=", field.name))
            .collect()
    }
}

/// Whether `name` marks its object or directory as hidden, by the convention
/// table tools share: staging areas, markers and other bookkeeping that is
/// not a table's, named with a leading `_` or `.`.
fn has_hidden_name(name: &str) -> bool {
    name.starts_with('_') || name.starts_with('.')
}

/// Whether `path`, as an Iceberg table names a file, lies under the table's
/// own directory: a relative path that never steps out of it. A file kept
/// elsewhere in the store is named by its absolute key.
fn is_under_table_directory(path: &ObjectPath) -> bool {
    !path.is_absolute()
        && path
            .as_str()
            .split('/')
            .all(|segment| !segment.is_empty() && segment != "." && segment != "..")
}

impl Table for IcebergTable {
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

    fn count_files(&self) -> crate::Result<usize> {
        Ok(self.table.file_index()?.len())
    }

    /// The table's file index in its own order: by partition, then by where
    /// a file starts on the sort key, then by path.
    fn walk_files(&self, visit: &mut VisitFile<'_>) -> crate::Result<()> {
        for file in self.table.file_index()?.files() {
            let walked = WalkedFile {
                file: &file.file,
                rows: Some(file.record_count()),
                partition: file.key().partition(),
                sort_range: file.sort_range(),
            };
            if visit(walked).is_break() {
                break;
            }
        }
        Ok(())
    }

    fn contains_file(&self, path: &ObjectPath) -> crate::Result<bool> {
        Ok(self.table.file_index()?.find(path).is_some())
    }

    /// Built from the file's record in the table's index.
    fn find_file(&self, path: &ObjectPath) -> crate::Result<Option<FileEntry>> {
        Ok(self.table.find_datastore_entry(path)?)
    }

    /// Read back from the file's record in the table's index.
    fn file_stats(&self, path: &ObjectPath) -> crate::Result<Option<Arc<FileStats>>> {
        Ok(self.table.file_stats(path)?.map(Arc::new))
    }

    /// Counted from the files' manifest entries: a column is free of NULLs
    /// when every file's entry records none in it.
    fn nullability(&self) -> crate::Result<Vec<bool>> {
        Ok(self.table.nullability()?)
    }

    fn total_rows(&self) -> crate::Result<u64> {
        Ok(self.table.total_rows()?)
    }

    /// Counted from the manifests: the data files' rows less those their
    /// deletion vectors delete.
    fn count_rows(&self) -> crate::Result<Option<u64>> {
        Ok(self
            .table
            .row_count()
            .and_then(|rows| u64::try_from(rows).ok()))
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

    /// Read over the pool for this call; the table keeps none of them.
    fn row_groups(&self, files: &[FileRef]) -> crate::Result<Vec<Vec<Arc<RowGroupMetadata>>>> {
        let location = self.table.location();
        let descriptors = files
            .iter()
            .map(|file| {
                Ok(FileDescriptor {
                    path: location.uri_for(&file.path)?,
                    length: file.size,
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

    fn plan_scan(&self, filters: &[Expression]) -> crate::Result<Arc<ParquetTable>> {
        Ok(Arc::new(self.table.plan_scan(filters)?))
    }

    /// One short listing from the version held; a newer version is read
    /// whole.
    fn refresh(&self) -> crate::Result<Option<Arc<dyn Table>>> {
        let Some(next) = self.table.refresh()? else {
            return Ok(None);
        };
        Ok(Some(Arc::new(self.advanced_to(next)?)))
    }

    fn commit(
        &self,
        removed: &[ObjectPath],
        added: &[TableFile],
        data_change: bool,
        _now_ms: u64,
    ) -> crate::Result<Option<Arc<dyn Table>>> {
        // An Iceberg snapshot says what a commit did by its shape: an append
        // when it removes nothing, else a replace, which changes no row.
        if data_change && !removed.is_empty() {
            return Err(Error::RowChangingRemoval {
                table: self.name.to_string(),
            }
            .into());
        }
        let removed_entries = removed
            .iter()
            .map(|removed| {
                self.table
                    .find_entry(removed)?
                    .ok_or_else(|| Error::UncommittedRemoval {
                        file: removed.to_string(),
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let added_entries: Vec<DataFileEntry> = added
            .iter()
            .map(|file| self.build_uncommitted_entry(file))
            .collect();
        let Some(next) = self.table.commit(&removed_entries, added_entries)? else {
            return Ok(None);
        };
        // The commit has succeeded: a failure from here on is of loading its
        // version, never of the commit.
        let table = self.advanced_to(next).map_err(|error| {
            iceberg_tables::Error::CommittedTableUnloaded {
                table: self.name.to_string(),
                source: Box::new(error),
            }
        })?;
        Ok(Some(Arc::new(table)))
    }

    fn with_files_swapped(&self, removed: &[ObjectPath], added: Vec<TableFile>) -> Arc<dyn Table> {
        let swapped = self
            .table
            .with_files_swapped(
                removed,
                added
                    .iter()
                    .map(|file| self.build_uncommitted_entry(file))
                    .collect(),
            )
            .expect("a swap keeps the metadata this version was built from");
        Arc::new(
            Self::new(self.name.clone(), swapped)
                .expect("a swap keeps the files this version indexed"),
        )
    }

    /// The table's own retention property, or the default.
    fn deleted_file_retention(&self) -> crate::Result<Duration> {
        Ok(self.table.snapshot_retention()?)
    }

    /// Expire every snapshot and every previous metadata file that has
    /// outlived the retention window, as one metadata-only commit, and hand
    /// back the deletion of those files and of what only those snapshots
    /// used. A lost race reloads and looks again: whatever the
    /// winner expired is the winner's to delete, and only what this commit
    /// itself dropped is deleted here. A commit that fails, or whose outcome
    /// is unknown, hands back nothing, so nothing is deleted for it.
    fn expire_snapshots(
        &self,
        storage: &TableStorage,
        now_ms: u64,
    ) -> crate::Result<Option<SnapshotExpiry>> {
        let now_ms = i64::try_from(now_ms).unwrap_or(i64::MAX);
        let mut refreshed: Option<IcebergTable> = None;
        for _ in 0..COMMIT_RETRIES {
            let table: &IcebergTable = refreshed.as_ref().unwrap_or(self);
            if let Some(expiry) = table.table.expire_snapshots(now_ms)? {
                let committed = Arc::new(table.advanced_to(expiry.table)?);
                let (expired, storage) = (expiry.expired, storage.clone());
                let deleting = Arc::clone(&committed);
                return Ok(Some(SnapshotExpiry {
                    table: committed,
                    delete_expired_objects: Box::new(move || {
                        deleting.delete_objects_of_expiry(&storage, &expired)
                    }),
                }));
            }
            let Some(next) = table.table.refresh()? else {
                return Ok(None);
            };
            refreshed = Some(table.advanced_to(next)?);
        }
        Ok(None)
    }

    /// An Iceberg table retires a file by committing a snapshot without it,
    /// and what a snapshot retired is deleted when the snapshot expires, as
    /// a metadata file is when the one that replaced it has aged past the
    /// window ([`expire_snapshots`](Self::expire_snapshots)): nothing else
    /// is deleted routinely.
    fn delete_expired_objects(
        &self,
        _name: &SchemaQualifiedTableName,
        _storage: &TableStorage,
        _now_ms: u64,
    ) -> crate::Result<()> {
        Ok(())
    }

    /// Bring a lagging hint up to date, then list everything under the
    /// table's directory, however deep and of whatever kind, and delete each
    /// object the table does not use
    /// ([`collect_objects_in_use`](iceberg_tables::Table::collect_objects_in_use))
    /// whose modification time is older than the safety delay
    /// ([`UNUSED_OBJECT_SAFETY_DELAY`], or the table's retention window when
    /// that is longer). Hidden objects and directories (a leading `_` or
    /// `.`) are left alone, except a partition directory of a field so
    /// named. Only the table's own directory is listed, so a file the table
    /// uses from elsewhere in the store is never met. The hint is advanced
    /// before anything is deleted, so a reader never follows it to a version
    /// this sweep removes.
    fn sweep_unused_objects(
        &self,
        name: &SchemaQualifiedTableName,
        storage: &TableStorage,
        now_ms: u64,
    ) -> crate::Result<()> {
        // A table that turned `gc.enabled` off keeps every object it has.
        if !self.table.is_gc_enabled()? {
            return Ok(());
        }
        self.advance_lagging_hint()?;

        let delay_ms = u64::try_from(self.unused_object_delay()?.as_millis()).unwrap_or(u64::MAX);
        let cutoff = now_ms.saturating_sub(delay_ms);
        let in_use = self.table.collect_objects_in_use()?;
        let partition_prefixes = self.list_hidden_looking_partition_prefixes();
        let is_hidden = |object_name: &str| {
            has_hidden_name(object_name)
                && !partition_prefixes
                    .iter()
                    .any(|prefix| object_name.starts_with(prefix.as_str()))
        };

        let mut deleted = 0u64;
        let mut first_error: Option<crate::Error> = None;
        for (path, modified_ms) in storage.list_objects_recursively(is_hidden)? {
            if modified_ms >= cutoff || in_use.contains(&path) {
                continue;
            }
            match storage.delete_data_file(&path) {
                Ok(()) => deleted += 1,
                Err(error) => {
                    warn!(table = %name, object = %path, error = %error, "vacuum: deleting an unused object failed");
                    first_error.get_or_insert(error);
                }
            }
        }
        if deleted > 0 {
            info!(table = %name, objects = deleted, "vacuumed unused objects");
        }
        first_error.map_or(Ok(()), Err)
    }
}
