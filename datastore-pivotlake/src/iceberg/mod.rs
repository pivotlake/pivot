//! The Iceberg format: per table, numbered metadata files under
//! `<directory>/metadata/`, each version committed with a conditional write
//! that admits exactly one writer, so writers in several processes share a
//! table with no catalog service. Readable by any Iceberg reader from the
//! table's path.
//!
//! The table itself is the `iceberg-tables` crate's, with its directory as
//! its metadata store; this module adapts it to the format seam.
//! A version is loaded from its metadata file and its manifest list alone.
//! Its live files are indexed a partition at a time, when a scan, a commit
//! or a maintenance pass first asks for the partition, from the manifests
//! that hold a file of it; the indexes of every table are held in one
//! cache. A version holds no footer: a scan, a merge or a layout
//! measurement takes the row groups of the files it works on and lets go
//! of them when it is done. The row groups of the footers it read stay in
//! the same cache, within the same budget, where the next one takes them
//! as they are, with no file opened.
//! The open of a database of few files plans a scan of each table, so the
//! cache holds every index and every footer before the first query.
//!
//! Compaction looks only at the partitions a commit added a file to
//! ([`DirtyPartitions`]), so a routine round indexes no partition that has
//! nothing new in it.

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use catalog::datastore::DatastoreTableMetadata;
use dispatch::DataFlowDispatcher;
use iceberg_tables::commit::{COMMIT_RETRIES, TableCreation};
use iceberg_tables::files::data_files::DataFileEntry;
use iceberg_tables::maintenance::expired_objects::ExpiredHistory;
use iceberg_tables::metadata::pivotlake_metadata_store::metadata_location_version;
use iceberg_tables::{FileIndex, IndexedFile, PivotlakeMetadataStore, TableLocation};
use object_storage::{FileRef, ObjectPath, ObjectStore};
use parquet_engine::{RowGroupMetadata, ScanPlan};
use planner::catalog::{Column, SchemaQualifiedTableName};
use planner::expression::Expression;
use tracing::{info, warn};
use uuid::Uuid;

use crate::file_entry::{FileEntry, constant_partition};
use crate::file_layout::{VisitFile, WalkedFile};
use crate::table::{TableFile, TableStorage};
use crate::table_format::{RemovedFiles, SnapshotExpiry, Table, TableFactory};

pub(crate) use iceberg_tables::{Error, Result};

/// What creates and loads the Iceberg tables of one database.
pub(crate) struct IcebergTableFactory {
    store: Arc<dyn ObjectStore>,
    dispatcher: DataFlowDispatcher,
}

impl IcebergTableFactory {
    pub(crate) fn new(store: Arc<dyn ObjectStore>, dispatcher: DataFlowDispatcher) -> Self {
        Self { store, dispatcher }
    }
}

impl TableFactory for IcebergTableFactory {
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
        let location = TableLocation::in_datastore_store(
            &name.to_string(),
            self.store.clone(),
            directory.clone(),
        )?;
        let entries = adopted
            .iter()
            .map(|file| adopted_entry(file, &partition_by))
            .collect::<Result<Vec<_>>>()?;
        let table = iceberg_tables::Table::create(
            &self.dispatcher,
            location,
            name,
            TableCreation {
                table_uuid: id,
                columns: &columns,
                partition_by: &partition_by,
                sort_by: &sort_by,
                adopted: entries,
            },
        )?;
        let created = IcebergTable::new(name.clone(), table, Arc::default());
        // The files a table is created with were added by its first commit.
        for partition in created.table.partitions()? {
            created.dirty.mark(partition.key(), created.version());
        }
        Ok(Arc::new(created))
    }

    /// Find the table's current metadata version and read it and its
    /// manifest list. No manifest is read until something asks for a
    /// partition's files, and no footer until something needs a file's row
    /// groups.
    fn load_table(
        &self,
        directory: &ObjectPath,
        name: &SchemaQualifiedTableName,
    ) -> crate::Result<Arc<dyn Table>> {
        let location = TableLocation::in_datastore_store(
            &name.to_string(),
            self.store.clone(),
            directory.clone(),
        )?;
        let table = iceberg_tables::Table::load(
            &self.dispatcher,
            Arc::new(PivotlakeMetadataStore::new(location.clone())),
            location,
            name,
        )?;
        Ok(Arc::new(IcebergTable::new(
            name.clone(),
            table,
            Arc::default(),
        )))
    }
}

/// The partitions of one table a routine compaction round has to look at:
/// those a commit added a file to since a round last looked at them and
/// found nothing to do. Shared by every version of the table this process
/// derived from the one it loaded, and empty when the table is loaded: a
/// partition waits for its next write.
#[derive(Default)]
struct DirtyPartitions {
    /// Per partition, by the bytes a walk names it by, the latest version
    /// that added a file to it.
    added_at: Mutex<HashMap<Vec<u8>, u64>>,
}

impl DirtyPartitions {
    /// Record that the version `version` added a file to `partition`.
    fn mark(&self, partition: &[u8], version: u64) {
        let mut added_at = self.added_at.lock().unwrap();
        let latest = added_at.entry(partition.to_vec()).or_default();
        *latest = (*latest).max(version);
    }

    /// Record that a round looking at the version `looked_at` found nothing
    /// to do in `partition`, and say whether the partition is then clean:
    /// it stays dirty when a later version added a file to it.
    fn settle(&self, partition: &[u8], looked_at: u64) -> bool {
        let mut added_at = self.added_at.lock().unwrap();
        if added_at
            .get(partition)
            .is_some_and(|latest| *latest > looked_at)
        {
            return false;
        }
        added_at.remove(partition);
        true
    }

    /// The dirty partitions, in the order of their bytes.
    fn list(&self) -> Vec<Vec<u8>> {
        let mut partitions: Vec<Vec<u8>> = self.added_at.lock().unwrap().keys().cloned().collect();
        partitions.sort_unstable();
        partitions
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
        first_row_id: None,
    })
}

/// One Iceberg table at one metadata version, as the format seam sees it.
struct IcebergTable {
    /// The qualified name the table is indexed under, for messages.
    name: SchemaQualifiedTableName,
    table: iceberg_tables::Table,
    /// The directory the table's versions are numbered in, for the version
    /// hint an expiry brings up to date.
    metadata_store: PivotlakeMetadataStore,
    /// The partitions a routine compaction round has to look at.
    dirty: Arc<DirtyPartitions>,
}

impl IcebergTable {
    /// The table at the version `table` is at, whose dirty partitions are
    /// kept in `dirty`.
    fn new(
        name: SchemaQualifiedTableName,
        table: iceberg_tables::Table,
        dirty: Arc<DirtyPartitions>,
    ) -> Self {
        let metadata_store = PivotlakeMetadataStore::new(table.location().clone());
        Self {
            name,
            table,
            metadata_store,
            dirty,
        }
    }

    /// Hand each file of `index`, the files of one partition, to `visit`;
    /// `Break` when the visit ended the walk.
    fn walk_index(index: &FileIndex, visit: &mut VisitFile<'_>) -> ControlFlow<()> {
        for file in index.files() {
            visit(WalkedFile {
                file: &file.file,
                rows: Some(file.record_count()),
                partition: file.key().partition(),
                sort_range: file.sort_range(),
            })?;
        }
        ControlFlow::Continue(())
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
            first_row_id: None,
        }
    }

    /// This table at the version `next`, which was derived from this one.
    /// The partitions a commit added a file to between the two become
    /// dirty: the data manifests that joined the manifest list name them,
    /// and those of another writer's commits are read for it.
    fn advanced_to(&self, next: iceberg_tables::Table) -> Result<Self> {
        let added = next.list_partitions_added_since(&self.table)?;
        let advanced = Self::new(self.name.clone(), next, Arc::clone(&self.dirty));
        for partition in added {
            self.dirty.mark(partition.key(), advanced.version());
        }
        Ok(advanced)
    }

    /// Delete, through `storage`, the objects the snapshots `expired` used
    /// that this version, the one their expiry committed, does not, and the
    /// previous metadata files `expired` that it no longer names. An object
    /// outside the table's own directory (a file the table adopted where it
    /// lay) is not the table's to delete and is left. The hint is brought up
    /// to this version before a metadata file is deleted, so a reader never
    /// follows it to a version deleted here. Nothing here fails the
    /// maintenance step: what is not deleted stays in the table's
    /// directory, named by no metadata.
    fn delete_objects_of_expiry(&self, storage: &TableStorage, expired: &ExpiredHistory) {
        if expired.metadata_file_count() > 0
            && let Err(error) = self.advance_lagging_hint()
        {
            warn!(
                table = %self.name,
                error = %error,
                "bringing the version hint up to date failed; the expiry's objects are left in place"
            );
            return;
        }
        let objects = match self.table.collect_expired_objects(expired) {
            Ok(objects) => objects,
            Err(error) => {
                warn!(
                    table = %self.name,
                    error = %error,
                    "finding the objects of an expiry failed; they are left in place"
                );
                return;
            }
        };
        let own_objects: Vec<ObjectPath> = objects
            .into_iter()
            .filter(is_under_table_directory)
            .collect();
        // A footer the cache holds keeps its file open, and a deleted file
        // that is open keeps its blocks.
        if let Err(error) = self.table.forget_footers(&own_objects) {
            warn!(
                table = %self.name,
                error = %error,
                "dropping the cached footers of an expiry's files failed; its objects are left in place"
            );
            return;
        }
        let mut deleted = 0u64;
        for path in &own_objects {
            match storage.delete_data_file(path) {
                Ok(()) => deleted += 1,
                Err(error) => {
                    warn!(table = %self.name, object = %path, error = %error, "deleting an expired object failed; it is left in place");
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
            .metadata_store
            .read_version_hint()?
            .is_none_or(|hinted| hinted < self.version())
        {
            self.metadata_store.write_version_hint(self.version())?;
        }
        Ok(())
    }
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

    /// The table's partitions in key order, each in its index's order: by
    /// where a file starts on the sort key, then by path. A few partitions
    /// are indexed at a time.
    fn walk_files(&self, visit: &mut VisitFile<'_>) -> crate::Result<()> {
        Ok(self
            .table
            .visit_partitions(&mut |_, index| Ok(Self::walk_index(index, visit)))?)
    }

    /// The key of every partition a data manifest of the version holds a
    /// file of; the manifests nothing decoded yet are read for it.
    fn list_partitions(&self) -> crate::Result<Vec<Vec<u8>>> {
        Ok(self
            .table
            .partitions()?
            .iter()
            .map(|partition| partition.key().to_vec())
            .collect())
    }

    fn list_dirty_partitions(&self) -> crate::Result<Vec<Vec<u8>>> {
        Ok(self.dirty.list())
    }

    fn settle_partition(&self, partition: &[u8]) -> bool {
        self.dirty.settle(partition, self.version())
    }

    /// The partition's index in its own order; only that partition is
    /// indexed.
    fn walk_partition_files(
        &self,
        partition: &[u8],
        visit: &mut VisitFile<'_>,
    ) -> crate::Result<()> {
        if let Some(partition) = self.table.find_partition(partition)? {
            let _ = Self::walk_index(&*self.table.partition_files(&partition)?, visit);
        }
        Ok(())
    }

    /// Built from the files' records in the partition's index.
    fn find_files(
        &self,
        partition: &[u8],
        paths: &[ObjectPath],
    ) -> crate::Result<Vec<Option<FileEntry>>> {
        let Some(partition) = self.table.find_partition(partition)? else {
            return Ok(vec![None; paths.len()]);
        };
        let index = self.table.partition_files(&partition)?;
        Ok(paths
            .iter()
            .map(|path| {
                index
                    .find(path)
                    .map(|file| self.table.build_datastore_entry(file))
                    .transpose()
            })
            .collect::<Result<_>>()?)
    }

    /// Read from the table's file index a partition at a time, each file
    /// named by its path under the table.
    fn describe(
        &self,
        name: SchemaQualifiedTableName,
        id: String,
    ) -> crate::Result<DatastoreTableMetadata> {
        let name_file = |file: &IndexedFile| file.file.path.as_str().to_string();
        Ok(self.table.describe(name, id, &name_file)?)
    }

    /// Summed from the manifest list's entries.
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

    /// Summed from the manifest list's entries, which count their
    /// manifests' files unless a writer of format version 1 left the
    /// counts out.
    fn count_files(&self) -> Option<u64> {
        self.table.count_data_files()
    }

    /// Plans an unfiltered scan and drops the plan: the partition indexes
    /// it built and the row groups of the footers it read stay in the cache
    /// of the tables' file metadata, where the next plan takes them from.
    fn preload_scan_metadata(&self) -> crate::Result<()> {
        self.table.plan_scan(&[])?;
        Ok(())
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

    /// The row groups the cache of the tables' file metadata holds for each
    /// file, shared with it and with no row deleted; a footer is read only
    /// when the cache does not hold it, and is then kept there.
    fn row_groups(&self, files: &[FileRef]) -> crate::Result<Vec<Vec<Arc<RowGroupMetadata>>>> {
        Ok(self
            .table
            .load_footers(files)?
            .into_iter()
            .map(|footer| footer.row_groups)
            .collect())
    }

    /// The columns the table's current schema requires.
    fn list_never_null_columns(&self) -> Vec<bool> {
        self.table.list_never_null_columns()
    }

    /// Nullability is answered for the files the filters keep, from their
    /// manifest entries.
    fn plan_scan(&self, filters: &[Expression]) -> crate::Result<ScanPlan> {
        Ok(self.table.plan_scan(filters)?)
    }

    /// One short listing from the version held; of a newer version, its
    /// metadata file and its manifest list are read.
    fn refresh(&self) -> crate::Result<Option<Arc<dyn Table>>> {
        let Some(next) = self.table.refresh()? else {
            return Ok(None);
        };
        Ok(Some(Arc::new(self.advanced_to(next)?)))
    }

    fn commit(
        &self,
        removed: RemovedFiles<'_>,
        added: &[TableFile],
        _data_change: bool,
    ) -> crate::Result<Option<Arc<dyn Table>>> {
        // An Iceberg snapshot says what a commit did by its shape: an append
        // when it removes nothing, else a replace, which changes no row.
        let removed_entries = removed
            .paths
            .iter()
            .map(|path| {
                Ok(self
                    .table
                    .find_entry(removed.partition, path)?
                    .expect("a file to remove is live in the version it is removed from"))
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

    fn with_files_swapped(
        &self,
        removed: RemovedFiles<'_>,
        added: Vec<TableFile>,
    ) -> Arc<dyn Table> {
        let swapped = self
            .table
            .with_files_swapped(
                removed.partition,
                removed.paths,
                added
                    .iter()
                    .map(|file| self.build_uncommitted_entry(file))
                    .collect(),
            )
            .expect("a swap keeps the metadata this version was built from");
        Arc::new(Self::new(
            self.name.clone(),
            swapped,
            Arc::clone(&self.dirty),
        ))
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
    /// is deleted.
    fn delete_expired_objects(
        &self,
        _name: &SchemaQualifiedTableName,
        _storage: &TableStorage,
        _now_ms: u64,
    ) -> crate::Result<()> {
        Ok(())
    }
}
