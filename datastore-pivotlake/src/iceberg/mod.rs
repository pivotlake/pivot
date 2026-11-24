//! The Iceberg format: per table, numbered metadata files under
//! `<directory>/metadata/`, each version committed with a conditional write
//! that admits exactly one writer, so writers in several processes share a
//! table with no catalog service. An Iceberg reader reads a version from
//! the path of its metadata file.
//!
//! The table itself is the `iceberg-tables` crate's, with its directory as
//! its metadata store ([`DirectoryMetadataStore`]); this module adapts it to
//! the format seam.
//! A version is loaded from its metadata file and its manifest list alone.
//! Its live files are indexed a partition at a time, when a scan, a commit
//! or a maintenance pass first asks for the partition, from the manifests
//! whose manifest-list entries do not rule the partition out; the indexes
//! of every table are held in one cache, and nothing else is kept of a
//! manifest. A version holds no footer: a scan, a merge or a layout
//! measurement takes the row groups of the files it works on and lets go
//! of them when it is done. The row groups of the footers it read stay in
//! the same cache, within the same budget, where the next one takes them
//! as they are, with no file opened.
//! The open of a database of few files selects a scan of each table, so the
//! cache holds every index and every footer before the first query.
//!
//! A version stands among its table's commits at its last sequence number,
//! and a manifest entry records the sequence number of the commit that
//! added its file, so the partitions a file was added to since a version
//! are read from the manifests written since, whoever wrote them: what a
//! routine compaction round examines.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use catalog::datastore::DatastoreTableMetadata;
use dispatch::DataFlowDispatcher;
use iceberg_tables::commit::{COMMIT_RETRIES, TableCreation};
use iceberg_tables::files::data_files::DataFileEntry;
use iceberg_tables::maintenance::expired_objects::ExpiredSnapshots;
use iceberg_tables::{IndexedFile, PartitionKeys, TableLocation, check_partition_terms};
use object_storage::{FileRef, ObjectPath, ObjectStore};
use parquet_engine::{RowGroupMetadata, ScanSelection};
use planner::catalog::{Column, SchemaQualifiedTableName};
use planner::expression::Expression;
use table_file_index::PartitionFiles;
use tracing::{info, warn};
use uuid::Uuid;

use crate::file_entry::{FileEntry, constant_partition};
use crate::iceberg::directory_metadata_store::{DirectoryMetadataStore, metadata_location_version};
use crate::table::{TableFile, TableStorage};
use crate::table_format::{
    PreparedCommit, RemovedFiles, SnapshotExpiry, Table, TableFactory, VisitPartition,
};

pub(crate) use iceberg_tables::{Error, Result};

pub(crate) mod directory_metadata_store;

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
    /// Each term is a column, or a transform of one that Iceberg takes over
    /// the column's type and an INSERT computes. A table that adopts files
    /// is partitioned by columns alone: an adopted file sits in one
    /// partition when its statistics prove every partition column constant,
    /// which they do not prove of a transform.
    fn check_partition_by(
        &self,
        name: &SchemaQualifiedTableName,
        columns: &[Column],
        partition_by: &[String],
        adopts_files: bool,
    ) -> crate::Result<()> {
        check_partition_terms(&name.to_string(), columns, partition_by)?;
        // A term that passed the check and names no column is a transform.
        let transform = partition_by
            .iter()
            .find(|term| columns.iter().all(|column| &column.name != *term));
        match transform {
            Some(term) if adopts_files => {
                Err(crate::Error::AdoptingIntoPartitionTransform { term: term.clone() })
            }
            _ => Ok(()),
        }
    }

    /// Write the table's first metadata version, holding the adopted files as
    /// its first snapshot when there are any. An adopted file of a partitioned
    /// table must sit in one partition, which its statistics prove of the
    /// partition columns; an Iceberg tuple has no "unknown" value, so one
    /// that does not is refused.
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
            Arc::new(DirectoryMetadataStore::new(location.clone())),
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
        Ok(Arc::new(IcebergTable::new(name.clone(), table)))
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
            Arc::new(DirectoryMetadataStore::new(location.clone())),
            location,
            name,
        )?;
        Ok(Arc::new(IcebergTable::new(name.clone(), table)))
    }

    /// The partition indexes and the footers the cache of the tables' file
    /// metadata holds for the table leave it, whichever of the table's
    /// versions stored them: nothing else lets go of them before the
    /// cache's budget does.
    fn forget_cached_table(&self, directory: &ObjectPath) -> crate::Result<()> {
        let location = TableLocation::in_datastore_store(
            directory.as_str(),
            self.store.clone(),
            directory.clone(),
        )?;
        iceberg_tables::forget_cached_table(&location);
        Ok(())
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
        partition_spec_id: None,
        stats: Some(stats),
        // Nobody here ordered the file's rows, so its entry names no order.
        sort_order_id: None,
        first_row_id: None,
        manifest_path: None,
    })
}

/// One Iceberg table at one metadata version, as the format seam sees it.
struct IcebergTable {
    /// The qualified name the table is indexed under, for messages.
    name: SchemaQualifiedTableName,
    table: iceberg_tables::Table,
    /// When each snapshot this version lists was written, read from the
    /// metadata the first time a commit's time is asked for.
    snapshot_times: OnceLock<SnapshotTimes>,
}

/// When the snapshots one version of a table lists were written.
struct SnapshotTimes {
    /// The timestamp of each snapshot, by its sequence number.
    by_sequence_number: HashMap<i64, i64>,
    /// The timestamp of the oldest snapshot, 0 for a table with none.
    oldest: i64,
}

impl IcebergTable {
    /// The table at the version `table` is at.
    fn new(name: SchemaQualifiedTableName, table: iceberg_tables::Table) -> Self {
        Self {
            name,
            table,
            snapshot_times: OnceLock::new(),
        }
    }

    /// Whether this version lists more data manifests than a table whose
    /// small files are being merged ever does: a merge's commit merges the
    /// small manifests as well, so they pile up only on a table that gains
    /// files and has none merged. An insert merges manifests itself then,
    /// and otherwise leaves them, since merging them would take from under
    /// a merge being committed the manifests it was prepared on.
    fn has_manifests_piled_up(&self) -> bool {
        /// A commit merges the newest manifests once there are a hundred of
        /// them, so a table kept merged lists few more than that.
        const MOST_DATA_MANIFESTS_OF_A_TABLE_KEPT_MERGED: usize = 500;

        self.table.manifest_list().data_manifests().count()
            > MOST_DATA_MANIFESTS_OF_A_TABLE_KEPT_MERGED
    }

    /// The entries of the files `removed` names, looked up together:
    /// [`Error::CommitConflict`] naming the first this version no longer
    /// holds.
    fn find_removed_entries(&self, removed: RemovedFiles<'_>) -> crate::Result<Vec<DataFileEntry>> {
        let found = self.table.find_entries(removed.partition, removed.paths)?;
        removed
            .paths
            .iter()
            .zip(found)
            .map(|(path, entry)| {
                entry.ok_or_else(|| crate::Error::CommitConflict {
                    location: self.table.location().directory().as_str().to_string(),
                    file: path.to_string(),
                })
            })
            .collect()
    }

    fn build_uncommitted_entries(&self, files: &[TableFile]) -> Vec<DataFileEntry> {
        files
            .iter()
            .map(|file| self.build_uncommitted_entry(file))
            .collect()
    }

    /// The entry a commit records for `file`, which a writer just uploaded:
    /// its sort order is the table's when the writer reports the rows in
    /// that order, and the unsorted one otherwise.
    fn build_uncommitted_entry(&self, file: &TableFile) -> DataFileEntry {
        DataFileEntry {
            file: file.entry.file.clone(),
            partition: file.entry.partition.clone(),
            partition_spec_id: None,
            stats: Some(file.entry.stats.clone().unwrap_or_else(|| {
                Arc::new(parquet_engine::aggregate_file_stats(&file.row_groups))
            })),
            sort_order_id: Some(
                self.table
                    .find_sort_order_id_of_rows_sorted_by(&file.entry.sorted_by),
            ),
            first_row_id: None,
            manifest_path: None,
        }
    }

    /// This table at the version `next`, which was derived from this one.
    fn advanced_to(&self, next: iceberg_tables::Table) -> Self {
        Self::new(self.name.clone(), next)
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

    /// Planned from the default partition spec: a column's identity field
    /// named after it is a key the rows carry, every other field a key
    /// computed from a row by the field's transform.
    fn plan_partition_keys(&self) -> crate::Result<PartitionKeys> {
        Ok(self.table.plan_partition_keys()?)
    }

    fn sort_by(&self) -> &[String] {
        self.table.sort_by()
    }

    /// The table's partitions in key order, a few indexed at a time.
    fn visit_partitions(&self, visit: &mut VisitPartition<'_>) -> crate::Result<()> {
        Ok(self.table.visit_partitions(&mut |partition, index| {
            Ok(visit(&partition.encode_key(), index.file_index()))
        })?)
    }

    /// The key of every partition a data manifest of the version holds a
    /// file of; the manifests whose manifest-list entries do not name their
    /// one partition are read for it, and nothing of them is kept.
    fn list_partitions(&self) -> crate::Result<Vec<Vec<u8>>> {
        Ok(self
            .table
            .partitions()?
            .iter()
            .map(|partition| partition.encode_key().to_vec())
            .collect())
    }

    /// The sequence number of the version's latest commit.
    fn commit_position(&self) -> u64 {
        u64::try_from(self.table.metadata().last_sequence_number()).unwrap_or(0)
    }

    /// Read from the data manifests written after the commit with that
    /// sequence number, as the manifest list numbers them: each entry
    /// records the sequence number of the commit that added its file.
    fn list_partitions_added_after(&self, commit_position: u64) -> crate::Result<Vec<Vec<u8>>> {
        let sequence_number = i64::try_from(commit_position).unwrap_or(i64::MAX);
        Ok(self
            .table
            .list_partitions_added_after(sequence_number)?
            .iter()
            .map(|partition| partition.encode_key().to_vec())
            .collect())
    }

    /// The partitions of the table's default partition spec: a rewrite's
    /// output is written under that spec with the partition values of its
    /// inputs, which only the files of that spec have. The partitions of
    /// another spec are left out, and logged once for the spec.
    fn select_rewritable_partitions(
        &self,
        partitions: Vec<Vec<u8>>,
    ) -> crate::Result<Vec<Vec<u8>>> {
        let default_spec_id = self.table.metadata().default_partition_spec_id();
        let mut rewritable = Vec::with_capacity(partitions.len());
        let mut left_out_per_spec: BTreeMap<i32, usize> = BTreeMap::new();
        for partition in partitions {
            let spec_id = self.table.find_partition(&partition)?.spec_id();
            if spec_id == default_spec_id {
                rewritable.push(partition);
            } else {
                *left_out_per_spec.entry(spec_id).or_default() += 1;
            }
        }
        for (spec_id, partition_count) in left_out_per_spec {
            warn!(
                table = %self.name,
                partition_spec_id = spec_id,
                default_partition_spec_id = default_spec_id,
                partitions = partition_count,
                "the partitions written under a partition spec other than the table's default are not compacted"
            );
        }
        Ok(rewritable)
    }

    /// The partitions are indexed together, each manifest that holds
    /// several of them decoded once for them all.
    fn index_partitions(&self, partitions: &[Vec<u8>]) -> crate::Result<()> {
        Ok(self.table.index_partitions(partitions)?)
    }

    /// The timestamp of the snapshot each commit wrote. A commit whose
    /// snapshot has been expired is given the timestamp of the oldest
    /// snapshot the table still lists: it was made no later than that.
    fn read_commit_times_ms(&self, commit_sequences: &[u64]) -> Vec<u64> {
        let snapshot_times = self.snapshot_times.get_or_init(|| {
            let by_sequence_number: HashMap<i64, i64> = self
                .table
                .metadata()
                .snapshots()
                .map(|snapshot| (snapshot.sequence_number(), snapshot.timestamp_ms()))
                .collect();
            let oldest = by_sequence_number.values().copied().min().unwrap_or(0);
            SnapshotTimes {
                by_sequence_number,
                oldest,
            }
        });
        commit_sequences
            .iter()
            .map(|commit_sequence| {
                let committed_at = i64::try_from(*commit_sequence)
                    .ok()
                    .and_then(|sequence_number| {
                        snapshot_times
                            .by_sequence_number
                            .get(&sequence_number)
                            .copied()
                    })
                    .unwrap_or(snapshot_times.oldest);
                u64::try_from(committed_at).unwrap_or(0)
            })
            .collect()
    }

    /// The partition's index; only that partition is indexed.
    fn load_partition_files(
        &self,
        partition: &[u8],
    ) -> crate::Result<Option<Arc<dyn PartitionFiles>>> {
        let partition = self.table.find_partition(partition)?;
        let index: Arc<dyn PartitionFiles> = self.table.partition_files(&partition)?;
        Ok((index.count_files() > 0).then_some(index))
    }

    /// Built from the files' records in the partition's index.
    fn find_file_entries(
        &self,
        partition: &[u8],
        paths: &[ObjectPath],
    ) -> crate::Result<Vec<Option<FileEntry>>> {
        // A commit that only adds files looks for none, and names no
        // partition.
        if paths.is_empty() {
            return Ok(Vec::new());
        }
        let partition = self.table.find_partition(partition)?;
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
        let name_file = |file: &IndexedFile| file.facts.file().path.as_str().to_string();
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

    /// Selects an unfiltered scan and drops the selection: the partition
    /// indexes it built and the row groups of the footers it read stay in
    /// the cache of the tables' file metadata, where the next scan takes
    /// them from.
    fn preload_scan_metadata(&self) -> crate::Result<()> {
        self.table.select_scan(&[])?;
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

    /// Every column but those the table's current schema requires.
    fn declared_nullability(&self) -> Vec<bool> {
        self.table.declared_nullability()
    }

    /// Nullability is answered for the files the filters keep, from their
    /// manifest entries.
    fn select_scan(&self, filters: &[Expression]) -> crate::Result<ScanSelection> {
        Ok(self.table.select_scan(filters)?)
    }

    /// One short listing from the version held; of a newer version, its
    /// metadata file and its manifest list are read.
    fn refresh(&self) -> crate::Result<Option<Arc<dyn Table>>> {
        let Some(next) = self.table.refresh()? else {
            return Ok(None);
        };
        Ok(Some(Arc::new(self.advanced_to(next))))
    }

    fn commit(
        &self,
        removed: RemovedFiles<'_>,
        added: &[TableFile],
        _data_change: bool,
        prepared: Option<PreparedCommit>,
    ) -> crate::Result<Option<Arc<dyn Table>>> {
        // An Iceberg snapshot says what a commit did by its shape: an append
        // when it removes nothing, else a replace, which changes no row.
        let added_entries = self.build_uncommitted_entries(added);
        let committed = match prepared {
            // The entries found when the commit was prepared are this
            // version's too while it lists the manifests they were read
            // from.
            Some(prepared) if prepared.rewrite.stands_on(&self.table) => self
                .table
                .commit_prepared(&prepared.removed_entries, added_entries, prepared.rewrite)?,
            Some(prepared) => self.table.commit_prepared(
                &self.find_removed_entries(removed)?,
                added_entries,
                prepared.rewrite,
            )?,
            None if removed.paths.is_empty() && !self.has_manifests_piled_up() => {
                self.table.append_without_merging_manifests(added_entries)?
            }
            None => self
                .table
                .commit(&self.find_removed_entries(removed)?, added_entries)?,
        };
        let Some(next) = committed else {
            return Ok(None);
        };
        Ok(Some(Arc::new(self.advanced_to(next))))
    }

    /// A replace has the manifests of its version written ahead: reading the
    /// manifests that list the files it removes and writing them again is
    /// most of its commit. An append has nothing written ahead.
    fn prepare_commit(
        &self,
        removed: RemovedFiles<'_>,
        added: &[TableFile],
    ) -> crate::Result<Option<PreparedCommit>> {
        if removed.paths.is_empty() {
            return Ok(None);
        }
        let removed_entries = self.find_removed_entries(removed)?;
        let rewrite = self
            .table
            .prepare_rewrite(&removed_entries, &self.build_uncommitted_entries(added))?;
        Ok(Some(PreparedCommit {
            rewrite,
            removed_entries,
        }))
    }

    fn does_prepared_commit_stand(&self, prepared: &PreparedCommit) -> bool {
        prepared.rewrite.stands_on(&self.table)
    }

    /// The table's own retention property, or the default.
    fn deleted_file_retention(&self) -> crate::Result<Duration> {
        Ok(self.table.snapshot_retention()?)
    }

    /// Read from the metadata this version holds.
    fn has_expired_history(&self, now_ms: u64) -> crate::Result<bool> {
        let now_ms = i64::try_from(now_ms).unwrap_or(i64::MAX);
        Ok(self.table.has_expired_history(now_ms)?)
    }

    /// Expire every snapshot and every previous metadata file that has
    /// outlived the retention window, as one metadata-only commit, and hand
    /// back the version that wrote with what it dropped. A version nothing
    /// has expired in is left with nothing read. A lost race reloads and
    /// looks again: whatever the winner expired is the winner's to delete,
    /// and only what this commit itself dropped is handed back. A commit
    /// that fails, or whose outcome is unknown, hands back nothing, so
    /// nothing is deleted for it. A table whose history is not linear (it
    /// has a branch or a tag besides `main`, or a snapshot that is not an
    /// ancestor of the current one) is refused before anything is
    /// committed.
    fn expire_snapshots(&self, now_ms: u64) -> crate::Result<Option<SnapshotExpiry>> {
        let now_ms = i64::try_from(now_ms).unwrap_or(i64::MAX);
        let mut refreshed: Option<IcebergTable> = None;
        for _ in 0..COMMIT_RETRIES {
            let table: &IcebergTable = refreshed.as_ref().unwrap_or(self);
            if !table.table.has_expired_history(now_ms)? {
                return Ok(None);
            }
            if let Some(expiry) = table.table.expire_snapshots(now_ms)? {
                return Ok(Some(SnapshotExpiry {
                    table: Arc::new(table.advanced_to(expiry.table)),
                    expired: expiry.expired,
                }));
            }
            // Something had expired and the commit handed nothing back:
            // another writer committed the next version first.
            let Some(next) = table.table.refresh()? else {
                return Ok(None);
            };
            refreshed = Some(table.advanced_to(next));
        }
        Ok(None)
    }

    /// Delete, through `storage`, what `expired`, the expiry that committed
    /// this version, left to delete: the files its snapshots retired, the
    /// manifests and manifest lists only they name, and the statistics
    /// files and previous metadata files it dropped
    /// ([`collect_expired_objects`](iceberg_tables::Table::collect_expired_objects)).
    /// They are found from the dropped snapshots and the oldest snapshot
    /// kept, the table's history being linear. An object
    /// outside the table's own directory (a file the table adopted where it
    /// lay) is not the table's to delete and is left. The version hint may
    /// name a metadata file deleted here: a store lists on from the name
    /// to the versions after it. Nothing here fails the maintenance step:
    /// what is not deleted stays in the table's directory, named by no
    /// metadata.
    fn delete_snapshot_objects(&self, storage: &TableStorage, expired: &ExpiredSnapshots) {
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
            "expired snapshots and previous metadata files and deleted what they left unused"
        );
    }

    /// An Iceberg table retires a file by committing a snapshot without it,
    /// and what a snapshot retired is deleted once every snapshot before it
    /// has expired, as a metadata file is when the one that replaced it has
    /// aged past the window
    /// ([`delete_snapshot_objects`](Self::delete_snapshot_objects)): nothing
    /// else is deleted.
    fn delete_expired_objects(
        &self,
        _name: &SchemaQualifiedTableName,
        _storage: &TableStorage,
        _now_ms: u64,
    ) -> crate::Result<()> {
        Ok(())
    }
}
