//! The Delta Lake format: a `_delta_log` per table, read and written through
//! Delta Kernel, with every live file's footer held in memory beside its log
//! entry so a query's scan view is built with no I/O.

pub(crate) mod log;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use dispatch::DataFlowDispatcher;
use object_storage::{self, DataFile, FileRef, ObjectPath, ObjectStore};
use parquet_engine::{FileStats, ParquetTable, PushedPredicate, RowGroupMetadata};
use planner::catalog::{Column, SchemaQualifiedTableName};
use planner::expression::{CompareType, Expression};
use tracing::{info, warn};
use uuid::Uuid;

use crate::Result;
use crate::file_entry::{ColumnStatFilter, FileEntry, PartitionEqFilter, constant_partition};
use crate::file_layout::{EntryLayout, SortRangeEncoder, VisitFile};
use crate::table::{TableFile, TableStorage};
use crate::table_format::{SnapshotExpiry, Table, TableFormat};

/// What every Delta table of one database shares: the Kernel engine its log is
/// read and written through (built once for the store, so a refresh, a commit
/// or a vacuum sweep reuses one object-store client and task executor), the
/// store, and the pool footers are fetched on.
struct DeltaShared {
    engine: log::DeltaEngine,
    store: Arc<dyn ObjectStore>,
    dispatcher: DataFlowDispatcher,
}

impl DeltaShared {
    /// Fetch the footers for `entries`, one per entry in order, each file's
    /// parsed schema reconciled by name against `columns`, the schema declared
    /// by the version the entries come from.
    fn fetch_footers(
        &self,
        location: &ObjectPath,
        entries: &[FileEntry],
        columns: &[Column],
    ) -> Result<Vec<parquet_engine::FileRowGroups>> {
        let to_fetch: Vec<DataFile> = entries
            .iter()
            .map(|entry| {
                entry
                    .file
                    .clone()
                    .into_data_file(self.store.as_ref(), location)
            })
            .collect::<object_storage::Result<_>>()?;
        Ok(parquet_engine::load_file_row_groups(
            &self.dispatcher,
            &to_fetch,
            parquet_engine::TableColumns::by_name(columns.to_vec()),
        )?)
    }
}

/// The Delta backend of one database.
pub(crate) struct DeltaFormat {
    shared: Arc<DeltaShared>,
}

impl DeltaFormat {
    pub(crate) fn new(store: Arc<dyn ObjectStore>, dispatcher: DataFlowDispatcher) -> Result<Self> {
        let engine = log::DeltaEngine::new(store.as_ref())?;
        Ok(Self {
            shared: Arc::new(DeltaShared {
                engine,
                store,
                dispatcher,
            }),
        })
    }
}

impl TableFormat for DeltaFormat {
    /// Write Delta version 0 with its protocol, metadata and initial `Add`
    /// actions. An adopted file whose partition columns are constant is
    /// stamped with that partition from its stats, so it is recorded in the log
    /// and prunes like a written file; one whose values vary is adopted
    /// without a tuple.
    fn create_table(
        &self,
        directory: &ObjectPath,
        _name: &SchemaQualifiedTableName,
        _id: Uuid,
        columns: Vec<Column>,
        partition_by: Vec<String>,
        sort_by: Vec<String>,
        mut adopted: Vec<TableFile>,
    ) -> Result<Arc<dyn Table>> {
        if !partition_by.is_empty() {
            for file in &mut adopted {
                file.entry.partition = file
                    .entry
                    .stats
                    .as_ref()
                    .and_then(|stats| constant_partition(stats, &partition_by));
            }
        }
        let entries: Vec<FileEntry> = adopted.iter().map(|file| file.entry.clone()).collect();
        // Version 0's commit yields the snapshot the new table starts at, so the
        // log it just wrote is not read back to open it.
        let snapshot = log::initialize_table(
            &self.shared.engine,
            self.shared.store.as_ref(),
            directory,
            &columns,
            &partition_by,
            &sort_by,
            &entries,
        )?;
        Ok(Arc::new(DeltaTable {
            shared: self.shared.clone(),
            location: directory.clone(),
            snapshot,
            columns,
            partition_by,
            sort_by,
            entries,
            footers: footers_by_path(adopted),
            tombstones: HashMap::new(),
            layout: OnceLock::new(),
        }))
    }

    /// Read the table's log at its latest version and fetch every live file's
    /// footer.
    fn load_table(
        &self,
        directory: &ObjectPath,
        _name: &SchemaQualifiedTableName,
        _id: Uuid,
    ) -> Result<Arc<dyn Table>> {
        let delta_uri = log::table_uri(&self.shared.store.location_uri(), directory)?;
        let state = log::load_table(&delta_uri, &self.shared.engine)?;
        let fetched = self
            .shared
            .fetch_footers(directory, &state.file_entries, &state.columns)?;
        let footers = state
            .file_entries
            .iter()
            .zip(fetched)
            .map(|(entry, footer)| (entry.file.path.clone(), footer.row_groups))
            .collect();
        Ok(Arc::new(DeltaTable {
            shared: self.shared.clone(),
            location: directory.clone(),
            snapshot: state.snapshot,
            columns: state.columns,
            partition_by: state.partition_by,
            sort_by: state.sort_by,
            entries: state.file_entries,
            footers,
            tombstones: state.tombstones,
            layout: OnceLock::new(),
        }))
    }
}

type Footers = HashMap<ObjectPath, Vec<Arc<RowGroupMetadata>>>;

fn footers_by_path(files: Vec<TableFile>) -> Footers {
    files
        .into_iter()
        .map(|file| (file.entry.file.path, file.row_groups))
        .collect()
}

/// One Delta table at one log version: the snapshot its files were built from
/// and that every commit from it is written on top of, the live entries, their
/// footers, and the retired files the log still remembers.
struct DeltaTable {
    shared: Arc<DeltaShared>,
    location: ObjectPath,
    snapshot: Arc<delta_kernel::Snapshot>,
    columns: Vec<Column>,
    partition_by: Vec<String>,
    sort_by: Vec<String>,
    entries: Vec<FileEntry>,
    /// Every live file's row groups, by path.
    footers: Footers,
    /// The files the log retired but still remembers, each by the Unix ms its
    /// commit retired it: what vacuum ages a retired file by, so the retention
    /// window runs from the moment the file stopped being referenced rather
    /// than from its storage mtime. Rebuilt from the log on a reload, which
    /// bounds it to what the log still carries, and extended by every commit
    /// made from this value's lineage, which drops the entries a window older
    /// than itself.
    tombstones: HashMap<ObjectPath, u64>,
    /// The entries as maintenance walks them, derived when first asked for.
    layout: OnceLock<EntryLayout>,
}

impl DeltaTable {
    /// The entries grouped by partition, each with the stretch of the sort
    /// key its statistics bound, and where each path sits among them.
    fn layout(&self) -> Result<&EntryLayout> {
        if let Some(layout) = self.layout.get() {
            return Ok(layout);
        }
        let sort_types = self
            .sort_by
            .iter()
            .filter_map(|name| self.columns.iter().find(|column| &column.name == name))
            .map(|column| planner::types::physical_arrow_type(&column.col_type))
            .collect();
        let sort_ranges = SortRangeEncoder::new(&self.sort_by, sort_types)?;
        let layout = EntryLayout::of(&self.entries, &self.partition_by, &sort_ranges)?;
        Ok(self.layout.get_or_init(|| layout))
    }

    /// The live entry at `path`.
    fn find_entry(&self, path: &ObjectPath) -> Result<Option<&FileEntry>> {
        Ok(self
            .layout()?
            .find_position(path)
            .map(|position| &self.entries[position]))
    }

    /// This table at the version `state` describes, keeping the footers already
    /// held (carrying forward their sort bounds, which the log does not
    /// persist) and fetching only the ones for files not yet held. Built whole
    /// before anything is handed back, so a failed fetch leaves the caller at
    /// the version whose files it holds.
    fn at_version(&self, state: log::DeltaTableState) -> Result<Self> {
        let to_fetch: Vec<FileEntry> = state
            .file_entries
            .iter()
            .filter(|entry| !self.footers.contains_key(&entry.file.path))
            .cloned()
            .collect();
        let mut fetched = self
            .shared
            .fetch_footers(&self.location, &to_fetch, &state.columns)?
            .into_iter();
        let mut footers = Footers::with_capacity(state.file_entries.len());
        for entry in &state.file_entries {
            let row_groups = match self.footers.get(&entry.file.path) {
                Some(held) => held.clone(),
                None => {
                    fetched
                        .next()
                        .expect("a footer is fetched for every file not held")
                        .row_groups
                }
            };
            footers.insert(entry.file.path.clone(), row_groups);
        }
        Ok(Self {
            shared: self.shared.clone(),
            location: self.location.clone(),
            snapshot: state.snapshot,
            columns: state.columns,
            partition_by: state.partition_by,
            sort_by: state.sort_by,
            entries: state.file_entries,
            footers,
            tombstones: state.tombstones,
            layout: OnceLock::new(),
        })
    }

    /// This table with `removed` dropped and `added` adopted, at `snapshot`.
    fn with_swap(
        &self,
        snapshot: Arc<delta_kernel::Snapshot>,
        removed: &[ObjectPath],
        added: &[TableFile],
        tombstones: HashMap<ObjectPath, u64>,
    ) -> Self {
        let mut entries: Vec<FileEntry> = self
            .entries
            .iter()
            .filter(|entry| !removed.contains(&entry.file.path))
            .cloned()
            .collect();
        let mut footers: Footers = self
            .footers
            .iter()
            .filter(|(path, _)| !removed.contains(path))
            .map(|(path, row_groups)| (path.clone(), row_groups.clone()))
            .collect();
        for file in added {
            entries.push(file.entry.clone());
            footers.insert(file.entry.file.path.clone(), file.row_groups.clone());
        }
        Self {
            shared: self.shared.clone(),
            location: self.location.clone(),
            snapshot,
            columns: self.columns.clone(),
            partition_by: self.partition_by.clone(),
            sort_by: self.sort_by.clone(),
            entries,
            footers,
            tombstones,
            layout: OnceLock::new(),
        }
    }

    /// The tombstones after a commit at `retired_at_ms` retired `removed`: a
    /// clock reading taken just before the transaction was opened, back to back
    /// with the one Kernel stamps their `Remove` tombstones with, so they are
    /// dated within milliseconds of what a reload from the log would date them.
    /// The same reading retires the tombstones a full retention window older
    /// than this commit: any sweep run after it sees them past the window, so
    /// they date nothing, and dropping them is what bounds a writer's copy,
    /// which only ever extends its tombstones between reloads, to one window's
    /// worth of retirements.
    fn tombstones_after(
        &self,
        removed: &[ObjectPath],
        retired_at_ms: u64,
    ) -> HashMap<ObjectPath, u64> {
        let mut tombstones = self.tombstones.clone();
        if removed.is_empty() {
            return tombstones;
        }
        let cutoff = retired_at_ms
            .saturating_sub(log::deleted_file_retention(&self.snapshot).as_millis() as u64);
        tombstones.retain(|_, at| *at > cutoff);
        tombstones.extend(removed.iter().map(|path| (path.clone(), retired_at_ms)));
        tombstones
    }

    /// The equality predicates among `predicates` as partition-filter
    /// candidates: the column name and the typed scalar. A file's entry
    /// intersects these with the table's partition columns, so yielding every
    /// equality (not just ones on partition columns) is fine; a non-partition
    /// column prunes no files.
    fn partition_filter_candidates(
        &self,
        predicates: &[PushedPredicate],
    ) -> Vec<PartitionEqFilter> {
        predicates
            .iter()
            .filter(|predicate| matches!(predicate.compare_type, CompareType::Equal))
            .filter_map(|predicate| {
                Some(PartitionEqFilter {
                    column: self.columns.get(predicate.column_idx)?.name.clone(),
                    value: predicate.value.clone(),
                })
            })
            .collect()
    }

    /// The predicates among `predicates` that bound a file's stats: a plain
    /// top-level column comparison (no variant path) paired with its typed
    /// constant. Variant paths are excluded: their stats live in a shredded
    /// leaf, pruned per row group, not in the file's column stats. A column the
    /// file's stats don't bound simply prunes no files.
    fn stat_filter_candidates(&self, predicates: &[PushedPredicate]) -> Vec<ColumnStatFilter> {
        predicates
            .iter()
            .filter(|predicate| predicate.path.is_empty())
            .filter_map(|predicate| {
                Some(ColumnStatFilter {
                    column: self.columns.get(predicate.column_idx)?.name.clone(),
                    compare_type: predicate.compare_type,
                    value: predicate.value.clone(),
                })
            })
            .collect()
    }
}

impl DeltaTable {
    /// The row groups of every live file, from the footers held.
    fn live_row_groups(&self) -> impl Iterator<Item = &Arc<RowGroupMetadata>> {
        self.entries
            .iter()
            .filter_map(|entry| self.footers.get(&entry.file.path))
            .flatten()
    }
}

impl Table for DeltaTable {
    fn version(&self) -> u64 {
        self.snapshot.version()
    }

    fn columns(&self) -> &[Column] {
        &self.columns
    }

    fn partition_by(&self) -> &[String] {
        &self.partition_by
    }

    fn sort_by(&self) -> &[String] {
        &self.sort_by
    }

    fn count_files(&self) -> Result<usize> {
        Ok(self.entries.len())
    }

    /// The entries in the log's order, grouped by partition.
    fn walk_files(&self, visit: &mut VisitFile<'_>) -> Result<()> {
        self.layout()?.walk(&self.entries, visit);
        Ok(())
    }

    fn contains_file(&self, path: &ObjectPath) -> Result<bool> {
        Ok(self.find_entry(path)?.is_some())
    }

    /// An entry holds everything the log recorded for its file.
    fn find_file(&self, path: &ObjectPath) -> Result<Option<FileEntry>> {
        Ok(self.find_entry(path)?.cloned())
    }

    /// An entry holds everything the log recorded for its file.
    fn file_stats(&self, path: &ObjectPath) -> Result<Option<Arc<FileStats>>> {
        Ok(self.find_entry(path)?.and_then(|entry| entry.stats.clone()))
    }

    /// From the footers held: a column is free of NULLs when every row
    /// group of every file has it and says so; a file missing the column
    /// reads as all-NULL.
    fn nullability(&self) -> Result<Vec<bool>> {
        let row_groups: Vec<&Arc<RowGroupMetadata>> = self.live_row_groups().collect();
        Ok(self
            .columns
            .iter()
            .map(|column| {
                self.entries.is_empty()
                    || row_groups.iter().any(|row_group| {
                        match row_group.schema.index_of(&column.name) {
                            Ok(position) => row_group.column_may_hold_nulls(position),
                            Err(_) => true,
                        }
                    })
            })
            .collect())
    }

    /// The sum of the held footers' row-group counts.
    fn total_rows(&self) -> Result<u64> {
        Ok(self
            .live_row_groups()
            .map(|row_group| row_group.num_rows.max(0) as u64)
            .sum())
    }

    /// The held footers count every row.
    fn count_rows(&self) -> Result<Option<u64>> {
        Ok(Some(
            self.live_row_groups()
                .map(|row_group| row_group.live_rows().max(0) as u64)
                .sum(),
        ))
    }

    /// Delta files resolve columns by name.
    fn field_ids(&self) -> Option<Vec<i32>> {
        None
    }

    /// Every live file's footer is held, so this is a lookup.
    fn row_groups(&self, files: &[FileRef]) -> Result<Vec<Vec<Arc<RowGroupMetadata>>>> {
        Ok(files
            .iter()
            .map(|file| {
                self.footers
                    .get(&file.path)
                    .expect("a file of this version has its footer held")
                    .clone()
            })
            .collect())
    }

    /// The files whose recorded partition tuple and column bounds admit a
    /// row the filters keep, in entry order, with the footers held.
    fn plan_scan(&self, filters: &[Expression]) -> Result<Arc<ParquetTable>> {
        let predicates: Vec<PushedPredicate> = filters
            .iter()
            .flat_map(PushedPredicate::from_filter)
            .collect();
        let partition_filters = self.partition_filter_candidates(&predicates);
        let stat_filters = self.stat_filter_candidates(&predicates);
        let surviving: Vec<&FileEntry> = self
            .entries
            .iter()
            .filter(|entry| entry.maybe_matches_partition(&self.partition_by, &partition_filters))
            .filter(|entry| entry.maybe_matches_stats(&stat_filters))
            .collect();
        let surviving: Vec<FileRef> = surviving
            .into_iter()
            .map(|entry| entry.file.clone())
            .collect();
        let row_groups = self.row_groups(&surviving)?.into_iter().flatten().collect();
        Ok(Arc::new(ParquetTable::new(row_groups)))
    }

    /// One log listing when nothing changed (Kernel updates the snapshot in
    /// hand rather than rebuilding it); otherwise the new version with the
    /// footers of its new files fetched.
    fn refresh(&self) -> Result<Option<Arc<dyn Table>>> {
        let Some(state) = log::refresh_table(&self.snapshot, &self.shared.engine)? else {
            return Ok(None);
        };
        Ok(Some(Arc::new(self.at_version(state)?)))
    }

    fn commit(
        &self,
        removed: &[ObjectPath],
        added: &[TableFile],
        data_change: bool,
        now_ms: u64,
    ) -> Result<Option<Arc<dyn Table>>> {
        let removed_entries = removed
            .iter()
            .map(|path| {
                self.find_entry(path)?
                    .cloned()
                    .ok_or_else(|| crate::Error::CommitConflict {
                        location: self.location.as_str().to_string(),
                        file: path.to_string(),
                    })
            })
            .collect::<Result<Vec<FileEntry>>>()?;
        let added_entries: Vec<FileEntry> = added.iter().map(|file| file.entry.clone()).collect();
        let Some(committed) = log::commit_file_changes(
            &self.shared.engine,
            &self.snapshot,
            &removed_entries,
            &added_entries,
            data_change,
        )?
        else {
            return Ok(None);
        };
        let tombstones = self.tombstones_after(removed, now_ms);
        Ok(Some(Arc::new(
            self.with_swap(committed, removed, added, tombstones),
        )))
    }

    fn with_files_swapped(&self, removed: &[ObjectPath], added: Vec<TableFile>) -> Arc<dyn Table> {
        Arc::new(self.with_swap(
            self.snapshot.clone(),
            removed,
            &added,
            self.tombstones.clone(),
        ))
    }

    /// The table's `delta.deletedFileRetentionDuration`, or the datastore's
    /// 4-hour default when unset.
    fn deleted_file_retention(&self) -> Result<Duration> {
        Ok(log::deleted_file_retention(&self.snapshot))
    }

    /// A Delta table has nothing to expire here: its checkpoints are written
    /// on the commit path, and what a checkpoint supersedes is deleted by
    /// [`delete_expired_objects`](Self::delete_expired_objects).
    fn expire_snapshots(
        &self,
        _storage: &TableStorage,
        _now_ms: u64,
    ) -> Result<Option<SnapshotExpiry>> {
        Ok(None)
    }

    /// Delete every data file this version does not reference once it has been
    /// unreferenced for the retention window: a retired file's window runs from
    /// the tombstone held for it, an orphan's (an upload no commit adopted) from
    /// its storage mtime. Then delete the commits, checkpoints and checksums a
    /// later checkpoint has made redundant, once past the log-retention window.
    /// Failed deletions are retried by the next sweep.
    fn delete_expired_objects(
        &self,
        name: &SchemaQualifiedTableName,
        storage: &TableStorage,
        now_ms: u64,
    ) -> Result<()> {
        let cutoff =
            now_ms.saturating_sub(log::deleted_file_retention(&self.snapshot).as_millis() as u64);
        let live: HashSet<&ObjectPath> =
            self.entries.iter().map(|entry| &entry.file.path).collect();

        let mut deleted = 0u64;
        let mut first_error = None;
        for (path, modified_ms) in storage.list_data_files()? {
            if live.contains(&path) {
                continue;
            }
            // A retired file is dated by the commit that retired it, which keeps
            // it whatever the storage mtime says: it may have been live for days
            // before a merge or delete retired it, and a reader on the prior
            // snapshot still needs the bytes. A file with no tombstone, an
            // orphan upload or one retired before the window, is dated the only
            // way left, by when its bytes landed in storage.
            let unreferenced_since_ms = self.tombstones.get(&path).copied().unwrap_or(modified_ms);
            if unreferenced_since_ms > cutoff {
                continue;
            }
            if let Err(error) = storage.delete_data_file(&path) {
                warn!(table = %name, file = %path, error = %error, "vacuum: deleting file failed");
                first_error.get_or_insert(error);
                continue;
            }
            deleted += 1;
        }
        if deleted > 0 {
            info!(table = %name, files = deleted, "vacuumed files");
        }

        match log::cleanup_log(
            self.shared.store.as_ref(),
            &self.location,
            &self.snapshot,
            now_ms,
        ) {
            Ok(cleaned) if cleaned > 0 => {
                info!(table = %name, log_files = cleaned, "cleaned up superseded log files");
            }
            Ok(_) => {}
            Err(error) => {
                warn!(table = %name, error = %error, "vacuum: log cleanup failed");
                first_error.get_or_insert(error.into());
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// A Delta table's routine deletion
    /// ([`delete_expired_objects`](Self::delete_expired_objects)) lists the
    /// table's directory itself, so a sweep has nothing further to find.
    fn sweep_unused_objects(
        &self,
        _name: &SchemaQualifiedTableName,
        _storage: &TableStorage,
        _now_ms: u64,
    ) -> Result<()> {
        Ok(())
    }
}
