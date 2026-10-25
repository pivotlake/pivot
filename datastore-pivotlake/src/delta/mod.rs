//! The Delta Lake format: a `_delta_log` per table, read and written through
//! Delta Kernel, with every live file's footer held in memory beside its log
//! entry so a query's scan view is built with no I/O.

pub(crate) mod log;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use dispatch::DataFlowDispatcher;
use object_storage::{self, DataFile, ObjectPath, ObjectStore};
use parquet_engine::{ParquetTable, PushedPredicate, RowGroupMetadata};
use planner::catalog::{Column, SchemaQualifiedTableName};
use planner::expression::Expression;
use tracing::{info, warn};
use uuid::Uuid;

use crate::Result;
use crate::file_entry::{
    FileEntry, constant_partition, partition_filter_candidates, stat_filter_candidates,
};
use crate::format::{Format, FormatTable, TableFormat};
use crate::table::{TableFile, TableStorage};

/// What every Delta table of one database shares: the Kernel engine its log is
/// read and written through (built once for the store, so a refresh, a commit
/// or a vacuum sweep reuses one object-store client and task executor), the
/// store, and the pool footers are fetched on.
struct DeltaShared {
    engine: log::DeltaEngine,
    store: Arc<dyn ObjectStore>,
    dispatcher: DataFlowDispatcher,
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

impl Format for DeltaFormat {
    fn name(&self) -> TableFormat {
        TableFormat::Delta
    }

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
    ) -> Result<Arc<dyn FormatTable>> {
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
        }))
    }

    /// Read the table's log at its latest version and fetch every live file's
    /// footer.
    fn load_table(
        &self,
        directory: &ObjectPath,
        _name: &SchemaQualifiedTableName,
        _id: Uuid,
    ) -> Result<Arc<dyn FormatTable>> {
        let delta_uri = log::table_uri(&self.shared.store.location_uri(), directory)?;
        let state = log::load_table(&delta_uri, &self.shared.engine)?;
        let fetched = fetch_footers(&self.shared, directory, &state.file_entries, &state.columns)?;
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
        }))
    }
}

/// Fetch the footers for `entries`, one per entry in order, each file's parsed
/// schema reconciled by name against `columns`, the schema declared by the
/// version the entries come from.
fn fetch_footers(
    shared: &DeltaShared,
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
                .into_data_file(shared.store.as_ref(), location)
        })
        .collect::<object_storage::Result<_>>()?;
    Ok(parquet_engine::load_file_row_groups(
        &shared.dispatcher,
        &to_fetch,
        parquet_engine::TableColumns::by_name(columns.to_vec()),
    )?)
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
}

impl DeltaTable {
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
        let mut fetched =
            fetch_footers(&self.shared, &self.location, &to_fetch, &state.columns)?.into_iter();
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
}

impl FormatTable for DeltaTable {
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

    fn entries(&self) -> &[FileEntry] {
        &self.entries
    }

    /// Delta files resolve columns by name.
    fn field_ids(&self) -> Option<Vec<i32>> {
        None
    }

    /// Every live file's footer is held, so this is a lookup.
    fn row_groups(&self, files: &[&FileEntry]) -> Result<Vec<Vec<Arc<RowGroupMetadata>>>> {
        Ok(files
            .iter()
            .map(|entry| {
                self.footers
                    .get(&entry.file.path)
                    .expect("a file of this version has its footer held")
                    .clone()
            })
            .collect())
    }

    /// The files whose recorded partition tuple and column bounds admit a
    /// row the filters keep, in entry order, with the footers held.
    fn scan(&self, filters: &[Expression]) -> Result<Arc<ParquetTable>> {
        let predicates: Vec<PushedPredicate> = filters
            .iter()
            .flat_map(PushedPredicate::from_filter)
            .collect();
        let partition_filters = partition_filter_candidates(&predicates, &self.columns);
        let stat_filters = stat_filter_candidates(&predicates, &self.columns);
        let surviving: Vec<&FileEntry> = self
            .entries
            .iter()
            .filter(|entry| entry.maybe_matches_partition(&self.partition_by, &partition_filters))
            .filter(|entry| entry.maybe_matches_stats(&stat_filters))
            .collect();
        let row_groups = self.row_groups(&surviving)?.into_iter().flatten().collect();
        Ok(Arc::new(ParquetTable::new(row_groups)))
    }

    /// One log listing when nothing changed (Kernel updates the snapshot in
    /// hand rather than rebuilding it); otherwise the new version with the
    /// footers of its new files fetched.
    fn refresh(&self) -> Result<Option<Arc<dyn FormatTable>>> {
        let Some(state) = log::refresh_table(&self.snapshot, &self.shared.engine)? else {
            return Ok(None);
        };
        Ok(Some(Arc::new(self.at_version(state)?)))
    }

    fn commit(
        &self,
        removed: &[FileEntry],
        added: &[TableFile],
        data_change: bool,
        now_ms: u64,
    ) -> Result<Option<Arc<dyn FormatTable>>> {
        let added_entries: Vec<FileEntry> = added.iter().map(|file| file.entry.clone()).collect();
        let Some(committed) = log::commit_file_changes(
            &self.shared.engine,
            &self.snapshot,
            removed,
            &added_entries,
            data_change,
        )?
        else {
            return Ok(None);
        };
        let removed_paths: Vec<ObjectPath> = removed
            .iter()
            .map(|entry| entry.file.path.clone())
            .collect();
        let tombstones = self.tombstones_after(&removed_paths, now_ms);
        Ok(Some(Arc::new(self.with_swap(
            committed,
            &removed_paths,
            added,
            tombstones,
        ))))
    }

    fn with_files_swapped(
        &self,
        removed: &[ObjectPath],
        added: Vec<TableFile>,
    ) -> Arc<dyn FormatTable> {
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

    /// A Delta table's log housekeeping (checkpointing) happens on the commit
    /// path, and what a checkpoint supersedes is deleted by the sweep.
    fn maintain(&self, _now_ms: u64) -> Result<Option<Arc<dyn FormatTable>>> {
        Ok(None)
    }

    /// Delete every data file this version does not reference once it has been
    /// unreferenced for the retention window: a retired file's window runs from
    /// the tombstone held for it, an orphan's (an upload no commit adopted) from
    /// its storage mtime. Then delete the commits, checkpoints and checksums a
    /// later checkpoint has made redundant, once past the log-retention window.
    /// Failed deletions are retried by the next sweep.
    fn sweep(
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
}
