//! Compaction of the catalog's tables through the shared compacter: this
//! datastore as the host that names its tables and lands a swap through the
//! catalog, and each table as the copy a round refreshes, merges through and
//! folds its swaps into.
//!
//! A table whose current snapshot holds live delete files is not compacted:
//! a merge reads its inputs' rows as the files hold them, and a deletion
//! vector or an equality delete would be dropped on the floor, resurrecting
//! the rows it deletes. The engine that wrote the deletes rewrites such a
//! table.

use std::sync::Arc;

use arrow_schema::SchemaRef;
use dispatch::DataFlowDispatcher;
use iceberg_tables::commit::{COMMIT_RETRIES, now_unix_ms};
use iceberg_tables::data_files::DataFileEntry;
use iceberg_tables::sink::write_schema;
use iceberg_tables::{FileDescriptor, TableCatalog};
use object_storage::{ObjectPath, ObjectStore};
use parquet_engine::RowGroupMetadata;
use parquet_engine::file_entry::FileEntry;
use planner::catalog::{Column, SchemaQualifiedTableName};
use table_compaction::{
    CompacterHandle, CompactionError, CompactionHost, CompactionTable, DEFAULT_COMPACT_BYTES,
    DEFAULT_COMPACT_PARALLELISM, DEFAULT_LAYOUT_CLIQUE_SIZE, DEFAULT_MIN_FILES_TO_MERGE, TableFile,
    default_merge_target_bytes,
};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::write::writable_target;
use crate::{Error, IcebergDatastore, LoadedTable, Result, TableEntry};

/// One catalog table at one version, as the compacter holds it: the shared
/// table beside what a merge needs that it does not say, its live files in
/// the compacter's shape, where under the store its outputs go, and the
/// schema they are encoded under.
#[derive(Clone)]
pub struct CompactionTarget {
    name: SchemaQualifiedTableName,
    table: Arc<LoadedTable>,
    entries: Vec<FileEntry>,
    /// The table's directory, which a merge writes under and reports its
    /// outputs relative to, and the data path below it.
    directory: ObjectPath,
    file_prefix: ObjectPath,
    /// The declared columns under the table's field ids and nullability:
    /// what a merged file is stamped with, whatever its inputs carried.
    written_schema: SchemaRef,
}

impl CompactionTarget {
    /// `table` as the compacter works on it, or why it cannot: live delete
    /// files in the current snapshot, or a shape this writer refuses (see
    /// [`writable_target`]).
    pub(crate) fn load(name: &SchemaQualifiedTableName, table: Arc<LoadedTable>) -> Result<Self> {
        if table.has_live_delete_files() {
            return Err(Error::DeleteFilesNotCompacted {
                table: name.to_string(),
            });
        }
        let (insert_target, _) = writable_target(name, table.metadata(), table.store())?;
        let written_schema = write_schema(
            &insert_target.columns,
            &insert_target.nullability,
            insert_target.field_ids.as_deref(),
        );
        let entries = table.entries()?.iter().map(to_file_entry).collect();
        Ok(Self {
            name: name.clone(),
            table,
            entries,
            directory: insert_target.directory,
            file_prefix: insert_target.file_prefix,
            written_schema,
        })
    }

    /// This target over `table`, the same metadata version with its files
    /// swapped, keeping what was derived from the metadata.
    fn with_table(&self, table: LoadedTable) -> Result<Self> {
        let entries = table.entries()?.iter().map(to_file_entry).collect();
        Ok(Self {
            name: self.name.clone(),
            table: Arc::new(table),
            entries,
            directory: self.directory.clone(),
            file_prefix: self.file_prefix.clone(),
            written_schema: self.written_schema.clone(),
        })
    }
}

/// A manifest's entry as the compacter holds it: the same identity, tuple
/// and statistics, without where the manifest records it.
fn to_file_entry(entry: &DataFileEntry) -> FileEntry {
    FileEntry {
        file: entry.file.clone(),
        partition: entry.partition.clone(),
        stats: entry.stats.clone(),
    }
}

/// The entry a commit records for a file a merge just uploaded.
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

impl CompactionTable for CompactionTarget {
    type Error = Error;

    fn id(&self) -> Uuid {
        self.table.uuid()
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

    /// Read over the pool the footers not yet held, once; answer the rest
    /// from memory.
    fn row_groups(&self, entries: &[&FileEntry]) -> Result<Vec<Vec<Arc<RowGroupMetadata>>>> {
        let location = self.table.location();
        let descriptors = entries
            .iter()
            .map(|entry| {
                Ok(FileDescriptor {
                    path: location.uri_for(&entry.file.path)?,
                    length: entry.file.size,
                })
            })
            .collect::<iceberg_tables::Result<Vec<_>>>()?;
        Ok(self
            .table
            .read_footers(&descriptors)?
            .into_iter()
            .map(|footer| footer.row_groups)
            .collect())
    }

    /// Ask the catalog for a newer version; one is read whole, keeping the
    /// footers held for the files it still has.
    fn refresh(&mut self) -> Result<bool> {
        let Some(next) = self.table.refresh()? else {
            return Ok(false);
        };
        next.carry_footers_from(&self.table)?;
        *self = Self::load(&self.name, Arc::new(next))?;
        Ok(true)
    }

    fn apply_file_swap(&mut self, removed: &[ObjectPath], added: Vec<TableFile>) {
        let swapped = self
            .table
            .with_files_swapped(removed, added.iter().map(uncommitted_entry).collect())
            .expect("a swap keeps the metadata this version was built from");
        for file in &added {
            swapped
                .remember_footer(&file.entry.file.path, file.row_groups.clone())
                .expect("a swapped-in file is under the table's store");
        }
        *self = self
            .with_table(swapped)
            .expect("a swap keeps the manifests this version decoded");
    }

    fn dispatcher(&self) -> &DataFlowDispatcher {
        self.table.dispatcher()
    }

    fn store(&self) -> Arc<dyn ObjectStore> {
        self.table.store().store().clone()
    }

    fn directory(&self) -> ObjectPath {
        self.directory.clone()
    }

    fn file_prefix(&self) -> ObjectPath {
        self.file_prefix.clone()
    }

    fn written_schema(&self) -> Option<SchemaRef> {
        Some(self.written_schema.clone())
    }

    /// `path` is one of a merge's outputs, relative to the table's directory
    /// as the upload reported it.
    fn delete_data_file(&self, path: &ObjectPath) -> Result<()> {
        self.table
            .store()
            .store()
            .delete(&self.directory.join(path.as_str()))?;
        Ok(())
    }

    fn location(&self) -> String {
        self.table.metadata().location().to_string()
    }
}

impl CompactionError for Error {
    fn is_commit_conflict(&self) -> bool {
        matches!(
            self,
            Error::Compaction(table_compaction::Error::CommitConflict { .. })
        )
    }
}

impl CompactionHost for IcebergDatastore {
    type Table = CompactionTarget;
    type Error = Error;

    /// Every table of the index the compacter can work on. One it cannot
    /// (delete files, a shape this writer refuses) is left out of the round
    /// and said so at debug level: the reason stands until the table
    /// changes, so a round is not the place to repeat it.
    fn tables(&self) -> Vec<(SchemaQualifiedTableName, CompactionTarget)> {
        let index = self.index();
        let tables = match self.load_all_tables(&index) {
            Ok(tables) => tables,
            Err(error) => {
                warn!(error = %error, "compaction: loading the catalog's tables failed");
                return Vec::new();
            }
        };
        tables
            .into_iter()
            .filter_map(|(name, table)| match CompactionTarget::load(&name, table) {
                Ok(target) => Some((name, target)),
                Err(error) => {
                    debug!(table = %name, error = %error, "compaction: table left out of the round");
                    None
                }
            })
            .collect()
    }

    fn table_by_id(&self, id: &Uuid) -> Option<CompactionTarget> {
        let index = self.index();
        let name = index.table_name_by_uuid(id)?;
        let table = match self.load_table(&index, &name) {
            Ok(table) => table?,
            Err(error) => {
                warn!(table = %name, error = %error, "compaction: loading the table failed");
                return None;
            }
        };
        match CompactionTarget::load(&name, table) {
            Ok(target) => Some(target),
            Err(error) => {
                warn!(table = %name, error = %error, "compaction: the table cannot be compacted");
                None
            }
        }
    }

    /// Commit the swap as one `replace` snapshot through the catalog, built
    /// again on the winner each time another writer lands first, and publish
    /// the table at the landed version into the index. A removed file that
    /// the current version no longer holds is a conflict: another writer
    /// swapped it out, and the merge's outputs must not be added on top.
    fn commit_swap(&self, id: Uuid, removed: &[ObjectPath], added: Vec<TableFile>) -> Result<()> {
        let index = self.index();
        let name = index
            .table_name_by_uuid(&id)
            .ok_or_else(|| Error::TableNotFound(id.to_string()))?;
        let held = self
            .load_table(&index, &name)?
            .ok_or_else(|| Error::TableNotFound(name.to_string()))?;
        let table_name = name.to_string();
        let catalog: Arc<dyn TableCatalog> = self.rest_table_catalog(&name);
        let load = || {
            LoadedTable::load(
                held.dispatcher(),
                Arc::clone(&catalog),
                held.location().clone(),
                held.store().clone(),
                &table_name,
            )
        };
        let added_entries: Vec<DataFileEntry> = added.iter().map(uncommitted_entry).collect();
        let mut table = load()?;
        for _ in 0..COMMIT_RETRIES {
            let live = table.entries()?;
            let removed_entries = removed
                .iter()
                .map(|path| {
                    live.iter()
                        .find(|entry| entry.file.path == *path)
                        .cloned()
                        .ok_or_else(|| {
                            table_compaction::Error::CommitConflict {
                                location: table.metadata().location().to_string(),
                                file: path.to_string(),
                            }
                            .into()
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            match table.commit(
                &removed_entries,
                added_entries.clone(),
                false,
                now_unix_ms(),
            )? {
                Some(committed) => {
                    self.publish_entry(
                        &name,
                        TableEntry {
                            metadata: Arc::clone(committed.metadata()),
                            metadata_location: committed.metadata_location().to_string(),
                            store: held.store().clone(),
                            manifest_list_size: committed.manifest_list_size(),
                        },
                    );
                    return Ok(());
                }
                None => table = load()?,
            }
        }
        Err(iceberg_tables::Error::CommitRacesExhausted {
            attempts: COMMIT_RETRIES,
        }
        .into())
    }
}

impl IcebergDatastore {
    /// The datastore's one compaction actor, serving both the configured
    /// rounds and explicit `COMPACT` statements; spawned on first use, so it
    /// exists even when no rounds are configured.
    pub(crate) fn compacter(self: &Arc<Self>) -> CompacterHandle<IcebergDatastore> {
        let mut compacter = self.compacter.lock().unwrap();
        if let Some(compacter) = compacter.as_ref() {
            return compacter.clone();
        }
        let (
            target_bytes,
            merge_target_bytes,
            min_files,
            max_concurrent_merges,
            layout_clique_size,
            poll_interval,
        ) = match &self.compaction {
            Some(config) => (
                config.target_bytes,
                config.merge_target_bytes,
                config.min_files,
                config.max_concurrent_merges,
                config.layout_clique_size,
                Some(config.poll_interval),
            ),
            None => (
                DEFAULT_COMPACT_BYTES,
                default_merge_target_bytes(DEFAULT_COMPACT_BYTES),
                DEFAULT_MIN_FILES_TO_MERGE,
                DEFAULT_COMPACT_PARALLELISM,
                DEFAULT_LAYOUT_CLIQUE_SIZE,
                None,
            ),
        };
        let (handle, actor) = CompacterHandle::new(
            target_bytes,
            merge_target_bytes,
            min_files,
            max_concurrent_merges,
            layout_clique_size,
            poll_interval,
            Arc::clone(self),
        );
        let task = tokio::spawn(actor.run());
        self.maintenance_tasks
            .lock()
            .unwrap()
            .push(task.abort_handle());
        *compacter = Some(handle.clone());
        handle
    }
}
