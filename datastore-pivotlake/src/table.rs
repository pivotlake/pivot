//! The catalog's in-memory handle to one table at one committed version,
//! whichever format keeps it: what every reader, writer and maintenance sweep
//! holds and evolves, over the format's own [`Table`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::Error;
use crate::file_entry::{
    ColumnStatFilter, FileEntry, FileStats, PartitionEqFilter, PartitionValues,
};
use crate::table_format::Table;
use arrow_array::{Array, ArrayRef, Datum, RecordBatch};
use arrow_cast::display::{ArrayFormatter, FormatOptions};
use arrow_schema::DataType;
use catalog::datastore::{DatastoreColumnMetadata, DatastoreFileMetadata, DatastoreTableMetadata};
use crossbeam_deque::{Injector, Steal};
use dispatch::{DataFlowDispatcher, Projection};
use object_storage::{FileRef, ObjectPath, ObjectStore};
use parquet_engine::{ParquetTable, RowGroupMetadata};
use planner::catalog::{Column, SchemaQualifiedTableName};
use planner::expression::Expression;
use uuid::Uuid;

/// One data file of a table, materialized: its metadata entry ([`FileEntry`]:
/// identity, partition tuple, stats) paired with the row groups read from its
/// footer (file-local order). What a writer hands a commit: the entry is the
/// metadata the commit records, the row groups the content the next scan
/// reads.
#[derive(Clone)]
pub struct TableFile {
    pub(crate) entry: FileEntry,
    pub(crate) row_groups: Vec<Arc<RowGroupMetadata>>,
}

impl TableFile {
    /// Pair a file's entry with the row groups read from its footer.
    ///
    /// Derive the file's stats from its footers when the entry carries none, so
    /// file-level range pruning always has min/max bounds. Usually the entry
    /// already has them (a fresh INSERT records them, and a reload reads the
    /// persisted stats), so this covers only a file with neither: one adopted
    /// at CREATE, or whose metadata recorded no stats.
    pub(crate) fn new(mut entry: FileEntry, row_groups: Vec<Arc<RowGroupMetadata>>) -> Self {
        if entry.stats.is_none() && !row_groups.is_empty() {
            entry.stats = Some(Arc::new(parquet_engine::aggregate_file_stats(&row_groups)));
        }
        Self { entry, row_groups }
    }

    /// The file's store identity (path and size), for a caller that names the
    /// file outside the table's own module (reporting a merge's outputs,
    /// deleting one whose commit failed).
    pub(crate) fn file_ref(&self) -> &FileRef {
        &self.entry.file
    }
}

/// What a file's bytes hold decoded, summed over its column chunks.
fn uncompressed_size(row_groups: &[Arc<RowGroupMetadata>]) -> u64 {
    row_groups
        .iter()
        .flat_map(|row_group| row_group.columns.iter())
        .map(|chunk| chunk.total_uncompressed_size.max(0) as u64)
        .sum()
}

/// A table's own storage: the directory under the database root its data and
/// metadata live in, and the store it is read and deleted through. The one
/// place a table's objects are listed and deleted, so a deletion never reaches
/// outside the directory the table owns.
#[derive(Clone)]
pub(crate) struct TableStorage {
    store: Arc<dyn ObjectStore>,
    location: ObjectPath,
}

impl TableStorage {
    pub(crate) fn new(store: Arc<dyn ObjectStore>, location: ObjectPath) -> Self {
        Self { store, location }
    }

    /// The Parquet data objects physically present under the table location,
    /// each with its storage modification time (Unix ms): what a vacuum sweep
    /// lists to find files no metadata references. Data files are flat under
    /// the location (partition values live in the metadata, not the path), so a
    /// one-level listing is complete; non-`.parquet` objects (the metadata) are
    /// excluded. Paths are relative to the location, matching the table's
    /// [`FileRef`]s and what [`delete_data_file`](Self::delete_data_file)
    /// expects.
    pub(crate) fn list_data_files(&self) -> crate::Result<Vec<(ObjectPath, u64)>> {
        Ok(self
            .store
            .list(&self.location)?
            .objects
            .into_iter()
            .filter(|object| object.file.path.as_str().ends_with(".parquet"))
            .map(|object| (object.file.path, object.modified_unix_ms))
            .collect())
    }

    /// Delete a data file. `path` resolves against the table's location like
    /// any [`FileRef`] path, and must land inside it: that directory is the
    /// only storage the table owns.
    ///
    /// A file the table *adopted* (`with_pre_existing_parquets`) lives wherever
    /// its owner put it, so it is never the table's to delete: compaction may
    /// copy its rows into the table's own storage and drop it from the
    /// metadata, but the object itself stays. The test is where the file lands
    /// rather than how the path is spelled, so a path that names the table's
    /// own directory outright is still the table's to delete.
    pub(crate) fn delete_data_file(&self, path: &ObjectPath) -> crate::Result<()> {
        let resolved = self.location.resolve(path);
        if !self.is_in_own_storage(&resolved)? {
            return Err(Error::DeletingOutsideStorage {
                location: self.location.as_str().to_string(),
                file: path.as_str().to_string(),
            });
        }
        self.store.delete(&resolved)?;
        Ok(())
    }

    /// Whether `key` names an object under the table's own location. Both sides
    /// are taken to their store-root-absolute form first, so a key that resolves
    /// into the table's directory counts however it was written.
    pub(crate) fn is_in_own_storage(&self, key: &ObjectPath) -> crate::Result<bool> {
        let owned = self.store.absolute_key(&self.location)?;
        let owned = format!("{}/", owned.as_str().trim_end_matches('/'));
        Ok(self
            .store
            .absolute_key(key)?
            .as_str()
            .starts_with(owned.as_str()))
    }
}

/// A table at one committed version: its durable identity and storage, the
/// format's own view of that version, and the lock in-process writers take to
/// commit on top of it.
///
/// A copy handed out for reading drifts as soon as anything commits; it is
/// never a commit base, since writes go through the datastore, which commits
/// on the live copy under the lock and publishes the result.
#[derive(Clone)]
pub struct CatalogTable {
    /// The table's durable identity from the database index, minted once at
    /// creation and stable across renames and every commit. The catalog
    /// indexes by this; the format's own metadata independently owns whatever
    /// id it mints.
    id: Uuid,
    storage: TableStorage,
    /// The version this copy is at, as its format holds it. Replaced whole by
    /// a refresh, a commit or a local swap.
    table: Arc<dyn Table>,
    /// The pool a merge decodes on, so the mutators need no dispatcher passed
    /// in.
    dispatcher: DataFlowDispatcher,
    /// This exists only for throughput: the format's own conditional write
    /// already ensures correctness, but concurrent in-process writers can all
    /// start at `V` and race for `V + 1`, forcing every loser to reload and
    /// retry.
    commit_lock: Arc<std::sync::Mutex<()>>,
}

impl CatalogTable {
    pub(super) fn new(
        id: Uuid,
        storage: TableStorage,
        table: Arc<dyn Table>,
        dispatcher: DataFlowDispatcher,
    ) -> Self {
        Self {
            id,
            storage,
            table,
            dispatcher,
            commit_lock: Arc::default(),
        }
    }

    /// An owned handle to this table's
    /// [commit lock](field@Self::commit_lock).
    pub(super) fn commit_lock(&self) -> Arc<std::sync::Mutex<()>> {
        self.commit_lock.clone()
    }

    /// Advance this copy to the latest committed version. Returns whether it
    /// advanced; `Ok(false)` means this copy was already current.
    pub fn refresh(&mut self) -> crate::Result<bool> {
        let Some(table) = self.table.refresh()? else {
            return Ok(false);
        };
        self.table = table;
        Ok(true)
    }

    /// Fold a file swap this process already committed through the datastore
    /// into this read copy without a metadata read: drop `removed` and adopt
    /// `added`, whose footers the writer holds. The copy stays at its version;
    /// a later [`refresh`](Self::refresh) reconciles it to the committed
    /// metadata.
    pub(crate) fn apply_file_swap(&mut self, removed: &[ObjectPath], added: Vec<TableFile>) {
        self.table = self.table.with_files_swapped(removed, added);
    }

    /// Commit file changes against this copy's version. Each attempt first
    /// verifies that every `removed` path is still live, returning
    /// [`Error::CommitConflict`] if another commit removed one. After a
    /// successful commit, the already-loaded `added` files fold directly into
    /// this copy. A version conflict refreshes the copy and retries.
    ///
    /// In-process callers follow the table's
    /// [commit-lock protocol](field@Self::commit_lock), so this retry path
    /// normally handles contention from another process on a shared remote
    /// store.
    ///
    /// Runs off the dispatch workers: the retry's `refresh` may drive a
    /// footer-fetch dataflow, and every commit path commits from a blocking
    /// thread, never a pinned worker (which would deadlock driving a dataflow
    /// from inside one).
    pub(crate) fn commit_files(
        &mut self,
        removed: &[ObjectPath],
        added: Vec<TableFile>,
        data_change: bool,
    ) -> crate::Result<()> {
        loop {
            let removed_entries = self.live_entries_for(removed)?;
            let now_ms = crate::vacuum::now_unix_ms();
            if let Some(table) = self
                .table
                .commit(&removed_entries, &added, data_change, now_ms)?
            {
                self.table = table;
                return Ok(());
            }
            self.refresh()?;
        }
    }

    /// The live entries at `paths`, or [`Error::CommitConflict`] naming the
    /// first path this version no longer holds.
    fn live_entries_for(&self, paths: &[ObjectPath]) -> crate::Result<Vec<FileEntry>> {
        paths
            .iter()
            .map(|path| {
                self.table
                    .entries()
                    .iter()
                    .find(|entry| entry.file.path.as_str() == path.as_str())
                    .cloned()
                    .ok_or_else(|| Error::CommitConflict {
                        location: self.storage.location.as_str().to_string(),
                        file: path.to_string(),
                    })
            })
            .collect()
    }

    /// Commit bare `entries` whose footers are not yet in hand: like
    /// [`commit_files`](Self::commit_files), but the added files' footers are
    /// read only if the commit wins (a losing swap errors before touching its
    /// output). Used by test/seed paths; production writers (INSERT,
    /// compaction) already hold the footers and use `commit_files`. Only its
    /// `test-support` caller ([`test_support`](crate::test_support)) needs it,
    /// so it is gated with that feature to stay out of a plain library build.
    #[cfg(feature = "test-support")]
    pub(crate) fn commit_entries(
        &mut self,
        removed: &[ObjectPath],
        added: &[FileEntry],
        data_change: bool,
    ) -> crate::Result<()> {
        let bare: Vec<TableFile> = added
            .iter()
            .map(|entry| TableFile {
                entry: entry.clone(),
                row_groups: Vec::new(),
            })
            .collect();
        self.commit_files(removed, bare, data_change)?;
        // We won; only now read the added files' footers and fold them in
        // under the same version.
        let footers = self.fetch_footers(added)?;
        let paths: Vec<ObjectPath> = added.iter().map(|entry| entry.file.path.clone()).collect();
        let files: Vec<TableFile> = added
            .iter()
            .zip(footers)
            .map(|(entry, footer)| TableFile::new(entry.clone(), footer.row_groups))
            .collect();
        self.apply_file_swap(&paths, files);
        Ok(())
    }

    /// Read the footers of `entries`, files under this table's location that
    /// its metadata does not hold yet, one per entry in order, each reconciled
    /// by name against the declared columns.
    #[cfg(feature = "test-support")]
    fn fetch_footers(
        &self,
        entries: &[FileEntry],
    ) -> crate::Result<Vec<parquet_engine::FileRowGroups>> {
        let to_fetch: Vec<object_storage::DataFile> = entries
            .iter()
            .map(|entry| {
                entry
                    .file
                    .clone()
                    .into_data_file(self.storage.store.as_ref(), &self.storage.location)
            })
            .collect::<object_storage::Result<_>>()?;
        Ok(parquet_engine::load_file_row_groups(
            &self.dispatcher,
            &to_fetch,
            parquet_engine::TableColumns::by_name(self.columns()),
        )?)
    }

    /// Expire the snapshots the table's retention allows as of `now_ms`, under
    /// the commit lock the caller holds. Returns whether this copy advanced.
    pub(crate) fn expire_snapshots(&mut self, now_ms: u64) -> crate::Result<bool> {
        let Some(table) = self.table.expire_snapshots(now_ms)? else {
            return Ok(false);
        };
        self.table = table;
        Ok(true)
    }

    /// Delete what this version no longer needs under the table's directory and
    /// has not needed for longer than the retention window, as of `now_ms`.
    pub(crate) fn delete_expired_objects(
        &self,
        name: &SchemaQualifiedTableName,
        now_ms: u64,
    ) -> crate::Result<()> {
        self.table
            .delete_expired_objects(name, &self.storage, now_ms)
    }

    /// The table's durable identity from the database index, stable across
    /// renames and commits. The catalog indexes by this.
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// The table's partition columns, in order (empty = unpartitioned). A
    /// partitioning writer routes each row to a file by these columns' values.
    pub fn partition_by(&self) -> &[String] {
        self.table.partition_by()
    }

    /// The table's sort columns, in order (empty = unsorted). A writer sorts each
    /// file's rows by these before encoding.
    pub fn sort_by(&self) -> &[String] {
        self.table.sort_by()
    }

    /// The table's current files as [`FileRef`]s — what a compacter scans to pick
    /// merge candidates, and names in a compaction swap.
    pub fn file_refs(&self) -> Vec<FileRef> {
        self.table
            .entries()
            .iter()
            .map(|entry| entry.file.clone())
            .collect()
    }

    /// Every live data file at this version, with its partition tuple and
    /// stats as the metadata recorded them.
    pub(crate) fn entries(&self) -> &[FileEntry] {
        self.table.entries()
    }

    /// The row groups of each of `entries` (entries of this very version), in
    /// the order given: what a merge and a layout measurement read.
    pub(crate) fn row_groups(
        &self,
        entries: &[&FileEntry],
    ) -> crate::Result<Vec<Vec<Arc<RowGroupMetadata>>>> {
        self.table.row_groups(entries)
    }

    /// This table as the cross-datastore catalog describes it, under the `name`
    /// it is currently indexed by: the columns it declares with what they cost,
    /// and the files it holds. Reads every file's footer the format does not
    /// already hold.
    pub(crate) fn metadata(
        &self,
        name: SchemaQualifiedTableName,
    ) -> crate::Result<DatastoreTableMetadata> {
        let entries: Vec<&FileEntry> = self.table.entries().iter().collect();
        let footers = self.table.row_groups(&entries)?;
        let columns = self.table.columns();
        let partition_by = self.table.partition_by();
        let sort_by = self.table.sort_by();

        let column_bytes = self.column_bytes(&footers);
        let column_metadata = columns
            .iter()
            .enumerate()
            .map(|(position, column)| {
                let (bytes, bytes_uncompressed) = column_bytes[position];
                DatastoreColumnMetadata {
                    name: column.name.clone(),
                    column_type: column.col_type.clone(),
                    position,
                    bytes,
                    bytes_uncompressed,
                    is_partition_key: partition_by.contains(&column.name),
                    is_sort_key: sort_by.contains(&column.name),
                }
            })
            .collect();

        let files = entries
            .iter()
            .zip(&footers)
            .map(|(entry, row_groups)| DatastoreFileMetadata {
                path: entry.file.path.as_str().to_string(),
                bytes: entry.file.size,
                bytes_uncompressed: uncompressed_size(row_groups),
                partition: self.format_partition(&entry.partition),
                min_max_stats: format_min_max_stats(entry.stats.as_deref()),
            })
            .collect();

        Ok(DatastoreTableMetadata {
            name,
            id: self.id.to_string(),
            columns: column_metadata,
            sort_by: sort_by.to_vec(),
            partition_by: partition_by.to_vec(),
            total_rows: footers
                .iter()
                .flatten()
                .map(|row_group| row_group.num_rows.max(0) as u64)
                .sum(),
            bytes: entries.iter().map(|entry| entry.file.size).sum(),
            bytes_uncompressed: footers
                .iter()
                .map(|row_groups| uncompressed_size(row_groups))
                .sum(),
            files,
        })
    }

    /// What each declared column costs across `footers`, the row groups of
    /// the table's committed files: the bytes it occupies in storage, and what
    /// they hold decoded. Chunks are per leaf, so a nested column is charged
    /// for every leaf it spans; a file that does not carry a column
    /// contributes nothing to it.
    fn column_bytes(&self, footers: &[Vec<Arc<RowGroupMetadata>>]) -> Vec<(u64, u64)> {
        let columns = self.table.columns();
        let position_by_name: HashMap<&str, usize> = columns
            .iter()
            .enumerate()
            .map(|(position, column)| (column.name.as_str(), position))
            .collect();

        let mut totals = vec![(0, 0); columns.len()];
        for row_group in footers.iter().flatten() {
            let mut leaf = 0;
            for field in row_group.schema.fields() {
                let leaves = parquet_engine::leaf_count(field);
                if let Some(&position) = position_by_name.get(field.name().as_str()) {
                    for chunk in &row_group.columns[leaf..leaf + leaves] {
                        totals[position].0 += chunk.total_compressed_size.max(0) as u64;
                        totals[position].1 += chunk.total_uncompressed_size.max(0) as u64;
                    }
                }
                leaf += leaves;
            }
        }
        totals
    }

    /// A file's partition as the catalog reports it: `column=value` pairs in the
    /// table's partition order, comma-separated. Empty for an unpartitioned
    /// table, and for a file adopted before the table was partitioned, which
    /// carries no tuple (or none for some of the columns).
    fn format_partition(&self, partition: &Option<PartitionValues>) -> String {
        let Some(values) = partition else {
            return String::new();
        };
        self.table
            .partition_by()
            .iter()
            .filter_map(|column| {
                let (array, _) = values.get(column)?.get();
                let formatter = ArrayFormatter::try_new(array, &FormatOptions::default()).ok()?;
                Some(format!("{column}={}", formatter.value(0)))
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Each committed file paired with the typed partition tuple recorded for
    /// it (or `None`), reflecting the current committed version.
    pub fn file_partitions(&self) -> Vec<(ObjectPath, Option<PartitionValues>)> {
        self.table
            .entries()
            .iter()
            .map(|entry| (entry.file.path.clone(), entry.partition.clone()))
            .collect()
    }

    /// Delete a data file (a compaction input swapped out of the metadata, or a
    /// merge output whose commit failed). See
    /// `TableStorage::delete_data_file`.
    pub fn delete_data_file(&self, path: &ObjectPath) -> crate::Result<()> {
        self.storage.delete_data_file(path)
    }

    /// How long a file this table no longer references is kept before vacuum may
    /// delete it. Vacuum ages unreferenced files against this window.
    pub fn deleted_file_retention(&self) -> crate::Result<Duration> {
        self.table.deleted_file_retention()
    }

    /// See `TableStorage::list_data_files`.
    pub fn list_data_files(&self) -> crate::Result<Vec<(ObjectPath, u64)>> {
        self.storage.list_data_files()
    }

    /// The committed version this copy is at. Monotonic per table; used to
    /// decide whether a published copy is newer than the catalog's.
    pub fn version(&self) -> u64 {
        self.table.version()
    }

    /// A flat scan view of the files a query's pushed-down predicates cannot rule
    /// out — every surviving file's row groups concatenated in metadata order,
    /// where a row group's global index is simply its position. A file survives
    /// when its recorded partition tuple can still match every `partition_filters`
    /// *and* its stats can still match every `stat_filters`. Empty filters keep
    /// every file.
    ///
    /// Built from the entries this copy holds, with the surviving files' row
    /// groups from the format: in memory for one that holds every footer, read
    /// once per version for one that reads them on demand.
    ///
    /// The row group's global index is its position in this returned flat list, so
    /// a scan and its materialize must build it from the *same* filters (they do:
    /// both go through the binding's predicates) to address the same groups.
    /// The row groups a scan over `filters`, the predicates the query pushed
    /// into it, reads: see `Table::plan_scan`.
    pub fn plan_scan(&self, filters: &[Expression]) -> crate::Result<Arc<ParquetTable>> {
        self.table.plan_scan(filters)
    }

    /// Every file's row groups, less the files `partition_filters` and
    /// `stat_filters` rule out by the metadata this version records for
    /// them: the view of the whole table a merge or a measurement reads.
    pub fn build_scan_view(
        &self,
        partition_filters: &[PartitionEqFilter],
        stat_filters: &[ColumnStatFilter],
    ) -> crate::Result<Arc<ParquetTable>> {
        let partition_by = self.table.partition_by();
        let surviving: Vec<&FileEntry> = self
            .table
            .entries()
            .iter()
            .filter(|entry| entry.maybe_matches_partition(partition_by, partition_filters))
            .filter(|entry| entry.maybe_matches_stats(stat_filters))
            .collect();
        let row_groups = self
            .table
            .row_groups(&surviving)?
            .into_iter()
            .flatten()
            .collect();
        Ok(Arc::new(ParquetTable::new(row_groups)))
    }

    /// The table's columns (schema), as the planner's [`Column`]s.
    pub fn columns(&self) -> Vec<Column> {
        self.table.columns().to_vec()
    }

    /// Whether each column (in [`columns`](Self::columns) order) can hold SQL
    /// NULLs, computed from the files' footers rather than the declared
    /// schema: the planner uses the flags to route between the branch-free and
    /// the null-aware execution paths, so they must reflect whether the data
    /// can actually hold NULLs. An empty table reports every column nullable.
    /// Reads every footer the format does not already hold.
    pub fn nullability(&self) -> crate::Result<Vec<bool>> {
        let entries: Vec<&FileEntry> = self.table.entries().iter().collect();
        let footers = self.table.row_groups(&entries)?;
        let row_groups: Vec<&Arc<RowGroupMetadata>> = footers.iter().flatten().collect();
        Ok(self
            .table
            .columns()
            .iter()
            .map(|column| entries.is_empty() || column_may_hold_nulls(&row_groups, &column.name))
            .collect())
    }

    /// The field id each column is written under in new files, for a format
    /// that resolves columns by id.
    pub(crate) fn field_ids(&self) -> Option<Vec<i32>> {
        self.table.field_ids()
    }

    /// Where the table's data lives, relative to the database root (an absolute
    /// path escapes to the store root). Combine with the store's own root to
    /// know the physical location.
    pub fn location(&self) -> &str {
        self.storage.location.as_str()
    }

    pub(crate) fn store(&self) -> Arc<dyn ObjectStore> {
        self.storage.store.clone()
    }

    /// Re-encode `inputs`' rows into fresh target-sized files and return them,
    /// uploaded but not yet part of the table: the read half of a compaction,
    /// which [`compact_table_files`](crate::compact_table_files) then swaps in
    /// for the inputs. The merged files are written over the
    /// shared io_uring ring by the same upload operators an INSERT uses, and
    /// their row groups come straight from the writer's own footer metadata, so
    /// no footer is re-read.
    ///
    /// Nothing here touches the metadata, so it runs off any read copy that
    /// holds the inputs, and a merge of a large batch never stalls another
    /// writer's commit. An input the copy no longer lists is a commit conflict:
    /// another writer swapped it out first.
    /// `inputs` must share one partition tuple; the caller batches them so.
    pub(crate) fn merge_files(
        &self,
        inputs: &[FileRef],
        target_rows_per_group: usize,
        max_output_file_size: u64,
    ) -> crate::Result<Vec<TableFile>> {
        let input_paths: Vec<ObjectPath> = inputs.iter().map(|input| input.path.clone()).collect();
        let input_entries = self.live_entries_for(&input_paths)?;
        let row_groups = self
            .table
            .row_groups(&input_entries.iter().collect::<Vec<_>>())?
            .into_iter()
            .flatten()
            .collect();
        let parquet = Arc::new(ParquetTable::new(row_groups));
        let columns = parquet.schema().fields().len();
        let scan = parquet_engine::table_input(
            &self.dispatcher,
            &parquet,
            Projection::all(columns),
            false,
        );
        // The selector guarantees that all inputs share one partition, so its
        // recorded tuple can key every decoded batch without repartitioning it.
        let partition = input_entries
            .first()
            .and_then(|entry| entry.partition.clone());
        // A format that resolves columns by field id gets its outputs stamped
        // with the declared schema: the inputs are reassembled to the declared
        // shape (an adopted file may carry no field ids, and a variant column
        // is re-shredded for the file it lands in) and encoded under it.
        // Otherwise the outputs keep the inputs' own schema.
        let (scan, output_schema) = match self.table.field_ids() {
            None => (scan, parquet.schema().clone()),
            Some(field_ids) => {
                let schema = iceberg_tables::sink::write_schema(
                    self.table.columns(),
                    &self.declared_nullability(),
                    Some(&field_ids),
                );
                let projected = parquet_engine::writing::unshred_batches_spec(scan).project({
                    let schema = schema.clone();
                    move || {
                        let schema = schema.clone();
                        move |batch: RecordBatch| {
                            Ok(RecordBatch::try_new(
                                schema.clone(),
                                batch.columns().to_vec(),
                            )?)
                        }
                    }
                });
                (projected, schema)
            }
        };
        let max_output_file_size = usize::try_from(max_output_file_size).unwrap_or(usize::MAX);
        let uploaded_files = Arc::new(Injector::new());
        let encoded = parquet_engine::writing::encode_compaction_batches_spec(
            scan,
            output_schema,
            partition,
            Arc::from(self.partition_by()),
            Arc::from(self.sort_by()),
            target_rows_per_group,
            max_output_file_size,
        );
        // Merged files sit flat in the table's directory.
        let spec = parquet_engine::writing::upload::upload_files_spec(
            self.store(),
            self.storage.location.clone(),
            ObjectPath::new(""),
            self.columns().into(),
            uploaded_files.clone(),
            None,
            encoded,
        );
        // Drive encode → upload to completion; the emitted row-count batch is
        // ignored, and the uploaded files arrive on `uploaded_files`.
        let outcome = spec.collect().map(|_| ()).map_err(crate::Error::Merge);

        let mut added: Vec<TableFile> = Vec::new();
        loop {
            match uploaded_files.steal() {
                Steal::Success(uploaded) => {
                    let parquet_engine::writing::upload::UploadedFile {
                        file,
                        partition,
                        row_groups,
                    } = uploaded;
                    let stats = Some(std::sync::Arc::new(parquet_engine::aggregate_file_stats(
                        &row_groups,
                    )));
                    let entry = FileEntry {
                        file,
                        partition,
                        stats,
                    };
                    added.push(TableFile::new(entry, row_groups));
                }
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }

        // A merge that failed part way has uploaded some of its outputs. They
        // will never be committed, so delete them rather than leave orphans.
        if let Err(error) = outcome {
            for file in &added {
                if let Err(cleanup_error) = self.delete_data_file(&file.entry.file.path) {
                    tracing::warn!(
                        merge_error = %error,
                        cleanup_error = %cleanup_error,
                        file = %file.entry.file.path,
                        "compaction: deleting an output of a merge that will not commit failed (orphan left)"
                    );
                }
            }
            return Err(error);
        }
        Ok(added)
    }

    /// Whether each column (in [`columns`](Self::columns) order) may be
    /// written a NULL: every one, as this datastore declares every column
    /// optional whatever the files written so far hold.
    fn declared_nullability(&self) -> Vec<bool> {
        vec![true; self.table.columns().len()]
    }

    /// What an INSERT into this table writes: files flat under the table's
    /// directory, in the declared columns, with the field ids the format
    /// assigns when it resolves columns by id, cut by the partition columns
    /// (every partition field of a table created here is the identity of a
    /// column, named after it).
    pub(crate) fn insert_target(&self) -> iceberg_tables::sink::InsertTarget {
        iceberg_tables::sink::InsertTarget {
            store: self.store(),
            directory: self.storage.location.clone(),
            file_prefix: ObjectPath::new(""),
            columns: self.columns(),
            nullability: self.declared_nullability(),
            field_ids: self.field_ids(),
            partition_keys: iceberg_tables::PartitionKeys::by_columns(self.partition_by()),
            sort_by: self.sort_by().to_vec(),
        }
    }
}

/// Whether any of `row_groups` can hold a NULL in the column named `name`; a
/// file missing the column entirely reads as all-NULL.
fn column_may_hold_nulls(row_groups: &[&Arc<RowGroupMetadata>], name: &str) -> bool {
    row_groups
        .iter()
        .any(|row_group| match row_group.schema.index_of(name) {
            Ok(field_idx) => row_group.column_may_hold_nulls(field_idx),
            Err(_) => true,
        })
}

/// Render a file's typed column bounds as a stable JSON object. Each column is
/// one field containing its `min` and `max`; values stay JSON numbers/booleans
/// where their Arrow type has that shape, while strings and temporal values are
/// quoted. A column whose pair cannot be rendered is omitted.
fn format_min_max_stats(stats: Option<&FileStats>) -> String {
    let Some(stats) = stats else {
        return "{}".to_string();
    };
    let mut names: Vec<_> = stats.min_values.keys().collect();
    names.sort_unstable();

    let fields = names
        .into_iter()
        .filter_map(|name| {
            let min = format_bound(stats.min_values.get(name)?)?;
            let max = format_bound(stats.max_values.get(name)?)?;
            let name = serde_json::to_string(name).expect("a Rust string is valid JSON");
            Some(format!(r#"{name}:{{"min":{min},"max":{max}}}"#))
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{fields}}}")
}

/// Render the single value held by a file statistic as one JSON value.
fn format_bound(array: &ArrayRef) -> Option<String> {
    if array.len() != 1 || array.is_null(0) {
        return None;
    }
    let formatter = ArrayFormatter::try_new(array.as_ref(), &FormatOptions::default()).ok()?;
    let rendered = formatter.value(0).to_string();
    match array.data_type() {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float16
        | DataType::Float32
        | DataType::Float64
        | DataType::Decimal32(_, _)
        | DataType::Decimal64(_, _)
        | DataType::Decimal128(_, _)
        | DataType::Decimal256(_, _) => serde_json::from_str::<serde_json::Value>(&rendered)
            .ok()
            .map(|_| rendered),
        _ => serde_json::to_string(&rendered).ok(),
    }
}
