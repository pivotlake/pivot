//! The catalog's in-memory handle to one table at one committed version,
//! whichever format keeps it: what every reader, writer and maintenance sweep
//! holds and evolves, over the format's own [`Table`].

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::sync::Arc;
use std::time::Duration;

use crate::Error;
use crate::file_entry::{
    ColumnStatFilter, FileEntry, FileStats, PartitionEqFilter, PartitionValues,
};
use crate::file_layout::VisitFile;
use crate::table_format::{ExpiredObjectsDeletion, Table};
use arrow_array::{Array, ArrayRef, Datum, RecordBatch};
use arrow_cast::display::{ArrayFormatter, FormatOptions};
use arrow_schema::DataType;
use catalog::datastore::{DatastoreColumnMetadata, DatastoreFileMetadata, DatastoreTableMetadata};
use dispatch::{DataFlowDispatcher, Projection};
use object_storage::{FileRef, ObjectPath, ObjectStore};
use parquet_engine::writing::upload::UploadedFiles;
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

    /// Every object under the table location, however deep, each with its
    /// path relative to the location and its storage modification time (Unix
    /// ms): what a sweep lists to find objects no metadata names. A
    /// directory or an object whose own name `is_hidden` is left out, and a
    /// hidden directory is not walked into. A listing is one level deep, so
    /// the subtree is walked one directory at a time.
    pub(crate) fn list_objects_recursively(
        &self,
        is_hidden: impl Fn(&str) -> bool,
    ) -> crate::Result<Vec<(ObjectPath, u64)>> {
        let mut objects = Vec::new();
        // Each pending directory as its path relative to the location, the
        // location itself being the empty one.
        let mut pending: Vec<Option<ObjectPath>> = vec![None];
        while let Some(relative_directory) = pending.pop() {
            let directory = match &relative_directory {
                None => self.location.clone(),
                Some(relative) => self.location.join(relative.as_str()),
            };
            let under_directory = |name: &str| match &relative_directory {
                None => ObjectPath::new(name),
                Some(relative) => relative.join(name),
            };
            let listing = self.store.list(&directory)?;
            for object in listing.objects {
                let name = object.file.path.as_str();
                if !is_hidden(name) {
                    objects.push((under_directory(name), object.modified_unix_ms));
                }
            }
            for prefix in listing.prefixes {
                let name = prefix.as_str().trim_end_matches('/');
                if !is_hidden(name) {
                    pending.push(Some(under_directory(name)));
                }
            }
        }
        Ok(objects)
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
    /// this copy. A version conflict refreshes the copy and retries. A commit
    /// whose version is current but could not be loaded has succeeded: the
    /// copy is refreshed onto it, or stays where it was for the next refresh.
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
            self.check_files_are_live(removed)?;
            let now_ms = crate::vacuum::now_unix_ms();
            match self.table.commit(removed, &added, data_change, now_ms) {
                Ok(Some(table)) => {
                    self.table = table;
                    return Ok(());
                }
                Ok(None) => {
                    self.refresh()?;
                }
                Err(Error::Iceberg(
                    error @ iceberg_tables::Error::CommittedTableUnloaded { .. },
                )) => {
                    tracing::warn!(error = %error, "a commit succeeded without its version loading; refreshing onto it");
                    if let Err(refresh_error) = self.refresh() {
                        tracing::warn!(
                            error = %refresh_error,
                            "refreshing onto a committed version failed; the next refresh loads it"
                        );
                    }
                    return Ok(());
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// [`Error::CommitConflict`] naming the first of `paths` this version no
    /// longer holds.
    fn check_files_are_live(&self, paths: &[ObjectPath]) -> crate::Result<()> {
        for path in paths {
            if !self.table.contains_file(path)? {
                return Err(self.conflict_at(path));
            }
        }
        Ok(())
    }

    fn conflict_at(&self, path: &ObjectPath) -> Error {
        Error::CommitConflict {
            location: self.storage.location.as_str().to_string(),
            file: path.to_string(),
        }
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
    /// the commit lock the caller holds. When that committed a version, this
    /// copy advances to it and the deletion of what the expired snapshots
    /// alone used is handed back, for the caller to run once it has released
    /// the lock.
    pub(crate) fn expire_snapshots(
        &mut self,
        now_ms: u64,
    ) -> crate::Result<Option<ExpiredObjectsDeletion>> {
        let Some(expiry) = self.table.expire_snapshots(&self.storage, now_ms)? else {
            return Ok(None);
        };
        self.table = expiry.table;
        Ok(Some(expiry.delete_expired_objects))
    }

    /// Delete what this version retired longer ago than the retention window,
    /// as of `now_ms`.
    pub(crate) fn delete_expired_objects(
        &self,
        name: &SchemaQualifiedTableName,
        now_ms: u64,
    ) -> crate::Result<()> {
        self.table
            .delete_expired_objects(name, &self.storage, now_ms)
    }

    /// List the table's directory and delete every object in it that this
    /// version does not use, as of `now_ms`.
    pub(crate) fn sweep_unused_objects(
        &self,
        name: &SchemaQualifiedTableName,
        now_ms: u64,
    ) -> crate::Result<()> {
        self.table.sweep_unused_objects(name, &self.storage, now_ms)
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

    /// The table's current files as [`FileRef`]s, in the order a walk visits
    /// them.
    pub fn file_refs(&self) -> crate::Result<Vec<FileRef>> {
        let mut files = Vec::with_capacity(self.table.count_files()?);
        self.walk_files(&mut |walked| {
            files.push(walked.file.clone());
            ControlFlow::Continue(())
        })?;
        Ok(files)
    }

    /// Visit every live data file at this version, partition by partition;
    /// see `Table::walk_files`.
    pub(crate) fn walk_files(&self, visit: &mut VisitFile<'_>) -> crate::Result<()> {
        self.table.walk_files(visit)
    }

    /// The live file at `path` with its partition tuple and its bounds on
    /// the sort columns, or `None` when this version has no such file.
    pub(crate) fn find_file(&self, path: &ObjectPath) -> crate::Result<Option<FileEntry>> {
        self.table.find_file(path)
    }

    /// The row groups of each of `files` (live files of this very version),
    /// in the order given: what a merge and a layout measurement read.
    pub(crate) fn row_groups(
        &self,
        files: &[FileRef],
    ) -> crate::Result<Vec<Vec<Arc<RowGroupMetadata>>>> {
        self.table.row_groups(files)
    }

    /// This table as the cross-datastore catalog describes it, under the `name`
    /// it is currently indexed by: the columns it declares with what they cost,
    /// and the files it holds. Reads every file's footer, a batch of files
    /// at a time, keeping of each only what it reports.
    pub(crate) fn metadata(
        &self,
        name: SchemaQualifiedTableName,
    ) -> crate::Result<DatastoreTableMetadata> {
        /// How many files' footers are held at a time.
        const FILES_PER_READ: usize = 1024;

        let all_files = self.file_refs()?;
        let columns = self.table.columns();
        let partition_by = self.table.partition_by();
        let sort_by = self.table.sort_by();

        let mut column_bytes = vec![(0, 0); columns.len()];
        let mut files = Vec::with_capacity(all_files.len());
        let mut total_rows = 0;
        let mut bytes_uncompressed = 0;
        for batch in all_files.chunks(FILES_PER_READ) {
            let footers = self.table.row_groups(batch)?;
            self.add_column_bytes(&mut column_bytes, &footers);
            for (file, row_groups) in batch.iter().zip(&footers) {
                let entry = self
                    .table
                    .find_file(&file.path)?
                    .expect("a walked file is live in the version walked");
                let file_bytes_uncompressed = uncompressed_size(row_groups);
                total_rows += row_groups
                    .iter()
                    .map(|row_group| row_group.num_rows.max(0) as u64)
                    .sum::<u64>();
                bytes_uncompressed += file_bytes_uncompressed;
                files.push(DatastoreFileMetadata {
                    path: file.path.as_str().to_string(),
                    bytes: file.size,
                    bytes_uncompressed: file_bytes_uncompressed,
                    partition: self.format_partition(&entry.partition),
                    min_max_stats: format_min_max_stats(
                        self.table.file_stats(&file.path)?.as_deref(),
                    ),
                });
            }
        }

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

        Ok(DatastoreTableMetadata {
            name,
            id: self.id.to_string(),
            columns: column_metadata,
            sort_by: sort_by.to_vec(),
            partition_by: partition_by.to_vec(),
            total_rows,
            bytes: all_files.iter().map(|file| file.size).sum(),
            bytes_uncompressed,
            files,
        })
    }

    /// Add to `totals`, per declared column, what the column costs across
    /// `footers`, the row groups of some of the table's committed files: the
    /// bytes it occupies in storage, and what they hold decoded. Chunks are
    /// per leaf, so a nested column is charged for every leaf it spans; a
    /// file that does not carry a column contributes nothing to it.
    fn add_column_bytes(&self, totals: &mut [(u64, u64)], footers: &[Vec<Arc<RowGroupMetadata>>]) {
        let columns = self.table.columns();
        let position_by_name: HashMap<&str, usize> = columns
            .iter()
            .enumerate()
            .map(|(position, column)| (column.name.as_str(), position))
            .collect();

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
    pub fn file_partitions(&self) -> crate::Result<Vec<(ObjectPath, Option<PartitionValues>)>> {
        Ok(self
            .live_entries()?
            .into_iter()
            .map(|entry| (entry.file.path, entry.partition))
            .collect())
    }

    /// Every live data file at this version with its partition tuple and its
    /// bounds on the sort columns, in the order a walk visits them.
    pub(crate) fn live_entries(&self) -> crate::Result<Vec<FileEntry>> {
        self.file_refs()?
            .iter()
            .map(|file| {
                Ok(self
                    .table
                    .find_file(&file.path)?
                    .expect("a walked file is live in the version walked"))
            })
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
    /// for this call by one that holds none.
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
        let surviving: Vec<FileRef> = self
            .live_entries()?
            .into_iter()
            .filter(|entry| entry.maybe_matches_partition(partition_by, partition_filters))
            .filter(|entry| entry.maybe_matches_stats(stat_filters))
            .map(|entry| entry.file)
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
    /// NULLs, as the table's data shows it rather than as the schema
    /// declares it: the planner uses the flags to route between the
    /// branch-free and the null-aware execution paths, so they must reflect
    /// whether the data can actually hold NULLs. An empty table reports
    /// every column nullable.
    pub fn nullability(&self) -> crate::Result<Vec<bool>> {
        self.table.nullability()
    }

    /// The rows of every live data file at this version, with nothing read.
    pub fn total_rows(&self) -> crate::Result<u64> {
        self.table.total_rows()
    }

    /// The rows a scan of the whole table returns, when this version's
    /// metadata counts them exactly; `None` when only a scan can.
    pub fn count_rows(&self) -> crate::Result<Option<u64>> {
        self.table.count_rows()
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
        let input_entries = input_paths
            .iter()
            .map(|path| {
                self.table
                    .find_file(path)?
                    .ok_or_else(|| self.conflict_at(path))
            })
            .collect::<crate::Result<Vec<FileEntry>>>()?;
        let row_groups = self
            .table
            .row_groups(inputs)?
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
            .and_then(|first| first.partition.clone());
        // The merge keeps each input's rows in the order the input had them,
        // so its outputs are in the table's sort order only when every input
        // is recorded as written in it: a file adopted as it lay is not.
        let sort_by = self.sort_by();
        let inputs_are_sorted = !sort_by.is_empty()
            && input_entries
                .iter()
                .all(|input| input.sorted_by.as_ref() == sort_by);
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
                    &self.schema_nullability(),
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
        // Merged files sit flat in the table's directory.
        let uploaded_files = UploadedFiles::new(self.store(), self.storage.location.clone());
        let encoded = parquet_engine::writing::encode_compaction_batches_spec(
            scan,
            output_schema,
            partition,
            Arc::from(self.partition_by()),
            Arc::from(sort_by),
            inputs_are_sorted,
            target_rows_per_group,
            max_output_file_size,
        );
        let spec = parquet_engine::writing::upload::upload_files_spec(
            uploaded_files.clone(),
            self.columns().into(),
            None,
            encoded,
        );
        // Drive encode → upload to completion; the emitted row-count batch is
        // ignored, and the uploaded files arrive on `uploaded_files`.
        let outcome = spec.collect().map(|_| ()).map_err(crate::Error::Merge);

        // The files were written into the table's directory, so the path
        // each is reported by is the one the table names it by.
        let added: Vec<TableFile> = uploaded_files
            .take()
            .into_iter()
            .map(|uploaded| {
                let parquet_engine::writing::upload::UploadedFile {
                    file,
                    partition,
                    row_groups,
                    sorted_by,
                } = uploaded;
                let stats = Some(std::sync::Arc::new(parquet_engine::aggregate_file_stats(
                    &row_groups,
                )));
                let entry = FileEntry {
                    file,
                    partition,
                    stats,
                    sorted_by,
                };
                TableFile::new(entry, row_groups)
            })
            .collect();

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
    fn schema_nullability(&self) -> Vec<bool> {
        vec![true; self.table.columns().len()]
    }

    /// What an INSERT into this table writes: files directly in the table's
    /// directory, in the declared columns, with the field ids the format
    /// assigns when it resolves columns by id, cut by the partition columns
    /// (every partition field of a table created here is the identity of a
    /// column, named after it).
    pub(crate) fn insert_target(&self) -> iceberg_tables::sink::InsertTarget {
        iceberg_tables::sink::InsertTarget {
            uploaded_files: UploadedFiles::new(self.store(), self.storage.location.clone()),
            columns: self.columns(),
            nullability: self.schema_nullability(),
            field_ids: self.field_ids(),
            partition_keys: iceberg_tables::PartitionKeys::by_columns(self.partition_by()),
            sort_by: self.sort_by().to_vec(),
        }
    }
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
