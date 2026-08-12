//! Background compaction of a table's small Parquet files.
//!
//! Frequent writes litter a table with small files, and scans pay per file. The
//! [`Compacter`] merges them: whenever the table's files smaller than the target
//! size add up to at least one target-sized output, it rewrites that batch as
//! one file (with full-size row groups) and swaps it into the table in a single
//! log commit.
//!
//! The compacter is **location-agnostic**: candidates come from the table's
//! manifest (paths + sizes, no directory scanning), and the merge (read,
//! upload, swap) runs through [`compact_table_files`]. A table under an `s3://`
//! database root compacts through the exact same code path as a local one.
//!
//! The swapped-out inputs are not deleted with the swap: a query that loaded the
//! prior version is still reading them, and the swap's Delta `Remove` actions
//! are their durable tombstones. The objects stay in place until the vacuum
//! sweep ages them out.
//!
//! Merging is **one dataflow** on the dispatch worker pool: the scan stages
//! decode the inputs' row groups, the encode stages repack them into full-size
//! row groups, and the upload stage writes each merged file over the shared
//! io_uring ring, the same operators an INSERT uses. The compacter's own task
//! only drives that dataflow (from a blocking thread) and, once the uploads
//! land, commits the swap.
//!
//! Crash safety comes from the table log, not from ordering tricks: a file is
//! part of the table iff the current log version lists it. The merged files are
//! uploaded, then swapped in for the inputs in one Delta commit (labelled a
//! rearrangement, not a data change); a crash before the commit leaves only
//! *orphan* objects (unlogged merged files), never a double read.
//!
//! Merging always writes into the table's own directory, so compacting a file
//! the table adopted (`with_pre_existing_parquets`, which leaves the file where its owner
//! put it) copies those rows under the database root. The adopted original is
//! dropped from the log but never deleted: it is not the table's to delete, and
//! the vacuum sweep only reclaims files under the table's own location. A table
//! that adopts a directory of small files and then compacts them therefore ends
//! up storing that data twice, once in the user's directory and once in its own.
//! That is the price of never writing into a directory the user owns.
//!
//! A [`DeltaDatastore`] self-manages its own compaction: when opened with a
//! [`MaintenanceConfig`] that enables it, the datastore spawns a [`Compacter`]
//! loop bound to itself. The loop polls, each round reloading a table to its
//! latest log version before scanning, and stops when the datastore's dispatch
//! pool begins to shut down (so an in-flight merge finishes before the workers
//! tear down).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::time::MissedTickBehavior;
use tracing::{error, info, warn};

use crate::store::ObjectPath;
use crate::{CatalogTable, DeltaDatastore, FileRef, scalar_values_equal};
use planner::catalog::SchemaQualifiedTableName;

/// Rows per row group in a merged file.
const ROW_GROUP_ROWS: usize = 128 * 1024;

/// Default compaction target: files smaller than this are merge candidates,
/// and a merge runs once their combined size reaches it.
pub const DEFAULT_COMPACT_BYTES: u64 = 64 * 1024 * 1024;

/// Default cadence for re-checking the tables' logs. Candidates only change
/// when a flush commits a new version, so seconds-scale is plenty.
pub const DEFAULT_COMPACT_POLL: Duration = Duration::from_secs(10);

/// Default cadence for reloading the tables from the store. This bounds how
/// stale a query's view of externally committed data in a shared remote store
/// is, so it trades freshness against the store's listing traffic.
pub const DEFAULT_REFRESH_INTERVAL: Duration = Duration::from_secs(30);

/// A partition with at least this many small files is merged even if they don't
/// yet add up to a full output, otherwise partitions whose data never reaches
/// the byte target accumulate small files without bound. Kept well above the
/// count that fills one byte-target output, so a busy partition always merges on
/// the byte trigger (full-size outputs); this count trigger is only the safety
/// net for a partition whose data trickles in, and a higher bar there means
/// fewer small sub-target merges.
pub const DEFAULT_MIN_FILES_TO_MERGE: usize = 4;

/// What background maintenance a [`DeltaDatastore`] runs for itself once opened.
/// The datastore spawns its own tasks from this on the ambient tokio runtime;
/// they stop when its dispatch pool begins shutting down.
#[derive(Clone)]
pub struct MaintenanceConfig {
    /// How often the datastore reloads its tables from the store (new Delta
    /// versions and new files' footers), so externally committed data in a
    /// shared remote store becomes visible to queries.
    pub refresh_interval: Duration,
    /// Compaction settings, or `None` to run no compacter (a read-only server
    /// or a remote-store deployment where another process owns compaction).
    pub compaction: Option<CompactionConfig>,
    /// Vacuum settings, or `None` to run no vacuumer (a read-only server, or a
    /// remote-store deployment where another process owns physical cleanup).
    /// Deletes tombstoned data files and superseded commit JSONs past their
    /// retention.
    pub vacuum: Option<crate::vacuum::VacuumConfig>,
}

/// Tuning for a datastore's self-managed compaction loop.
#[derive(Clone)]
pub struct CompactionConfig {
    /// Candidate threshold and merge trigger (see [`DEFAULT_COMPACT_BYTES`]).
    pub target_bytes: u64,
    /// Count trigger for a sub-target partition's pile (see
    /// [`DEFAULT_MIN_FILES_TO_MERGE`]).
    pub min_files: usize,
    /// How often to re-check the tables' logs for newly-accumulated files.
    pub poll_interval: Duration,
}

/// Compacts every table of a datastore into target-sized Parquet files,
/// entirely off the tables' logs: poll, reload, merge what's eligible. Holds
/// nothing but a datastore handle, so the hosting process is a deployment
/// detail: the datastore bundles one when maintenance enables it, and a
/// dedicated process could run another over the same database root.
pub struct Compacter {
    /// Candidate threshold and merge trigger (see [`DEFAULT_COMPACT_BYTES`]).
    target_bytes: u64,
    /// Count trigger for a sub-target partition's pile (see
    /// [`DEFAULT_MIN_FILES_TO_MERGE`]): merge a low-traffic partition's small
    /// files once this many accumulate, even below `target_bytes`.
    min_files: usize,
    /// How often to re-check the tables' logs for newly-accumulated files.
    poll_interval: Duration,
    datastore: Arc<DeltaDatastore>,
}

impl Compacter {
    pub fn new(
        target_bytes: u64,
        min_files: usize,
        poll_interval: Duration,
        datastore: Arc<DeltaDatastore>,
    ) -> Self {
        Self {
            target_bytes,
            min_files: min_files.max(2),
            poll_interval,
            datastore,
        }
    }

    /// The compaction loop: every `poll_interval`, sweep the datastore's tables
    /// and merge whatever is eligible. Runs until the task is aborted on shutdown.
    /// The first tick fires immediately, so leftovers from a previous run are
    /// handled at startup.
    pub async fn run(self: Arc<Self>) {
        let mut tick = tokio::time::interval(self.poll_interval);
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            tick.tick().await;
            self.compact_all().await;
        }
    }

    /// One poll round over every table the datastore knows. (Public so tests,
    /// and a future standalone compacter binary, can drive one round without
    /// the loop.)
    pub async fn compact_all(&self) {
        for (name, table) in self.datastore.tables() {
            self.compact_table(&name, table).await;
        }
    }

    /// One manual sweep of the single table `name`, for the `COMPACT`
    /// statement. Returns whether the table exists; the sweep itself reports
    /// nothing, so the caller judges progress by the table's log version.
    pub async fn sweep_table(&self, name: &SchemaQualifiedTableName) -> bool {
        let Some((name, table)) = self
            .datastore
            .tables()
            .into_iter()
            .find(|(candidate, _)| candidate == name)
        else {
            return false;
        };
        self.compact_table(&name, table).await;
        true
    }

    /// One table's round: reload it to its latest log version (this is what
    /// lets a compacter on a shared remote store in *another process* see files
    /// the server registered), then merge batches of eligible files until none
    /// remain.
    /// Each batch is capped at roughly one output file's worth, so a long
    /// backlog (e.g. after a restart) is worked off with bounded memory.
    /// Errors are logged and end the table's round, the next poll retries.
    async fn compact_table(&self, name: &SchemaQualifiedTableName, mut table: CatalogTable) {
        if let Err(e) = table.refresh() {
            warn!(table = %name, error = %e, "compaction: table refresh failed");
            return;
        }
        let id = table.id();
        while let Some(inputs) = self.next_batch(&table) {
            // `spawn_blocking` needs a `'static` closure, so it gets its own
            // handle rather than a borrow of `self`.
            let datastore = self.datastore.clone();
            match tokio::task::spawn_blocking(move || swap_batch(&datastore, id, &inputs)).await {
                Ok(Ok((merged, committed))) => {
                    info!(
                        table = %name,
                        files = merged.len(),
                        "compacted batch"
                    );
                    // Continue from the copy the swap published, so the next
                    // batch is picked from the merged file set.
                    table = committed;
                }
                Ok(Err(e)) => {
                    error!(table = %name, error = %e, "compaction merge failed");
                    return;
                }
                Err(e) => {
                    error!(table = %name, error = %e, "compaction job panicked");
                    return;
                }
            }
        }
    }

    /// The next batch of `table`'s files to merge, straight from its current
    /// log version: files under the target size, oldest-named first, cut off
    /// once they amount to one output file. `None` when there's nothing worth
    /// doing (no table, fewer than two small files, or not enough bytes for a
    /// full output yet, merging earlier would just rewrite the same rows
    /// again on the next flush).
    fn next_batch(&self, table: &CatalogTable) -> Option<Vec<FileRef>> {
        // Compaction merges within a single partition: across partitions it
        // couldn't stamp the merged file with one tuple, and the write pipeline
        // would just re-split it back out, making no progress. Group by the
        // typed partition tuple and fill a batch from one group.
        let partition_of: HashMap<ObjectPath, _> = table.file_partitions().into_iter().collect();
        let mut by_partition: Vec<(Option<crate::PartitionValues>, Vec<FileRef>)> = Vec::new();
        for file in table.file_refs() {
            if file.size >= self.target_bytes {
                continue;
            }
            let key = partition_of.get(&file.path).cloned().unwrap_or(None);
            match by_partition
                .iter_mut()
                .find(|(partition, _)| partition_values_equal(partition, &key))
            {
                Some((_, files)) => files.push(file),
                None => by_partition.push((key, vec![file])),
            }
        }

        for (_, mut files) in by_partition {
            if files.len() < 2 {
                continue;
            }
            // Oldest first (file names embed a timestamp + sequence).
            files.sort_by(|a, b| a.path.as_str().cmp(b.path.as_str()));

            // Take as many small files as it takes to fill one ~target-sized
            // output, and merge them.
            let mut total = 0u64;
            let mut batch = Vec::new();
            for file in files {
                total += file.size;
                batch.push(file);
                if total >= self.target_bytes {
                    return Some(batch);
                }
            }

            // The partition's small files don't add up to a full output (e.g. a
            // low-traffic partition of a many-partition table). Merge the pile
            // anyway once enough have accumulated, otherwise such a partition
            // accumulates small files forever, which is what bloats a table to
            // tens of thousands of tiny files. (The merged file is itself a
            // candidate, so it keeps growing toward the target as more arrive.)
            if batch.len() >= self.min_files {
                return Some(batch);
            }
        }
        None
    }
}

/// Swap one batch of merged files into the table with identity `id`, and hand
/// back the copy the swap published alongside the files it wrote.
///
/// That copy is what lets the caller's loop advance past its own swap: batches
/// are picked off the table's file list, so a caller still holding its pre-merge
/// copy would re-pick the very files this swap replaced, and the next removed-files
/// check would reject them. Picking up another writer's commit is incidental; the
/// sweep's own `refresh` would catch that on the next round anyway.
///
/// It is re-read rather than returned from the commit because
/// [`DeltaDatastore::publish_table`] consumes the committed copy, so returning it
/// would put a deep clone on the INSERT path to save one here. Reading it on this
/// blocking thread rather than in the caller at least keeps that clone off the
/// reactor.
fn swap_batch(
    datastore: &DeltaDatastore,
    id: uuid::Uuid,
    inputs: &[FileRef],
) -> Result<(Vec<FileRef>, CatalogTable), crate::Error> {
    let merged = compact_table_files(datastore, id, inputs, ROW_GROUP_ROWS)?;
    let committed = datastore
        .table_handle_by_id(&id)
        .ok_or_else(|| crate::Error::TableNotFound(id.to_string()))?;
    Ok((merged, committed))
}

/// Merge `inputs` of the table with identity `id` into fresh target-sized files
/// and swap them in for the inputs, in one log commit labelled a rearrangement.
/// `inputs` must share one partition tuple; the caller batches them so. Returns
/// the merged files.
///
/// The merge itself (decode, re-encode, upload) runs off a plain read copy of
/// the table with no commit lock held, so it never blocks a concurrent INSERT's
/// commit; only the swap that follows takes it. If that swap fails for any
/// reason, including another writer having swapped the inputs out first, the
/// uncommitted merge outputs are deleted before the error is returned.
///
/// Blocking store I/O, so callers run it on the blocking pool.
pub fn compact_table_files(
    datastore: &DeltaDatastore,
    id: uuid::Uuid,
    inputs: &[FileRef],
    target_rows_per_group: usize,
) -> Result<Vec<FileRef>, crate::Error> {
    let table = datastore
        .table_handle_by_id(&id)
        .ok_or_else(|| crate::Error::TableNotFound(id.to_string()))?;
    let merged = table.merge_files(inputs, target_rows_per_group)?;

    let files: Vec<FileRef> = merged.iter().map(|file| file.file_ref().clone()).collect();
    let removed: Vec<ObjectPath> = inputs.iter().map(|file| file.path.clone()).collect();
    if let Err(error) = datastore.commit_to_table(id, &removed, merged, false) {
        for file in &files {
            if let Err(cleanup_error) = table.delete_data_file(&file.path) {
                warn!(
                    commit_error = %error,
                    cleanup_error = %cleanup_error,
                    file = %file.path,
                    "compaction: deleting output after commit failure failed (orphan left)"
                );
            }
        }
        return Err(error);
    }
    Ok(files)
}

fn partition_values_equal(
    left: &Option<crate::PartitionValues>,
    right: &Option<crate::PartitionValues>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => scalar_values_equal(left, right),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeltaDatastore;
    use crate::parquet::ParquetTable;
    use arrow_array::{Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch};
    use parquet::arrow::ArrowWriter;
    use planner::catalog::SchemaQualifiedTableName;
    use std::path::Path;

    const RING_BUFFERS: usize = 64 * 1024 * 1024 / BUFFER_SIZE;

    /// Write `values` as a small SNAPPY Parquet file (single Int64 `Timestamp`
    /// column) into `dir` under `file_name`. A table created over `dir` picks it
    /// up as a small input for the compacter to merge.
    fn write_parquet_file(dir: &Path, file_name: &str, values: Vec<i64>) {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "Timestamp",
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(values)) as _],
        )
        .unwrap();
        let file = std::fs::File::create(dir.join(file_name)).unwrap();
        // SNAPPY, like every pivot-written file -- the decompressor expects it.
        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// `CREATE TABLE <name> (Timestamp Int64) WITH (with_pre_existing_parquets = dir)`
    /// against `datastore`, the way the server would run it -- or, when `dir` is
    /// `None`, an empty table adopting nothing.
    fn create_table(
        datastore: &Arc<DeltaDatastore>,
        dispatcher: &DataFlowDispatcher,
        name: &str,
        dir: Option<&Path>,
    ) {
        use datastore::DatastoreTransaction as _;
        use planner::catalog::{Column, CreateTableRequest};
        let options = match dir {
            Some(dir) => std::collections::HashMap::from([(
                "with_pre_existing_parquets".to_string(),
                dir.to_str().unwrap().to_string(),
            )]),
            None => std::collections::HashMap::new(),
        };
        let request = CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: name.to_string(),
            columns: vec![Column {
                name: "Timestamp".to_string(),
                col_type: planner::types::Type::Int64,
            }],
            options,
            if_not_exists: false,
        };
        let transaction = datastore.clone().begin_transaction();
        transaction
            .bind_create_table(request)
            .unwrap()
            .compile(dispatcher)
            .unwrap()
            .execute()
            .collect()
            .unwrap();
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(transaction.commit())
            .unwrap();
    }

    /// Refresh `name` to the latest committed manifest (a writer evolves a
    /// cloned-out handle, so the datastore's own copy lags until a refresh), then
    /// return its current row groups.
    fn fresh_parquet(datastore: &DeltaDatastore, name: &str) -> Arc<ParquetTable> {
        let mut table = datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema(name))
            .expect("table exists");
        table.refresh().expect("manifest reload");
        table.build_scan_view(&[], &[]).expect("build scan view")
    }

    /// Drive one full compaction sweep to completion on a temporary runtime.
    fn run_one_sweep(compacter: &Compacter) {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(compacter.compact_all());
    }

    /// End-to-end compaction: several registered small files merge into one, the
    /// datastore swaps to the merged file in one log commit, and it decodes back to
    /// all the original rows.
    #[test]
    fn compaction_merges_registered_files_and_swaps_datastore() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();
        write_parquet_file(&adopted_dir, "a.parquet", vec![1, 2, 3]);
        write_parquet_file(&adopted_dir, "b.parquet", vec![4, 5]);
        write_parquet_file(&adopted_dir, "c.parquet", vec![6, 7, 8, 9]);

        let datastore =
            DeltaDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
        );
        assert_eq!(
            datastore
                .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
                .unwrap()
                .len(),
            3
        );

        let total: u64 = datastore
            .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap()
            .iter()
            .map(|f| f.size)
            .sum();
        let compacter = Compacter::new(
            total,
            DEFAULT_MIN_FILES_TO_MERGE,
            std::time::Duration::from_secs(1),
            datastore.clone(),
        );
        run_one_sweep(&compacter);

        assert_eq!(
            datastore
                .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
                .unwrap()
                .len(),
            1
        );
        let parquet = fresh_parquet(&datastore, "events");
        assert_eq!(parquet.row_groups().len(), 1);
        assert_eq!(parquet.row_groups()[0].num_rows, 9);
        assert!(
            datastore
                .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
                .unwrap()
                .iter()
                .all(|f| f.path.as_str().starts_with("pivot-"))
        );

        dispatch.exit();
    }

    /// Compaction is location-agnostic: a table whose adopted files were listed
    /// **store-relative** (a directory under the database root, resolved through
    /// the store -- the same path a remote `s3://` root takes) compacts through
    /// the exact same code, with writes and deletes going through the table's
    /// store handle.
    #[test]
    fn compaction_works_on_store_relative_tables() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();

        // Two small files under a prefix inside the database root.
        write_parquet_file(&adopted_dir, "a.parquet", vec![1, 2]);
        write_parquet_file(&adopted_dir, "b.parquet", vec![3]);

        let datastore =
            DeltaDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        // A relative adoption path: the files are read at `events` under the
        // store root, rather than at an absolute path of their own.
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
        );
        let table_dir = db.path().join(
            datastore
                .table_handle(&SchemaQualifiedTableName::in_default_schema("events"))
                .unwrap()
                .location(),
        );

        let total: u64 = datastore
            .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap()
            .iter()
            .map(|f| f.size)
            .sum();
        let compacter = Compacter::new(
            total,
            DEFAULT_MIN_FILES_TO_MERGE,
            std::time::Duration::from_secs(1),
            datastore.clone(),
        );
        run_one_sweep(&compacter);

        // One merged object replaced the two inputs, in the store and in the
        // datastore's table view.
        let files = datastore
            .table_files(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].path.as_str().starts_with("pivot-"));
        let on_disk: Vec<_> = std::fs::read_dir(&table_dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".parquet"))
            .collect();
        // The merged object is written to the table's own directory; the two
        // adopted inputs are left where they were.
        assert!(on_disk.contains(&files[0].path.as_str().to_string()));
        assert!(adopted_dir.join("a.parquet").exists());
        assert!(adopted_dir.join("b.parquet").exists());

        let rows = fresh_parquet(&datastore, "events")
            .row_groups()
            .iter()
            .map(|rg| rg.num_rows)
            .sum::<i64>();
        assert_eq!(rows, 3);

        dispatch.exit();
    }
}
