//! Background compaction and layout optimization of a table's Parquet files.
//!
//! Frequent writes litter a table with small files, and scans pay per file. The
//! [`CompacterHandle`] merges them: small files are accumulated to a comfortably full
//! output, with a file-count fallback for pathological piles, while already
//! large files whose sort-key ranges overlap heavily are rewritten together to
//! improve their physical layout. Each rewrite swaps its inputs into the table
//! in a single log commit.
//!
//! The compacter is **location-agnostic**: candidates come from the table's
//! manifest (paths, sizes, partition values, and stats; no directory scanning), and the merge (read,
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
//! A [`DeltaDatastore`] owns one [`CompacterHandle`] actor. Configured timer ticks and
//! explicit `COMPACT` commands share its queue, so their rewrites never race in
//! one process. A timed round reloads each table to its latest log version;
//! commands carry the table snapshot captured by their query transaction. The
//! actor stops before the datastore's dispatch pool shuts down.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};
use tokio::time::MissedTickBehavior;
use tracing::{error, info, warn};

use crate::manifest::DeltaFileEntry;
use crate::store::ObjectPath;
use crate::{CatalogTable, DeltaDatastore, FileRef, scalar_values_equal};
use planner::catalog::SchemaQualifiedTableName;

mod overlap;

/// Rows per row group in a merged file.
const ROW_GROUP_ROWS: usize = 128 * 1024;

/// Default output-size target. Files smaller than this enter small-file
/// compaction; files at least this large enter layout optimization.
pub const DEFAULT_COMPACT_BYTES: u64 = 64 * 1024 * 1024;

/// Default headroom for compression gains: a group whose input bytes only just
/// reach the desired output size can encode to another undersized file.
const DEFAULT_MERGE_TARGET_MULTIPLIER: f64 = 1.3;

/// Default count at which a sub-target pile may take the balance-check path.
pub const DEFAULT_MIN_FILES_TO_MERGE: usize = 100;

/// Per-file fixed cost in the balance test. This prevents many similarly small
/// files from being rejected merely because their byte total is low.
const FIXED_FILE_COST_BYTES: f64 = (5 * 1024 * 1024) as f64;
const MIN_BALANCE_RATIO: f64 = 5.0;

const LAYOUT_OVERLAP_THRESHOLD: f64 = 0.3;

/// Default cadence for re-checking the tables' logs. Candidates only change
/// when a flush commits a new version, so seconds-scale is plenty.
pub const DEFAULT_COMPACT_POLL: Duration = Duration::from_secs(10);

/// Default cadence for reloading the tables from the store. This bounds how
/// stale a query's view of externally committed data in a shared remote store
/// is, so it trades freshness against the store's listing traffic.
pub const DEFAULT_REFRESH_INTERVAL: Duration = Duration::from_secs(30);

/// Default accumulated input size for a small-file merge.
pub fn default_merge_target_bytes(target_bytes: u64) -> u64 {
    (target_bytes as f64 * DEFAULT_MERGE_TARGET_MULTIPLIER) as u64
}

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
    /// Output-size target and small/large boundary (see [`DEFAULT_COMPACT_BYTES`]).
    pub target_bytes: u64,
    /// Accumulated small-file bytes that trigger a merge.
    pub merge_target_bytes: u64,
    /// Count at which a sub-target pile may take the balance-check path.
    pub min_files: usize,
    /// How often to re-check the tables' logs for newly-accumulated files.
    pub poll_interval: Duration,
}

/// Handle to the datastore's single compaction actor. Both periodic maintenance
/// and explicit `COMPACT` statements run through that actor, so two compaction
/// rounds in one process can never select and rewrite files concurrently.
#[derive(Clone)]
pub struct CompacterHandle {
    commands: mpsc::UnboundedSender<CompactionCommand>,
}

struct CompactionCommand {
    name: SchemaQualifiedTableName,
    table: CatalogTable,
    final_sweep: bool,
    result: oneshot::Sender<crate::Result<u64>>,
}

pub(crate) struct CompacterActor {
    /// Output-size target and small/large boundary (see [`DEFAULT_COMPACT_BYTES`]).
    target_bytes: u64,
    merge_target_bytes: u64,
    min_files: usize,
    /// How often to re-check the tables' logs, or `None` when only explicit
    /// commands drive this actor.
    poll_interval: Option<Duration>,
    datastore: Arc<DeltaDatastore>,
    commands: mpsc::UnboundedReceiver<CompactionCommand>,
}

impl CompacterHandle {
    pub(crate) fn new(
        target_bytes: u64,
        merge_target_bytes: u64,
        min_files: usize,
        poll_interval: Option<Duration>,
        datastore: Arc<DeltaDatastore>,
    ) -> (Self, CompacterActor) {
        let (sender, receiver) = mpsc::unbounded_channel();
        (
            Self { commands: sender },
            CompacterActor {
                target_bytes,
                merge_target_bytes,
                min_files: min_files.max(2),
                poll_interval,
                datastore,
                commands: receiver,
            },
        )
    }

    /// Enqueue one table from the caller's transaction snapshot. `FINAL` keeps
    /// selecting from each committed result until a round can do no more work,
    /// without applying the normal merge-size, file-count, balance, or 30%
    /// overlap guards.
    pub(crate) async fn compact(
        &self,
        name: SchemaQualifiedTableName,
        table: CatalogTable,
        final_sweep: bool,
    ) -> crate::Result<u64> {
        let (result, answer) = oneshot::channel();
        self.commands
            .send(CompactionCommand {
                name,
                table,
                final_sweep,
                result,
            })
            .map_err(|_| crate::Error::CompacterStopped)?;
        answer.await.map_err(|_| crate::Error::CompacterStopped)?
    }
}

impl CompacterActor {
    /// Run timer ticks and commands serially. The first configured tick fires
    /// immediately, preserving the background compacter's startup sweep.
    pub(crate) async fn run(mut self) {
        let mut tick = self.poll_interval.map(|d| {
            let mut t = tokio::time::interval(d);
            t.set_missed_tick_behavior(MissedTickBehavior::Skip);
            t
        });

        loop {
            tokio::select! {
                biased;
                command = self.commands.recv() => match command {
                    Some(command) => {
                        let res = self
                            .compact_table(&command.name, command.table, command.final_sweep)
                            .await;
                        let _ = command.result.send(res.map(|(_, count)| count));
                    },
                    None => return,
                },
                _ = async {
                    match tick.as_mut() {
                        Some(t) => t.tick().await,
                        None => std::future::pending().await,
                    }
                } => self.compact_all().await,
            }
        }
    }

    async fn compact_all(&self) {
        for (name, mut table) in self.datastore.tables() {
            if let Err(error) = table.refresh() {
                warn!(table = %name, error = %error, "compaction: table refresh failed");
                continue;
            }
            if let Err(error) = self.compact_table(&name, table, false).await {
                error!(table = %name, error = %error, "compaction round failed");
            }
        }
    }

    /// Compact at most one small-file group and then at most one overlapping
    /// large-file pair, choosing the second batch from the first batch's commit.
    async fn compact_table(
        &self,
        name: &SchemaQualifiedTableName,
        mut table: CatalogTable,
        final_sweep: bool,
    ) -> crate::Result<(CatalogTable, u64)> {
        let mut sweeps = 0;
        let apply_guards = !final_sweep;
        loop {
            let id = table.id();
            let mut compacted = false;

            if let Some(inputs) = self.next_small_batch(&table, apply_guards) {
                table = self.merge_batch(name, id, inputs, "small files").await?;
                compacted = true;
            }

            if let Some(inputs) = self.next_layout_optimization(&table, apply_guards) {
                table = self
                    .merge_batch(name, id, inputs, "layout optimization")
                    .await?;
                compacted = true;
            }

            sweeps += 1;
            if !final_sweep || !compacted {
                return Ok((table, sweeps));
            }
        }
    }

    async fn merge_batch(
        &self,
        name: &SchemaQualifiedTableName,
        id: uuid::Uuid,
        inputs: Vec<FileRef>,
        kind: &'static str,
    ) -> crate::Result<CatalogTable> {
        let input_sizes: Vec<u64> = inputs.iter().map(|file| file.size).collect();
        // `spawn_blocking` needs a `'static` closure, so it gets its own
        // datastore handle rather than a borrow of `self`.
        let datastore = self.datastore.clone();
        let (merged, committed) = tokio::task::spawn_blocking(move || {
            let merged = compact_table_files(&datastore, id, &inputs, ROW_GROUP_ROWS)?;
            let committed = datastore
                .table_handle_by_id(&id)
                .ok_or_else(|| crate::Error::TableNotFound(id.to_string()))?;
            Ok::<_, crate::Error>((merged, committed))
        })
        .await
        .map_err(|error| crate::Error::CompactionJobPanicked(error.to_string()))??;
        let output_sizes: Vec<u64> = merged.iter().map(|file| file.size).collect();
        info!(
            table = %name,
            kind,
            inputs = input_sizes.len(),
            input_sizes = ?input_sizes,
            input_bytes = input_sizes.iter().sum::<u64>(),
            outputs = output_sizes.len(),
            output_sizes = ?output_sizes,
            output_bytes = output_sizes.iter().sum::<u64>(),
            "compacted batch"
        );
        Ok(committed)
    }

    /// Pick one group of files below the target size from a single partition.
    /// A normal merge stops as soon as its input reaches the configured merge
    /// target. A pile that reaches the configured file-count threshold first
    /// uses the balance test, repeatedly discarding its largest file until the
    /// remainder is balanced enough to merge or no useful group remains.
    fn next_small_batch(&self, table: &CatalogTable, apply_guards: bool) -> Option<Vec<FileRef>> {
        let mut by_partition: Vec<(Option<crate::PartitionValues>, Vec<&DeltaFileEntry>)> =
            Vec::new();
        for entry in table.file_entries() {
            if entry.file.size >= self.target_bytes {
                continue;
            }
            match by_partition
                .iter_mut()
                .find(|(partition, _)| partition_values_equal(partition, &entry.partition))
            {
                Some((_, files)) => files.push(entry),
                None => by_partition.push((entry.partition.clone(), vec![entry])),
            }
        }

        by_partition.into_iter().find_map(|(_, files)| {
            if apply_guards {
                small_file_batch(files, self.merge_target_bytes, self.min_files)
            } else {
                (files.len() >= 2)
                    .then(|| files.into_iter().map(|entry| entry.file.clone()).collect())
            }
        })
    }

    /// Find the highest-overlap pair of already-large files. Pairs never cross
    /// partitions. Sort columns are considered in table order; a column only
    /// defers to the next one when both files are the same singleton on it.
    /// Guardless selection still requires positive overlap: rewriting disjoint
    /// ranges cannot improve their layout.
    fn next_layout_optimization(
        &self,
        table: &CatalogTable,
        apply_guards: bool,
    ) -> Option<Vec<FileRef>> {
        if table.sort_by().is_empty() {
            return None;
        }

        let mut by_partition: Vec<(Option<crate::PartitionValues>, Vec<&DeltaFileEntry>)> =
            Vec::new();
        for entry in table.file_entries() {
            if entry.file.size < self.target_bytes {
                continue;
            }
            match by_partition
                .iter_mut()
                .find(|(partition, _)| partition_values_equal(partition, &entry.partition))
            {
                Some((_, files)) => files.push(entry),
                None => by_partition.push((entry.partition.clone(), vec![entry])),
            }
        }

        let mut best: Option<(f64, Vec<FileRef>)> = None;
        for (_, files) in by_partition {
            let Some((score, left, right)) = overlap::highest_scoring_pair(&files, table.sort_by())
            else {
                continue;
            };
            if score > 0.0
                && (!apply_guards || score >= LAYOUT_OVERLAP_THRESHOLD)
                && best.as_ref().is_none_or(|(best, _)| score > *best)
            {
                best = Some((score, vec![left.file.clone(), right.file.clone()]));
            }
        }
        best.map(|(_, files)| files)
    }
}

fn small_file_batch(
    files: Vec<&DeltaFileEntry>,
    merge_target_bytes: u64,
    min_files: usize,
) -> Option<Vec<FileRef>> {
    let mut total = 0u64;
    let mut batch = Vec::with_capacity(files.len());
    for file in files {
        total += file.file.size;
        batch.push(file);
        if batch.len() >= 2 && total >= merge_target_bytes {
            return Some(batch.into_iter().map(|entry| entry.file.clone()).collect());
        }
    }

    if batch.len() < min_files.max(2) {
        return None;
    }

    // A few disproportionately large files can make the whole pile a poor
    // merge. Remove the current largest and run the same balance test again on
    // everything that remains; the count threshold is only the entry condition.
    while batch.len() >= 2 {
        let largest = batch
            .iter()
            .enumerate()
            .max_by_key(|(_, entry)| entry.file.size)
            .map(|(index, _)| index)
            .expect("a non-empty batch has a largest file");
        let largest_size = batch[largest].file.size;
        let ratio = (total as f64 + FIXED_FILE_COST_BYTES * batch.len() as f64)
            / (largest_size as f64 + FIXED_FILE_COST_BYTES);
        if ratio >= MIN_BALANCE_RATIO {
            return Some(batch.into_iter().map(|entry| entry.file.clone()).collect());
        }
        total -= largest_size;
        batch.swap_remove(largest);
    }
    None
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
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, Scalar};
    use arrow_schema::{DataType, Field, Schema};
    use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch};
    use parquet::arrow::ArrowWriter;
    use planner::catalog::SchemaQualifiedTableName;
    use std::path::Path;

    const RING_BUFFERS: usize = 64 * 1024 * 1024 / BUFFER_SIZE;

    fn file_entry(path: &str, size: u64) -> DeltaFileEntry {
        DeltaFileEntry::new(FileRef {
            path: ObjectPath::new(path),
            size,
        })
    }

    #[test]
    fn small_files_merge_at_configured_target() {
        let files = [file_entry("a", 70), file_entry("b", 59), file_entry("c", 1)];

        let batch = small_file_batch(files.iter().collect(), 130, 100).unwrap();

        assert_eq!(
            batch.iter().map(|file| file.size).collect::<Vec<_>>(),
            [70, 59, 1]
        );
    }

    #[test]
    fn small_file_fallback_removes_largest_until_balanced() {
        const MIB: u64 = 1024 * 1024;
        let mut files = vec![file_entry("outlier", 500 * MIB)];
        files.extend((0..99).map(|index| file_entry(&format!("small-{index}"), MIB)));

        let batch = small_file_batch(files.iter().collect(), 2 * 1024 * MIB, 100).unwrap();

        assert_eq!(batch.len(), 99, "the check continues below 100 files");
        assert!(batch.iter().all(|file| file.path.as_str() != "outlier"));
    }

    #[test]
    fn small_file_fallback_uses_configured_minimum() {
        let files: Vec<_> = (0..5)
            .map(|index| file_entry(&format!("small-{index}"), 1))
            .collect();

        let allowed = small_file_batch(files.iter().collect(), 100, 5);
        let waiting = small_file_batch(files.iter().collect(), 100, 6);

        assert_eq!(allowed.unwrap().len(), 5);
        assert!(waiting.is_none());
    }

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

    fn write_partitioned_parquet_file(
        dir: &Path,
        file_name: &str,
        timestamps: Vec<i64>,
        partition: i64,
    ) {
        let row_count = timestamps.len();
        let schema = Arc::new(Schema::new(vec![
            Field::new("Timestamp", DataType::Int64, false),
            Field::new("Partition", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(timestamps)) as _,
                Arc::new(Int64Array::from(vec![partition; row_count])) as _,
            ],
        )
        .unwrap();
        let file = std::fs::File::create(dir.join(file_name)).unwrap();
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
        sorted: bool,
    ) {
        use datastore::DatastoreTransaction as _;
        use planner::catalog::{Column, CreateTableRequest};
        let mut options = match dir {
            Some(dir) => std::collections::HashMap::from([(
                "with_pre_existing_parquets".to_string(),
                dir.to_str().unwrap().to_string(),
            )]),
            None => std::collections::HashMap::new(),
        };
        if sorted {
            options.insert("sort_by".to_string(), "Timestamp".to_string());
        }
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

    /// Create an empty table partitioned by its `Partition` column.
    fn create_partitioned_table(
        datastore: &Arc<DeltaDatastore>,
        dispatcher: &DataFlowDispatcher,
        name: &str,
    ) {
        use datastore::DatastoreTransaction as _;
        use planner::catalog::{Column, CreateTableRequest};
        let request = CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: name.to_string(),
            columns: vec![
                Column {
                    name: "Timestamp".to_string(),
                    col_type: planner::types::Type::Int64,
                },
                Column {
                    name: "Partition".to_string(),
                    col_type: planner::types::Type::Int64,
                },
            ],
            options: std::collections::HashMap::from([(
                "partition_by".to_string(),
                "Partition".to_string(),
            )]),
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

    fn partition_value(value: i64) -> crate::PartitionValues {
        std::collections::HashMap::from([(
            "Partition".to_string(),
            Scalar::new(Arc::new(Int64Array::from(vec![value])) as ArrayRef),
        )])
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
    fn run_one_sweep(target_bytes: u64, datastore: Arc<DeltaDatastore>) {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                let tables = datastore.tables();
                let (compacter, actor) = CompacterHandle::new(
                    target_bytes,
                    default_merge_target_bytes(target_bytes),
                    DEFAULT_MIN_FILES_TO_MERGE,
                    None,
                    datastore,
                );
                let actor = tokio::spawn(actor.run());
                for (name, table) in tables {
                    compacter.compact(name, table, false).await.unwrap();
                }
                actor.abort();
            });
    }

    fn run_command(
        target_bytes: u64,
        merge_target_bytes: u64,
        min_files: usize,
        datastore: Arc<DeltaDatastore>,
        name: SchemaQualifiedTableName,
        table: CatalogTable,
        final_sweep: bool,
    ) -> u64 {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async move {
                let (compacter, actor) = CompacterHandle::new(
                    target_bytes,
                    merge_target_bytes,
                    min_files,
                    None,
                    datastore,
                );
                let actor = tokio::spawn(actor.run());
                let sweeps = compacter.compact(name, table, final_sweep).await.unwrap();
                actor.abort();
                sweeps
            })
    }

    #[test]
    fn command_uses_the_supplied_table_snapshot() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_parquet_file(&adopted, "a.parquet", vec![1]);
        write_parquet_file(&adopted, "b.parquet", vec![2]);
        let datastore =
            DeltaDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(&adopted),
            false,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let captured = datastore.table_handle(&name).unwrap();
        let target = captured.file_refs().iter().map(|file| file.size).sum();
        write_parquet_file(&adopted, "c.parquet", vec![3]);
        let mut latest = captured.clone();
        latest
            .append_data_file(
                ObjectPath::new("c.parquet"),
                &std::fs::read(adopted.join("c.parquet")).unwrap(),
                None,
            )
            .unwrap();
        datastore.refresh_from_store().unwrap();

        let sweeps = run_command(
            target,
            default_merge_target_bytes(target),
            DEFAULT_MIN_FILES_TO_MERGE,
            datastore.clone(),
            name.clone(),
            captured,
            false,
        );

        assert_eq!(sweeps, 1);
        assert_eq!(datastore.table_handle(&name).unwrap().file_refs().len(), 3);
        dispatch.exit();
    }

    #[test]
    fn final_command_bypasses_small_file_guards() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let encoded = tempfile::tempdir().unwrap();
        let datastore =
            DeltaDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_partitioned_table(&datastore, dispatch.dispatcher(), "events");
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let mut table = datastore.table_handle(&name).unwrap();
        for (file, partition) in [("a", 1), ("b", 1), ("c", 2), ("d", 2)] {
            let file = format!("{file}.parquet");
            write_partitioned_parquet_file(encoded.path(), &file, vec![partition], partition);
            table
                .append_data_file(
                    ObjectPath::new(&file),
                    &std::fs::read(encoded.path().join(&file)).unwrap(),
                    Some(partition_value(partition)),
                )
                .unwrap();
        }
        datastore.refresh_from_store().unwrap();
        let table = datastore.table_handle(&name).unwrap();
        let target = table
            .file_refs()
            .iter()
            .map(|file| file.size)
            .max()
            .unwrap()
            + 1;

        let guarded_sweeps = run_command(
            target,
            u64::MAX,
            usize::MAX,
            datastore.clone(),
            name.clone(),
            table.clone(),
            false,
        );
        let guarded_files = datastore.table_handle(&name).unwrap().file_refs().len();
        let final_sweeps = run_command(
            target,
            u64::MAX,
            usize::MAX,
            datastore.clone(),
            name.clone(),
            table,
            true,
        );

        assert_eq!(guarded_sweeps, 1);
        assert_eq!(guarded_files, 4);
        assert_eq!(final_sweeps, 3);
        assert_eq!(datastore.table_handle(&name).unwrap().file_refs().len(), 2);
        dispatch.exit();
    }

    #[test]
    fn final_command_accepts_only_positive_layout_overlap() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted = db.path().join("events");
        std::fs::create_dir_all(&adopted).unwrap();
        write_parquet_file(&adopted, "a.parquet", vec![0, 17]);
        write_parquet_file(&adopted, "b.parquet", vec![15, 32]);
        write_parquet_file(&adopted, "c.parquet", vec![100, 110]);
        let datastore =
            DeltaDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(&adopted),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let table = datastore.table_handle(&name).unwrap();

        let guarded_sweeps = run_command(
            1,
            u64::MAX,
            usize::MAX,
            datastore.clone(),
            name.clone(),
            table.clone(),
            false,
        );
        let guarded_files = datastore.table_handle(&name).unwrap().file_refs().len();
        let final_sweeps = run_command(
            1,
            u64::MAX,
            usize::MAX,
            datastore.clone(),
            name.clone(),
            table,
            true,
        );

        assert_eq!(guarded_sweeps, 1);
        assert_eq!(guarded_files, 3);
        assert_eq!(final_sweeps, 2);
        let files = datastore.table_handle(&name).unwrap().file_refs();
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|file| file.path.name() == "c.parquet"));
        dispatch.exit();
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
            false,
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
        let target = (total as f64 / DEFAULT_MERGE_TARGET_MULTIPLIER).floor() as u64;
        run_one_sweep(target, datastore.clone());

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

    #[test]
    fn layout_optimization_merges_the_highest_overlap_large_pair() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let adopted_dir = db.path().join("events");
        std::fs::create_dir_all(&adopted_dir).unwrap();
        write_parquet_file(&adopted_dir, "a.parquet", vec![0, 100]);
        write_parquet_file(&adopted_dir, "b.parquet", vec![50, 150]);
        write_parquet_file(&adopted_dir, "c.parquet", vec![1_000, 1_100]);

        let datastore =
            DeltaDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_table(
            &datastore,
            dispatch.dispatcher(),
            "events",
            Some(Path::new("events")),
            true,
        );
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let files = datastore.table_files(&name).unwrap();
        let target = files.iter().map(|file| file.size).min().unwrap();
        run_one_sweep(target, datastore.clone());

        let files = datastore.table_files(&name).unwrap();
        assert_eq!(files.len(), 2);
        assert!(
            files.iter().any(|file| file.path.name() == "c.parquet"),
            "the disjoint file is not part of the chosen pair"
        );
        assert!(
            files
                .iter()
                .all(|file| file.path.name() != "a.parquet" && file.path.name() != "b.parquet")
        );

        dispatch.exit();
    }

    /// Partitioned compaction never mixes partitions, and both the replacement
    /// Adds and input tombstones retain the partition values needed to reload a
    /// valid Delta snapshot.
    #[test]
    fn compaction_preserves_partitioned_table_metadata_and_rows() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS, None);
        let db = tempfile::tempdir().unwrap();
        let encoded = tempfile::tempdir().unwrap();
        let datastore =
            DeltaDatastore::open(db.path().to_str().unwrap(), dispatch.dispatcher()).unwrap();
        create_partitioned_table(&datastore, dispatch.dispatcher(), "events");

        // Two inputs in each partition. If batching crossed the partition
        // boundary, the merged file could not truthfully carry one constant.
        let inputs = [
            ("one-a.parquet", 1),
            ("one-b.parquet", 1),
            ("two-a.parquet", 2),
            ("two-b.parquet", 2),
        ];
        let name = SchemaQualifiedTableName::in_default_schema("events");
        let mut table = datastore.table_handle(&name).unwrap();
        for (file_name, value) in inputs {
            write_partitioned_parquet_file(
                encoded.path(),
                file_name,
                vec![value * 10, value * 10 + 1],
                value,
            );
            table
                .append_data_file(
                    ObjectPath::new(file_name),
                    &std::fs::read(encoded.path().join(file_name)).unwrap(),
                    Some(partition_value(value)),
                )
                .unwrap();
        }
        let before_version = table.version();
        assert_eq!(table.file_refs().len(), 4);
        assert!(datastore.refresh_from_store().unwrap());

        let target = table
            .file_refs()
            .into_iter()
            .map(|file| file.size)
            .max()
            .unwrap()
            + 1;
        run_one_sweep(target, datastore.clone());

        // A table round performs only one small-file merge. The next round
        // handles the other partition.
        assert_eq!(datastore.table_handle(&name).unwrap().file_refs().len(), 3);
        run_one_sweep(target, datastore.clone());

        // Reload from Delta rather than trusting the in-memory swap. Exactly one
        // output per partition remains, and every original row is still present.
        let mut reloaded = datastore.table_handle(&name).unwrap();
        reloaded.refresh().unwrap();
        assert_eq!(reloaded.file_refs().len(), 2);
        let partitions = reloaded.file_partitions();
        assert_eq!(partitions.len(), 2);
        for expected in [partition_value(1), partition_value(2)] {
            assert!(partitions.iter().any(|(_, actual)| {
                actual
                    .as_ref()
                    .is_some_and(|actual| scalar_values_equal(actual, &expected))
            }));
        }
        let parquet = reloaded.build_scan_view(&[], &[]).unwrap();
        assert_eq!(
            parquet
                .row_groups()
                .iter()
                .map(|group| group.num_rows)
                .sum::<i64>(),
            8
        );

        // Inspect every compaction commit. Each removes and adds files from one
        // partition only, and Kernel serialized that value into both actions.
        let log_dir = db.path().join(reloaded.location()).join("_delta_log");
        let mut compacted_partitions = std::collections::BTreeSet::new();
        for version in before_version + 1..=reloaded.version() {
            let commit =
                std::fs::read_to_string(log_dir.join(format!("{version:020}.json"))).unwrap();
            let actions: Vec<serde_json::Value> = commit
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            let remove_values: std::collections::BTreeSet<_> = actions
                .iter()
                .filter_map(|action| action.get("remove"))
                .map(|remove| remove["partitionValues"]["Partition"].as_str().unwrap())
                .collect();
            assert_eq!(
                actions
                    .iter()
                    .filter(|action| action.get("remove").is_some())
                    .count(),
                2
            );
            let add_values: std::collections::BTreeSet<_> = actions
                .iter()
                .filter_map(|action| action.get("add"))
                .map(|add| add["partitionValues"]["Partition"].as_str().unwrap())
                .collect();
            assert_eq!(
                actions
                    .iter()
                    .filter(|action| action.get("add").is_some())
                    .count(),
                1
            );
            assert_eq!(remove_values.len(), 1, "one partition per compacted batch");
            assert_eq!(
                add_values, remove_values,
                "replacement stays in its partition"
            );
            compacted_partitions.extend(remove_values.into_iter().map(str::to_string));
        }
        assert_eq!(
            compacted_partitions,
            std::collections::BTreeSet::from(["1".to_string(), "2".to_string()])
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
            false,
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
        let target = (total as f64 / DEFAULT_MERGE_TARGET_MULTIPLIER).floor() as u64;
        run_one_sweep(target, datastore.clone());

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
