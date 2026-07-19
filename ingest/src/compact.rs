//! Background compaction of a table's small Parquet files.
//!
//! Frequent flushes keep ingest latency low but litter a table with small
//! files, and scans pay per file. The [`Compacter`] merges them: whenever the
//! table's files smaller than the target size add up to at least one
//! target-sized output, it rewrites that batch as one file (with full-size row
//! groups) and swaps it into the table in a single log commit.
//!
//! The compacter is **location-agnostic**: candidates come from the table's
//! manifest (paths + sizes — no directory scanning), and the merge (read,
//! upload, swap) runs through [`CatalogTable::compact_files`]. A table under an
//! `s3://` database root compacts through the exact same code path as a local
//! one.
//!
//! Merging is **one dataflow** on the dispatch worker pool: the scan stages
//! decode the inputs' row groups, the encode stages repack them into full-size
//! row groups, and the upload stage writes each merged file over the shared
//! io_uring ring — the same operators an INSERT uses. The compacter's own task
//! only drives that dataflow (from a blocking thread) and, once the uploads
//! land, commits the swap.
//!
//! Crash safety comes from the table log, not from ordering tricks: a file is
//! part of the table iff the current log version lists it. The merged files are
//! uploaded, then swapped in for the inputs in one Delta commit (labelled a
//! rearrangement, not a data change); a crash before the commit leaves only
//! *orphan* objects (unlogged merged files), never a double read.
//!
//! The compacter is **deployment-agnostic** for the same reason: it talks to
//! nothing but the catalog (and through it, the table log and store), so it
//! can run inside the server or as a separate process over the same database
//! root. It covers every table of the catalog it is handed and polls — each
//! round reloads a table to its latest log version before scanning — rather
//! than being woken by the ingest path; the sinks don't know it exists.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use arrow_array::{ArrayRef, Scalar};
use catalog::store::ObjectPath;
use catalog::{CatalogTable, FileRef, ParquetCatalog, scalar_values_equal};
use tokio::sync::watch;
use tokio::time::MissedTickBehavior;
use tracing::{error, info, warn};

use crate::sink::{ROW_GROUP_ROWS, ROW_GROUPS_PER_FILE};

/// Default compaction target: files smaller than this are merge candidates,
/// and a merge runs once their combined size reaches it.
pub const DEFAULT_COMPACT_BYTES: u64 = 64 * 1024 * 1024;

/// Default cadence for re-checking the tables' logs. Candidates only change
/// when a flush commits a new version, so seconds-scale is plenty.
pub const DEFAULT_COMPACT_POLL: Duration = Duration::from_secs(10);

/// A partition with at least this many small files is merged even if they don't
/// yet add up to a full output — otherwise partitions whose data never reaches
/// the byte target accumulate small files without bound. Kept well above the
/// count that fills one byte-target output, so a busy partition always merges on
/// the byte trigger (full-size outputs); this count trigger is only the safety
/// net for a partition whose data trickles in, and a higher bar there means
/// fewer small sub-target merges.
pub const DEFAULT_MIN_FILES_TO_MERGE: usize = 4;

/// Compacts every table of a catalog into target-sized Parquet files,
/// entirely off the tables' logs: poll, reload, merge what's eligible. Holds
/// nothing but a catalog handle, so the hosting process is a deployment
/// detail — the server bundles one, and a dedicated process can run another
/// over the same database root.
pub struct Compacter {
    /// Candidate threshold and merge trigger (see [`DEFAULT_COMPACT_BYTES`]).
    target_bytes: u64,
    /// Count trigger for a sub-target partition's pile (see
    /// [`DEFAULT_MIN_FILES_TO_MERGE`]): merge a low-traffic partition's small
    /// files once this many accumulate, even below `target_bytes`.
    min_files: usize,
    /// How often to re-check the tables' logs for newly-accumulated files.
    poll_interval: Duration,
    catalog: Arc<ParquetCatalog>,
    /// Cumulative work counters, for the introspection API.
    stats: CompactStats,
}

/// The compacter's cumulative work counters, tracked per table behind a mutex
/// (merges are infrequent, so the lock is uncontended).
#[derive(Default)]
struct CompactStats {
    inner: Mutex<CompactState>,
}

#[derive(Default)]
struct CompactState {
    per_table: HashMap<String, CompactStatsSnapshot>,
    /// When the last full sweep ran (regardless of whether it merged anything).
    last_sweep_unix_ms: u64,
}

/// One table's compaction counters.
#[derive(Debug, Clone, Default)]
pub struct CompactStatsSnapshot {
    /// Merge batches successfully committed.
    pub compactions: u64,
    /// Small input files merged away.
    pub files_merged_in: u64,
    /// Target-sized files written.
    pub files_written: u64,
    pub bytes_written: u64,
    /// Unix-epoch milliseconds of this table's last merge, or 0 if none.
    pub last_run_unix_ms: u64,
}

/// A point-in-time read of the compacter's work, per table plus the last sweep.
#[derive(Debug, Clone, Default)]
pub struct CompactSnapshot {
    pub per_table: HashMap<String, CompactStatsSnapshot>,
    pub last_sweep_unix_ms: u64,
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl Compacter {
    pub fn new(
        target_bytes: u64,
        min_files: usize,
        poll_interval: Duration,
        catalog: Arc<ParquetCatalog>,
    ) -> Self {
        Self {
            target_bytes,
            min_files: min_files.max(2),
            poll_interval,
            catalog,
            stats: CompactStats::default(),
        }
    }

    /// A snapshot of the compacter's per-table work counters.
    pub fn snapshot(&self) -> CompactSnapshot {
        let state = self.stats.inner.lock().unwrap();
        CompactSnapshot {
            per_table: state.per_table.clone(),
            last_sweep_unix_ms: state.last_sweep_unix_ms,
        }
    }

    /// Record a committed merge against `table`.
    fn record_merge(&self, table: &str, files_merged_in: u64, written: &[FileRef]) {
        let bytes: u64 = written.iter().map(|f| f.size).sum();
        let mut state = self.stats.inner.lock().unwrap();
        let entry = state.per_table.entry(table.to_string()).or_default();
        entry.compactions += 1;
        entry.files_merged_in += files_merged_in;
        entry.files_written += written.len() as u64;
        entry.bytes_written += bytes;
        entry.last_run_unix_ms = now_unix_ms();
    }

    /// The compaction loop: every `poll_interval`, sweep the catalog's tables
    /// and merge whatever is eligible, until shutdown flips. The first tick
    /// fires immediately, so leftovers from a previous run are handled at
    /// startup.
    pub async fn run(self: Arc<Self>, mut shutdown_rx: watch::Receiver<bool>) {
        let mut tick = tokio::time::interval(self.poll_interval);
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                _ = tick.tick() => self.compact_all().await,
                res = shutdown_rx.changed() => {
                    if res.is_err() || *shutdown_rx.borrow() {
                        return;
                    }
                }
            }
        }
    }

    /// One poll round over every table the catalog knows. (Public so tests —
    /// and a future standalone compacter binary — can drive one round without
    /// the loop.)
    pub async fn compact_all(&self) {
        self.stats.inner.lock().unwrap().last_sweep_unix_ms = now_unix_ms();
        for table in self.catalog.tables() {
            self.compact_table(table).await;
        }
    }

    /// One table's round: reload it to its latest log version (this is what
    /// lets a compacter in *another process* see files the server
    /// registered), then merge batches of eligible files until none remain.
    /// Each batch is capped at roughly one output file's worth, so a long
    /// backlog (e.g. after a restart) is worked off with bounded memory.
    /// Errors are logged and end the table's round — the next poll retries.
    async fn compact_table(&self, mut table: CatalogTable) {
        if let Err(e) = table.refresh() {
            warn!(table = table.name(), error = %e, "compaction: table refresh failed");
            return;
        }
        while let Some(inputs) = self.next_batch(&table) {
            let input_count = inputs.len() as u64;
            let job = CompactJob {
                table: table.clone(),
            };
            match tokio::task::spawn_blocking(move || job.compact(inputs)).await {
                Ok(Ok((committed, merged))) => {
                    // Continue from the copy that performed the swap: it
                    // already holds the committed version, and it is the only
                    // copy carrying the merged files' sort bounds (the Delta
                    // log does not persist them). Publish it so the next
                    // query's snapshot reads the merged file instead of the
                    // swapped-out inputs.
                    table = committed;
                    // `merged` is empty when another writer won the swap (no real
                    // work happened); only count an actual merge.
                    if !merged.is_empty() {
                        self.record_merge(table.name(), input_count, &merged);
                    }
                    info!(
                        table = table.name(),
                        files = merged.len(),
                        "compacted batch"
                    );
                    self.catalog.publish_table(table.clone());
                }
                Ok(Err(e)) => {
                    error!(table = table.name(), error = %e, "compaction merge failed");
                    return;
                }
                Err(e) => {
                    error!(table = table.name(), error = %e, "compaction job panicked");
                    return;
                }
            }
        }
    }

    /// The next batch of `table`'s files to merge, straight from its current
    /// log version: files under the target size, oldest-named first, cut off
    /// once they amount to one output file. `None` when there's nothing worth
    /// doing (no table, fewer than two small files, or not enough bytes for a
    /// full output yet — merging earlier would just rewrite the same rows
    /// again on the next flush).
    fn next_batch(&self, table: &CatalogTable) -> Option<Vec<FileRef>> {
        // Compaction merges within a single partition: across partitions it
        // couldn't stamp the merged file with one tuple, and the write pipeline
        // would just re-split it back out, making no progress. Group by the
        // typed partition tuple and fill a batch from one group.
        let partition_of: HashMap<ObjectPath, _> = table.file_partitions().into_iter().collect();
        let mut by_partition: Vec<(Option<HashMap<String, Scalar<ArrayRef>>>, Vec<FileRef>)> =
            Vec::new();
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
            // Oldest first (sink file names embed a timestamp + sequence).
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
            // anyway once enough have accumulated — otherwise such a partition
            // accumulates small files forever, which is what bloats a sink to
            // tens of thousands of tiny files. (The merged file is itself a
            // candidate, so it keeps growing toward the target as more arrive.)
            if batch.len() >= self.min_files {
                return Some(batch);
            }
        }
        None
    }
}

fn partition_values_equal(
    left: &Option<HashMap<String, Scalar<ArrayRef>>>,
    right: &Option<HashMap<String, Scalar<ArrayRef>>>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => scalar_values_equal(left, right),
        _ => false,
    }
}

/// One merge, run on a blocking thread (it drives a dataflow, which blocks the
/// driving thread while the work itself runs on the dispatch workers).
struct CompactJob {
    table: CatalogTable,
}

impl CompactJob {
    /// Merge `inputs` into target-sized files and swap them into the table in one
    /// log commit ([`CatalogTable::compact_files`]). Returns the table copy that
    /// holds the committed version alongside the merged files written (empty when
    /// another writer won the swap).
    ///
    /// The swapped-out inputs are not deleted here: a query that loaded the prior
    /// version is still reading them, and the swap's Delta Remove actions are
    /// their durable tombstones. The objects stay in place until the table grows
    /// VACUUM-style physical cleanup.
    fn compact(
        mut self,
        inputs: Vec<FileRef>,
    ) -> Result<(CatalogTable, Vec<FileRef>), catalog::Error> {
        let merged = self
            .table
            .compact_files(&inputs, ROW_GROUP_ROWS, ROW_GROUPS_PER_FILE)?;
        if merged.is_empty() {
            // Another writer (a second compacter over the same root) already
            // swapped these inputs out; our merge was discarded.
            warn!(
                table = self.table.name(),
                "compaction: inputs already swapped by another writer; discarding merge"
            );
        }
        Ok((self.table, merged))
    }
}
