//! Background compaction of a table's small Parquet files.
//!
//! Frequent flushes keep ingest latency low but litter a table with small
//! files, and scans pay per file. The [`Compacter`] merges them: whenever the
//! table's files smaller than the target size add up to at least one
//! target-sized output, it rewrites that batch as one file (with full-size row
//! groups) and swaps it into the table in a single log commit.
//!
//! The compacter is **location-agnostic**: candidates come from the table's
//! manifest (paths + sizes — no directory scanning), reads resolve through the
//! catalog (a local path or a presigned URL alike), and writes/deletes go
//! through the table's [`CatalogTable::write_data_file`] /
//! [`CatalogTable::delete_data_file`]. A table under an `s3://` database root
//! compacts through the exact same code path as a local one.
//!
//! Merging is **one dataflow** on the dispatch worker pool: the scan stages
//! decode the inputs' row groups and the write pipeline's encode stages
//! consume those batches directly ([`parquet_writing::encode_record_batches`]) — the data
//! never leaves the pool until finished files stream out. The compacter's own
//! task only drives that dataflow (from a blocking thread) and does the
//! log/store bookkeeping.
//!
//! Crash safety comes from the table log, not from ordering tricks: a file is
//! part of the table iff the current log version lists it. The commit order is
//! write the merged file(s) → commit the swap ([`replace_data_files`]) →
//! delete the inputs; a crash anywhere in between leaves only *orphan* objects
//! (an unlogged merged file, or already-swapped-out inputs), never a double
//! read.
//!
//! The compacter is **deployment-agnostic** for the same reason: it talks to
//! nothing but the catalog (and through it, the table log and store), so it
//! can run inside the server or as a separate process over the same database
//! root. It covers every table of the catalog it is handed and polls — each
//! round reloads a table to its latest log version before scanning — rather
//! than being woken by the ingest path; the sinks don't know it exists.
//!
//! [`replace_data_files`]: catalog::CatalogTable::replace_data_files

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use catalog::parquet::table_input;
use catalog::store::ObjectPath;
use catalog::{CatalogTable, FileRef, ManifestEntry, ParquetCatalog};
use dispatch::{DataFlowDispatcher, DataFlowError, Projection};
use tokio::sync::watch;
use tokio::time::MissedTickBehavior;
use tracing::{error, info, warn};

use crate::parquet_writing;
use crate::sink::{ROW_GROUP_ROWS, ROW_GROUPS_PER_FILE};

/// Why one compaction merge failed: either the scan→encode dataflow, or a
/// catalog write/swap/delete. Both already carry typed causes.
#[derive(Debug, thiserror::Error)]
enum CompactError {
    #[error("compaction scan/encode: {0}")]
    Encode(#[from] DataFlowError),
    #[error(transparent)]
    Catalog(#[from] catalog::Error),
}

/// Default compaction target: files smaller than this are merge candidates,
/// and a merge runs once their combined size reaches it.
pub const DEFAULT_COMPACT_BYTES: u64 = 64 * 1024 * 1024;

/// Default cadence for re-checking the tables' logs. Candidates only change
/// when a flush commits a new version, so seconds-scale is plenty.
pub const DEFAULT_COMPACT_POLL: Duration = Duration::from_secs(10);

/// A partition with at least this many small files is merged even if they don't
/// yet add up to a full output — otherwise partitions whose data never reaches
/// the byte target accumulate small files without bound.
const MIN_FILES_TO_MERGE: usize = 8;

/// Compacts every table of a catalog into target-sized Parquet files,
/// entirely off the tables' logs: poll, reload, merge what's eligible. Holds
/// nothing but a catalog handle, so the hosting process is a deployment
/// detail — the server bundles one, and a dedicated process can run another
/// over the same database root.
pub struct Compacter {
    /// Candidate threshold and merge trigger (see [`DEFAULT_COMPACT_BYTES`]).
    target_bytes: u64,
    /// How often to re-check the tables' logs for newly-accumulated files.
    poll_interval: Duration,
    catalog: Arc<ParquetCatalog>,
    /// Monotonic sequence for merged-file names (same collision guard as the
    /// sink's).
    seq: AtomicU64,
}

impl Compacter {
    pub fn new(target_bytes: u64, poll_interval: Duration, catalog: Arc<ParquetCatalog>) -> Self {
        Self {
            target_bytes,
            poll_interval,
            catalog,
            seq: AtomicU64::new(0),
        }
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
            let job = CompactJob {
                dispatcher: self.catalog.dispatcher().clone(),
                table: table.clone(),
                seq: self.seq.fetch_add(1, Ordering::Relaxed),
            };
            match tokio::task::spawn_blocking(move || job.compact(inputs)).await {
                Ok(Ok(merged)) => {
                    info!(
                        table = table.name(),
                        files = merged.len(),
                        "compacted batch"
                    );
                    // Pull in the swap we just committed before scanning again.
                    if let Err(e) = table.refresh() {
                        warn!(table = table.name(), error = %e, "compaction: refresh after merge failed");
                        return;
                    }
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
        // Reclaim the manifest versions left behind by this round's swaps (and
        // by any appends since the last sweep). Best-effort: a failure here just
        // leaves the old version files for the next round.
        if let Err(e) = table.prune_old_versions() {
            warn!(table = table.name(), error = %e, "compaction: pruning old manifest versions failed");
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
        // would just re-split it back out, making no progress. So group the small
        // files by partition (keyed by the tuple's JSON, `""` for unpartitioned)
        // and fill a batch from one group.
        let partition_of: HashMap<ObjectPath, _> = table.file_partitions().into_iter().collect();
        let mut by_partition: BTreeMap<String, Vec<FileRef>> = BTreeMap::new();
        for file in table.file_refs() {
            if file.size >= self.target_bytes {
                continue;
            }
            let key = partition_of
                .get(&file.path)
                .and_then(|p| p.as_ref())
                .map(|tuple| tuple.to_string())
                .unwrap_or_default();
            by_partition.entry(key).or_default().push(file);
        }

        for mut files in by_partition.into_values() {
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
            if batch.len() >= MIN_FILES_TO_MERGE {
                return Some(batch);
            }
        }
        None
    }
}

/// One merge, run on a blocking thread (it drives a dataflow, which blocks the
/// driving thread while the work itself runs on the dispatch workers).
struct CompactJob {
    dispatcher: DataFlowDispatcher,
    table: CatalogTable,
    seq: u64,
}

impl CompactJob {
    /// Decode `inputs` and re-encode them as one stream of target-sized row
    /// groups — a single scan→encode dataflow — then commit: write the merged
    /// file(s), swap them for the inputs in one manifest version, delete the
    /// inputs. Returns the merged files written.
    fn compact(self, inputs: Vec<FileRef>) -> Result<Vec<FileRef>, CompactError> {
        // Scan just the input files and re-encode their rows into target-sized
        // merged files — one scan→encode dataflow on the worker pool. The inputs
        // share one partition (see `next_batch`), so re-applying the table's
        // partition/sort spec reproduces that partition tuple and recomputes the
        // merged file's sort bounds.
        let parquet = self.table.parquet_table_for(&inputs);
        let columns = parquet.schema().fields().len();
        let scan = table_input(&self.dispatcher, &parquet, Projection::all(columns), false);
        let merged = parquet_writing::encode_record_batches(
            scan,
            Arc::from(self.table.partition_by()),
            Arc::from(self.table.sort_by()),
            ROW_GROUP_ROWS,
            ROW_GROUPS_PER_FILE,
        );

        // Write each merged file into the table's data location as it streams out
        // — so we never hold them all in memory at once — keeping the partition
        // tuple and sort bounds the pipeline recorded for it. Only the lightweight
        // manifest entries are gathered, for the one atomic swap below.
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let mut added = Vec::new();
        for (i, encoded) in merged.enumerate() {
            let encoded = encoded?;
            let name = format!("compacted-{millis}-{:06}-{i}.parquet", self.seq);
            let file = self
                .table
                .write_data_file(ObjectPath::new(name), &encoded.bytes)?;
            added.push(ManifestEntry {
                file,
                partition: encoded.partition,
                sort_bounds: encoded.sort_bounds,
            });
        }

        // Commit the swap (one manifest version: inputs out, merged in), then
        // delete the now-orphaned inputs. The order is crash-safe: a file is part
        // of the table iff the committed manifest lists it, so a crash before the
        // deletes leaves only orphan objects, never a double read.
        let removed: Vec<ObjectPath> = inputs.iter().map(|f| f.path.clone()).collect();
        let mut table = self.table;
        let committed = table.replace_data_files(&removed, &added)?;
        if !committed {
            // Another writer (a second compacter over the same root) already
            // swapped these inputs out. Our merged output was never committed —
            // delete it so it doesn't linger as an orphan, and leave the inputs
            // alone: they belong to the swap that won.
            warn!(
                table = table.name(),
                "compaction: inputs already swapped by another writer; discarding merge"
            );
            for entry in &added {
                if let Err(e) = table.delete_data_file(&entry.file.path) {
                    warn!(error = %e, file = %entry.file.path, "compaction: deleting discarded merge output failed (orphan left)");
                }
            }
            return Ok(Vec::new());
        }
        for input in &inputs {
            if let Err(e) = table.delete_data_file(&input.path) {
                warn!(error = %e, file = %input.path, "compaction: deleting merged-away input failed (orphan left)");
            }
        }
        Ok(added.into_iter().map(|entry| entry.file).collect())
    }
}
