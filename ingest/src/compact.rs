//! Background compaction of a table's small Parquet files.
//!
//! Frequent flushes keep ingest latency low but litter a table with small
//! files, and scans pay per file. The [`Compacter`] merges them: whenever the
//! table's files smaller than the target size add up to at least one
//! target-sized output, it rewrites that batch as one file (with full-size row
//! groups) and swaps it into the table in a single log commit.
//!
//! The compacter is **location-agnostic**: candidates come from the table's
//! log (names + sizes — no directory scanning), reads resolve through the
//! catalog (a local path or a presigned URL alike), and writes/deletes go
//! through the catalog's [`write_data_file`]/[`delete_data_file`]. A table under
//! an `s3://` database root compacts through the exact same code path as a local
//! one.
//!
//! [`write_data_file`]: ParquetCatalog::write_data_file
//! [`delete_data_file`]: ParquetCatalog::delete_data_file
//!
//! Merging is **one dataflow** on the dispatch worker pool: the scan stages
//! decode the inputs' row groups and the write pipeline's encode stages
//! consume those batches directly ([`parquet_writing::encode`]) — the data
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
//! [`replace_data_files`]: ParquetCatalog::replace_data_files

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use dispatch::{DataFlowDispatcher, Projection};
use goose::parquet::{ParquetTable, table_input};
use goose::{CatalogTable, FileRef, ParquetCatalog};
use tokio::sync::watch;
use tokio::time::MissedTickBehavior;
use tracing::{debug, error, info, warn};

use crate::parquet_writing;
use crate::sink::{ROW_GROUP_ROWS, ROW_GROUPS_PER_FILE};

/// Default compaction target: files smaller than this are merge candidates,
/// and a merge runs once their combined size reaches it.
pub const DEFAULT_COMPACT_BYTES: u64 = 64 * 1024 * 1024;

/// Default cadence for re-checking the tables' logs. Candidates only change
/// when a flush commits a new version, so seconds-scale is plenty.
pub const DEFAULT_COMPACT_POLL: Duration = Duration::from_secs(10);

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
    async fn compact_table(&self, table: &CatalogTable) {
        // Refresh table, compact it (in dispatch from a spawn blocking thread), as long as there is 
        // what to compact for this table
        todo!()
    }

    /// The next batch of `table`'s files to merge, straight from its current
    /// log version: files under the target size, oldest-named first, cut off
    /// once they amount to one output file. `None` when there's nothing worth
    /// doing (no table, fewer than two small files, or not enough bytes for a
    /// full output yet — merging earlier would just rewrite the same rows
    /// again on the next flush).
    fn next_batch(&self, table: &CatalogTable) -> Option<Vec<FileRef>> {
        let files = table.files()?;
        let mut small: Vec<FileRef> = files
            .into_iter()
            .filter(|f| f.size < self.target_bytes)
            .collect();
        if small.len() < 2 {
            return None;
        }
        // Oldest first (sink file names embed a timestamp + sequence),
        // stopping once the batch fills one output.
        small.sort_by(|a, b| a.name.cmp(&b.name));
        let mut total = 0u64;
        let mut batch = Vec::new();
        for file in small {
            total += file.size;
            batch.push(file);
            if total >= self.target_bytes {
                return Some(batch);
            }
        }
        debug!(
            table,
            files = batch.len(),
            bytes = total,
            "small files below compaction threshold; waiting for more"
        );
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
    /// file(s), swap them for the inputs in the table log, delete the inputs.
    fn compact(self, inputs: Vec<FileRef>) -> Result<Vec<FileRef>, String> {
        // todo!()
    }
}
