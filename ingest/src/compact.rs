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
//! through the table's [`goose::TableData`] store handle. A table under an `s3://`
//! database root compacts through the exact same code path as a local one.
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
//! [`replace_data_files`]: ParquetCatalog::replace_data_files

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use dispatch::Projection;
use goose::parquet::{ParquetTable, table_input};
use goose::{LoggedFile, ParquetCatalog};
use tokio::sync::{Notify, watch};
use tracing::{debug, error, info};

use crate::parquet_writing;
use crate::sink::{ROW_GROUP_ROWS, ROW_GROUPS_PER_FILE};

/// Default compaction target: files smaller than this are merge candidates,
/// and a merge runs once their combined size reaches it.
pub const DEFAULT_COMPACT_BYTES: u64 = 64 * 1024 * 1024;

/// Compacts one table into target-sized Parquet files. Shared between the
/// sink (which [`notify`](Self::notify)s it after every registered flush) and
/// its background task ([`run`](Self::run)).
pub(crate) struct Compacter {
    /// The catalog table whose files are compacted; also the output file-name
    /// prefix.
    name: String,
    /// Candidate threshold and merge trigger (see [`DEFAULT_COMPACT_BYTES`]).
    target_bytes: u64,
    catalog: Arc<ParquetCatalog>,
    /// Pinged by the sink after each registered flush.
    wakeup: Notify,
    /// Monotonic sequence for merged-file names (same collision guard as the
    /// sink's).
    seq: AtomicU64,
}

impl Compacter {
    pub(crate) fn new(
        name: impl Into<String>,
        target_bytes: u64,
        catalog: Arc<ParquetCatalog>,
    ) -> Self {
        Self {
            name: name.into(),
            target_bytes,
            catalog,
            wakeup: Notify::new(),
            seq: AtomicU64::new(0),
        }
    }

    /// Wake the compaction task to re-check the table (called by the sink
    /// after a flushed file is registered).
    pub(crate) fn notify(&self) {
        self.wakeup.notify_one();
    }

    /// The compaction loop: merge whatever is already eligible (so leftovers
    /// from a previous run are handled at startup), then sleep until the sink
    /// flushes again or shutdown flips.
    pub(crate) async fn run(self: Arc<Self>, mut shutdown_rx: watch::Receiver<bool>) {
        loop {
            if *shutdown_rx.borrow() {
                return;
            }
            self.compact_eligible().await;
            tokio::select! {
                _ = self.wakeup.notified() => {}
                res = shutdown_rx.changed() => {
                    if res.is_err() || *shutdown_rx.borrow() {
                        return;
                    }
                }
            }
        }
    }

    /// Merge batches of eligible files until none remain. Each batch is capped
    /// at roughly one output file's worth, so a long backlog (e.g. after a
    /// restart) is worked off with bounded memory. Errors are logged and stop
    /// this round — the next flush retries. (`pub(crate)` so tests can drive
    /// one compaction round without the background loop.)
    pub(crate) async fn compact_eligible(&self) {
        loop {
            let Some(inputs) = self.next_batch() else {
                return;
            };
            let job = CompactJob {
                name: self.name.clone(),
                catalog: self.catalog.clone(),
                seq: self.seq.fetch_add(1, Ordering::Relaxed),
            };
            let count = inputs.len();
            match tokio::task::spawn_blocking(move || job.compact(inputs)).await {
                Ok(Ok(outputs)) => {
                    info!(
                        table = %self.name,
                        inputs = count,
                        outputs = outputs.len(),
                        "compacted parquet files"
                    );
                }
                Ok(Err(e)) => {
                    error!(table = %self.name, error = %e, "compaction failed");
                    return;
                }
                Err(e) => {
                    error!(table = %self.name, error = %e, "compaction task panicked");
                    return;
                }
            }
        }
    }

    /// The next batch of files to merge, straight from the table's current
    /// log version: files under the target size, oldest-named first, cut off
    /// once they amount to one output file. `None` when there's nothing worth
    /// doing (no table, fewer than two small files, or not enough bytes for a
    /// full output yet — merging earlier would just rewrite the same rows
    /// again on the next flush).
    fn next_batch(&self) -> Option<Vec<LoggedFile>> {
        let files = self.catalog.table_files(&self.name)?;
        let mut small: Vec<LoggedFile> = files
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
            table = %self.name,
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
    name: String,
    catalog: Arc<ParquetCatalog>,
    seq: u64,
}

impl CompactJob {
    /// Decode `inputs` and re-encode them as one stream of target-sized row
    /// groups — a single scan→encode dataflow — then commit: write the merged
    /// file(s), swap them for the inputs in the table log, delete the inputs.
    fn compact(self, inputs: Vec<LoggedFile>) -> Result<Vec<LoggedFile>, String> {
        let dispatcher = self.catalog.dispatcher().clone();
        let resolved = self
            .catalog
            .resolve_data_files(&self.name, &inputs)
            .map_err(|e| format!("resolving inputs: {e}"))?
            .ok_or_else(|| format!("table `{}` vanished", self.name))?;
        let table = Arc::new(
            ParquetTable::from_locations(&dispatcher, resolved)
                .map_err(|e| format!("reading input footers: {e}"))?,
        );
        let data = self
            .catalog
            .table_data(&self.name)
            .ok_or_else(|| format!("table `{}` vanished", self.name))?;

        // One dataflow: scan stages decode the inputs, encode stages cut and
        // compress full-size row groups, finished files stream out here.
        let columns = table.schema().fields().len();
        let scan = table_input(&dispatcher, &table, Projection::all(columns), false);
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let mut outputs: Vec<LoggedFile> = Vec::new();
        for (idx, bytes) in
            parquet_writing::encode(scan, ROW_GROUP_ROWS, ROW_GROUPS_PER_FILE).enumerate()
        {
            let bytes = match bytes {
                Ok(bytes) => bytes,
                Err(e) => {
                    // Unlogged files are invisible; deleting them is hygiene.
                    remove_all(&data, &outputs);
                    return Err(format!("re-encoding: {e}"));
                }
            };
            let name = format!(
                "{}-compacted-{}-{:06}-{:03}.parquet",
                self.name, millis, self.seq, idx
            );
            if let Err(e) = data.put(&name, &bytes) {
                remove_all(&data, &outputs);
                return Err(format!("writing {name}: {e}"));
            }
            outputs.push(LoggedFile {
                name,
                size: bytes.len() as u64,
            });
        }

        // The commit: one log version replaces the inputs with the outputs.
        let removed: Vec<String> = inputs.iter().map(|f| f.name.clone()).collect();
        match self
            .catalog
            .replace_data_files(&self.name, &removed, &outputs)
        {
            Ok(true) => {}
            Ok(false) => {
                remove_all(&data, &outputs);
                return Err(format!("table `{}` vanished before the swap", self.name));
            }
            Err(e) => {
                remove_all(&data, &outputs);
                return Err(format!("log swap: {e}"));
            }
        }
        // The inputs are out of the log; deleting them reclaims space. A
        // failure (or a crash) just leaves invisible orphans.
        for input in &inputs {
            if let Err(e) = data.delete(&input.name) {
                error!(
                    table = %self.name, file = %input.name, error = %e,
                    "failed deleting compacted input"
                );
            }
        }
        Ok(outputs)
    }
}

/// Best-effort cleanup of staged (never-logged, hence invisible) outputs.
fn remove_all(data: &goose::TableData, files: &[LoggedFile]) {
    for file in files {
        let _ = data.delete(&file.name);
    }
}
