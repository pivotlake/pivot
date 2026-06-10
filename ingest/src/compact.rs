//! Background compaction of a sink's small Parquet files.
//!
//! Frequent flushes keep ingest latency low but litter a table's directory
//! with small files, and scans pay per file. The [`Compacter`] merges them:
//! whenever the *registered* files smaller than the target size add up to at
//! least one target-sized output, it rewrites that batch as one file (with
//! full-size row groups) and atomically swaps it into the catalog.
//!
//! All the CPU happens on the **dispatch worker pool**, like a flush: the
//! inputs are decoded back to Arrow batches by the engine's scan dataflow, and
//! re-encoded by the same [`parquet_writing`] pipeline
//! a flush uses. The compacter's own task only scans directories, drives those
//! dataflows from a blocking thread, and does the file/catalog bookkeeping.
//!
//! Only files the catalog already holds are candidates — the catalog is the
//! barrier that proves the sink finished registering a file, so a merge can
//! never race a registration into double-counting rows. The commit order is:
//! rename the merged file(s) into the directory, swap old-for-new in the
//! catalog (one atomic table update — no binding sees rows doubled or
//! missing), then unlink the inputs. A crash between the rename and the
//! unlinks can leave both the merged file and its inputs on disk — a restart
//! would then double-read them; that window is accepted for now (the
//! alternative ordering risks data *loss*, which is worse).

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use dispatch::{DataFlowDispatcher, Projection};
use goose::ParquetCatalog;
use goose::parquet::{ParquetTable, table_input};
use tokio::sync::{Notify, watch};
use tracing::{debug, error, info};

use crate::parquet_writing;
use crate::sink::{INFLIGHT_DIR, ROW_GROUP_ROWS, ROW_GROUPS_PER_FILE};

/// Default compaction target: files smaller than this are merge candidates,
/// and a merge runs once their combined size reaches it.
pub const DEFAULT_COMPACT_BYTES: u64 = 64 * 1024 * 1024;

/// Compacts one sink's directory into target-sized Parquet files. Shared
/// between the sink (which [`notify`](Self::notify)s it after every flush) and
/// its background task ([`run`](Self::run)).
pub(crate) struct Compacter {
    /// Sink/table name: the catalog table whose files are compacted, and the
    /// output file-name prefix.
    name: String,
    /// The sink's (local) output directory.
    dir: PathBuf,
    /// Candidate threshold and merge trigger (see [`DEFAULT_COMPACT_BYTES`]).
    target_bytes: u64,
    dispatcher: DataFlowDispatcher,
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
        dir: PathBuf,
        target_bytes: u64,
        dispatcher: DataFlowDispatcher,
        catalog: Arc<ParquetCatalog>,
    ) -> Self {
        Self {
            name: name.into(),
            dir,
            target_bytes,
            dispatcher,
            catalog,
            wakeup: Notify::new(),
            seq: AtomicU64::new(0),
        }
    }

    /// Wake the compaction task to re-check the directory (called by the sink
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
            let compacter = CompactJob {
                name: self.name.clone(),
                dir: self.dir.clone(),
                dispatcher: self.dispatcher.clone(),
                catalog: self.catalog.clone(),
                seq: self.seq.fetch_add(1, Ordering::Relaxed),
            };
            let count = inputs.len();
            match tokio::task::spawn_blocking(move || compacter.compact(inputs)).await {
                Ok(Ok(outputs)) => {
                    info!(
                        sink = %self.name,
                        inputs = count,
                        outputs = outputs.len(),
                        "compacted parquet files"
                    );
                }
                Ok(Err(e)) => {
                    error!(sink = %self.name, error = %e, "compaction failed");
                    return;
                }
                Err(e) => {
                    error!(sink = %self.name, error = %e, "compaction task panicked");
                    return;
                }
            }
        }
    }

    /// The next batch of files to merge: registered files under the target
    /// size, oldest-named first, cut off once they amount to one output file.
    /// `None` when there's nothing worth doing (fewer than two small files, or
    /// not enough bytes for a full output yet — merging earlier would just
    /// rewrite the same rows again on the next flush).
    fn next_batch(&self) -> Option<Vec<PathBuf>> {
        // The catalog is the source of truth: a file it doesn't hold may still
        // be mid-registration, and a missing table means nothing is queryable
        // (so compaction can wait too).
        let table = self.catalog.parquet_table(&self.name)?;
        let mut seen = HashSet::new();
        let mut small: Vec<(PathBuf, u64)> = Vec::new();
        for rg in table.parquet.row_groups() {
            let Some(path) = rg.source.local_path() else {
                continue;
            };
            if path.parent() != Some(self.dir.as_path()) || !seen.insert(path.to_path_buf()) {
                continue;
            }
            // A vanished file (already compacted away by a previous swap this
            // binding predates) is simply not a candidate.
            let Ok(meta) = std::fs::metadata(path) else {
                continue;
            };
            if meta.len() < self.target_bytes {
                small.push((path.to_path_buf(), meta.len()));
            }
        }
        if small.len() < 2 {
            return None;
        }
        // Oldest first (sink file names embed a timestamp + sequence), stopping
        // once the batch fills one output.
        small.sort();
        let mut total = 0u64;
        let mut batch = Vec::new();
        for (path, size) in small {
            total += size;
            batch.push(path);
            if total >= self.target_bytes {
                return Some(batch);
            }
        }
        debug!(
            sink = %self.name,
            files = batch.len(),
            bytes = total,
            "small files below compaction threshold; waiting for more"
        );
        None
    }
}

/// One merge, run on a blocking thread (it drives dataflows, which block the
/// driving thread while the work itself runs on the dispatch workers).
struct CompactJob {
    name: String,
    dir: PathBuf,
    dispatcher: DataFlowDispatcher,
    catalog: Arc<ParquetCatalog>,
    seq: u64,
}

impl CompactJob {
    /// Decode `inputs` (scan dataflow), re-encode them as one stream of
    /// target-sized row groups (write pipeline), and commit: rename the merged
    /// file(s) in, swap them for the inputs in the catalog, unlink the inputs.
    fn compact(self, inputs: Vec<PathBuf>) -> Result<Vec<PathBuf>, String> {
        let table = Arc::new(
            ParquetTable::from_files(&self.dispatcher, &inputs)
                .map_err(|e| format!("reading input footers: {e}"))?,
        );
        let columns = table.schema().fields().len();
        let batches = table_input(&self.dispatcher, &table, Projection::all(columns), false)
            .collect()
            .map_err(|e| format!("decoding inputs: {e}"))?;
        // Drop the inputs' open fds before the unlink below, not at scope end.
        drop(table);

        // Re-encode on the worker pool; stage every output in `.inflight`
        // first, so a failure mid-merge leaves the directory untouched.
        let inflight_dir = self.dir.join(INFLIGHT_DIR);
        std::fs::create_dir_all(&inflight_dir)
            .map_err(|e| format!("creating {}: {e}", inflight_dir.display()))?;
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let mut staged: Vec<(PathBuf, PathBuf)> = Vec::new();
        for (idx, bytes) in parquet_writing::run(
            &self.dispatcher,
            batches,
            ROW_GROUP_ROWS,
            ROW_GROUPS_PER_FILE,
        )
        .enumerate()
        {
            let bytes = match bytes {
                Ok(bytes) => bytes,
                Err(e) => {
                    remove_all(staged.iter().map(|(s, _)| s));
                    return Err(format!("re-encoding: {e}"));
                }
            };
            let file_name = format!(
                "{}-compacted-{}-{:06}-{:03}.parquet",
                self.name, millis, self.seq, idx
            );
            let inflight = inflight_dir.join(&file_name);
            if let Err(e) = std::fs::write(&inflight, &bytes) {
                remove_all(staged.iter().map(|(s, _)| s));
                return Err(format!("write {}: {e}", inflight.display()));
            }
            staged.push((inflight, self.dir.join(&file_name)));
        }

        // Commit. Renames first; if the catalog swap then fails, roll the
        // merged files back out so a restart can't double-read the rows.
        for (inflight, dest) in &staged {
            if let Err(e) = std::fs::rename(inflight, dest) {
                remove_all(staged.iter().flat_map(|(s, d)| [s, d]));
                return Err(format!("rename into {}: {e}", dest.display()));
            }
        }
        let outputs: Vec<PathBuf> = staged.into_iter().map(|(_, dest)| dest).collect();
        if let Err(e) =
            self.catalog
                .replace_data_files(&self.dispatcher, &self.name, &inputs, &outputs)
        {
            remove_all(outputs.iter());
            return Err(format!("catalog swap: {e}"));
        }
        for input in &inputs {
            if let Err(e) = std::fs::remove_file(input) {
                // The rows now live (twice) on disk until the next restart
                // reconciles the directory; queries are unaffected — the
                // catalog already dropped this file.
                error!(
                    sink = %self.name, file = %input.display(), error = %e,
                    "failed unlinking compacted input"
                );
            }
        }
        Ok(outputs)
    }
}

/// Best-effort cleanup of partially-staged outputs.
fn remove_all<'a>(paths: impl Iterator<Item = &'a PathBuf>) {
    for path in paths {
        let _ = std::fs::remove_file(path);
    }
}
