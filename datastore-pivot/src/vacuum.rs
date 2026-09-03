//! Physical cleanup of a table's unreferenced Parquet files and superseded log
//! commits.
//!
//! Compaction (and, in time, DELETE) retires a file by writing a Delta `Remove`
//! action for it, but leaves the object in place: a query that loaded the prior
//! version is still reading it. A write can also leave a file behind with no log
//! record at all -- an upload a writer crashed before committing. Either way the
//! table's storage grows without bound, and the [`Vacuumer`] is the back half:
//! it deletes every object the current table version no longer references, once
//! that object's storage mtime is older than the table's
//! `deletedFileRetentionDuration`. In the same sweep it also deletes the commit
//! JSONs a checkpoint has folded in that are past the log-retention window (the
//! checkpoints are written inline on the commit path, in [`crate::log`], not
//! here), so the `_delta_log` does not grow unbounded either. A dropped table
//! follows the same shape one level up: `DROP TABLE` removes only the catalog
//! entries and leaves a manifest tombstone, and the sweep deletes the whole
//! table's storage once the tombstone is older than the table's retention.
//!
//! Like the compacter it is **location-agnostic** and **deployment-agnostic**:
//! it holds nothing but a datastore handle, reads each table's directory, and
//! deletes through the table's store -- so a table under an `s3://` root vacuums
//! through the exact same code as a local one. It is self-managed by the
//! [`PivotDatastore`] alongside the compacter, and unlike the merge it touches no
//! dispatch workers: it only lists the directory and deletes objects.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::time::MissedTickBehavior;
use tracing::{info, warn};

use planner::catalog::SchemaQualifiedTableName;

use crate::{CatalogTable, PivotDatastore};
use object_storage::ObjectPath;

/// Default cadence for re-scanning the tables for newly-expired files and
/// superseded commits. Both expire on the retention timescale (hours, by the
/// default `deletedFileRetentionDuration`), so hourly polling is ample.
pub const DEFAULT_VACUUM_POLL: Duration = Duration::from_secs(60 * 60);

/// Tuning for a datastore's self-managed vacuum loop. The deletion window is not
/// configured here: each table's own `delta.deletedFileRetentionDuration` (4
/// hours by default) governs how long an unreferenced file is kept.
#[derive(Clone)]
pub struct VacuumConfig {
    /// How often to re-scan the tables.
    pub poll_interval: Duration,
}

impl Default for VacuumConfig {
    fn default() -> Self {
        Self {
            poll_interval: DEFAULT_VACUUM_POLL,
        }
    }
}

pub(crate) fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Deletes every datastore table's unreferenced Parquet objects and superseded
/// commit JSONs, entirely off the tables' directories and logs. Holds nothing
/// but a datastore handle, so it is a deployment detail: the datastore bundles
/// one when maintenance enables it, and a dedicated process could run another
/// over the same database root.
pub struct Vacuumer {
    /// How often to re-scan the tables for files to delete.
    poll_interval: Duration,
    datastore: Arc<PivotDatastore>,
}

impl Vacuumer {
    pub fn new(poll_interval: Duration, datastore: Arc<PivotDatastore>) -> Self {
        Self {
            poll_interval,
            datastore,
        }
    }

    /// The vacuum loop: every `poll_interval`, sweep the datastore's tables and
    /// delete whatever has expired. Runs until the task is aborted on shutdown.
    /// The first tick fires immediately, so a backlog from a previous run is
    /// handled at startup.
    pub async fn run(self: Arc<Self>) {
        let mut tick = tokio::time::interval(self.poll_interval);
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            tick.tick().await;
            let vacuumer = self.clone();
            // Store list/get/delete and the log reads are synchronous (blocking
            // HTTP for S3), so the sweep runs off the reactor.
            if let Err(e) =
                tokio::task::spawn_blocking(move || vacuumer.vacuum_all(now_unix_ms())).await
            {
                warn!(error = %e, "vacuum sweep panicked");
            }
        }
    }

    /// One poll round over every table the datastore knows, evaluated against
    /// `now_ms` (threaded in so tests are deterministic -- there is no mockable
    /// clock). Public so tests and a standalone vacuumer binary can drive one
    /// round without the loop.
    pub fn vacuum_all(&self, now_ms: u64) {
        for (name, table) in self.datastore.tables() {
            self.vacuum_table(&name, table, now_ms);
        }
        // Dropped tables are no longer in the live set above; their storage is
        // reclaimed off the manifest's tombstones once each retention window
        // has passed.
        match self.datastore.reclaim_dropped_tables(now_ms) {
            Ok(reclaimed) if reclaimed > 0 => {
                info!(tables = reclaimed, "reclaimed dropped tables' storage");
            }
            Ok(_) => {}
            Err(e) => warn!(error = %e, "vacuum: reclaiming dropped tables failed"),
        }
    }

    /// One table's round: reload it to its latest log version (so a vacuumer on
    /// a shared remote store sees commits another process wrote), then delete
    /// every data file the current version does not reference whose storage
    /// mtime is older than the deletion window (the table's
    /// `deletedFileRetentionDuration`).
    /// Finally delete the superseded commit JSONs past the log-retention window.
    /// Errors are logged and end the round; the next poll retries.
    fn vacuum_table(&self, name: &SchemaQualifiedTableName, mut table: CatalogTable, now_ms: u64) {
        if let Err(e) = table.refresh() {
            warn!(table = %name, error = %e, "vacuum: table refresh failed");
            return;
        }

        let cutoff = now_ms.saturating_sub(table.deleted_file_retention().as_millis() as u64);

        // The current version's live files. Everything else in the directory is
        // unreferenced -- a file the log superseded, or an orphan upload no commit
        // ever adopted -- and therefore a deletion candidate.
        let live: HashSet<ObjectPath> = table.file_refs().into_iter().map(|f| f.path).collect();

        let mut deleted = 0u64;
        for file in table.list_data_files() {
            let (path, modified_ms) = match file {
                Ok(file) => file,
                Err(error) => {
                    warn!(table = %name, error = %error, "vacuum: listing data files failed");
                    return;
                }
            };
            // Keep a live file. Keep any unreferenced file still within the
            // window: it may be an upload a writer has not committed yet, or a
            // just-superseded file a reader on the prior snapshot still needs.
            if live.contains(&path) || modified_ms > cutoff {
                continue;
            }
            if let Err(e) = table.delete_data_file(&path) {
                warn!(table = %name, file = %path, error = %e, "vacuum: deleting file failed");
                continue;
            }
            deleted += 1;
        }
        if deleted > 0 {
            info!(table = %name, files = deleted, "vacuumed files");
        }

        // Reclaim log storage in the same sweep: delete commit JSONs a checkpoint
        // has folded in that are past the table's log-retention window. The
        // checkpoints themselves are written inline on the commit path; this is
        // only the deletion of what they supersede.
        match table.cleanup_log(now_ms) {
            Ok(cleaned) if cleaned > 0 => {
                info!(table = %name, log_files = cleaned, "cleaned up superseded commits");
            }
            Ok(_) => {}
            Err(e) => warn!(table = %name, error = %e, "vacuum: log cleanup failed"),
        }
    }
}
