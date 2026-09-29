//! Physical cleanup of a table's unreferenced Parquet files and the log files a
//! checkpoint has superseded.
//!
//! Compaction (and, in time, DELETE) retires a file by writing a Delta `Remove`
//! action for it, but leaves the object in place: a query that loaded the prior
//! version is still reading it. A write can also leave a file behind with no log
//! record at all -- an upload a writer crashed before committing. Either way the
//! table's storage grows without bound, and the [`Vacuumer`] is the back half:
//! it deletes every object the current table version no longer references, once
//! that object has been unreferenced for longer than the table's
//! `deletedFileRetentionDuration`. A retired file is dated by the `Remove`
//! tombstone its commit wrote, which the table carries with its version, so the
//! window starts when the file stopped being referenced, however long it was
//! live before that; a file the log never adopted has no tombstone, so its
//! storage mtime dates it instead. In the same sweep it also deletes every log
//! file past the log-retention window that sits below the oldest checkpoint the
//! log still needs (writing a checkpoint stays on the commit path, in
//! [`crate::log`], not here), so the `_delta_log` does not grow unbounded
//! either. A dropped table follows the
//! same shape one level up: `DROP TABLE` removes only the catalog entries and
//! leaves a manifest tombstone, and the sweep deletes the whole table's storage
//! once the tombstone is older than the table's retention.
//!
//! Like the compacter it is **location-agnostic** and **deployment-agnostic**:
//! it holds nothing but a datastore handle, reads each table's directory, and
//! deletes through the table's store -- so a table under an `s3://` root vacuums
//! through the exact same code as a local one. It is self-managed by the
//! [`PivotlakeDatastore`] alongside the compacter, and unlike the merge it touches no
//! dispatch workers: it only lists the directory and deletes objects.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::time::MissedTickBehavior;
use tracing::{info, warn};

use planner::catalog::SchemaQualifiedTableName;

use crate::{CatalogTable, PivotlakeDatastore, Result};
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
    datastore: Arc<PivotlakeDatastore>,
}

impl Vacuumer {
    pub fn new(poll_interval: Duration, datastore: Arc<PivotlakeDatastore>) -> Self {
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
            match tokio::task::spawn_blocking(move || vacuumer.vacuum_all(now_unix_ms())).await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => warn!(error = %error, "vacuum sweep failed"),
                Err(error) => warn!(error = %error, "vacuum sweep panicked"),
            }
        }
    }

    /// One poll round over every table the datastore knows, evaluated against
    /// `now_ms` (threaded in so tests are deterministic -- there is no mockable
    /// clock). Public so tests and a standalone vacuumer binary can drive one
    /// round without the loop. Continues after individual failures and returns
    /// the first error, so a manual sweep can report incomplete cleanup.
    pub fn vacuum_all(&self, now_ms: u64) -> Result<()> {
        let mut first_error = None;
        for (name, table) in self.datastore.tables() {
            if let Err(error) = self.vacuum_table(&name, table, now_ms) {
                warn!(table = %name, error = %error, "vacuum: table cleanup failed");
                first_error.get_or_insert(error);
            }
        }
        // Dropped tables are no longer in the live set above; their storage is
        // reclaimed off the manifest's tombstones once each retention window
        // has passed.
        match self.datastore.reclaim_dropped_tables(now_ms) {
            Ok(reclaimed) if reclaimed > 0 => {
                info!(tables = reclaimed, "reclaimed dropped tables' storage");
            }
            Ok(_) => {}
            Err(error) => {
                warn!(error = %error, "vacuum: reclaiming dropped tables failed");
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// One table's round: reload it to its latest log version (so a vacuumer on
    /// a shared remote store sees commits another process wrote), then delete
    /// every data file the current version does not reference once it has been
    /// unreferenced for the deletion window (the table's
    /// `deletedFileRetentionDuration`): a retired file's window runs from the
    /// tombstone the table holds for it, an orphan's from its storage mtime.
    /// Finally delete the superseded commit JSONs past the log-retention window.
    /// Failed deletions are retried by the next sweep.
    fn vacuum_table(
        &self,
        name: &SchemaQualifiedTableName,
        mut table: CatalogTable,
        now_ms: u64,
    ) -> Result<()> {
        table.refresh()?;

        let cutoff = now_ms.saturating_sub(table.deleted_file_retention().as_millis() as u64);

        // The current version's live files. Everything else in the directory is
        // unreferenced -- a file the log retired, or an orphan upload no commit
        // ever adopted -- and therefore a deletion candidate.
        let live: HashSet<ObjectPath> = table.file_refs().into_iter().map(|f| f.path).collect();

        let files = table.list_data_files()?;
        let mut deleted = 0u64;
        let mut first_error = None;
        for (path, modified_ms) in files {
            if live.contains(&path) {
                continue;
            }
            // A retired file is dated by the commit that retired it, which keeps
            // it whatever the storage mtime says: it may have been live for days
            // before a merge or delete retired it, and a reader on the prior
            // snapshot, or a recovery from a commit that should not have retired
            // it, still needs the bytes. A file with no tombstone, an orphan
            // upload or one retired before the window, is dated the only way
            // left, by when its bytes landed in storage.
            let unreferenced_since_ms = table
                .tombstones()
                .get(&path)
                .copied()
                .unwrap_or(modified_ms);
            // Keep a file still inside the window: an upload a writer has not
            // committed yet, or a retired file a reader on the prior snapshot
            // still needs.
            if unreferenced_since_ms > cutoff {
                continue;
            }
            if let Err(e) = table.delete_data_file(&path) {
                warn!(table = %name, file = %path, error = %e, "vacuum: deleting file failed");
                first_error.get_or_insert(e);
                continue;
            }
            deleted += 1;
        }
        if deleted > 0 {
            info!(table = %name, files = deleted, "vacuumed files");
        }

        // Reclaim log storage in the same sweep: delete the commits, checkpoints
        // and checksums a later checkpoint has made redundant, once they are
        // past the table's log-retention window. Writing a checkpoint stays on
        // the commit path; this is only the deletion of what one supersedes.
        match table.cleanup_log(now_ms) {
            Ok(cleaned) if cleaned > 0 => {
                info!(table = %name, log_files = cleaned, "cleaned up superseded log files");
            }
            Ok(_) => {}
            Err(error) => {
                warn!(table = %name, error = %error, "vacuum: log cleanup failed");
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}
