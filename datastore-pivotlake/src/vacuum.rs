//! Physical cleanup of what a table's current version no longer needs.
//!
//! Compaction (and, in time, DELETE) retires a file by committing a version
//! without it, but leaves the object in place: a query that loaded the prior
//! version is still reading it. A write can also leave a file behind with no
//! metadata record at all, an upload a writer crashed before committing.
//! Either way the table's storage grows without bound, and the [`Vacuumer`]
//! is the back half.
//!
//! One round ([`Vacuumer::vacuum_all`]) is what the datastore's timer runs
//! and what the SQL `VACUUM` command runs. What it deletes is the format's
//! to say: an Iceberg table expires the snapshots and the previous metadata
//! files its retention allows and deletes exactly those files and the
//! objects only those snapshots used, found from the table's own metadata
//! with no listing of its directory
//! ([`Table::expire_snapshots`](crate::table_format::Table::expire_snapshots)),
//! so a file no metadata ever named is not reclaimed; a Delta table lists
//! its directory, dates a retired file by its `Remove` tombstone and an
//! orphan by its storage mtime, deletes each past the table's deleted-file
//! retention and deletes the log files a checkpoint superseded
//! ([`Table::delete_expired_objects`](crate::table_format::Table::delete_expired_objects)).
//! A dropped table follows the same shape one level up: `DROP TABLE` removes
//! only the catalog entries and leaves a manifest tombstone, and the round
//! deletes the whole table's storage once the tombstone is older than the
//! table's retention.
//!
//! Like the compacter it is **location-agnostic** and **deployment-agnostic**:
//! it holds nothing but a datastore handle and deletes through the table's
//! store, so a table under an `s3://` root is cleaned up through the exact
//! same code as a local one. It is self-managed by the [`PivotlakeDatastore`]
//! alongside the compacter.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::time::MissedTickBehavior;
use tracing::{info, warn};

use planner::catalog::SchemaQualifiedTableName;

use crate::{CatalogTable, PivotlakeDatastore, Result};

/// Default cadence of the vacuum round. Objects expire on the retention
/// timescale (hours, by the default retention), so hourly polling is ample.
pub const DEFAULT_VACUUM_POLL: Duration = Duration::from_secs(60 * 60);

/// Tuning for a datastore's self-managed vacuum loop. The deletion window is
/// not configured here: each table's own retention (4 hours by default)
/// governs how long what it retired is kept.
#[derive(Clone)]
pub struct VacuumConfig {
    /// How often to run a round.
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

/// Deletes every datastore table's expired objects. Holds nothing but a
/// datastore handle, so it is a deployment detail: the datastore bundles one
/// when maintenance enables it, and a dedicated process could run another
/// over the same database root.
pub struct Vacuumer {
    /// How often the loop runs a round.
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

    /// The vacuum loop: every `poll_interval`, run one round over the
    /// datastore's tables. Runs until the task is aborted on shutdown. The
    /// first tick fires immediately, so a backlog from a previous run is
    /// handled at startup.
    pub async fn run(self: Arc<Self>) {
        let mut tick = tokio::time::interval(self.poll_interval);
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            tick.tick().await;
            let vacuumer = self.clone();
            // Store list/get/delete and the metadata reads are synchronous
            // (blocking HTTP for S3), so the round runs off the reactor.
            match tokio::task::spawn_blocking(move || vacuumer.vacuum_all(now_unix_ms())).await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => warn!(error = %error, "vacuum sweep failed"),
                Err(error) => warn!(error = %error, "vacuum sweep panicked"),
            }
        }
    }

    /// One round over every table the datastore knows, evaluated against
    /// `now_ms` (threaded in so tests are deterministic -- there is no
    /// mockable clock): each table's own deletions, then the storage of
    /// dropped tables past their retention. What the loop runs and what the
    /// SQL `VACUUM` command runs; public so tests and a standalone binary
    /// can drive one round without the loop. Continues after individual
    /// failures and returns the first error, so a manual round can report
    /// incomplete cleanup.
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

    /// One table's round: reload it to its latest version (so a vacuumer on a
    /// shared remote store sees commits another process wrote), expire its
    /// snapshots under the commit lock and publish the version that wrote,
    /// then, with the lock released, delete what the expired snapshots alone
    /// used and whatever else the format retires.
    fn vacuum_table(
        &self,
        name: &SchemaQualifiedTableName,
        mut table: CatalogTable,
        now_ms: u64,
    ) -> Result<()> {
        table.refresh()?;
        let delete_expired_snapshot_objects = {
            // An expiry is a commit like any other in-process write, so it
            // takes the table's commit lock; the format retries a lost race
            // inside.
            let commit_lock = table.commit_lock();
            let _committing = commit_lock.lock().unwrap();
            let deletion = table.expire_snapshots(now_ms)?;
            if deletion.is_some() {
                self.datastore.publish_table(table.clone());
            }
            deletion
        };
        if let Some(delete_expired_snapshot_objects) = delete_expired_snapshot_objects {
            delete_expired_snapshot_objects();
        }
        table.delete_expired_objects(name, now_ms)
    }
}
