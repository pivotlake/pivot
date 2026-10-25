//! Physical cleanup of what a table's current version no longer needs.
//!
//! Compaction (and, in time, DELETE) retires a file by committing a version
//! without it, but leaves the object in place: a query that loaded the prior
//! version is still reading it. A write can also leave a file behind with no
//! metadata record at all, an upload a writer crashed before committing.
//! Either way the table's storage grows without bound, and the [`Vacuumer`]
//! is the back half: it deletes every object the current table version no
//! longer references, once that object has been unreferenced for longer than
//! the table's deleted-file retention. How a retired file is dated, and which
//! metadata objects a version leaves behind, is the format's to say (see
//! [`FormatTable::sweep`](crate::format::FormatTable::sweep)): a Delta table
//! dates a retired file by its `Remove` tombstone and deletes the log files a
//! checkpoint superseded; an Iceberg table first expires the snapshots its
//! retention allows, then deletes every data file, manifest and metadata file
//! no retained snapshot reaches. A dropped table follows the same shape one
//! level up: `DROP TABLE` removes only the catalog entries and leaves a
//! manifest tombstone, and the sweep deletes the whole table's storage once
//! the tombstone is older than the table's retention.
//!
//! Like the compacter it is **location-agnostic** and **deployment-agnostic**:
//! it holds nothing but a datastore handle, reads each table's directory, and
//! deletes through the table's store, so a table under an `s3://` root vacuums
//! through the exact same code as a local one. It is self-managed by the
//! [`PivotlakeDatastore`] alongside the compacter, and unlike the merge it
//! touches no dispatch workers: it only lists the directory and deletes
//! objects.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::time::MissedTickBehavior;
use tracing::{info, warn};

use planner::catalog::SchemaQualifiedTableName;

use crate::{CatalogTable, PivotlakeDatastore, Result};

/// Default cadence for re-scanning the tables for newly-expired objects. They
/// expire on the retention timescale (hours, by the default deleted-file
/// retention), so hourly polling is ample.
pub const DEFAULT_VACUUM_POLL: Duration = Duration::from_secs(60 * 60);

/// Tuning for a datastore's self-managed vacuum loop. The deletion window is not
/// configured here: each table's own deleted-file retention (4 hours by
/// default) governs how long an unreferenced object is kept.
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

/// Deletes every datastore table's expired objects, entirely off the tables'
/// directories and metadata. Holds nothing but a datastore handle, so it is a
/// deployment detail: the datastore bundles one when maintenance enables it,
/// and a dedicated process could run another over the same database root.
pub struct Vacuumer {
    /// How often to re-scan the tables for objects to delete.
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
            // Store list/get/delete and the metadata reads are synchronous
            // (blocking HTTP for S3), so the sweep runs off the reactor.
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

    /// One table's round: reload it to its latest version (so a vacuumer on a
    /// shared remote store sees commits another process wrote), run the
    /// format's own metadata housekeeping under the commit lock and publish
    /// what it wrote, then delete every object the version no longer needs
    /// once it has been unneeded for the retention window. Failed deletions
    /// are retried by the next sweep.
    fn vacuum_table(
        &self,
        name: &SchemaQualifiedTableName,
        mut table: CatalogTable,
        now_ms: u64,
    ) -> Result<()> {
        table.refresh()?;
        {
            // Housekeeping is a commit like any other in-process write, so it
            // takes the table's commit lock; the format retries a lost race
            // inside.
            let commit_lock = table.commit_lock();
            let _committing = commit_lock.lock().unwrap();
            if table.maintain(now_ms)? {
                self.datastore.publish_table(table.clone());
            }
        }
        table.sweep(name, now_ms)
    }
}
