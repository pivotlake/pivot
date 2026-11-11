//! How long a managed table keeps what its current snapshot no longer
//! references, and which snapshots and previous metadata files have outlived
//! that.
//!
//! A snapshot stays readable while a reader that loaded it may still be
//! reading: at least until its successor is older than the table's
//! retention window. Expiring it deletes the files only it used, so the
//! window is the only rule: no count of newer snapshots expires one that a
//! reader inside the window may hold. A table whose `gc.enabled` property
//! is `false` expires nothing.
//!
//! A table's history is taken to be linear: its current snapshot and that
//! snapshot's ancestors, oldest first, each the parent of the next, which
//! is its successor. A linear history gives up snapshots from its oldest
//! only: a snapshot expires when its successor is older than the window
//! and every snapshot before it has expired, so what expires is every
//! snapshot before the first one still inside the window. In a history
//! whose snapshots are each dated after their parent, as Pivot dates every
//! one it commits, the second condition follows from the first: it decides
//! only where the clocks of a table's writers disagreed. A table whose
//! history is not linear, because it has a branch or a tag besides `main`
//! or a snapshot that is not an ancestor of the current one, is refused,
//! and nothing of it expires, its previous metadata files included: what
//! its expired snapshots retired could not be told from what the rest of
//! it still reads.
//!
//! A previous metadata file follows the same rule: the metadata log names
//! it until the file that replaced it is older than the window, and no
//! count of newer files drops it sooner. A writer that loaded a version
//! inside the window therefore finds the file after it still in place, which
//! is how it learns that it lost.

use std::time::Duration;

use iceberg::spec::{TableMetadata, TableProperties};

use crate::maintenance::linear_history::list_linear_history;
use crate::table::Table;
use crate::{Error, Result};

/// The table property holding the retention window in milliseconds, under
/// the name Iceberg's own expiry reads.
pub const SNAPSHOT_RETENTION_PROPERTY: &str = "history.expire.max-snapshot-age-ms";
/// The retention window of a table that sets none: how long a reader may
/// keep reading a snapshot after a later commit retired its files.
pub const DEFAULT_SNAPSHOT_RETENTION: Duration = Duration::from_secs(4 * 60 * 60);

impl Table {
    /// The table's retention window: its own property, or the default.
    pub fn snapshot_retention(&self) -> Result<Duration> {
        snapshot_retention(self.metadata())
    }

    /// The snapshots that have expired as of `now_ms`, oldest first: the
    /// oldest snapshots of the table's linear history, each one's successor
    /// older than the retention window. Successors are taken in the
    /// history's order, whatever clocks dated them, and the history gives
    /// up snapshots from its oldest only: the first snapshot still inside
    /// the window keeps every snapshot after it. What expires is therefore
    /// the nearest ancestors of the oldest snapshot kept, none of them kept
    /// itself, which is what
    /// [`collect_expired_objects`](Self::collect_expired_objects) finds the
    /// retired files of. The current snapshot never expires, and nothing
    /// expires in a table that turned `gc.enabled` off.
    ///
    /// A table whose history is not linear is refused
    /// ([`Error::NonLinearHistory`]): it has a branch or a tag besides
    /// `main`, or a snapshot that is not an ancestor of the current one.
    pub fn expired_snapshot_ids(&self, now_ms: i64) -> Result<Vec<i64>> {
        let metadata = self.metadata();
        if !self.is_gc_enabled()? {
            return Ok(Vec::new());
        }
        let history = list_linear_history(&self.name().to_string(), metadata)?;
        let retention_ms =
            i64::try_from(snapshot_retention(metadata)?.as_millis()).unwrap_or(i64::MAX);
        let mut expired = Vec::new();
        for (snapshot, successor) in history.iter().zip(history.iter().skip(1)) {
            let past_window = successor.timestamp_ms().saturating_add(retention_ms) <= now_ms;
            if !past_window {
                break;
            }
            expired.push(snapshot.snapshot_id());
        }
        Ok(expired)
    }

    /// The previous metadata files that have expired as of `now_ms`, as the
    /// metadata log names them. An entry expires when its successor, the
    /// file of the next entry or, for the newest entry, this version's own
    /// metadata file, was written longer ago than the retention window.
    /// Successors are taken in the log's order, whatever clocks dated them,
    /// and the log gives up entries from its oldest only: the first entry
    /// still inside the window keeps every entry after it. Nothing expires
    /// in a table that turned `gc.enabled` off.
    ///
    /// A table whose history is not linear is refused as its snapshots
    /// are ([`expired_snapshot_ids`](Self::expired_snapshot_ids)): nothing
    /// of such a table is deleted, so its log gives up no file either.
    pub fn expired_metadata_files(&self, now_ms: i64) -> Result<Vec<String>> {
        let metadata = self.metadata();
        if !self.is_gc_enabled()? {
            return Ok(Vec::new());
        }
        list_linear_history(&self.name().to_string(), metadata)?;
        let retention_ms =
            i64::try_from(snapshot_retention(metadata)?.as_millis()).unwrap_or(i64::MAX);
        let log = metadata.metadata_log();
        let successor_timestamps_ms = log
            .iter()
            .skip(1)
            .map(|entry| entry.timestamp_ms)
            .chain(std::iter::once(metadata.last_updated_ms()));
        let mut expired = Vec::new();
        for (entry, successor_timestamp_ms) in log.iter().zip(successor_timestamps_ms) {
            let past_window = successor_timestamp_ms.saturating_add(retention_ms) <= now_ms;
            if !past_window {
                break;
            }
            expired.push(entry.metadata_file.clone());
        }
        Ok(expired)
    }

    /// Whether the table lets its snapshots expire and its unused files be
    /// deleted: its `gc.enabled` property, `true` when unset.
    pub fn is_gc_enabled(&self) -> Result<bool> {
        match self
            .metadata()
            .properties()
            .get(TableProperties::PROPERTY_GC_ENABLED)
        {
            None => Ok(TableProperties::PROPERTY_GC_ENABLED_DEFAULT),
            Some(raw) => raw.parse::<bool>().map_err(|_| Error::MalformedProperty {
                property: TableProperties::PROPERTY_GC_ENABLED.to_string(),
                message: format!("`{raw}` is neither `true` nor `false`"),
            }),
        }
    }
}

/// The retention window of the table `metadata` describes: its own property,
/// or the default. Asked of a metadata document alone by a catalog judging
/// the age of a commit's base, which holds no [`Table`].
pub fn snapshot_retention(metadata: &TableMetadata) -> Result<Duration> {
    match metadata.properties().get(SNAPSHOT_RETENTION_PROPERTY) {
        None => Ok(DEFAULT_SNAPSHOT_RETENTION),
        Some(raw) => {
            raw.parse::<u64>()
                .map(Duration::from_millis)
                .map_err(|_| Error::MalformedProperty {
                    property: SNAPSHOT_RETENTION_PROPERTY.to_string(),
                    message: format!("`{raw}` is not a number of milliseconds"),
                })
        }
    }
}
