//! How long a managed table keeps what its current snapshot no longer
//! references, and which snapshots and previous metadata files have outlived
//! that.
//!
//! A snapshot stays readable while a reader that loaded it may still be
//! reading: until its successor is older than the table's retention window.
//! Expiring it deletes the files only it used, so the window is the only
//! rule: no count of newer snapshots expires one that a reader inside the
//! window may hold. A snapshot a branch or a tag points at is kept for as
//! long as the reference is, and a table whose `gc.enabled` property is
//! `false` expires nothing.
//!
//! A previous metadata file follows the same rule: the metadata log names
//! it until the file that replaced it is older than the window, and no
//! count of newer files drops it sooner. A writer that loaded a version
//! inside the window therefore finds the file after it still in place, which
//! is how it learns that it lost.

use std::collections::HashSet;
use std::time::Duration;

use iceberg::spec::{TableMetadata, TableProperties};

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

    /// The snapshots that have expired as of `now_ms`: every non-current one
    /// whose successor is older than the retention window. The current
    /// snapshot never expires, and neither does one a branch or a tag points
    /// at: the spec crate's expiry refuses to drop the head of a reference
    /// it keeps. Nothing expires in a table that turned `gc.enabled` off.
    pub fn expired_snapshot_ids(&self, now_ms: i64) -> Result<Vec<i64>> {
        let metadata = self.metadata();
        if !self.is_gc_enabled()? {
            return Ok(Vec::new());
        }
        let referenced = referenced_snapshot_ids(metadata);
        let retention_ms =
            i64::try_from(snapshot_retention(metadata)?.as_millis()).unwrap_or(i64::MAX);
        let mut ordered: Vec<_> = metadata.snapshots().collect();
        ordered.sort_by_key(|snapshot| (snapshot.sequence_number(), snapshot.timestamp_ms()));
        let current = metadata.current_snapshot_id();
        let mut expired = Vec::new();
        for (position, snapshot) in ordered.iter().enumerate() {
            if Some(snapshot.snapshot_id()) == current {
                continue;
            }
            if referenced.contains(&snapshot.snapshot_id()) {
                continue;
            }
            let Some(successor) = ordered.get(position + 1) else {
                continue;
            };
            let past_window = successor.timestamp_ms().saturating_add(retention_ms) <= now_ms;
            if past_window {
                expired.push(snapshot.snapshot_id());
            }
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
    pub fn expired_metadata_files(&self, now_ms: i64) -> Result<Vec<String>> {
        let metadata = self.metadata();
        if !self.is_gc_enabled()? {
            return Ok(Vec::new());
        }
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

/// The snapshots the branches and tags of the table `metadata` describes
/// point at.
fn referenced_snapshot_ids(metadata: &TableMetadata) -> HashSet<i64> {
    metadata
        .refs()
        .values()
        .map(|reference| reference.snapshot_id)
        .collect()
}
