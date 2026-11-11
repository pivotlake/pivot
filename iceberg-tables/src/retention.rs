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
pub(crate) fn snapshot_retention(metadata: &TableMetadata) -> Result<Duration> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::build_table_at;
    use iceberg::spec::{
        FormatVersion, MAIN_BRANCH, NestedField, Operation, PrimitiveType, Schema, Snapshot,
        SnapshotReference, SnapshotRetention, SortOrder, Summary, TableMetadataBuilder, Type,
        UnboundPartitionSpec,
    };
    use std::collections::HashMap;

    /// The metadata of a table with one snapshot per entry of
    /// `timestamps_ms`, in order.
    fn build_metadata_with_snapshots(
        timestamps_ms: &[i64],
        properties: HashMap<String, String>,
    ) -> TableMetadata {
        let schema = Schema::builder()
            .with_fields(vec![
                NestedField::optional(1, "v", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap();
        let mut builder = TableMetadataBuilder::new(
            schema,
            UnboundPartitionSpec::builder().build(),
            SortOrder::unsorted_order(),
            "file:///t".to_string(),
            FormatVersion::V3,
            properties,
        )
        .unwrap();
        let mut parent = None;
        for (index, timestamp_ms) in timestamps_ms.iter().enumerate() {
            let snapshot_id = index as i64 + 1;
            let snapshot = Snapshot::builder()
                .with_snapshot_id(snapshot_id)
                .with_parent_snapshot_id(parent)
                .with_sequence_number(snapshot_id)
                .with_timestamp_ms(*timestamp_ms)
                .with_manifest_list(format!("file:///t/metadata/snap-{snapshot_id}.avro"))
                .with_summary(Summary {
                    operation: Operation::Append,
                    additional_properties: HashMap::new(),
                })
                .with_schema_id(0)
                .with_row_range(0, 0)
                .build();
            builder = builder.set_branch_snapshot(snapshot, MAIN_BRANCH).unwrap();
            parent = Some(snapshot_id);
        }
        builder.build().unwrap().metadata
    }

    /// The metadata of a table whose metadata file `v{n}` was written at the
    /// `n`th entry of `timestamps_ms`, the last of them its current one and
    /// the rest named by its metadata log.
    fn build_metadata_with_metadata_files(
        timestamps_ms: &[i64],
        properties: HashMap<String, String>,
    ) -> TableMetadata {
        let mut metadata = build_metadata_with_snapshots(&timestamps_ms[..1], properties);
        for (index, timestamp_ms) in timestamps_ms.iter().enumerate().skip(1) {
            let snapshot_id = index as i64 + 1;
            let snapshot = Snapshot::builder()
                .with_snapshot_id(snapshot_id)
                .with_parent_snapshot_id(Some(snapshot_id - 1))
                .with_sequence_number(snapshot_id)
                .with_timestamp_ms(*timestamp_ms)
                .with_manifest_list(format!("file:///t/metadata/snap-{snapshot_id}.avro"))
                .with_summary(Summary {
                    operation: Operation::Append,
                    additional_properties: HashMap::new(),
                })
                .with_schema_id(0)
                .with_row_range(0, 0)
                .build();
            metadata = metadata
                .into_builder(Some(format!("v{index}")))
                .set_branch_snapshot(snapshot, MAIN_BRANCH)
                .unwrap()
                .retain_metadata_log_except([])
                .build()
                .unwrap()
                .metadata;
        }
        metadata
    }

    const HOUR_MS: i64 = 60 * 60 * 1000;
    const SECOND_MS: i64 = 1000;

    #[test]
    fn a_snapshot_expires_once_its_successor_is_older_than_the_window() {
        let table = build_table_at(build_metadata_with_snapshots(
            &[0, HOUR_MS, 2 * HOUR_MS],
            HashMap::new(),
        ));

        // The second snapshot superseded the first one hour in; the window is
        // four hours, so the first expires five hours in, the second never
        // (its successor is current, superseded by nothing).
        assert_eq!(
            table.expired_snapshot_ids(4 * HOUR_MS).unwrap(),
            Vec::<i64>::new()
        );
        assert_eq!(table.expired_snapshot_ids(5 * HOUR_MS).unwrap(), vec![1]);
        assert_eq!(
            table.expired_snapshot_ids(100 * HOUR_MS).unwrap(),
            vec![1, 2]
        );
    }

    #[test]
    fn the_window_is_read_from_the_tables_property() {
        let table = build_table_at(build_metadata_with_snapshots(
            &[0, HOUR_MS],
            HashMap::from([(SNAPSHOT_RETENTION_PROPERTY.to_string(), "1000".to_string())]),
        ));

        assert_eq!(
            table.snapshot_retention().unwrap(),
            Duration::from_millis(1000)
        );
        assert_eq!(table.expired_snapshot_ids(HOUR_MS + 1000).unwrap(), vec![1]);
    }

    #[test]
    fn a_malformed_window_is_an_error() {
        let table = build_table_at(build_metadata_with_snapshots(
            &[0],
            HashMap::from([(SNAPSHOT_RETENTION_PROPERTY.to_string(), "soon".to_string())]),
        ));

        let error = table.snapshot_retention().unwrap_err();

        assert!(matches!(error, Error::MalformedProperty { .. }), "{error}");
    }

    #[test]
    fn a_snapshot_a_tag_points_at_does_not_expire() {
        let metadata = build_metadata_with_snapshots(&[0, HOUR_MS, 2 * HOUR_MS], HashMap::new())
            .into_builder(None)
            .set_ref(
                "release",
                SnapshotReference::new(
                    1,
                    SnapshotRetention::Tag {
                        max_ref_age_ms: None,
                    },
                ),
            )
            .unwrap()
            .build()
            .unwrap()
            .metadata;
        let table = build_table_at(metadata);

        let expired = table.expired_snapshot_ids(100 * HOUR_MS).unwrap();

        assert_eq!(expired, vec![2]);
    }

    #[test]
    fn a_table_with_gc_disabled_expires_nothing() {
        let table = build_table_at(build_metadata_with_snapshots(
            &[0, HOUR_MS, 2 * HOUR_MS],
            HashMap::from([("gc.enabled".to_string(), "false".to_string())]),
        ));

        let expired = table.expired_snapshot_ids(100 * HOUR_MS).unwrap();

        assert!(expired.is_empty());
    }

    #[test]
    fn no_number_of_newer_snapshots_expires_one_inside_the_window() {
        let timestamps: Vec<i64> = (0..5000).collect();
        let table = build_table_at(build_metadata_with_snapshots(&timestamps, HashMap::new()));

        let expired = table.expired_snapshot_ids(timestamps.len() as i64).unwrap();

        assert!(expired.is_empty());
    }

    #[test]
    fn a_metadata_file_expires_once_its_successor_is_older_than_the_window() {
        let table = build_table_at(build_metadata_with_metadata_files(
            &[0, HOUR_MS, 2 * HOUR_MS, 3 * HOUR_MS],
            HashMap::new(),
        ));

        // `v2` replaced `v1` one hour in and the window is four hours, so
        // `v1` expires five hours in; `v4` is current and never expires.
        assert_eq!(
            table.expired_metadata_files(5 * HOUR_MS - 1).unwrap(),
            Vec::<String>::new()
        );
        assert_eq!(table.expired_metadata_files(5 * HOUR_MS).unwrap(), ["v1"]);
        assert_eq!(
            table.expired_metadata_files(100 * HOUR_MS).unwrap(),
            ["v1", "v2", "v3"]
        );
    }

    #[test]
    fn a_metadata_file_inside_the_window_keeps_every_file_after_it() {
        // The writer of `v3` dated it thirty seconds before `v2`.
        let table = build_table_at(build_metadata_with_metadata_files(
            &[0, HOUR_MS, HOUR_MS - 30 * SECOND_MS, 2 * HOUR_MS],
            HashMap::new(),
        ));

        let expired = table
            .expired_metadata_files(5 * HOUR_MS - 10 * SECOND_MS)
            .unwrap();

        // `v2`'s successor is past the window by its own date, but `v1`'s is
        // not, and the log gives up its oldest entry first.
        assert!(expired.is_empty(), "{expired:?}");
    }

    #[test]
    fn no_number_of_newer_metadata_files_expires_one_inside_the_window() {
        let timestamps: Vec<i64> = (0..500).collect();
        let table = build_table_at(build_metadata_with_metadata_files(
            &timestamps,
            HashMap::new(),
        ));

        let expired = table
            .expired_metadata_files(timestamps.len() as i64)
            .unwrap();

        assert!(expired.is_empty());
        assert_eq!(table.metadata().metadata_log().len(), 499);
    }

    #[test]
    fn a_table_with_gc_disabled_expires_no_metadata_file() {
        let table = build_table_at(build_metadata_with_metadata_files(
            &[0, HOUR_MS, 2 * HOUR_MS],
            HashMap::from([("gc.enabled".to_string(), "false".to_string())]),
        ));

        let expired = table.expired_metadata_files(100 * HOUR_MS).unwrap();

        assert!(expired.is_empty());
    }
}
