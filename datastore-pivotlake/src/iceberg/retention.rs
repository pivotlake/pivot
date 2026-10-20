//! How long a managed table keeps what its current snapshot no longer
//! references, and which snapshots have outlived that.
//!
//! A snapshot stays readable while a reader that loaded it may still be
//! reading: until its successor is older than the table's retention window.
//! Expiring it is what lets vacuum delete the files only it referenced. A
//! hard cap on retained snapshots bounds the metadata file a hot table
//! rewrites on every commit.

use std::time::Duration;

use iceberg::spec::TableMetadata;

use super::{Error, Result};

/// The table property holding the retention window in milliseconds, under
/// the name Iceberg's own expiry reads.
pub(crate) const SNAPSHOT_RETENTION_PROPERTY: &str = "history.expire.max-snapshot-age-ms";
/// The table property bounding how many previous metadata files the
/// metadata log lists, which is also how many vacuum keeps.
pub(crate) const PREVIOUS_VERSIONS_PROPERTY: &str = "write.metadata.previous-versions-max";
pub(crate) const DEFAULT_PREVIOUS_VERSIONS: usize = 100;
/// The retention window of a table that sets none: how long a reader may
/// keep reading a snapshot after a later commit retired its files.
pub(crate) const DEFAULT_SNAPSHOT_RETENTION: Duration = Duration::from_secs(4 * 60 * 60);
/// A non-current snapshot expires once this many newer ones exist, whatever
/// its age: each retained snapshot costs the metadata file about half a
/// kilobyte on every commit, and a table committing every second would carry
/// the whole window's worth.
pub(crate) const MAX_RETAINED_SNAPSHOTS: usize = 1000;

/// The table's retention window: its own property, or the default.
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

/// The snapshots that have expired as of `now_ms`: every non-current one
/// whose successor is older than the retention window, and every one with
/// [`MAX_RETAINED_SNAPSHOTS`] or more newer than it. The current snapshot
/// never expires.
pub(crate) fn expired_snapshot_ids(metadata: &TableMetadata, now_ms: i64) -> Result<Vec<i64>> {
    let retention_ms = i64::try_from(snapshot_retention(metadata)?.as_millis()).unwrap_or(i64::MAX);
    let mut ordered: Vec<_> = metadata.snapshots().collect();
    ordered.sort_by_key(|snapshot| (snapshot.sequence_number(), snapshot.timestamp_ms()));
    let current = metadata.current_snapshot_id();
    let retained_from = ordered.len().saturating_sub(MAX_RETAINED_SNAPSHOTS);
    let mut expired = Vec::new();
    for (position, snapshot) in ordered.iter().enumerate() {
        if Some(snapshot.snapshot_id()) == current {
            continue;
        }
        let Some(successor) = ordered.get(position + 1) else {
            continue;
        };
        let past_window = successor.timestamp_ms().saturating_add(retention_ms) <= now_ms;
        let past_cap = position < retained_from;
        if past_window || past_cap {
            expired.push(snapshot.snapshot_id());
        }
    }
    Ok(expired)
}

#[cfg(test)]
mod tests {
    use super::*;
    use iceberg::spec::{
        FormatVersion, MAIN_BRANCH, NestedField, Operation, PrimitiveType, Schema, Snapshot,
        SortOrder, Summary, TableMetadataBuilder, Type, UnboundPartitionSpec,
    };
    use std::collections::HashMap;

    /// A table with one snapshot per entry of `timestamps_ms`, in order.
    fn table_with_snapshots(
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

    const HOUR_MS: i64 = 60 * 60 * 1000;

    #[test]
    fn a_snapshot_expires_once_its_successor_is_older_than_the_window() {
        let metadata = table_with_snapshots(&[0, HOUR_MS, 2 * HOUR_MS], HashMap::new());

        // The second snapshot superseded the first one hour in; the window is
        // four hours, so the first expires five hours in, the second never
        // (its successor is current, superseded by nothing).
        assert_eq!(
            expired_snapshot_ids(&metadata, 4 * HOUR_MS).unwrap(),
            Vec::<i64>::new()
        );
        assert_eq!(
            expired_snapshot_ids(&metadata, 5 * HOUR_MS).unwrap(),
            vec![1]
        );
        assert_eq!(
            expired_snapshot_ids(&metadata, 100 * HOUR_MS).unwrap(),
            vec![1, 2]
        );
    }

    #[test]
    fn the_window_is_read_from_the_tables_property() {
        let metadata = table_with_snapshots(
            &[0, HOUR_MS],
            HashMap::from([(SNAPSHOT_RETENTION_PROPERTY.to_string(), "1000".to_string())]),
        );

        assert_eq!(
            snapshot_retention(&metadata).unwrap(),
            Duration::from_millis(1000)
        );
        assert_eq!(
            expired_snapshot_ids(&metadata, HOUR_MS + 1000).unwrap(),
            vec![1]
        );
    }

    #[test]
    fn a_malformed_window_is_an_error() {
        let metadata = table_with_snapshots(
            &[0],
            HashMap::from([(SNAPSHOT_RETENTION_PROPERTY.to_string(), "soon".to_string())]),
        );

        let error = snapshot_retention(&metadata).unwrap_err();

        assert!(matches!(error, Error::MalformedProperty { .. }), "{error}");
    }

    #[test]
    fn only_the_newest_snapshots_are_kept_past_the_cap() {
        let timestamps: Vec<i64> = (0..(MAX_RETAINED_SNAPSHOTS as i64 + 3)).collect();
        let metadata = table_with_snapshots(&timestamps, HashMap::new());

        let expired = expired_snapshot_ids(&metadata, timestamps.len() as i64).unwrap();

        assert_eq!(expired, vec![1, 2, 3]);
    }
}
