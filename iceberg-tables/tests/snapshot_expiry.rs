//! Which snapshots and previous metadata files a table expires, and what an
//! expiry commits: only what has outlived the retention window, the oldest
//! snapshots of the table's linear history, nothing when garbage
//! collection is off, nothing of a table whose history is not linear (it
//! has a tag, a second branch or a snapshot that is no ancestor of the
//! current one), and never as a side effect of a commit of files.

mod support;

use std::collections::HashMap;
use std::time::Duration;

use iceberg::spec::{
    FormatVersion, MAIN_BRANCH, NestedField, Operation, PrimitiveType, Schema, Snapshot,
    SnapshotReference, SnapshotRetention, SortOrder, Summary, TableMetadata, TableMetadataBuilder,
    Type, UnboundPartitionSpec,
};
use iceberg_tables::commit::{ExpiryOutcome, SnapshotExpiry};
use iceberg_tables::maintenance::retention::{
    DEFAULT_SNAPSHOT_RETENTION, SNAPSHOT_RETENTION_PROPERTY,
};
use iceberg_tables::{Error, Table};
use support::{
    Fixture, HOUR_MS, append, avro_objects, create_table, entry, expire_at,
    is_non_linear_history_error, list_expired_objects, load, logged_versions, now_unix_ms, paths,
    publish_changed, replace, version, with_branch_on_current_snapshot,
    with_retention_of_one_millisecond, with_staged_snapshot, with_tag_on_current_snapshot,
};

const WINDOW_MS: i64 = DEFAULT_SNAPSHOT_RETENTION.as_millis() as i64;
const SECOND_MS: i64 = 1000;

#[test]
fn a_commit_drops_no_snapshot_however_far_past_the_window() {
    let (fixture, _created) = create_table(&[]);
    let table = with_retention_of_one_millisecond(&fixture);
    let table = append(&table, vec![entry("a", 1, 1, 1)]);
    let table = append(&table, vec![entry("b", 1, 2, 2)]);
    std::thread::sleep(Duration::from_millis(5));
    assert_eq!(table.expired_snapshot_ids(now_unix_ms()).unwrap().len(), 1);

    let table = append(&table, vec![entry("c", 1, 3, 3)]);

    assert_eq!(table.metadata().snapshots().len(), 3);
    assert_eq!(paths(&table), ["a.parquet", "b.parquet", "c.parquet"]);
}

/// What a try at expiring a table left of it.
struct RefusedExpiry {
    /// The error the try failed with; `None` when it did not fail.
    error: Option<Error>,
    /// Whether the table has the versions, the snapshots and the manifests
    /// and manifest lists it had before the try.
    is_table_untouched: bool,
}

/// Try to expire `table`, the fixture's current version, far past its
/// window.
fn try_to_expire(fixture: &Fixture, table: &Table) -> RefusedExpiry {
    let versions_before = fixture.metadata_store.count_versions();
    let snapshots_before = table.metadata().snapshots().len();
    let objects_before = avro_objects(&fixture.location);
    let error = table.expire_snapshots(now_unix_ms() + 100 * HOUR_MS).err();
    RefusedExpiry {
        error,
        is_table_untouched: fixture.metadata_store.count_versions() == versions_before
            && load(fixture).metadata().snapshots().len() == snapshots_before
            && avro_objects(&fixture.location) == objects_before,
    }
}

#[test]
fn a_table_with_a_tag_is_refused_before_anything_expires_and_commits_keep_landing() {
    let (fixture, created) = create_table(&[]);
    let table = append(&created, vec![entry("a", 1, 1, 1)]);
    let table = with_tag_on_current_snapshot(&fixture, &table);
    let table = append(&table, vec![entry("b", 1, 2, 2)]);
    let table = append(&table, vec![entry("c", 1, 3, 3)]);

    let refused = try_to_expire(&fixture, &table);
    let table = append(&table, vec![entry("d", 1, 4, 4)]);
    let table = replace(&table, &["a.parquet"], "e");

    assert!(
        refused
            .error
            .as_ref()
            .is_some_and(is_non_linear_history_error)
    );
    assert!(refused.is_table_untouched);
    assert_eq!(table.metadata().snapshots().len(), 5);
    assert_eq!(
        paths(&table),
        ["b.parquet", "c.parquet", "d.parquet", "e.parquet"]
    );
}

#[test]
fn a_table_with_a_second_branch_is_refused_before_anything_expires() {
    let (fixture, created) = create_table(&[]);
    let table = append(&created, vec![entry("a", 1, 1, 1)]);
    let table = with_branch_on_current_snapshot(&fixture, &table);
    let table = append(&table, vec![entry("b", 1, 2, 2)]);

    let refused = try_to_expire(&fixture, &table);

    let error = refused.error.expect("the table has a second branch");
    assert!(is_non_linear_history_error(&error), "{error}");
    assert!(error.to_string().contains("branch `audit`"), "{error}");
    assert!(refused.is_table_untouched);
}

#[test]
fn a_table_with_a_snapshot_outside_its_linear_history_is_refused_before_anything_expires() {
    let (fixture, created) = create_table(&[]);
    let table = append(&created, vec![entry("a", 1, 1, 1)]);
    let table = append(&table, vec![entry("b", 1, 2, 2)]);
    let table = with_staged_snapshot(&fixture, &table);

    let refused = try_to_expire(&fixture, &table);

    let error = refused
        .error
        .expect("the table has a snapshot outside its linear history");
    assert!(is_non_linear_history_error(&error), "{error}");
    assert!(error.to_string().contains("ancestor"), "{error}");
    assert!(refused.is_table_untouched);
}

#[test]
fn expiry_alone_commits_a_metadata_only_version() {
    // Every commit is dated by the wall clock, so only the sweep's clock
    // moves.
    let (fixture, created) = create_table(&[]);
    let table = append(&created, vec![entry("a", 1, 1, 1)]);
    let table = append(&table, vec![entry("b", 1, 2, 2)]);

    let nothing_yet = table.expire_snapshots(now_unix_ms() + 2 * HOUR_MS).unwrap();
    let expiry = expire_at(&table, now_unix_ms() + 6 * HOUR_MS);

    assert!(matches!(nothing_yet, ExpiryOutcome::NothingExpired));
    assert_eq!(expiry.expired.snapshot_count(), 1);
    let expired = expiry.table;
    assert_eq!(version(&fixture, &expired), version(&fixture, &table) + 1);
    assert_eq!(expired.metadata().snapshots().len(), 1);
    assert_eq!(
        expired.metadata().current_snapshot_id(),
        table.metadata().current_snapshot_id()
    );
    assert_eq!(paths(&expired), paths(&table));
}

/// The versions of the metadata files `expiry` leaves to delete.
fn expired_metadata_versions(fixture: &Fixture, expiry: &SnapshotExpiry) -> Vec<u64> {
    list_expired_objects(expiry)
        .iter()
        .filter_map(|object| fixture.metadata_store.find_version_number(object))
        .collect()
}

#[test]
fn an_expiry_with_only_metadata_files_past_the_window_commits_and_drops_them() {
    let (fixture, created) = create_table(&[]);
    let table = append(&created, vec![entry("a", 1, 1, 1)]);

    let expiry = expire_at(&table, now_unix_ms() + 100 * HOUR_MS);

    assert_eq!(expiry.expired.snapshot_count(), 0);
    assert_eq!(expired_metadata_versions(&fixture, &expiry), [1]);
    assert_eq!(logged_versions(&fixture, &expiry.table), [2]);
    assert_eq!(version(&fixture, &expiry.table), 3);
    assert_eq!(paths(&expiry.table), ["a.parquet"]);
}

#[test]
fn an_expiry_with_nothing_past_the_window_commits_nothing() {
    let (fixture, created) = create_table(&[]);
    let table = append(&created, vec![entry("a", 1, 1, 1)]);
    let table = append(&table, vec![entry("b", 1, 2, 2)]);

    let expiry = table
        .expire_snapshots(now_unix_ms() + WINDOW_MS - HOUR_MS)
        .unwrap();

    assert!(matches!(expiry, ExpiryOutcome::NothingExpired));
    assert_eq!(fixture.metadata_store.count_versions(), 3);
    assert_eq!(logged_versions(&fixture, &load(&fixture)), [1, 2]);
}

#[test]
fn a_table_with_gc_disabled_drops_no_metadata_file() {
    let (fixture, created) = create_table(&[]);
    let table = append(&created, vec![entry("a", 1, 1, 1)]);
    append(&table, vec![entry("b", 1, 2, 2)]);
    let table = publish_changed(&fixture, |builder| {
        builder
            .set_properties(HashMap::from([(
                "gc.enabled".to_string(),
                "false".to_string(),
            )]))
            .unwrap()
    });

    let expiry = table
        .expire_snapshots(now_unix_ms() + 100 * HOUR_MS)
        .unwrap();

    assert!(matches!(expiry, ExpiryOutcome::NothingExpired));
    assert_eq!(fixture.metadata_store.count_versions(), 4);
    assert_eq!(logged_versions(&fixture, &load(&fixture)), [1, 2, 3]);
}

/// A table directory holding one manifest list, a real commit's, which every
/// snapshot of the hand-made histories below names: a table is loaded with
/// the manifest list of its current snapshot and nothing else of its files.
struct HandMadeHistory {
    fixture: Fixture,
    manifest_list: String,
}

impl HandMadeHistory {
    fn new() -> Self {
        let (fixture, created) = create_table(&[]);
        let committed = append(&created, vec![entry("a", 1, 1, 1)]);
        let snapshot = committed.metadata().current_snapshot().unwrap();
        Self {
            manifest_list: snapshot.manifest_list().to_string(),
            fixture,
        }
    }

    /// A snapshot `snapshot_id`, committed at `timestamp_ms` on top of
    /// `parent`.
    fn build_snapshot(&self, snapshot_id: i64, parent: Option<i64>, timestamp_ms: i64) -> Snapshot {
        Snapshot::builder()
            .with_snapshot_id(snapshot_id)
            .with_parent_snapshot_id(parent)
            .with_sequence_number(snapshot_id)
            .with_timestamp_ms(timestamp_ms)
            .with_manifest_list(self.manifest_list.clone())
            .with_summary(Summary {
                operation: Operation::Append,
                additional_properties: HashMap::new(),
            })
            .with_schema_id(0)
            .with_row_range(0, 0)
            .build()
    }

    /// The metadata of a table with one snapshot per entry of
    /// `timestamps_ms`, in order.
    fn build_metadata_with_snapshots(
        &self,
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
            let snapshot = self.build_snapshot(snapshot_id, parent, *timestamp_ms);
            builder = builder.set_branch_snapshot(snapshot, MAIN_BRANCH).unwrap();
            parent = Some(snapshot_id);
        }
        builder.build().unwrap().metadata
    }

    /// The metadata of a table whose metadata file `v{n}` was written at the
    /// `n`th entry of `timestamps_ms`, the last of them its current one and
    /// the rest named by its metadata log.
    fn build_metadata_with_metadata_files(
        &self,
        timestamps_ms: &[i64],
        properties: HashMap<String, String>,
    ) -> TableMetadata {
        let mut metadata = self.build_metadata_with_snapshots(&timestamps_ms[..1], properties);
        for (index, timestamp_ms) in timestamps_ms.iter().enumerate().skip(1) {
            let snapshot_id = index as i64 + 1;
            let snapshot = self.build_snapshot(snapshot_id, Some(snapshot_id - 1), *timestamp_ms);
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

    /// The table `metadata` describes.
    fn build_table_at(&self, metadata: TableMetadata) -> Table {
        self.fixture.build_table_at(metadata)
    }

    /// The table with one snapshot per entry of `timestamps_ms`.
    fn build_table_with_snapshots(
        &self,
        timestamps_ms: &[i64],
        properties: HashMap<String, String>,
    ) -> Table {
        self.build_table_at(self.build_metadata_with_snapshots(timestamps_ms, properties))
    }

    /// The table with one metadata file per entry of `timestamps_ms`.
    fn build_table_with_metadata_files(
        &self,
        timestamps_ms: &[i64],
        properties: HashMap<String, String>,
    ) -> Table {
        self.build_table_at(self.build_metadata_with_metadata_files(timestamps_ms, properties))
    }
}

#[test]
fn a_snapshot_expires_once_its_successor_is_older_than_the_window() {
    let history = HandMadeHistory::new();
    let table = history.build_table_with_snapshots(&[0, HOUR_MS, 2 * HOUR_MS], HashMap::new());

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
    let history = HandMadeHistory::new();
    let table = history.build_table_with_snapshots(
        &[0, HOUR_MS],
        HashMap::from([(SNAPSHOT_RETENTION_PROPERTY.to_string(), "1000".to_string())]),
    );

    assert_eq!(
        table.snapshot_retention().unwrap(),
        Duration::from_millis(1000)
    );
    assert_eq!(table.expired_snapshot_ids(HOUR_MS + 1000).unwrap(), vec![1]);
}

#[test]
fn a_malformed_window_is_an_error() {
    let history = HandMadeHistory::new();
    let table = history.build_table_with_snapshots(
        &[0],
        HashMap::from([(SNAPSHOT_RETENTION_PROPERTY.to_string(), "soon".to_string())]),
    );

    let error = table.snapshot_retention().unwrap_err();

    assert!(matches!(error, Error::MalformedProperty { .. }), "{error}");
}

#[test]
fn a_snapshot_inside_the_window_keeps_every_snapshot_after_it() {
    let history = HandMadeHistory::new();
    // The writer of the third snapshot dated it thirty seconds before the
    // second.
    let table = history.build_table_with_snapshots(
        &[0, HOUR_MS, HOUR_MS - 30 * SECOND_MS, 2 * HOUR_MS],
        HashMap::new(),
    );

    let expired = table
        .expired_snapshot_ids(5 * HOUR_MS - 10 * SECOND_MS)
        .unwrap();

    // The second snapshot's successor is past the window by its own date,
    // but the first one's is not, and a history gives up its oldest
    // snapshot first.
    assert!(expired.is_empty(), "{expired:?}");
}

#[test]
fn a_table_with_a_tag_expires_no_snapshot_and_no_metadata_file() {
    let history = HandMadeHistory::new();
    let metadata = history
        .build_metadata_with_metadata_files(&[0, HOUR_MS, 2 * HOUR_MS], HashMap::new())
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
    let table = history.build_table_at(metadata);

    let snapshots_refusal = table.expired_snapshot_ids(100 * HOUR_MS).unwrap_err();
    let metadata_files_refusal = table.expired_metadata_files(100 * HOUR_MS).unwrap_err();

    assert!(
        is_non_linear_history_error(&snapshots_refusal),
        "{snapshots_refusal}"
    );
    assert!(
        snapshots_refusal.to_string().contains("tag `release`"),
        "{snapshots_refusal}"
    );
    assert!(
        is_non_linear_history_error(&metadata_files_refusal),
        "{metadata_files_refusal}"
    );
}

#[test]
fn a_table_with_gc_disabled_expires_nothing() {
    let history = HandMadeHistory::new();
    let table = history.build_table_with_snapshots(
        &[0, HOUR_MS, 2 * HOUR_MS],
        HashMap::from([("gc.enabled".to_string(), "false".to_string())]),
    );

    let expired = table.expired_snapshot_ids(100 * HOUR_MS).unwrap();

    assert!(expired.is_empty());
}

#[test]
fn no_number_of_newer_snapshots_expires_one_inside_the_window() {
    let history = HandMadeHistory::new();
    let timestamps: Vec<i64> = (0..5000).collect();
    let table = history.build_table_with_snapshots(&timestamps, HashMap::new());

    let expired = table.expired_snapshot_ids(timestamps.len() as i64).unwrap();

    assert!(expired.is_empty());
}

#[test]
fn a_metadata_file_expires_once_its_successor_is_older_than_the_window() {
    let history = HandMadeHistory::new();
    let table = history
        .build_table_with_metadata_files(&[0, HOUR_MS, 2 * HOUR_MS, 3 * HOUR_MS], HashMap::new());

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
    let history = HandMadeHistory::new();
    // The writer of `v3` dated it thirty seconds before `v2`.
    let table = history.build_table_with_metadata_files(
        &[0, HOUR_MS, HOUR_MS - 30 * SECOND_MS, 2 * HOUR_MS],
        HashMap::new(),
    );

    let expired = table
        .expired_metadata_files(5 * HOUR_MS - 10 * SECOND_MS)
        .unwrap();

    // `v2`'s successor is past the window by its own date, but `v1`'s is
    // not, and the log gives up its oldest entry first.
    assert!(expired.is_empty(), "{expired:?}");
}

#[test]
fn no_number_of_newer_metadata_files_expires_one_inside_the_window() {
    let history = HandMadeHistory::new();
    let timestamps: Vec<i64> = (0..500).collect();
    let table = history.build_table_with_metadata_files(&timestamps, HashMap::new());

    let expired = table
        .expired_metadata_files(timestamps.len() as i64)
        .unwrap();

    assert!(expired.is_empty());
    assert_eq!(table.metadata().metadata_log().len(), 499);
}

#[test]
fn a_table_with_gc_disabled_expires_no_metadata_file() {
    let history = HandMadeHistory::new();
    let table = history.build_table_with_metadata_files(
        &[0, HOUR_MS, 2 * HOUR_MS],
        HashMap::from([("gc.enabled".to_string(), "false".to_string())]),
    );

    let expired = table.expired_metadata_files(100 * HOUR_MS).unwrap();

    assert!(expired.is_empty());
}
