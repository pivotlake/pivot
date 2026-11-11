//! Which snapshots and previous metadata files a table expires, and what an
//! expiry commits and leaves to delete: only what has outlived the retention
//! window, never what a tag points at, nothing when garbage collection is off,
//! and never as a side effect of a commit of files.

mod support;

use std::collections::HashMap;
use std::time::Duration;

use iceberg::spec::{
    FormatVersion, MAIN_BRANCH, NestedField, Operation, PrimitiveType, Schema, Snapshot,
    SnapshotReference, SnapshotRetention, SortOrder, Summary, TableMetadata, TableMetadataBuilder,
    Type, UnboundPartitionSpec,
};
use iceberg_tables::commit::SnapshotExpiry;
use iceberg_tables::maintenance::retention::{
    DEFAULT_SNAPSHOT_RETENTION, SNAPSHOT_RETENTION_PROPERTY,
};
use iceberg_tables::{Error, Table};
use support::{
    Fixture, HOUR_MS, append, create_table, entry, live_entry, load, logged_versions, now_unix_ms,
    paths, publish_changed, version, with_retention_of_one_millisecond,
    with_tag_on_current_snapshot,
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

#[test]
fn a_tagged_snapshot_outlives_the_window_and_commits_keep_landing() {
    let (fixture, created) = create_table(&[]);
    let table = append(&created, vec![entry("a", 1, 1, 1)]);
    let tagged_snapshot_id = table.metadata().current_snapshot_id().unwrap();
    let table = with_tag_on_current_snapshot(&fixture, &table);
    let table = append(&table, vec![entry("b", 1, 2, 2)]);
    let untagged_snapshot_id = table.metadata().current_snapshot_id().unwrap();
    let table = append(&table, vec![entry("c", 1, 3, 3)]);

    let much_later = now_unix_ms() + 100 * HOUR_MS;
    let table = table.expire_snapshots(much_later).unwrap().unwrap().table;
    let table = append(&table, vec![entry("d", 1, 4, 4)]);

    let snapshot_ids: Vec<i64> = table
        .metadata()
        .snapshots()
        .map(|snapshot| snapshot.snapshot_id())
        .collect();
    assert!(snapshot_ids.contains(&tagged_snapshot_id));
    assert!(!snapshot_ids.contains(&untagged_snapshot_id));
    assert_eq!(
        paths(&table),
        ["a.parquet", "b.parquet", "c.parquet", "d.parquet"]
    );
}

#[test]
fn expiry_alone_commits_a_metadata_only_version() {
    // Every commit is dated by the wall clock, so only the sweep's clock
    // moves.
    let (fixture, created) = create_table(&[]);
    let table = append(&created, vec![entry("a", 1, 1, 1)]);
    let table = append(&table, vec![entry("b", 1, 2, 2)]);

    let nothing_yet = table.expire_snapshots(now_unix_ms() + 2 * HOUR_MS).unwrap();
    let expiry = table
        .expire_snapshots(now_unix_ms() + 6 * HOUR_MS)
        .unwrap()
        .unwrap();

    assert!(nothing_yet.is_none());
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

/// The table's names for the objects `expiry` leaves to delete, sorted.
fn objects_of(expiry: &SnapshotExpiry) -> Vec<String> {
    let mut objects: Vec<String> = expiry
        .table
        .collect_expired_objects(&expiry.expired)
        .unwrap()
        .iter()
        .map(|path| path.to_string())
        .collect();
    objects.sort_unstable();
    objects
}

/// The table's names for the objects an expiry `far_later` leaves to
/// delete for the snapshots it dropped, sorted: the metadata files it
/// dropped are left out.
fn expired_snapshot_objects(table: &Table) -> Vec<String> {
    let far_later = now_unix_ms() + 100 * HOUR_MS;
    let expiry = table.expire_snapshots(far_later).unwrap().unwrap();
    let mut objects = objects_of(&expiry);
    objects.retain(|object| !object.ends_with(".metadata.json"));
    objects
}

/// The versions of the metadata files `expiry` leaves to delete.
fn expired_metadata_versions(fixture: &Fixture, expiry: &SnapshotExpiry) -> Vec<u64> {
    objects_of(expiry)
        .iter()
        .filter_map(|object| fixture.metadata_store.find_version_number(object))
        .collect()
}

/// The table's names for the manifest list and the manifests of
/// `table`'s current snapshot.
fn current_snapshot_objects(table: &Table) -> Vec<String> {
    let location = table.location();
    let list = table.metadata().current_snapshot().unwrap().manifest_list();
    let manifests = table.manifest_list();
    std::iter::once(list)
        .chain(
            manifests
                .data_manifests()
                .chain(manifests.delete_manifests())
                .map(|manifest| manifest.manifest_path.as_str()),
        )
        .map(|uri| location.file_path_for_uri(uri).unwrap().to_string())
        .collect()
}

#[test]
fn an_expiry_leaves_to_delete_what_only_its_snapshots_used() {
    let (_fixture, created) = create_table(&[]);
    let appended = append(&created, vec![entry("a", 1, 1, 1), entry("b", 1, 2, 2)]);
    let removed = live_entry(&appended, "a.parquet");
    let replaced = appended
        .commit(&[removed], vec![entry("c", 1, 1, 1)])
        .unwrap()
        .unwrap();

    let objects = expired_snapshot_objects(&replaced);

    let mut expected = current_snapshot_objects(&appended);
    expected.push("a.parquet".to_string());
    expected.sort_unstable();
    assert_eq!(objects, expected);
}

#[test]
fn an_expiry_leaves_what_a_retained_snapshot_shares_with_an_expired_one() {
    let (_fixture, created) = create_table(&[]);
    let first = append(&created, vec![entry("a", 1, 1, 1)]);
    let second = append(&first, vec![entry("b", 1, 2, 2)]);

    let objects = expired_snapshot_objects(&second);

    let first_list = first.metadata().current_snapshot().unwrap().manifest_list();
    assert_eq!(
        objects,
        [second
            .location()
            .file_path_for_uri(first_list)
            .unwrap()
            .to_string()],
        "the first snapshot's manifest, and the file in it, are the second's too"
    );
}

#[test]
fn an_expiry_leaves_what_a_tagged_snapshot_uses() {
    let (fixture, created) = create_table(&[]);
    let appended = append(&created, vec![entry("a", 1, 1, 1)]);
    let tagged = with_tag_on_current_snapshot(&fixture, &appended);
    let removed = live_entry(&tagged, "a.parquet");
    let replaced = tagged
        .commit(&[removed], vec![entry("b", 1, 1, 1)])
        .unwrap()
        .unwrap();
    let replaced = append(&replaced, vec![entry("c", 1, 2, 2)]);

    let objects = expired_snapshot_objects(&replaced);

    let kept = current_snapshot_objects(&appended);
    assert!(!objects.contains(&"a.parquet".to_string()), "{objects:?}");
    assert!(
        kept.iter().all(|object| !objects.contains(object)),
        "{objects:?}"
    );
    assert!(!objects.is_empty(), "the untagged replace expired");
}

#[test]
fn an_expiry_with_only_metadata_files_past_the_window_commits_and_drops_them() {
    let (fixture, created) = create_table(&[]);
    let table = append(&created, vec![entry("a", 1, 1, 1)]);

    let expiry = table
        .expire_snapshots(now_unix_ms() + 100 * HOUR_MS)
        .unwrap()
        .unwrap();

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

    assert!(expiry.is_none());
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

    assert!(expiry.is_none());
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
fn a_snapshot_a_tag_points_at_does_not_expire() {
    let history = HandMadeHistory::new();
    let metadata = history
        .build_metadata_with_snapshots(&[0, HOUR_MS, 2 * HOUR_MS], HashMap::new())
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

    let expired = table.expired_snapshot_ids(100 * HOUR_MS).unwrap();

    assert_eq!(expired, vec![2]);
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
