//! What an expiry leaves to delete: the files the snapshots it dropped
//! retired, the manifests and manifest lists only they name, and the
//! metadata files it dropped; never a file or a manifest a kept snapshot
//! reads. It is found from the dropped snapshots and the oldest kept one
//! alone, and it is what reachability leaves: everything a dropped
//! snapshot reaches that no kept snapshot does, which each test works out
//! for itself from the table's manifests. The Puffin file of a removed
//! deletion vector is the one thing left in place, and a manifest the
//! current snapshot lists again refuses the whole collection.

mod support;

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, MAIN_BRANCH, Manifest,
    ManifestFile, ManifestList as SpecManifestList, ManifestListWriter, ManifestWriterBuilder,
    Operation, Snapshot, Summary,
};
use iceberg_tables::commit::SnapshotExpiry;
use iceberg_tables::maintenance::retention::DEFAULT_SNAPSHOT_RETENTION;
use iceberg_tables::test_support::Faults;
use iceberg_tables::{Error, Table, block_on};
use object_storage::ObjectPath;
use support::{
    Fixture, HOUR_MS, append, create_table, entry, is_non_linear_history_error,
    list_expired_objects, load, load_over_faulty_store, lower_manifest_merge_threshold,
    now_unix_ms, paths, pause_between_commits, publish_changed, replace,
    with_tag_on_current_snapshot,
};

const WINDOW_MS: i64 = DEFAULT_SNAPSHOT_RETENTION.as_millis() as i64;

/// Expire every snapshot of `table` but its current one.
fn expire_all_but_current(table: &Table) -> SnapshotExpiry {
    table
        .expire_snapshots(now_unix_ms() + 100 * HOUR_MS)
        .unwrap()
        .expect("no other writer")
}

/// The table's names for the objects `expiry` leaves to delete for the
/// snapshots it dropped, sorted: the metadata files it dropped are left
/// out.
fn list_expired_snapshot_objects(expiry: &SnapshotExpiry) -> Vec<String> {
    let mut objects = list_expired_objects(expiry);
    objects.retain(|object| !object.ends_with(".metadata.json"));
    objects
}

/// The data files among `objects`.
fn list_data_files(objects: &[String]) -> Vec<&str> {
    objects
        .iter()
        .map(String::as_str)
        .filter(|object| object.ends_with(".parquet"))
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

/// The table's names for everything `snapshot`, a snapshot of `table`,
/// reaches: its manifest list, every manifest the list names, and every
/// file live in one of them, each read from the store.
fn list_reachable_objects(table: &Table, snapshot: &Snapshot) -> BTreeSet<String> {
    let location = table.location();
    let name = |uri: &str| location.file_path_for_uri(uri).unwrap().to_string();
    let list_bytes = location
        .read_metadata_object(snapshot.manifest_list())
        .unwrap();
    let list = SpecManifestList::parse_with_version(&list_bytes, table.metadata().format_version())
        .unwrap();
    let mut objects = BTreeSet::from([name(snapshot.manifest_list())]);
    for listed in list.entries() {
        objects.insert(name(&listed.manifest_path));
        let manifest_bytes = location
            .read_metadata_object(&listed.manifest_path)
            .unwrap();
        let manifest = Manifest::parse_avro(&manifest_bytes).unwrap();
        for entry in manifest.entries().iter().filter(|entry| entry.is_alive()) {
            objects.insert(name(entry.file_path()));
        }
    }
    objects
}

/// What reachability leaves to delete of the expiry that took `before` to
/// `after`: everything a snapshot it dropped reaches that no snapshot it
/// kept does, sorted.
fn list_unreachable_objects(before: &Table, after: &Table) -> Vec<String> {
    let mut reached_by_dropped = BTreeSet::new();
    let mut reached_by_kept = BTreeSet::new();
    for snapshot in before.metadata().snapshots() {
        let is_kept = after
            .metadata()
            .snapshot_by_id(snapshot.snapshot_id())
            .is_some();
        let reached = list_reachable_objects(before, snapshot);
        if is_kept {
            reached_by_kept.extend(reached);
        } else {
            reached_by_dropped.extend(reached);
        }
    }
    reached_by_dropped
        .difference(&reached_by_kept)
        .cloned()
        .collect()
}

/// Delete `objects`, as the table names them, from the fixture's store,
/// the way a datastore deletes what an expiry leaves, and load the table
/// from what is left.
fn load_after_deleting(fixture: &Fixture, objects: &[String]) -> Table {
    let location = &fixture.location;
    for object in objects {
        let key = location.data_file_key(&ObjectPath::new(object.as_str()));
        location.store().delete(&key).unwrap();
    }
    load(fixture)
}

#[test]
fn an_expiry_leaves_to_delete_what_only_its_snapshots_used() {
    let (fixture, created) = create_table(&[]);
    let appended = append(&created, vec![entry("a", 1, 1, 1), entry("b", 1, 2, 2)]);
    let compacted = replace(&appended, &["a.parquet"], "c");

    let expiry = expire_all_but_current(&compacted);
    let unreachable = list_unreachable_objects(&compacted, &expiry.table);
    let objects = list_expired_snapshot_objects(&expiry);
    let reloaded = load_after_deleting(&fixture, &objects);

    let mut expected = current_snapshot_objects(&appended);
    expected.push("a.parquet".to_string());
    expected.sort_unstable();
    assert_eq!(objects, expected);
    assert_eq!(objects, unreachable);
    assert_eq!(paths(&reloaded), ["b.parquet", "c.parquet"]);
}

#[test]
fn an_expiry_of_the_snapshots_before_a_compaction_leaves_the_files_it_replaced() {
    let (fixture, created) = create_table(&[]);
    let appended = append(&created, vec![entry("a", 1, 1, 1), entry("b", 1, 2, 2)]);
    let compacted = replace(&appended, &["a.parquet"], "c");
    let window_start = pause_between_commits();
    let table = append(&compacted, vec![entry("d", 1, 3, 3)]);

    let expiry = table
        .expire_snapshots(window_start + WINDOW_MS)
        .unwrap()
        .unwrap();
    let unreachable = list_unreachable_objects(&table, &expiry.table);
    let objects = list_expired_snapshot_objects(&expiry);
    let reloaded = load_after_deleting(&fixture, &objects);

    let mut expected = current_snapshot_objects(&appended);
    expected.push("a.parquet".to_string());
    expected.sort_unstable();
    assert_eq!(expiry.table.metadata().snapshots().len(), 2);
    assert_eq!(objects, expected);
    assert_eq!(objects, unreachable);
    assert_eq!(paths(&reloaded), ["b.parquet", "c.parquet", "d.parquet"]);
}

#[test]
fn an_expiry_of_several_snapshots_leaves_every_file_the_compactions_among_them_replaced() {
    let (fixture, created) = create_table(&[]);
    let table = append(&created, vec![entry("a", 1, 1, 1), entry("b", 1, 2, 2)]);
    let table = replace(&table, &["a.parquet"], "c");
    let table = append(&table, vec![entry("d", 1, 3, 3)]);
    let table = replace(&table, &["b.parquet", "d.parquet"], "e");
    let table = append(&table, vec![entry("f", 1, 4, 4)]);

    let expiry = expire_all_but_current(&table);
    let unreachable = list_unreachable_objects(&table, &expiry.table);
    let objects = list_expired_snapshot_objects(&expiry);
    let reloaded = load_after_deleting(&fixture, &objects);

    assert_eq!(
        list_data_files(&objects),
        ["a.parquet", "b.parquet", "d.parquet"]
    );
    assert_eq!(objects, unreachable);
    assert_eq!(paths(&reloaded), ["c.parquet", "e.parquet", "f.parquet"]);
}

#[test]
fn an_expiry_leaves_the_manifests_merged_away_since_and_no_data_file() {
    let (fixture, created) = create_table(&[]);
    let first = append(&created, vec![entry("a", 1, 1, 1)]);
    let second = append(&first, vec![entry("b", 1, 2, 2)]);
    let merging = lower_manifest_merge_threshold(&fixture);
    let table = append(&merging, vec![entry("c", 1, 3, 3)]);

    let expiry = expire_all_but_current(&table);
    let unreachable = list_unreachable_objects(&table, &expiry.table);
    let objects = list_expired_snapshot_objects(&expiry);
    let reloaded = load_after_deleting(&fixture, &objects);

    let mut expected = current_snapshot_objects(&first);
    expected.extend(current_snapshot_objects(&second));
    expected.sort_unstable();
    expected.dedup();
    assert_eq!(table.manifest_list().data_manifests().count(), 1);
    assert_eq!(objects, expected);
    assert_eq!(objects, unreachable);
    assert_eq!(paths(&reloaded), ["a.parquet", "b.parquet", "c.parquet"]);
}

#[test]
fn an_expiry_leaves_a_replaced_file_whose_removal_record_a_later_commit_merged_away() {
    let (fixture, created) = create_table(&[]);
    let appended = append(&created, vec![entry("a", 1, 1, 1), entry("b", 1, 2, 2)]);
    replace(&appended, &["a.parquet"], "c");
    let merging = lower_manifest_merge_threshold(&fixture);
    let table = append(&merging, vec![entry("d", 1, 3, 3)]);

    let expiry = expire_all_but_current(&table);
    let unreachable = list_unreachable_objects(&table, &expiry.table);
    let objects = list_expired_snapshot_objects(&expiry);
    let reloaded = load_after_deleting(&fixture, &objects);

    assert_eq!(table.manifest_list().data_manifests().count(), 1);
    assert_eq!(list_data_files(&objects), ["a.parquet"]);
    assert_eq!(objects, unreachable);
    assert_eq!(paths(&reloaded), ["b.parquet", "c.parquet", "d.parquet"]);
}

#[test]
fn an_expiry_leaves_a_replaced_file_whose_removal_record_its_own_commit_merged() {
    let (fixture, created) = create_table(&[]);
    append(&created, vec![entry("a", 1, 1, 1), entry("b", 1, 2, 2)]);
    let merging = lower_manifest_merge_threshold(&fixture);
    let compacted = replace(&merging, &["a.parquet"], "c");
    let table = append(&compacted, vec![entry("d", 1, 3, 3)]);

    let expiry = expire_all_but_current(&table);
    let unreachable = list_unreachable_objects(&table, &expiry.table);
    let objects = list_expired_snapshot_objects(&expiry);
    let reloaded = load_after_deleting(&fixture, &objects);

    assert_eq!(compacted.manifest_list().data_manifests().count(), 1);
    assert_eq!(list_data_files(&objects), ["a.parquet"]);
    assert_eq!(objects, unreachable);
    assert_eq!(paths(&reloaded), ["b.parquet", "c.parquet", "d.parquet"]);
}

#[test]
fn an_expiry_leaves_what_a_retained_snapshot_shares_with_an_expired_one() {
    let (_fixture, created) = create_table(&[]);
    let first = append(&created, vec![entry("a", 1, 1, 1)]);
    let second = append(&first, vec![entry("b", 1, 2, 2)]);

    let objects = list_expired_snapshot_objects(&expire_all_but_current(&second));

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
fn an_expiry_of_appends_leaves_their_manifest_lists_and_metadata_files_and_no_data_file() {
    let (fixture, created) = create_table(&[]);
    let first = append(&created, vec![entry("a", 1, 1, 1)]);
    let second = append(&first, vec![entry("b", 1, 2, 2)]);
    let table = append(&second, vec![entry("c", 1, 3, 3)]);

    let expiry = expire_all_but_current(&table);
    let unreachable = list_unreachable_objects(&table, &expiry.table);
    let objects = list_expired_objects(&expiry);
    let reloaded = load_after_deleting(&fixture, &objects);

    let (metadata_files, snapshot_objects): (Vec<String>, Vec<String>) = objects
        .into_iter()
        .partition(|object| object.ends_with(".metadata.json"));
    let name_manifest_list = |table: &Table| current_snapshot_objects(table).swap_remove(0);
    let mut expected = vec![name_manifest_list(&first), name_manifest_list(&second)];
    expected.sort_unstable();
    assert_eq!(snapshot_objects, expected);
    assert_eq!(snapshot_objects, unreachable);
    assert_eq!(metadata_files.len(), 3);
    assert_eq!(paths(&reloaded), ["a.parquet", "b.parquet", "c.parquet"]);
}

#[test]
fn the_objects_of_an_expiry_are_refused_to_a_table_that_gained_a_tag() {
    let (fixture, created) = create_table(&[]);
    let table = append(&created, vec![entry("a", 1, 1, 1)]);
    let table = append(&table, vec![entry("b", 1, 2, 2)]);
    let expiry = expire_all_but_current(&table);
    let tagged = with_tag_on_current_snapshot(&fixture, &expiry.table);

    let refused = tagged.collect_expired_objects(&expiry.expired).err();

    assert!(refused.as_ref().is_some_and(is_non_linear_history_error));
}

#[test]
fn the_objects_of_an_expiry_are_refused_to_a_version_a_later_expiry_committed() {
    let (_fixture, created) = create_table(&[]);
    let table = append(&created, vec![entry("a", 1, 1, 1)]);
    let table = append(&table, vec![entry("b", 1, 2, 2)]);
    let first_window_start = pause_between_commits();
    let table = append(&table, vec![entry("c", 1, 3, 3)]);
    let second_window_start = pause_between_commits();
    let table = append(&table, vec![entry("d", 1, 4, 4)]);
    let first_expiry = table
        .expire_snapshots(first_window_start + WINDOW_MS)
        .unwrap()
        .unwrap();
    let second_expiry = first_expiry
        .table
        .expire_snapshots(second_window_start + WINDOW_MS)
        .unwrap()
        .unwrap();

    let refused = second_expiry
        .table
        .collect_expired_objects(&first_expiry.expired)
        .err();

    assert_eq!(first_expiry.expired.snapshot_count(), 1);
    assert_eq!(second_expiry.expired.snapshot_count(), 1);
    assert!(refused.as_ref().is_some_and(is_non_linear_history_error));
}

#[test]
fn the_objects_of_an_expiry_are_found_from_its_snapshots_and_the_oldest_kept_one_alone() {
    let (fixture, created) = create_table(&[]);
    let appended = append(&created, vec![entry("a", 1, 1, 1), entry("b", 1, 2, 2)]);
    let compacted = replace(&appended, &["a.parquet"], "c");
    let window_start = pause_between_commits();
    let later = append(&compacted, vec![entry("d", 1, 3, 3)]);
    let latest = append(&later, vec![entry("e", 1, 4, 4)]);
    let faults = Arc::new(Faults::default());
    let expiry = load_over_faulty_store(&fixture, &faults)
        .expire_snapshots(window_start + WINDOW_MS)
        .unwrap()
        .unwrap();
    let manifests_without_a_removal = appended
        .manifest_list()
        .data_manifests()
        .chain(latest.manifest_list().data_manifests())
        .filter(|manifest| manifest.deleted_files_count == Some(0));
    for manifest in manifests_without_a_removal {
        fixture.delete_manifest(&manifest.manifest_path);
    }
    fixture.delete_manifest(later.metadata().current_snapshot().unwrap().manifest_list());
    faults.take_requests();

    let objects = list_expired_snapshot_objects(&expiry);
    let requests = faults.take_requests();

    // The manifest lists of the dropped snapshot and of the oldest kept
    // one, the compaction's; the one manifest read, over the pool, is the
    // one the compaction recorded its removal in.
    assert_eq!(expiry.table.metadata().snapshots().len(), 3);
    assert_eq!((requests.object_reads, requests.listings), (2, 0));
    assert_eq!(list_data_files(&objects), ["a.parquet"]);
}

#[test]
fn an_expiry_whose_manifest_list_cannot_be_read_leaves_nothing_to_delete() {
    let (fixture, created) = create_table(&[]);
    let appended = append(&created, vec![entry("a", 1, 1, 1), entry("b", 1, 2, 2)]);
    replace(&appended, &["a.parquet"], "c");
    let faults = Arc::new(Faults::default());
    let expiry = expire_all_but_current(&load_over_faulty_store(&fixture, &faults));
    faults.fail_gets.store(true, Ordering::SeqCst);

    let collected = expiry.table.collect_expired_objects(&expiry.expired);

    assert!(collected.is_err());
}

#[test]
fn an_expiry_whose_removal_record_cannot_be_read_leaves_nothing_to_delete() {
    let (fixture, created) = create_table(&[]);
    let appended = append(&created, vec![entry("a", 1, 1, 1), entry("b", 1, 2, 2)]);
    let compacted = replace(&appended, &["a.parquet"], "c");
    let expiry = expire_all_but_current(&compacted);
    let recording_manifest = compacted
        .manifest_list()
        .data_manifests()
        .find(|manifest| manifest.deleted_files_count == Some(1))
        .expect("the compaction recorded its removal in a manifest");
    fixture.delete_manifest(&recording_manifest.manifest_path);

    let collected = expiry.table.collect_expired_objects(&expiry.expired);

    assert!(collected.is_err());
}

/// A deletion vector of the data file `data_file`, kept as one blob of the
/// Puffin file `puffin`, both named as the fixture's table names a file.
fn build_deletion_vector(fixture: &Fixture, puffin: &str, data_file: &str) -> DataFile {
    let uri = |name: &str| fixture.location.uri_for(&ObjectPath::new(name)).unwrap();
    DataFileBuilder::default()
        .content(DataContentType::PositionDeletes)
        .file_path(uri(puffin))
        .file_format(DataFileFormat::Puffin)
        .file_size_in_bytes(100)
        .record_count(1)
        .referenced_data_file(Some(uri(data_file)))
        .content_offset(Some(4))
        .content_size_in_bytes(Some(40))
        .build()
        .unwrap()
}

/// The id of the snapshot a test writes by hand on top of `table`.
fn choose_next_snapshot_id(table: &Table) -> i64 {
    let current = table.metadata().current_snapshot().unwrap();
    current.snapshot_id().wrapping_add(1)
}

/// The delete manifest of the snapshot a test writes by hand on top of
/// `table`, holding deletion vectors that are blobs of the Puffin file
/// `puffin`: the vector of `replaced_for` as removed by the snapshot, and,
/// when there is a `kept_for`, the vector of that data file as added by
/// it. What a writer of deletion vectors records when it replaces one
/// vector of a Puffin file, which may hold several.
fn write_deletion_vector_manifest(
    fixture: &Fixture,
    table: &Table,
    puffin: &str,
    replaced_for: &str,
    kept_for: Option<&str>,
) -> ManifestFile {
    let metadata = table.metadata();
    let parent = metadata.current_snapshot().unwrap();
    let location = &fixture.location;
    let file_io = location.build_file_io();
    block_on(async {
        let output = file_io
            .new_output(location.metadata_uri("deletion-vectors-m0.avro"))
            .unwrap();
        let mut writer = ManifestWriterBuilder::new(
            output,
            Some(choose_next_snapshot_id(table)),
            metadata.current_schema().clone(),
            metadata.default_partition_spec().as_ref().clone(),
        )
        .build_v3_deletes();
        writer
            .add_delete_file(
                build_deletion_vector(fixture, puffin, replaced_for),
                parent.sequence_number(),
                Some(parent.sequence_number()),
            )
            .unwrap();
        if let Some(kept_for) = kept_for {
            writer
                .add_file(
                    build_deletion_vector(fixture, puffin, kept_for),
                    metadata.last_sequence_number() + 1,
                )
                .unwrap();
        }
        writer.write_manifest_file().await.unwrap()
    })
}

/// Every manifest the current snapshot of `table` names, as its manifest
/// list in the store has them.
fn read_listed_manifests(table: &Table) -> Vec<ManifestFile> {
    let manifest_list = table.metadata().current_snapshot().unwrap().manifest_list();
    let bytes = table
        .location()
        .read_metadata_object(manifest_list)
        .unwrap();
    SpecManifestList::parse_with_version(&bytes, table.metadata().format_version())
        .unwrap()
        .entries()
        .to_vec()
}

/// Commit, on top of `table`, the snapshot [`choose_next_snapshot_id`] chose,
/// its manifest list naming `manifests`, the way a writer other than Pivot
/// commits one: the list is written with the spec crate's writer, and the
/// snapshot made the head of `main`.
fn commit_manifests(fixture: &Fixture, table: &Table, manifests: Vec<ManifestFile>) -> Table {
    let metadata = table.metadata();
    let parent = metadata.current_snapshot().unwrap();
    let snapshot_id = choose_next_snapshot_id(table);
    let sequence_number = metadata.last_sequence_number() + 1;
    let location = &fixture.location;
    let file_io = location.build_file_io();
    let manifest_list = location.metadata_uri(&format!("snap-{snapshot_id}-by-hand.avro"));
    block_on(async {
        let output = file_io.new_output(&manifest_list).unwrap();
        let mut writer = ManifestListWriter::v3(
            output.writer().await.unwrap(),
            snapshot_id,
            Some(parent.snapshot_id()),
            sequence_number,
            Some(metadata.next_row_id()),
        );
        writer.add_manifests(manifests.into_iter()).unwrap();
        writer.close().await.unwrap();
    });
    let snapshot = Snapshot::builder()
        .with_snapshot_id(snapshot_id)
        .with_parent_snapshot_id(Some(parent.snapshot_id()))
        .with_sequence_number(sequence_number)
        .with_timestamp_ms(now_unix_ms())
        .with_manifest_list(manifest_list)
        .with_summary(Summary {
            operation: Operation::Overwrite,
            additional_properties: HashMap::new(),
        })
        .with_schema_id(metadata.current_schema_id())
        .with_row_range(metadata.next_row_id(), 0)
        .build();
    publish_changed(fixture, |builder| {
        builder.set_branch_snapshot(snapshot, MAIN_BRANCH).unwrap()
    })
}

#[test]
fn an_expiry_leaves_the_puffin_file_of_a_removed_deletion_vector_in_place() {
    let (fixture, created) = create_table(&[]);
    let appended = append(&created, vec![entry("a", 1, 1, 1), entry("b", 1, 2, 2)]);
    let delete_manifest = write_deletion_vector_manifest(
        &fixture,
        &appended,
        "vectors.puffin",
        "a.parquet",
        Some("b.parquet"),
    );
    let carried = appended.manifest_list().data_manifests().cloned();
    let table = commit_manifests(
        &fixture,
        &appended,
        carried.chain([delete_manifest]).collect(),
    );

    let expiry = expire_all_but_current(&table);
    let unreachable = list_unreachable_objects(&table, &expiry.table);
    let objects = list_expired_snapshot_objects(&expiry);

    // The vector of `b.parquet` is still live in the Puffin file the
    // removed vector was a blob of.
    assert_eq!(
        objects,
        [current_snapshot_objects(&appended).swap_remove(0)]
    );
    assert_eq!(objects, unreachable);
}

#[test]
fn an_expiry_leaves_a_delete_manifest_without_a_live_file_that_the_oldest_kept_snapshot_names() {
    let (fixture, created) = create_table(&[]);
    let appended = append(&created, vec![entry("a", 1, 1, 1)]);
    let delete_manifest =
        write_deletion_vector_manifest(&fixture, &appended, "vectors.puffin", "a.parquet", None);
    let carried = appended.manifest_list().data_manifests().cloned();
    let removing = commit_manifests(
        &fixture,
        &appended,
        carried.chain([delete_manifest]).collect(),
    );
    let table = commit_manifests(&fixture, &removing, read_listed_manifests(&removing));

    let expiry = expire_all_but_current(&table);
    let unreachable = list_unreachable_objects(&table, &expiry.table);
    let objects = list_expired_snapshot_objects(&expiry);

    // The current snapshot names the delete manifest the snapshot before
    // it wrote, though the version holds none of it: no file is left for
    // it to apply.
    let name_manifest_list = |table: &Table| current_snapshot_objects(table).swap_remove(0);
    let mut expected = vec![name_manifest_list(&appended), name_manifest_list(&removing)];
    expected.sort_unstable();
    assert_eq!(table.manifest_list().delete_manifests().count(), 0);
    assert_eq!(read_listed_manifests(&table).len(), 2);
    assert_eq!(objects, expected);
    assert_eq!(objects, unreachable);
}

#[test]
fn an_expiry_is_refused_a_manifest_that_the_current_snapshot_lists_again() {
    let (fixture, created) = create_table(&[]);
    let first = append(&created, vec![entry("a", 1, 1, 1)]);
    let merging = lower_manifest_merge_threshold(&fixture);
    let merged = append(&merging, vec![entry("b", 1, 2, 2)]);
    let window_start = pause_between_commits();
    let listed_again = first.manifest_list().data_manifests().next().unwrap();
    let kept = merged.manifest_list().data_manifests().cloned();
    let table = commit_manifests(
        &fixture,
        &merged,
        kept.chain([listed_again.clone()]).collect(),
    );
    let expiry = table
        .expire_snapshots(window_start + WINDOW_MS)
        .unwrap()
        .unwrap();

    let refused = expiry.table.collect_expired_objects(&expiry.expired).err();

    // The second snapshot merged the first one's manifest away, and is the
    // oldest one kept; the current snapshot lists that manifest again.
    assert_eq!(merged.manifest_list().data_manifests().count(), 1);
    assert!(matches!(
        &refused,
        Some(Error::ExpiredManifestStillListed { manifest, .. }) if *manifest == listed_again.manifest_path
    ));
}
