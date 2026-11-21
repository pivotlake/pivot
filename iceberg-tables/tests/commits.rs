//! Committing to a table: an append or a replace lands as the next version,
//! of several writers from one version exactly one wins, what was committed
//! is read back whole, in order, and a commit reads of the store the
//! manifest list it wrote and, of the version it was built on, the manifest
//! list and the manifests it merges or looks for a replaced file in.

mod support;

use std::collections::HashMap;
use std::sync::Arc;

use iceberg::Runtime;
use iceberg::io::FileIO;
use iceberg::spec::{
    FormatVersion, Manifest, ManifestStatus, Operation, SortOrder, TableMetadataBuilder,
    UnboundPartitionSpec,
};
use iceberg::table::Table as SpecTable;
use iceberg::{NamespaceIdent, TableIdent};
use iceberg_tables::test_support::Faults;
use iceberg_tables::types::schema::build_schema;
use iceberg_tables::{Error, MetadataFile, Table, block_on};
use object_storage::ObjectPath;
use parquet_engine::{PartitionValues, scalar_equal};
use planner::catalog::Column;
use planner::types::Type;
use support::{
    append, avro_objects, columns, commit_replace, count_manifests, create_table, entry,
    find_partition_key, int_scalar, live_entry, load, load_over_faulty_store, partition_entry,
    paths, repartition, version,
};

/// What a version's manifests record of one live file, with what a
/// manifest leaves implicit inherited from its manifest-list entry.
struct RecordedFile {
    sequence_number: i64,
    first_row_id: Option<i64>,
    manifest_path: String,
}

/// What the data manifests `table` lists record of its live file
/// `path`, read from the store.
fn read_recorded_file(table: &Table, path: &str) -> RecordedFile {
    let location = table.location();
    for listed in table.manifest_list().data_manifests() {
        let manifest_file = location.file_path_for_uri(&listed.manifest_path).unwrap();
        let bytes = location
            .store()
            .get(&location.data_file_key(&manifest_file))
            .unwrap()
            .unwrap();
        let manifest = Manifest::parse_avro(&bytes).unwrap();
        let mut next_row_id = listed.first_row_id.map(|first| first as i64);
        for entry in manifest.entries().iter().filter(|entry| entry.is_alive()) {
            let data = entry.data_file();
            let first_row_id = data.first_row_id().or(next_row_id);
            if data.first_row_id().is_none() {
                next_row_id = next_row_id.map(|next| next + data.record_count() as i64);
            }
            if !data.file_path().ends_with(&format!("/{path}")) {
                continue;
            }
            let sequence_number = entry.sequence_number().unwrap_or_else(|| {
                assert_eq!(entry.status(), ManifestStatus::Added);
                listed.sequence_number
            });
            return RecordedFile {
                sequence_number,
                first_row_id,
                manifest_path: listed.manifest_path.clone(),
            };
        }
    }
    panic!("{path} is live");
}

#[test]
fn a_new_table_is_its_stores_first_version_with_no_snapshot() {
    let (fixture, table) = create_table(&[]);

    let loaded = load(&fixture);

    assert_eq!(version(&fixture, &table), 1);
    assert_eq!(fixture.metadata_store.count_versions(), 1);
    assert_eq!(version(&fixture, &loaded), 1);
    assert_eq!(loaded.metadata().format_version(), FormatVersion::V3);
    assert_eq!(loaded.uuid(), table.uuid());
    assert!(loaded.metadata().current_snapshot().is_none());
    assert!(loaded.entries().unwrap().is_empty());
    assert_eq!(loaded.columns(), columns().as_slice());
}

#[test]
fn appends_commit_at_successive_versions_with_contiguous_row_ids() {
    let (fixture, created) = create_table(&[]);

    let after_a = append(&created, vec![entry("a", 100, 1, 100)]);
    let after_b = append(&after_a, vec![entry("b", 50, 101, 150)]);
    let loaded = load(&fixture);

    assert_eq!(
        (version(&fixture, &after_a), version(&fixture, &after_b)),
        (2, 3)
    );
    assert_eq!(paths(&loaded), ["a.parquet", "b.parquet"]);
    let (a, b) = (
        read_recorded_file(&loaded, "a.parquet"),
        read_recorded_file(&loaded, "b.parquet"),
    );
    assert_eq!((a.sequence_number, b.sequence_number), (1, 2));
    assert_eq!((a.first_row_id, b.first_row_id), (Some(0), Some(100)));
    assert_eq!(loaded.metadata().next_row_id(), 150);
    assert_eq!(loaded.metadata().snapshots().len(), 2);
}

#[test]
fn the_second_writer_from_one_base_loses_and_leaves_no_objects() {
    let (fixture, created) = create_table(&[]);
    let after_a = append(&created, vec![entry("a", 1, 1, 1)]);
    let objects_before = avro_objects(&fixture.location);

    let lost = created.commit(&[], &[], vec![entry("b", 1, 2, 2)]).unwrap();

    assert!(lost.is_none());
    assert_eq!(fixture.metadata_store.count_versions(), 2);
    assert_eq!(avro_objects(&fixture.location), objects_before);
    assert_eq!(paths(&after_a), ["a.parquet"]);
}

#[test]
fn a_committed_version_is_loaded_with_one_read_of_the_store() {
    let (fixture, _created) = create_table(&[]);
    let faults = Arc::new(Faults::default());
    let table = load_over_faulty_store(&fixture, &faults);
    faults.take_requests();

    let committed = table.commit(&[], &[], vec![entry("a", 1, 1, 1)]).unwrap();
    let requests = faults.take_requests();

    let committed = committed.expect("the commit succeeded");
    assert_eq!(version(&fixture, &committed), 2);
    // The manifest list of the snapshot the commit added.
    assert_eq!((requests.object_reads, requests.listings), (1, 0));
    assert_eq!(paths(&committed), ["a.parquet"]);
    assert_eq!(avro_objects(&fixture.location), 2);
}

#[test]
fn a_commit_whose_manifest_list_cannot_be_read_back_stands_with_its_objects() {
    use std::sync::atomic::Ordering;
    let (fixture, _created) = create_table(&[]);
    let faults = Arc::new(Faults::default());
    let table = load_over_faulty_store(&fixture, &faults);
    faults.fail_gets.store(true, Ordering::SeqCst);

    let unloaded = table.commit(&[], &[], vec![entry("a", 1, 1, 1)]).err();

    let error = unloaded.expect("the committed version could not be loaded");
    assert!(matches!(error, Error::CommittedTableUnloaded { .. }));
    assert!(error.may_have_committed());
    assert_eq!(fixture.metadata_store.count_versions(), 2);
    assert_eq!(avro_objects(&fixture.location), 2);
    assert_eq!(paths(&load(&fixture)), ["a.parquet"]);
}

#[test]
fn a_removal_retires_the_file_and_carries_the_rest_with_their_numbers() {
    let (fixture, created) = create_table(&[]);
    let table = append(
        &created,
        vec![entry("a", 10, 1, 10), entry("b", 10, 11, 20)],
    );
    let table = append(&table, vec![entry("c", 10, 21, 30)]);
    let b_before = read_recorded_file(&table, "b.parquet");

    let replaced = commit_replace(&table, &["a.parquet"], vec![entry("d", 10, 1, 10)])
        .unwrap()
        .unwrap();
    let loaded = load(&fixture);

    assert_eq!(paths(&replaced), ["b.parquet", "c.parquet", "d.parquet"]);
    assert_eq!(paths(&loaded), paths(&replaced));
    let b = read_recorded_file(&loaded, "b.parquet");
    assert_eq!(b.sequence_number, b_before.sequence_number);
    assert_eq!(b.first_row_id, b_before.first_row_id);
    assert_ne!(b.manifest_path, b_before.manifest_path);
    let snapshot = loaded.metadata().current_snapshot().unwrap();
    assert_eq!(snapshot.summary().operation, Operation::Replace);
    assert_eq!(
        snapshot.summary().additional_properties["deleted-data-files"],
        "1"
    );
    assert_eq!(count_manifests(&loaded), 3);
}

#[test]
fn refresh_sees_a_version_committed_from_another_copy() {
    let (fixture, created) = create_table(&[]);
    let reader = load(&fixture);
    let writer = append(&created, vec![entry("a", 1, 1, 1)]);

    let refreshed = reader.refresh().unwrap().unwrap();
    let unchanged = refreshed.refresh().unwrap();

    assert_eq!(refreshed.metadata_location(), writer.metadata_location());
    assert_eq!(paths(&refreshed), ["a.parquet"]);
    assert!(unchanged.is_none());
}

#[test]
fn a_partitioned_table_records_each_files_tuple() {
    let (fixture, created) = create_table(&["v".to_string()]);
    let mut partitioned = entry("a", 3, 7, 7);
    partitioned.partition = Some(PartitionValues::from([("v".to_string(), int_scalar(7))]));

    let table = append(&created, vec![partitioned]);
    let loaded = load(&fixture);

    assert_eq!(table.partition_by(), ["v"]);
    let tuple = live_entry(&loaded, "a.parquet").partition.unwrap();
    assert_eq!(scalar_equal(&tuple["v"], &int_scalar(7)), Some(true));
}

#[test]
fn a_replace_of_a_file_written_under_an_earlier_partition_spec_is_refused() {
    let (fixture, created) = create_table(&["v".to_string()]);
    append(&created, vec![partition_entry("by-value", 7, 0)]);
    let unpartitioned = repartition(&fixture, &[]);
    let merged = vec![entry("merged", 10, 7, 7)];

    let refused = commit_replace(&unpartitioned, &["by-value.parquet"], merged).err();

    assert!(matches!(
        refused,
        Some(Error::ReplacedFileOfAnotherPartitionSpec { spec_id, default_spec_id, .. })
            if spec_id == created.metadata().default_partition_spec_id()
                && default_spec_id == unpartitioned.metadata().default_partition_spec_id()
    ));
    assert_eq!(paths(&load(&fixture)), ["by-value.parquet"]);
}

#[test]
fn a_replace_of_a_file_its_partition_does_not_hold_is_refused() {
    let (fixture, created) = create_table(&[]);
    let table = append(&created, vec![entry("a", 1, 1, 1)]);
    let partition = find_partition_key(&table, "a.parquet");
    let removed = [
        ObjectPath::new("a.parquet"),
        ObjectPath::new("gone.parquet"),
    ];

    let refused = table
        .commit(&partition, &removed, vec![entry("b", 1, 1, 1)])
        .err();

    assert!(matches!(
        refused,
        Some(Error::RemovedFileNotLive { file, .. }) if file == "gone.parquet"
    ));
    assert_eq!(paths(&load(&fixture)), ["a.parquet"]);
}

#[test]
fn a_version_built_on_an_earlier_one_reads_only_the_manifests_that_joined() {
    let (fixture, created) = create_table(&[]);
    let first = append(&created, vec![entry("a", 10, 1, 10)]);
    paths(&first);
    let first_manifest = first.manifest_list().data_manifests().next().unwrap();
    let first_manifest = first_manifest.manifest_path.clone();
    let second = append(&first, vec![entry("b", 10, 11, 20)]);
    let location = &fixture.location;
    let manifest_key =
        location.data_file_key(&location.file_path_for_uri(&first_manifest).unwrap());
    location.store().delete(&manifest_key).unwrap();

    let files = paths(&second);

    assert_eq!(files, ["a.parquet", "b.parquet"]);
}

#[test]
fn a_scan_reports_a_column_nullable_only_while_a_live_file_records_a_null_in_it() {
    let (fixture, created) = create_table(&[]);
    let write = |name: &str, low: i64| fixture.write_data_file(entry(name, 10, low, low + 9), low);
    let mut holed = write("holed", 11);
    Arc::get_mut(holed.stats.as_mut().unwrap())
        .unwrap()
        .null_counts
        .insert("v".to_string(), 3);

    let dense = append(&created, vec![write("dense", 1)]);
    let with_holes = append(&dense, vec![holed]);
    let rewritten = commit_replace(&with_holes, &["holed.parquet"], vec![write("filled", 11)])
        .unwrap()
        .unwrap();

    let scan_nullability = |table: &Table| table.select_scan(&[]).unwrap().nullability;
    assert_eq!(scan_nullability(&created), [true]);
    assert_eq!(scan_nullability(&dense), [false]);
    assert_eq!(scan_nullability(&with_holes), [true]);
    assert_eq!(scan_nullability(&rewritten), [false]);
}

#[test]
fn commits_merging_manifests_read_the_carried_ones_from_the_store() {
    let (fixture, _created) = create_table(&[]);
    let faults = Arc::new(Faults::default());
    let mut table = load_over_faulty_store(&fixture, &faults);
    let appends = 100;
    faults.take_requests();

    for number in 0..appends {
        let low = number * 10;
        table = append(
            &table,
            vec![entry(&format!("f{number:03}"), 10, low, low + 9)],
        );
    }
    let requests = faults.take_requests();

    assert_eq!(count_manifests(&table), 1);
    assert_eq!(paths(&table).len(), appends as usize);
    // Each commit reads the manifest list it wrote and, but for the first,
    // the one it is built on; the one that merged every manifest into one
    // reads them all, the 99 it carried and the one it wrote for its own
    // file.
    assert_eq!(requests.object_reads, 3 * appends as usize - 1);
    assert_eq!(requests.listings, 0);
}

#[test]
fn a_rewrite_reads_every_data_manifest_of_the_table_from_the_store() {
    let (fixture, _created) = create_table(&[]);
    let faults = Arc::new(Faults::default());
    let mut table = load_over_faulty_store(&fixture, &faults);
    let manifests = 20;
    for number in 0..manifests {
        table = append(
            &table,
            vec![entry(&format!("f{number:02}"), 10, number, number)],
        );
    }
    let partition = find_partition_key(&table, "f00.parquet");
    let replaced = [ObjectPath::new("f00.parquet")];
    faults.take_requests();

    let rewritten = table.commit(&partition, &replaced, vec![entry("merged", 10, 0, 0)]);
    let requests = faults.take_requests();

    assert_eq!(count_manifests(&table), manifests as usize);
    assert!(rewritten.unwrap().is_some());
    // The manifest list the rewrite is built on, once to look for deletes
    // committed since and once for the manifests it lists; each of those
    // manifests, for the entry of the file replaced; and the manifest list
    // it wrote.
    assert_eq!(requests.object_reads, manifests as usize + 3);
    assert_eq!(requests.listings, 0);
}

#[test]
fn many_appends_leave_few_manifests_and_every_file() {
    let (fixture, created) = create_table(&[]);
    let mut table = created;
    let appends = 100;

    for number in 0..appends {
        let low = number * 10;
        table = append(
            &table,
            vec![entry(&format!("f{number:03}"), 10, low, low + 9)],
        );
    }
    let loaded = load(&fixture);

    assert_eq!(paths(&loaded).len(), appends as usize);
    assert_eq!(paths(&loaded), paths(&table));
    assert!(count_manifests(&loaded) < 10);
}

#[test]
fn a_table_without_a_metadata_location_is_refused_under_its_identifier() {
    let id_column = Column {
        name: "id".to_string(),
        col_type: Type::Int64,
    };
    let metadata = TableMetadataBuilder::new(
        build_schema(&[id_column]).unwrap(),
        UnboundPartitionSpec::builder().build(),
        SortOrder::unsorted_order(),
        "file:///t".to_string(),
        FormatVersion::V3,
        HashMap::new(),
    )
    .unwrap()
    .build()
    .unwrap()
    .metadata;
    let staged = block_on(async {
        SpecTable::builder()
            .metadata(metadata)
            .identifier(TableIdent::new(
                NamespaceIdent::new("db".to_string()),
                "t".to_string(),
            ))
            .file_io(FileIO::new_with_memory())
            .runtime(Runtime::try_current().unwrap())
            .build()
            .unwrap()
    });

    let refused = MetadataFile::try_from(&staged);

    assert!(matches!(
        refused,
        Err(Error::TableMetadataMissing { table }) if table == "db.t"
    ));
}
