//! A managed table another tool repartitioned: the files written before
//! keep the partition spec they were written under, so each file's
//! partition values are read by its own spec's fields, compaction leaves
//! the files of an earlier spec as they are, and a commit that replaces
//! one of them is refused.

mod iceberg_support;

use arrow_array::Datum as _;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use datastore_pivotlake::test_support::{
    commit_external_replace, live_data_file_names, live_data_file_partitions, metadata_versions,
};
use iceberg_support::{Database, collect_i64s, int_parquet_bytes};
use iceberg_tables::iceberg::spec::{TableMetadata, Transform, UnboundPartitionSpec};

/// Make the identity of `column` the partition spec the table `name` of
/// `db` is written under from now on, by publishing its next metadata
/// version the way another tool alters a table, and bring the datastore to
/// that version.
fn repartition_by(db: &Database, name: &str, column: &str) {
    let metadata_dir = db.table_path(name).join("metadata");
    let versions = metadata_versions(&db.root_uri(), &db.table_directory(name)).unwrap();
    let current_version = *versions.last().unwrap();
    let current_file = metadata_dir.join(format!("v{current_version:020}.metadata.json"));
    let current: TableMetadata =
        serde_json::from_slice(&std::fs::read(&current_file).unwrap()).unwrap();
    let source_id = current.current_schema().field_id_by_name(column).unwrap();
    let by_column = UnboundPartitionSpec::builder()
        .add_partition_field(source_id, column, Transform::Identity)
        .unwrap()
        .build();
    let repartitioned = current
        .into_builder(Some(format!("file://{}", current_file.display())))
        .add_partition_spec(by_column)
        .unwrap()
        .set_default_partition_spec(-1)
        .unwrap()
        .build()
        .unwrap()
        .metadata;
    let next_file = metadata_dir.join(format!("v{:020}.metadata.json", current_version + 1));
    std::fs::write(next_file, serde_json::to_vec(&repartitioned).unwrap()).unwrap();
    db.datastore.refresh_from_store().unwrap();
}

/// The partition values of every live file of the table `name` of `db`,
/// each as its field's name and its `BIGINT` value, sorted.
fn list_partition_values(db: &Database, name: &str) -> Vec<Vec<(String, i64)>> {
    let partitions = live_data_file_partitions(&db.root_uri(), &db.table_directory(name));
    let mut tuples: Vec<Vec<(String, i64)>> = partitions
        .unwrap()
        .into_iter()
        .map(|(_, partition)| {
            let mut values: Vec<(String, i64)> = partition
                .expect("a partitioned table's file has a tuple")
                .iter()
                .map(|(field, value)| {
                    let (value, _) = value.get();
                    (field.clone(), value.as_primitive::<Int64Type>().value(0))
                })
                .collect();
            values.sort();
            values
        })
        .collect();
    tuples.sort();
    tuples
}

#[test]
fn a_file_written_before_a_repartition_keeps_the_partition_values_it_was_written_with() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (ts BIGINT, p BIGINT) PARTITIONED BY (p)");
    db.execute("INSERT INTO t VALUES (10, 1)");
    repartition_by(&db, "t", "ts");
    db.execute("INSERT INTO t VALUES (20, 2)");

    let tuples = list_partition_values(&db, "t");

    assert_eq!(
        tuples,
        [vec![("p".to_string(), 1)], vec![("ts".to_string(), 20)]]
    );
}

#[test]
fn compaction_leaves_the_files_of_an_earlier_partition_spec_as_they_are() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (ts BIGINT, p BIGINT) PARTITIONED BY (p)");
    db.execute("INSERT INTO t VALUES (10, 1)");
    db.execute("INSERT INTO t VALUES (11, 1)");
    let directory = db.table_directory("t");
    let written_by_p = live_data_file_names(&db.root_uri(), &directory).unwrap();
    repartition_by(&db, "t", "ts");
    db.execute("INSERT INTO t VALUES (20, 2)");
    db.execute("INSERT INTO t VALUES (20, 3)");

    db.compact("t", true);

    let live = live_data_file_names(&db.root_uri(), &directory).unwrap();
    assert_eq!(live.len(), 3, "{live:?}");
    assert!(written_by_p.iter().all(|name| live.contains(name)));
    let rows = db.execute("SELECT ts, p FROM t ORDER BY ts, p");
    assert_eq!(collect_i64s(&rows, 0), [10, 11, 20, 20]);
    assert_eq!(collect_i64s(&rows, 1), [1, 1, 2, 3]);
}

#[test]
fn a_commit_that_replaces_a_file_of_an_earlier_partition_spec_is_refused() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (ts BIGINT, p BIGINT) PARTITIONED BY (p)");
    db.execute("INSERT INTO t VALUES (10, 1)");
    let directory = db.table_directory("t");
    let written_by_p = live_data_file_names(&db.root_uri(), &directory).unwrap();
    repartition_by(&db, "t", "ts");

    let refusal = commit_external_replace(
        &db.root_uri(),
        &directory,
        &[written_by_p[0].as_str()],
        "replacement.parquet",
        &int_parquet_bytes(&[10]),
        1,
    )
    .unwrap_err();

    let refusal = refusal.to_string();
    assert!(
        refusal.contains("a file of another partition spec is not replaced"),
        "{refusal}"
    );
    let live = live_data_file_names(&db.root_uri(), &directory).unwrap();
    assert_eq!(live, written_by_p);
}
