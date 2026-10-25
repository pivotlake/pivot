//! End-to-end tests of the pivot-managed mode through SQL: a table created,
//! written and read through the catalog is an Iceberg v3 table in the
//! datastore's own directory, and it comes back after a reopen.

mod iceberg_support;

use arrow_array::{ArrayRef, Int64Array, Scalar};
use datastore_pivotlake::test_support::{
    commit_external_append, hinted_version, live_data_file_names, metadata_versions,
};
use iceberg_support::{
    Database, collect_i64s, collect_strings, names_in, parquet_files_in, single_i64,
};
use iceberg_tables::iceberg::spec::{FormatVersion, Manifest, ManifestList};
use parquet::file::reader::{FileReader, SerializedFileReader};
use planner::catalog::{BoundTable, SchemaQualifiedTableName};
use planner::expression::{Compare, CompareType, Expression, Ref};
use planner::types::Type;
use std::sync::Arc;

/// The metadata JSON of version `version` of the table at `table_path`.
fn metadata_json(table_path: &std::path::Path, version: u64) -> serde_json::Value {
    let bytes = std::fs::read(
        table_path
            .join("metadata")
            .join(format!("v{version:020}.metadata.json")),
    )
    .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[test]
fn a_created_and_filled_table_reads_back_and_survives_a_reopen() {
    let mut db = Database::new();
    db.execute("CREATE TABLE events (name VARCHAR, value BIGINT)");
    db.execute("INSERT INTO events VALUES ('a', 1), ('b', 2), ('c', 3)");

    let rows = db.execute("SELECT name, value FROM events ORDER BY value");
    let total = db.execute("SELECT CAST(sum(value) AS BIGINT) FROM events");

    assert_eq!(collect_strings(&rows, 0), ["a", "b", "c"]);
    assert_eq!(collect_i64s(&rows, 1), [1, 2, 3]);
    assert_eq!(single_i64(&total), 6);
    let directory = db.table_directory("events");
    assert_eq!(
        metadata_versions(&db.root_uri(), &directory).unwrap(),
        [1, 2]
    );
    assert_eq!(hinted_version(&db.root_uri(), &directory).unwrap(), Some(2));
    let metadata = metadata_json(&db.table_path("events"), 2);
    assert_eq!(metadata["format-version"], 3);
    assert_eq!(metadata["table-uuid"], directory);
    let fields = metadata["schemas"][0]["fields"].as_array().unwrap();
    assert!(fields.iter().all(|field| field["required"] == false));
    assert_eq!(fields[0]["type"], "string");
    assert_eq!(fields[1]["type"], "long");
    assert_eq!(metadata["snapshots"].as_array().unwrap().len(), 1);

    let mut reopened = Database::reopen(db.close());
    let rows = reopened.execute("SELECT value FROM events ORDER BY value");
    assert_eq!(collect_i64s(&rows, 0), [1, 2, 3]);
    assert_eq!(
        metadata_versions(&reopened.root_uri(), &directory).unwrap(),
        [1, 2]
    );
}

/// What the planner is told about each column of `name` holding NULLs.
fn bound_nullability(db: &Database, name: &SchemaQualifiedTableName) -> Vec<bool> {
    db.datastore
        .clone()
        .begin_transaction()
        .table(planner::DEFAULT_DATASTORE_NAME, name)
        .unwrap()
        .expect("the table exists")
        .nullability()
}

#[test]
fn a_column_binds_as_nullable_only_once_a_file_holds_a_null_in_it() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT, label VARCHAR)");
    db.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')");
    let name = SchemaQualifiedTableName::in_default_schema("t");

    let before_a_null = bound_nullability(&db, &name);
    db.execute("INSERT INTO t VALUES (3, NULL)");
    let after_a_null = bound_nullability(&db, &name);

    assert_eq!(before_a_null, [false, false]);
    assert_eq!(after_a_null, [false, true]);
}

#[test]
fn a_filter_on_a_columns_bounds_prunes_the_files_a_scan_reads() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT)");
    db.execute("INSERT INTO t VALUES (1), (2), (3)");
    db.execute("INSERT INTO t VALUES (100), (101), (102)");
    let table = db
        .datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap();
    let above_fifty = Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx: 0,
            return_type: Type::Int64,
            name: None,
        })),
        right: Box::new(Expression::Constant(Scalar::new(
            Arc::new(Int64Array::from(vec![50])) as ArrayRef,
        ))),
        compare_type: CompareType::Greater,
        return_type: Type::Boolean,
    });

    let whole = table.scan(&[]).unwrap();
    let filtered = table.scan(&[above_fifty]).unwrap();

    assert_eq!(whole.row_groups().len(), 2);
    assert_eq!(filtered.row_groups().len(), 1);
    assert_eq!(filtered.row_groups()[0].num_rows, 3);
}

#[test]
fn an_insert_writes_added_manifests_with_bounds_and_files_with_field_ids() {
    let mut db = Database::new();
    db.execute("CREATE TABLE events (name VARCHAR, value BIGINT)");

    db.execute("INSERT INTO events VALUES ('a', 1), ('b', 2), ('c', 3)");

    let metadata_dir = db.table_path("events").join("metadata");
    let names = names_in(&metadata_dir);
    let manifest_lists: Vec<_> = names.iter().filter(|n| n.starts_with("snap-")).collect();
    assert_eq!(manifest_lists.len(), 1, "{names:?}");
    let list_bytes = std::fs::read(metadata_dir.join(manifest_lists[0])).unwrap();
    let list = ManifestList::parse_with_version(&list_bytes, FormatVersion::V3).unwrap();
    let mut records = 0;
    let mut lowest = i64::MAX;
    let mut highest = i64::MIN;
    for manifest in list.entries() {
        assert!(manifest.first_row_id.is_some(), "row lineage is assigned");
        let name = manifest.manifest_path.rsplit('/').next().unwrap();
        let manifest =
            Manifest::parse_avro(&std::fs::read(metadata_dir.join(name)).unwrap()).unwrap();
        for entry in manifest.entries() {
            let file = entry.data_file();
            records += file.record_count();
            assert_eq!(file.null_value_counts()[&1], 0);
            assert_eq!(file.null_value_counts()[&2], 0);
            let lower = file.lower_bounds()[&2]
                .to_human_string()
                .parse::<i64>()
                .unwrap();
            let upper = file.upper_bounds()[&2]
                .to_human_string()
                .parse::<i64>()
                .unwrap();
            lowest = lowest.min(lower);
            highest = highest.max(upper);
            assert!(
                file.file_path().starts_with("file:///"),
                "{}",
                file.file_path()
            );
        }
    }
    assert_eq!((records, lowest, highest), (3, 1, 3));

    for file in parquet_files_in(&db.table_path("events")) {
        let reader = SerializedFileReader::new(
            std::fs::File::open(db.table_path("events").join(file)).unwrap(),
        )
        .unwrap();
        let fields = reader
            .metadata()
            .file_metadata()
            .schema_descr()
            .root_schema()
            .get_fields();
        let ids: Vec<i32> = fields
            .iter()
            .map(|field| field.get_basic_info().id())
            .collect();
        assert_eq!(ids, [1, 2]);
    }
}

#[test]
fn a_refresh_sees_a_version_committed_from_outside() {
    let mut db = Database::new();
    db.execute("CREATE TABLE events (name VARCHAR, value BIGINT)");
    db.execute("INSERT INTO events VALUES ('a', 1)");
    let written = parquet_files_in(&db.table_path("events"));
    let bytes = std::fs::read(db.table_path("events").join(&written[0])).unwrap();
    let directory = db.table_directory("events");

    let version =
        commit_external_append(&db.root_uri(), &directory, "external.parquet", &bytes, 1).unwrap();
    let changed = db.datastore.refresh_from_store().unwrap();
    let count = db.execute("SELECT count(*) FROM events");

    assert_eq!(version, 3);
    assert!(changed);
    assert_eq!(single_i64(&count), 2);
    let mut expected = vec![written[0].clone(), "external.parquet".to_string()];
    expected.sort();
    assert_eq!(
        live_data_file_names(&db.root_uri(), &directory).unwrap(),
        expected
    );
}

#[test]
fn dropping_a_table_removes_it_from_the_catalog_but_keeps_its_storage() {
    let mut db = Database::new();
    db.execute("CREATE TABLE events (value BIGINT)");
    db.execute("INSERT INTO events VALUES (1)");
    let table_path = db.table_path("events");

    db.execute("DROP TABLE events");

    let error = db.try_execute("SELECT * FROM events").unwrap_err();
    assert!(error.contains("events"), "{error}");
    assert_eq!(parquet_files_in(&table_path).len(), 1);
    assert!(table_path.join("metadata").is_dir());
}

#[test]
fn a_reopen_restores_partitioning_sorting_and_logical_types() {
    let mut db = Database::new();
    db.execute(
        "CREATE TABLE t (day DATE, id UBIGINT, tiny TINYINT, n BIGINT) \
         WITH (partition_by = 'day', sort_by = 'n')",
    );
    db.execute("INSERT INTO t VALUES (DATE '2026-01-01', 7, 1, 10), (DATE '2026-01-02', 8, 2, 20)");
    let metadata = metadata_json(&db.table_path("t"), 2);

    let mut reopened = Database::reopen(db.close());
    let table = reopened
        .datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap();
    let rows = reopened.execute("SELECT n FROM t WHERE day = DATE '2026-01-02' ORDER BY n");

    assert_eq!(table.partition_by(), ["day"]);
    assert_eq!(table.sort_by(), ["n"]);
    let types: Vec<Type> = table.columns().into_iter().map(|c| c.col_type).collect();
    assert_eq!(types, [Type::Date, Type::UInt64, Type::Int8, Type::Int64]);
    assert_eq!(collect_i64s(&rows, 0), [20]);
    assert_eq!(
        metadata["properties"]["pivot.logical-types"],
        r#"{"2":"UInt64","3":"Int8"}"#
    );
    assert_eq!(metadata["partition-specs"][0]["fields"][0]["name"], "day");
    assert_eq!(metadata["sort-orders"][0]["fields"][0]["source-id"], 4);
}
