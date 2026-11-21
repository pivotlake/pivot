//! End-to-end tests of the pivot-managed mode through SQL: a table created,
//! written and read through the catalog is an Iceberg v3 table in the
//! datastore's own directory, and it comes back after a reopen.

mod iceberg_support;

use arrow_array::{ArrayRef, Int64Array, Scalar, StringViewArray};
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
    assert_eq!(hinted_version(&db.root_uri(), &directory).unwrap(), None);
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

/// The filter `column 0 = value`, over a `BIGINT` column.
fn first_column_equals(value: i64) -> Expression {
    Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx: 0,
            return_type: Type::Int64,
            name: None,
        })),
        right: Box::new(Expression::Constant(Scalar::new(
            Arc::new(Int64Array::from(vec![value])) as ArrayRef,
        ))),
        compare_type: CompareType::Equal,
        return_type: Type::Boolean,
    })
}

#[test]
fn a_filter_that_keeps_only_partitions_without_nulls_binds_the_column_as_not_nullable() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (part BIGINT, label VARCHAR) PARTITIONED BY (part)");
    db.execute("INSERT INTO t VALUES (1, 'a')");
    db.execute("INSERT INTO t VALUES (2, NULL)");
    let name = SchemaQualifiedTableName::in_default_schema("t");
    let mut filtered = db
        .datastore
        .clone()
        .begin_transaction()
        .table(planner::DEFAULT_DATASTORE_NAME, &name)
        .unwrap()
        .expect("the table exists");
    filtered.pushdown_filter(first_column_equals(1)).unwrap();

    let nullability_of_the_partition = filtered.nullability();
    let nullability_of_the_table = bound_nullability(&db, &name);

    assert_eq!(nullability_of_the_partition, [false, false]);
    assert_eq!(nullability_of_the_table, [false, true]);
}

#[test]
fn binding_a_reopened_table_reads_none_of_its_data_files() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT, label VARCHAR)");
    db.execute("INSERT INTO t VALUES (1, 'a'), (2, NULL)");
    let name = SchemaQualifiedTableName::in_default_schema("t");
    let reopened = Database::reopen(db.close());
    let table_path = reopened.table_path("t");
    for file in parquet_files_in(&table_path) {
        std::fs::remove_file(table_path.join(file)).unwrap();
    }

    let bound = reopened
        .datastore
        .clone()
        .begin_transaction()
        .table(planner::DEFAULT_DATASTORE_NAME, &name)
        .unwrap()
        .expect("the table exists");

    assert_eq!(bound.estimate_row_count(), Some(2));
}

#[test]
fn a_scan_whose_files_cannot_be_read_binds_every_column_nullable_and_fails_the_query() {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (value BIGINT, label VARCHAR)");
    db.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')");
    let name = SchemaQualifiedTableName::in_default_schema("t");
    let table_path = db.table_path("t");
    for file in parquet_files_in(&table_path) {
        std::fs::remove_file(table_path.join(file)).unwrap();
    }
    let mut reopened = Database::reopen(db.close());

    let nullability = bound_nullability(&reopened, &name);
    let error = reopened
        .try_execute("SELECT count(*) FROM t a JOIN t b ON a.value = b.value")
        .unwrap_err();

    assert_eq!(nullability, [true, true]);
    assert!(error.contains("compiling table scan"), "{error}");
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

    let whole = table.select_scan(&[]).unwrap();
    let filtered = table.select_scan(&[above_fifty]).unwrap();

    assert_eq!(whole.row_groups().len(), 2);
    assert_eq!(filtered.row_groups().len(), 1);
    assert_eq!(filtered.row_groups()[0].num_rows, 3);
}

#[test]
fn a_filter_on_a_long_sort_string_prunes_a_file_whose_strings_share_its_prefix() {
    let mut db = Database::new();
    db.execute("CREATE TABLE pages (url VARCHAR) SORTED BY (url)");
    db.execute("INSERT INTO pages VALUES ('https://example.com/pages/a')");
    db.execute(
        "INSERT INTO pages VALUES \
         ('https://example.com/pages/b'), ('https://example.com/pages/c')",
    );
    let table = db
        .datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("pages"))
        .unwrap();
    let is_last_page = Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx: 0,
            return_type: Type::Utf8,
            name: None,
        })),
        right: Box::new(Expression::Constant(Scalar::new(
            Arc::new(StringViewArray::from(vec!["https://example.com/pages/c"])) as ArrayRef,
        ))),
        compare_type: CompareType::Equal,
        return_type: Type::Boolean,
    });

    let filtered = table.select_scan(&[is_last_page]).unwrap();

    assert_eq!(filtered.row_groups().len(), 1);
    assert_eq!(filtered.row_groups()[0].num_rows, 2);
}

#[test]
fn a_filter_on_a_sort_column_of_long_strings_returns_exactly_the_matching_rows() {
    let mut db = Database::new();
    db.execute("CREATE TABLE pages (url VARCHAR, visits BIGINT) SORTED BY (url)");
    db.execute(
        "INSERT INTO pages VALUES \
         ('https://example.com/pages/a', 1), ('https://example.com/pages/b', 2)",
    );
    db.execute(
        "INSERT INTO pages VALUES \
         ('https://example.com/pages/c', 3), ('https://example.com/pages/d', 4)",
    );
    let select_visits =
        |filter: &str| format!("SELECT visits FROM pages WHERE url {filter} ORDER BY visits");

    let equal = db.execute(&select_visits("= 'https://example.com/pages/c'"));
    let above = db.execute(&select_visits("> 'https://example.com/pages/b'"));
    let up_to = db.execute(&select_visits("<= 'https://example.com/pages/b'"));
    let absent = db.execute(&select_visits("= 'https://example.com/pages/bb'"));

    assert_eq!(collect_i64s(&equal, 0), [3]);
    assert_eq!(collect_i64s(&above, 0), [3, 4]);
    assert_eq!(collect_i64s(&up_to, 0), [1, 2]);
    assert!(collect_i64s(&absent, 0).is_empty());
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

/// The manifests each manifest list in `metadata_dir` names, by file name.
/// Every manifest list there is a snapshot's.
fn read_manifest_lists(metadata_dir: &std::path::Path) -> Vec<Vec<String>> {
    names_in(metadata_dir)
        .iter()
        .filter(|name| name.starts_with("snap-"))
        .map(|name| {
            let bytes = std::fs::read(metadata_dir.join(name)).unwrap();
            let list = ManifestList::parse_with_version(&bytes, FormatVersion::V3).unwrap();
            list.entries()
                .iter()
                .map(|manifest| {
                    manifest
                        .manifest_path
                        .rsplit('/')
                        .next()
                        .unwrap()
                        .to_string()
                })
                .collect()
        })
        .collect()
}

#[test]
fn inserts_whose_commits_merge_manifests_leave_no_manifest_that_no_snapshot_names() {
    let mut db = Database::new();
    db.execute("CREATE TABLE events (value BIGINT)");
    let inserts = 120;

    for value in 0..inserts {
        db.execute(&format!("INSERT INTO events VALUES ({value})"));
    }

    let metadata_dir = db.table_path("events").join("metadata");
    let manifest_lists = read_manifest_lists(&metadata_dir);
    let unnamed_manifests: Vec<String> = names_in(&metadata_dir)
        .into_iter()
        .filter(|name| name.ends_with(".avro") && !name.starts_with("snap-"))
        .filter(|name| !manifest_lists.iter().flatten().any(|named| named == name))
        .collect();
    // A commit merged manifests: no snapshot lists one for every insert.
    assert_eq!(manifest_lists.len(), inserts);
    assert!(manifest_lists.iter().all(|named| named.len() < inserts));
    assert_eq!(unnamed_manifests, Vec::<String>::new());
    let count = db.execute("SELECT count(*) FROM events");
    assert_eq!(single_i64(&count), inserts as i64);
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
         SORTED BY (n) PARTITIONED BY (day)",
    );
    db.execute("INSERT INTO t VALUES (DATE '2026-01-01', 7, 1, 10), (DATE '2026-01-02', 8, 2, 20)");
    let metadata = metadata_json(&db.table_path("t"), 2);

    let mut reopened = Database::reopen(db.close());
    let table = reopened
        .datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap();
    let rows = reopened.execute("SELECT n FROM t WHERE day = DATE '2026-01-02' ORDER BY n");

    assert_eq!(reopened.describe_partition_by("t"), ["day"]);
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

#[test]
fn a_sort_by_option_is_refused_as_an_unknown_option() {
    let mut db = Database::new();

    let error = db
        .try_execute("CREATE TABLE t (n BIGINT) WITH (sort_by = 'n')")
        .unwrap_err();

    assert!(error.contains("`sort_by` is not a table option"), "{error}");
}

#[test]
fn a_partition_by_option_is_refused_as_an_unknown_option() {
    let mut db = Database::new();

    let error = db
        .try_execute("CREATE TABLE t (n BIGINT) WITH (partition_by = 'n')")
        .unwrap_err();

    assert!(
        error.contains("`partition_by` is not a table option"),
        "{error}"
    );
}

/// The leaves the fields of a variant column were shredded into, by their
/// path in the schema of the one data file under `table_path`.
fn list_shredded_leaves(table_path: &std::path::Path) -> Vec<String> {
    let file = parquet_files_in(table_path).remove(0);
    let reader =
        SerializedFileReader::new(std::fs::File::open(table_path.join(file)).unwrap()).unwrap();
    let schema = reader.metadata().file_metadata().schema_descr();
    schema
        .columns()
        .iter()
        .map(|leaf| leaf.path().string())
        .filter(|path| path.ends_with("typed_value"))
        .collect()
}

#[test]
fn a_variant_column_is_declared_as_a_variant_field_and_reads_back_after_a_reopen() {
    let mut db = Database::new();
    db.execute("CREATE TABLE events (id BIGINT, doc VARIANT)");
    db.execute(
        r#"INSERT INTO events VALUES (1, '{"a": 1, "b": "x"}'), (2, '{"a": 2, "b": "y"}'), (3, NULL)"#,
    );

    let mut reopened = Database::reopen(db.close());
    let documents = reopened.execute(
        "SELECT CAST(doc AS VARCHAR), CAST(doc.a AS BIGINT) FROM events \
         WHERE doc IS NOT NULL ORDER BY id",
    );
    let without_a_document = reopened.execute("SELECT id FROM events WHERE doc IS NULL");

    assert_eq!(
        collect_strings(&documents, 0),
        [r#"{"a":1,"b":"x"}"#, r#"{"a":2,"b":"y"}"#]
    );
    assert_eq!(collect_i64s(&documents, 1), [1, 2]);
    assert_eq!(collect_i64s(&without_a_document, 0), [3]);
    let table_path = reopened.table_path("events");
    assert_eq!(
        list_shredded_leaves(&table_path),
        [
            "doc.typed_value.a.typed_value",
            "doc.typed_value.b.typed_value"
        ]
    );
    let metadata = metadata_json(&table_path, 2);
    assert_eq!(
        metadata["schemas"][0]["fields"][1],
        serde_json::json!({"id": 2, "name": "doc", "required": false, "type": "variant"})
    );
    assert_eq!(metadata["last-column-id"], 2);
    assert!(metadata["properties"].get("pivot.logical-types").is_none());
}
