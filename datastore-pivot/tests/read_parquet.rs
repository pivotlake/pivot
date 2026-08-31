mod common;

use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, OnceLock};

use arrow_array::{Int32Array, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use catalog::datastore::{DatastoreTableMetadata, DatastoreTransaction};
use catalog::{Datastore, PivotCatalog};
use datastore_pivot::test_support::{self, Backend};
use dispatch::{DataFlowDispatcher, Dispatch};
use object_storage::{AmbientExternalStoreFactory, ObjectPath};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use planner::catalog::{BoundTable, SchemaQualifiedTableName, TableRevision};
use planner::{DEFAULT_DATASTORE_NAME, Operator, PlanNode, Planner};
use tempfile::TempDir;

fn dispatcher() -> DataFlowDispatcher {
    static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
    DISPATCH
        .get_or_init(|| Dispatch::spin_up(1, 64, None))
        .dispatcher()
        .clone()
}

#[derive(Debug)]
struct EmptyDatastore;

impl Datastore for EmptyDatastore {
    fn begin_transaction(self: Arc<Self>) -> Arc<dyn DatastoreTransaction> {
        Arc::new(EmptyTransaction)
    }

    fn kind(&self) -> &'static str {
        "empty"
    }

    fn data_path(&self) -> String {
        String::new()
    }
}

#[derive(Debug)]
struct EmptyTransaction;

impl DatastoreTransaction for EmptyTransaction {
    fn does_schema_exist(&self, schema: &str) -> planner::catalog::Result<bool> {
        Ok(schema == planner::DEFAULT_SCHEMA_NAME)
    }

    fn bind_table(
        &self,
        _datastore: &str,
        _name: &SchemaQualifiedTableName,
    ) -> planner::catalog::Result<Option<Box<dyn BoundTable>>> {
        Ok(None)
    }

    fn table_revision(
        &self,
        _name: &SchemaQualifiedTableName,
    ) -> planner::catalog::Result<Option<TableRevision>> {
        Ok(None)
    }

    fn tables(&self) -> planner::catalog::Result<Vec<DatastoreTableMetadata>> {
        Ok(Vec::new())
    }
}

fn catalog() -> Arc<PivotCatalog> {
    Arc::new(
        PivotCatalog::new(
            HashMap::from([(
                DEFAULT_DATASTORE_NAME.to_string(),
                Arc::new(EmptyDatastore) as Arc<dyn Datastore>,
            )]),
            DEFAULT_DATASTORE_NAME.to_string(),
            common::trust_metastore(),
        )
        .unwrap()
        .with_external_parquet_read_context(&dispatcher(), Arc::new(AmbientExternalStoreFactory)),
    )
}

fn planner() -> Planner {
    Planner::from_datastore_names(
        vec![DEFAULT_DATASTORE_NAME.to_string()],
        DEFAULT_DATASTORE_NAME.to_string(),
    )
    .unwrap()
}

fn write_names_and_values(path: &std::path::Path, names: &[&str], values: &[i64]) {
    write_names_and_values_with_row_groups(path, names, values, None);
}

fn names_and_values_parquet(names: &[&str], values: &[i64]) -> Vec<u8> {
    let batch = common::strings_and_ints(names, values);
    let properties = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut bytes = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut bytes, batch.schema(), Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    bytes
}

fn write_names_and_values_with_row_groups(
    path: &std::path::Path,
    names: &[&str],
    values: &[i64],
    max_row_group_rows: Option<usize>,
) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let batch = common::strings_and_ints(names, values);
    let mut properties = WriterProperties::builder().set_compression(Compression::SNAPPY);
    if let Some(max_row_group_rows) = max_row_group_rows {
        properties = properties.set_max_row_group_row_count(Some(max_row_group_rows));
    }
    let mut writer = ArrowWriter::try_new(
        File::create(path).unwrap(),
        batch.schema(),
        Some(properties.build()),
    )
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_gzip_names_and_values(path: &std::path::Path, names: &[&str], values: &[i64]) {
    let batch = common::strings_and_ints(names, values);
    let properties = WriterProperties::builder()
        .set_compression(Compression::GZIP(Default::default()))
        .set_max_row_group_row_count(Some(1))
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(path).unwrap(),
        batch.schema(),
        Some(properties),
    )
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_incompatible_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "different",
        DataType::Int32,
        false,
    )]));
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
    let properties = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer =
        ArrowWriter::try_new(File::create(path).unwrap(), schema, Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_file_without_row_groups(path: &std::path::Path) {
    let schema = common::strings_and_ints(&[], &[]).schema();
    let writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.close().unwrap();
}

fn quote_sql_string(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn run_sql(sql: &str) -> Vec<RecordBatch> {
    let catalog = catalog();
    let transaction = catalog.begin_transaction();
    let mut planner = planner();
    planner
        .plan(sql, transaction.clone())
        .unwrap()
        .compile(&dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap()
}

fn plan_error(sql: &str) -> String {
    let catalog = catalog();
    let transaction = catalog.begin_transaction();
    let mut planner = planner();
    planner.plan(sql, transaction).unwrap_err().to_string()
}

fn first_int64(batches: &[RecordBatch]) -> i64 {
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0)
}

fn contains_materialize(node: &PlanNode) -> bool {
    matches!(node.operator, Operator::Materialize(_))
        || node.inputs.iter().any(contains_materialize)
}

#[test]
fn reads_one_external_parquet_file() {
    let external = TempDir::new().unwrap();
    let path = external.path().join("events.parquet");
    write_names_and_values(&path, &["a", "b"], &[2, 1]);
    let sql = format!(
        "SELECT name, value FROM read_parquet({}) ORDER BY value",
        quote_sql_string(&path.to_string_lossy())
    );

    let batches = run_sql(&sql);

    let names: Vec<&str> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap()
                .iter()
                .map(Option::unwrap)
        })
        .collect();
    let values: Vec<i64> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
        })
        .copied()
        .collect();
    assert_eq!(names, ["b", "a"]);
    assert_eq!(values, [1, 2]);
}

#[test]
fn reads_a_file_uri() {
    let external = TempDir::new().unwrap();
    let path = external.path().join("events.parquet");
    write_names_and_values(&path, &["a", "b"], &[1, 2]);
    let location = format!("file://{}", path.display());
    let sql = format!(
        "SELECT COUNT(*) FROM read_parquet({})",
        quote_sql_string(&location)
    );

    let batches = run_sql(&sql);

    assert_eq!(first_int64(&batches), 2);
}

#[test]
fn pushed_filter_prunes_every_row_group_and_preserves_the_schema() {
    let external = TempDir::new().unwrap();
    let path = external.path().join("events.parquet");
    write_gzip_names_and_values(&path, &["a", "b", "c"], &[1, 2, 3]);
    let sql = format!(
        "SELECT name, value FROM read_parquet({}) WHERE value = 999",
        quote_sql_string(&path.to_string_lossy())
    );
    let catalog = catalog();
    let transaction = catalog.begin_transaction();
    let mut planner = planner();

    let plan = planner.plan(&sql, transaction.clone()).unwrap();
    let output_names = plan.output_names.clone();
    let output_types = plan.root.output_types().unwrap();
    let batches = plan
        .compile(&dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap();

    // GZIP data pages are unsupported by Pivot. Successful execution proves
    // all three row groups were removed from their min/max statistics before
    // the scan tried to decode them.
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
    assert_eq!(output_names, ["name", "value"]);
    assert_eq!(
        output_types,
        [planner::types::Type::Utf8, planner::types::Type::Int64]
    );
}

#[test]
fn range_filter_prunes_every_row_group_and_preserves_the_schema() {
    let external = TempDir::new().unwrap();
    let path = external.path().join("events.parquet");
    write_gzip_names_and_values(&path, &["a", "b", "c"], &[501, 502, 503]);
    let sql = format!(
        "SELECT name, value FROM read_parquet({}) WHERE value < 500",
        quote_sql_string(&path.to_string_lossy())
    );
    let catalog = catalog();
    let transaction = catalog.begin_transaction();
    let mut planner = planner();

    let plan = planner.plan(&sql, transaction.clone()).unwrap();
    let output_names = plan.output_names.clone();
    let output_types = plan.root.output_types().unwrap();
    let batches = plan
        .compile(&dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap();

    // GZIP data pages are unsupported by Pivot. Successful execution proves
    // the less-than predicate removed all three row groups from their min/max
    // statistics before the scan tried to decode them.
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
    assert_eq!(output_names, ["name", "value"]);
    assert_eq!(
        output_types,
        [planner::types::Type::Utf8, planner::types::Type::Int64]
    );
}

#[test]
fn late_materializes_an_external_parquet_scan() {
    let external = TempDir::new().unwrap();
    let path = external.path().join("events.parquet");
    write_names_and_values_with_row_groups(&path, &["three", "one", "two"], &[3, 1, 2], Some(1));
    let sql = format!(
        "SELECT name FROM read_parquet({}) WHERE value > 1 ORDER BY value LIMIT 1",
        quote_sql_string(&path.to_string_lossy())
    );
    let catalog = catalog();
    let transaction = catalog.begin_transaction();
    let mut planner = planner();

    let plan = planner.plan(&sql, transaction.clone()).unwrap();
    assert!(
        contains_materialize(&plan.root),
        "expected read_parquet Top-N to use late materialization: {}",
        plan.root
    );
    let batches = plan
        .compile(&dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap();

    let names: Vec<&str> = batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap()
                .iter()
                .map(Option::unwrap)
        })
        .collect();
    assert_eq!(names, ["two"]);
}

#[test]
fn star_reads_matching_files_in_one_directory() {
    let external = TempDir::new().unwrap();
    write_names_and_values(&external.path().join("part-1.parquet"), &["a"], &[1]);
    write_names_and_values(
        &external.path().join("part-2.parquet"),
        &["b", "c"],
        &[2, 3],
    );
    write_names_and_values(&external.path().join("ignored.data"), &["x"], &[100]);
    let pattern = external.path().join("part-*.parquet");
    let sql = format!(
        "SELECT COUNT(*), CAST(SUM(value) AS BIGINT) FROM read_parquet({})",
        quote_sql_string(&pattern.to_string_lossy())
    );

    let batches = run_sql(&sql);

    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    let sum = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!((count, sum), (3, 6));
}

#[test]
fn no_matches_fail_during_binding() {
    let external = TempDir::new().unwrap();
    let pattern = external.path().join("*.parquet");
    let sql = format!(
        "SELECT * FROM read_parquet({})",
        quote_sql_string(&pattern.to_string_lossy())
    );

    let error = plan_error(&sql);

    assert!(error.contains("no files found matching"), "{error}");
}

#[test]
fn file_without_row_groups_fails_during_binding() {
    let external = TempDir::new().unwrap();
    let path = external.path().join("empty.parquet");
    write_file_without_row_groups(&path);
    let sql = format!(
        "SELECT * FROM read_parquet({})",
        quote_sql_string(&path.to_string_lossy())
    );

    let error = plan_error(&sql);

    assert!(error.contains("contains no row groups"), "{error}");
}

#[test]
fn stars_match_individual_directory_segments() {
    let external = TempDir::new().unwrap();
    let events = external.path().join("events");
    write_names_and_values(
        &events.join("2025").join("hello").join("part-1.parquet"),
        &["a"],
        &[1],
    );
    write_names_and_values(
        &events.join("2026").join("hello").join("part-2.parquet"),
        &["b"],
        &[2],
    );
    write_names_and_values(
        &events.join("2026").join("goodbye").join("ignored.parquet"),
        &["ignored"],
        &[100],
    );
    write_names_and_values(
        &events
            .join("2026")
            .join("hello")
            .join("nested")
            .join("ignored.parquet"),
        &["ignored"],
        &[100],
    );
    let pattern = events.join("*").join("hello").join("*.parquet");
    let sql = format!(
        "SELECT COUNT(*), CAST(SUM(value) AS BIGINT) FROM read_parquet({})",
        quote_sql_string(&pattern.to_string_lossy())
    );

    let batches = run_sql(&sql);
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    let sum = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);

    assert_eq!((count, sum), (2, 3));
}

fn remote_stars_match_individual_directory_segments(backend: &Backend) {
    for (path, names, values) in [
        ("events/2025/hello/part-1.parquet", &["a"][..], &[1][..]),
        ("events/2026/hello/part-2.parquet", &["b"][..], &[2][..]),
        (
            "events/2026/goodbye/ignored.parquet",
            &["ignored"][..],
            &[100][..],
        ),
        (
            "events/2026/hello/nested/ignored.parquet",
            &["ignored"][..],
            &[100][..],
        ),
    ] {
        backend
            .store
            .put(
                &ObjectPath::new(path),
                &names_and_values_parquet(names, values),
            )
            .unwrap();
    }
    let pattern = format!("{}/events/*/hello/*.parquet", backend.root);
    let sql = format!(
        "SELECT COUNT(*), CAST(SUM(value) AS BIGINT) FROM read_parquet({})",
        quote_sql_string(&pattern)
    );

    let batches = run_sql(&sql);
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    let sum = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);

    assert_eq!((count, sum), (2, 3));
}

#[test]
fn s3_stars_match_individual_directory_segments() {
    let Some(backend) = test_support::s3("read-parquet-nested-wildcard-s3") else {
        return;
    };
    remote_stars_match_individual_directory_segments(&backend);
}

#[test]
fn gcs_stars_match_individual_directory_segments() {
    let Some(backend) = test_support::gcs("read-parquet-nested-wildcard-gcs") else {
        return;
    };
    remote_stars_match_individual_directory_segments(&backend);
}

#[test]
fn matched_files_with_different_schemas_fail_during_binding() {
    let external = TempDir::new().unwrap();
    write_names_and_values(&external.path().join("a.parquet"), &["a"], &[1]);
    write_incompatible_file(&external.path().join("b.parquet"));
    let pattern = external.path().join("*.parquet");
    let sql = format!(
        "SELECT * FROM read_parquet({})",
        quote_sql_string(&pattern.to_string_lossy())
    );

    let error = plan_error(&sql);

    assert!(
        error.contains("has schema") && error.contains("expected"),
        "{error}"
    );
}

#[test]
fn binding_captures_the_matching_file_set() {
    let external = TempDir::new().unwrap();
    write_names_and_values(&external.path().join("part-1.parquet"), &["a"], &[1]);
    let pattern = external.path().join("part-*.parquet");
    let sql = format!(
        "SELECT COUNT(*) FROM read_parquet({})",
        quote_sql_string(&pattern.to_string_lossy())
    );
    let catalog = catalog();
    let transaction = catalog.begin_transaction();
    let mut planner = planner();

    let plan = planner.plan(&sql, transaction.clone()).unwrap();
    write_names_and_values(&external.path().join("part-2.parquet"), &["b"], &[2]);
    let batches = plan
        .compile(&dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap();

    assert_eq!(first_int64(&batches), 1);
    assert_eq!(first_int64(&run_sql(&sql)), 2);
}
