use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, Scalar, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use tempfile::TempDir;

use catalog::{ParquetCatalog, ParquetCatalogTable};
use planner::catalog::{Catalog as PlannerCatalog, Column, CreateTableRequest, Table};
use planner::expression::{Compare, CompareType, Expression, Ref, TableFilter};
use planner::types::Type;

/// Write a parquet file containing each batch as its own row group, so a
/// per-row-group min/max prune is observable.
fn write_one_row_per_group(batches: &[RecordBatch]) -> TempDir {
    let dir = TempDir::new().unwrap();
    let props = WriterProperties::builder()
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_max_row_group_row_count(Some(1))
        .build();
    let path = dir.path().join("data.parquet");
    let file = File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batches[0].schema(), Some(props)).unwrap();
    for batch in batches {
        writer.write(batch).unwrap();
    }
    writer.close().unwrap();
    dir
}

/// Three-row table with one (id, name) row per row group.
fn three_row_table() -> (TempDir, Vec<Column>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8View, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![10, 20, 30])) as ArrayRef,
            Arc::new(StringViewArray::from(vec!["a", "b", "c"])) as ArrayRef,
        ],
    )
    .unwrap();
    let dir = write_one_row_per_group(&[batch]);
    let columns = vec![
        Column {
            name: "id".to_string(),
            col_type: Type::Int32,
        },
        Column {
            name: "name".to_string(),
            col_type: Type::Utf8,
        },
    ];
    (dir, columns)
}

fn create_request(name: &str, path: &Path, columns: Vec<Column>) -> CreateTableRequest {
    let mut options = HashMap::new();
    options.insert("path".to_string(), path.to_string_lossy().into_owned());
    CreateTableRequest {
        name: name.to_string(),
        columns,
        options,
        if_not_exists: false,
    }
}

fn int_constant(v: i32) -> Scalar<ArrayRef> {
    Scalar::new(Arc::new(Int32Array::new_scalar(v).into_inner()) as ArrayRef)
}

fn col_neq_filter(column_idx: usize, constant: Scalar<ArrayRef>) -> TableFilter {
    constant_comparison(column_idx, CompareType::NotEqual, constant)
}

fn col_eq_filter(column_idx: usize, constant: Scalar<ArrayRef>) -> TableFilter {
    constant_comparison(column_idx, CompareType::Equal, constant)
}

fn constant_comparison(
    column_idx: usize,
    compare_type: CompareType,
    constant: Scalar<ArrayRef>,
) -> TableFilter {
    // Build the same shape DuckDB pushes through the C++ bridge:
    // `TableFilter::Expression(Compare { Ref, Constant })`.
    TableFilter::Expression(Box::new(Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx,
            return_type: Type::Int32,
        })),
        right: Box::new(Expression::Constant(constant)),
        compare_type,
        return_type: Type::Boolean,
    })))
}

fn row_group_count(table: &ParquetCatalogTable) -> usize {
    table.parquet.row_groups().len()
}

#[test]
fn create_table_succeeds_with_valid_path() {
    let (dir, columns) = three_row_table();
    let catalog = ParquetCatalog::new();
    catalog
        .create_table(create_request("t", dir.path(), columns))
        .unwrap();
    assert!(catalog.table("t").is_some());
}

#[test]
fn create_table_fails_when_path_option_missing() {
    let (_dir, columns) = three_row_table();
    let catalog = ParquetCatalog::new();
    let req = CreateTableRequest {
        name: "t".to_string(),
        columns,
        options: HashMap::new(),
        if_not_exists: false,
    };
    let err = catalog.create_table(req).unwrap_err().to_string();
    assert!(
        err.contains("path"),
        "expected error to mention path: {err}"
    );
}

#[test]
fn create_table_fails_when_path_does_not_exist() {
    let (_dir, columns) = three_row_table();
    let catalog = ParquetCatalog::new();
    let bogus = Path::new("/definitely/not/a/real/path/for/catalog/tests");
    let err = catalog
        .create_table(create_request("t", bogus, columns))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("does not exist"),
        "expected path-not-found error: {err}"
    );
}

#[test]
fn create_table_fails_when_path_is_a_file() {
    let (dir, columns) = three_row_table();
    let file_path = dir.path().join("data.parquet");
    let catalog = ParquetCatalog::new();
    let err = catalog
        .create_table(create_request("t", &file_path, columns))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("not a directory"),
        "expected not-a-directory error: {err}"
    );
}

#[test]
fn pushdown_filter_always_returns_false() {
    let (dir, columns) = three_row_table();
    let catalog = ParquetCatalog::new();
    catalog
        .create_table(create_request("t", dir.path(), columns))
        .unwrap();
    let mut table = catalog.parquet_table("t").unwrap();
    let pushed = table
        .pushdown_filter(col_neq_filter(0, int_constant(20)))
        .unwrap();
    assert!(!pushed);
}

#[test]
fn pushdown_filter_prunes_row_group_with_only_excluded_value() {
    let (dir, columns) = three_row_table();
    let catalog = ParquetCatalog::new();
    catalog
        .create_table(create_request("t", dir.path(), columns))
        .unwrap();

    let mut table = catalog.parquet_table("t").unwrap();
    assert_eq!(row_group_count(&table), 3);

    table
        .pushdown_filter(col_neq_filter(0, int_constant(20)))
        .unwrap();

    // Row group whose single value is 20 has min == max == 20 and is pruned.
    assert_eq!(row_group_count(&table), 2);
}

/// Each bind hands out a fresh clone, so pushdown applied to one binding
/// must not leak into a subsequent one.
#[test]
fn second_bind_is_independent_of_first_bind_pushdown() {
    let (dir, columns) = three_row_table();
    let catalog = ParquetCatalog::new();
    catalog
        .create_table(create_request("t", dir.path(), columns))
        .unwrap();

    let mut first = catalog.parquet_table("t").unwrap();
    first
        .pushdown_filter(col_neq_filter(0, int_constant(20)))
        .unwrap();
    assert_eq!(row_group_count(&first), 2);

    // A fresh bind starts from the master entry's full row group set.
    let second = catalog.parquet_table("t").unwrap();
    assert_eq!(row_group_count(&second), 3);
}

#[test]
fn pushdown_filter_eq_prunes_row_groups_when_constant_outside_range() {
    let (dir, columns) = three_row_table();
    let catalog = ParquetCatalog::new();
    catalog
        .create_table(create_request("t", dir.path(), columns))
        .unwrap();

    // `id = 999` lies outside every (min,max) → all three row groups drop.
    let mut table = catalog.parquet_table("t").unwrap();
    table
        .pushdown_filter(col_eq_filter(0, int_constant(999)))
        .unwrap();
    assert_eq!(row_group_count(&table), 0);
}

#[test]
fn pushdown_filter_eq_keeps_only_matching_row_group() {
    let (dir, columns) = three_row_table();
    let catalog = ParquetCatalog::new();
    catalog
        .create_table(create_request("t", dir.path(), columns))
        .unwrap();

    // `id = 20` matches only the row group whose single value is 20.
    let mut table = catalog.parquet_table("t").unwrap();
    table
        .pushdown_filter(col_eq_filter(0, int_constant(20)))
        .unwrap();
    assert_eq!(row_group_count(&table), 1);
}

#[test]
fn pushdown_filter_eq_returns_false() {
    let (dir, columns) = three_row_table();
    let catalog = ParquetCatalog::new();
    catalog
        .create_table(create_request("t", dir.path(), columns))
        .unwrap();
    let mut table = catalog.parquet_table("t").unwrap();
    let pushed = table
        .pushdown_filter(col_eq_filter(0, int_constant(20)))
        .unwrap();
    assert!(!pushed);
}

#[test]
fn pushdown_filter_keeps_row_groups_when_constant_outside_range() {
    let (dir, columns) = three_row_table();
    let catalog = ParquetCatalog::new();
    catalog
        .create_table(create_request("t", dir.path(), columns))
        .unwrap();

    let mut table = catalog.parquet_table("t").unwrap();
    table
        .pushdown_filter(col_neq_filter(0, int_constant(999)))
        .unwrap();

    assert_eq!(row_group_count(&table), 3);
}
