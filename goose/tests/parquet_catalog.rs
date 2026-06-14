use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use arrow_array::{ArrayRef, Int32Array, RecordBatch, Scalar, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{DataFlowDispatcher, Dispatch};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use tempfile::TempDir;

use goose::store::ObjectPath;
use goose::{ParquetCatalog, RegisterOutcome, TableBinding};
use planner::catalog::{
    Catalog as PlannerCatalog, Column, CreateTableRequest, Result as CatalogResult, Table,
};
use planner::expression::{Compare, CompareType, Expression, Ref, TableFilter};
use planner::types::Type;

/// A shared single-worker dispatch pool for the whole test binary, handed to
/// each `ParquetCatalog` so `create_table` can read footers once (via the
/// metadata-fetch dataflow) when the table is defined.
fn dispatcher() -> DataFlowDispatcher {
    static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
    DISPATCH
        .get_or_init(|| Dispatch::spin_up(1, 32))
        .dispatcher()
        .clone()
}

/// Create a table, mirroring how the server runs `CREATE TABLE`: `create_table`
/// reads the footers (on the coordinator) and returns the plan that writes the
/// table, which we then execute. Returns the catalog's own result so error-path
/// tests can still assert on the `Err`.
fn create_table(catalog: &Arc<ParquetCatalog>, request: CreateTableRequest) -> CatalogResult<()> {
    catalog
        .create_table(request, &dispatcher())?
        .execute()
        .collect()
        .map(|_| ())
        .map_err(|e| planner::catalog::Error::Other(Box::new(e)))
}

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

// Row groups that survive the binding's pushed-down predicates (what `compile`
// would scan). The row groups were materialized once at create time, so pruning
// is a pure in-memory filter — no dispatcher needed.
fn row_group_count(table: &TableBinding) -> usize {
    table.pruned_parquet().row_groups().len()
}

#[test]
fn create_table_succeeds_with_valid_path() {
    let (dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();
    assert!(catalog.table("t").is_some());
}

#[test]
fn create_table_without_a_path_makes_an_empty_table() {
    let (_dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    let req = CreateTableRequest {
        name: "t".to_string(),
        columns,
        options: HashMap::new(),
        if_not_exists: false,
    };
    // No path: the table lives under the (in-memory) database root with no data.
    create_table(&catalog, req).unwrap();
    let table = catalog.binding("t").unwrap();
    assert!(table.parquet.row_groups().is_empty());
}

#[test]
fn create_table_over_a_missing_path_yields_an_empty_table() {
    // A location with no files yields an empty table — the same as a relative or
    // no-path location. The catalog no longer eagerly stats the path (which only
    // made sense for a local store; on a bucket an absolute path is just a key).
    let (_dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    let bogus = Path::new("/definitely/not/a/real/path/for/catalog/tests");
    create_table(&catalog, create_request("t", bogus, columns)).unwrap();
    assert!(catalog.binding("t").unwrap().parquet.row_groups().is_empty());
}

#[test]
fn create_table_fails_when_path_is_a_file() {
    let (dir, columns) = three_row_table();
    let file_path = dir.path().join("data.parquet");
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    // Listing a file as a directory fails, so the create errors.
    let err = create_table(&catalog, create_request("t", &file_path, columns))
        .unwrap_err()
        .to_string();
    assert!(
        err.to_lowercase().contains("not a directory"),
        "expected not-a-directory error: {err}"
    );
}

#[test]
fn create_table_rejects_a_url_path() {
    let (_dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    // A table path is always a plain path — its storage is the database's, not the
    // path's — so a scheme is rejected regardless of the database's storage class.
    let req = CreateTableRequest {
        name: "t".to_string(),
        columns,
        options: HashMap::from([("path".to_string(), "s3://bucket/data".to_string())]),
        if_not_exists: false,
    };
    let err = create_table(&catalog, req).unwrap_err().to_string();
    assert!(
        err.contains("not a URL"),
        "expected scheme rejection: {err}"
    );
}

#[test]
fn pushdown_filter_always_returns_false() {
    let (dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();
    let mut table = catalog.binding("t").unwrap();
    let pushed = table
        .pushdown_filter(col_neq_filter(0, int_constant(20)))
        .unwrap();
    assert!(!pushed);
}

#[test]
fn pushdown_filter_prunes_row_group_with_only_excluded_value() {
    let (dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();

    let mut table = catalog.binding("t").unwrap();
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
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();

    let mut first = catalog.binding("t").unwrap();
    first
        .pushdown_filter(col_neq_filter(0, int_constant(20)))
        .unwrap();
    assert_eq!(row_group_count(&first), 2);

    // A fresh bind starts from the master entry's full row group set.
    let second = catalog.binding("t").unwrap();
    assert_eq!(row_group_count(&second), 3);
}

#[test]
fn pushdown_filter_eq_prunes_row_groups_when_constant_outside_range() {
    let (dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();

    // `id = 999` lies outside every (min,max) → all three row groups drop.
    let mut table = catalog.binding("t").unwrap();
    table
        .pushdown_filter(col_eq_filter(0, int_constant(999)))
        .unwrap();
    assert_eq!(row_group_count(&table), 0);
}

#[test]
fn pushdown_filter_eq_keeps_only_matching_row_group() {
    let (dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();

    // `id = 20` matches only the row group whose single value is 20.
    let mut table = catalog.binding("t").unwrap();
    table
        .pushdown_filter(col_eq_filter(0, int_constant(20)))
        .unwrap();
    assert_eq!(row_group_count(&table), 1);
}

#[test]
fn pushdown_filter_eq_returns_false() {
    let (dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();
    let mut table = catalog.binding("t").unwrap();
    let pushed = table
        .pushdown_filter(col_eq_filter(0, int_constant(20)))
        .unwrap();
    assert!(!pushed);
}

#[test]
fn pushdown_filter_keeps_row_groups_when_constant_outside_range() {
    let (dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();

    let mut table = catalog.binding("t").unwrap();
    table
        .pushdown_filter(col_neq_filter(0, int_constant(999)))
        .unwrap();

    assert_eq!(row_group_count(&table), 3);
}

/// Write one Parquet file of `ids` (single row group) into `dir`, mirroring the
/// `three_row_table` schema. Returns the file's path.
fn write_ids(dir: &Path, file_name: &str, ids: &[i32]) -> std::path::PathBuf {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8View, false),
    ]));
    let names: Vec<String> = ids.iter().map(|i| format!("n{i}")).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids.to_vec())) as ArrayRef,
            Arc::new(StringViewArray::from(
                names.iter().map(String::as_str).collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )
    .unwrap();
    let path = dir.join(file_name);
    let props = WriterProperties::builder()
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .build();
    let mut writer =
        ArrowWriter::try_new(File::create(&path).unwrap(), batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    path
}

/// Register `path` with table `name` through a cloned-out handle — the
/// table-level API a writer (ingest) uses: it CAS-commits a new manifest version
/// to the shared store. The catalog's own copy is not touched (a later `resolve`
/// reconciles it).
fn register(catalog: &ParquetCatalog, name: &str, path: &Path) -> RegisterOutcome {
    catalog
        .table_handle(name)
        .expect("table exists")
        .register_data_file(ObjectPath::new(path.to_string_lossy()))
        .unwrap()
}

/// Resolve `name` through the planner trait — what a query does on bind — so the
/// catalog refreshes its in-memory copy from the latest committed manifest.
fn resolve(catalog: &ParquetCatalog, name: &str) {
    let _ = PlannerCatalog::table(catalog, name);
}

/// The object path of `name` directly under `dir` — a local table's location, so
/// the path is the file's absolute filesystem path.
fn opath(dir: &Path, name: &str) -> ObjectPath {
    ObjectPath::new(dir.join(name).to_string_lossy())
}

/// A file flushed after `CREATE TABLE` becomes visible to new binds once
/// registered, with global row-group indices kept sequential.
#[test]
fn register_data_file_makes_new_file_visible_to_new_binds() {
    let (dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();
    assert_eq!(catalog.binding("t").unwrap().parquet.row_groups().len(), 3);

    let new_file = write_ids(dir.path(), "later.parquet", &[40, 50]);
    assert_eq!(register(&catalog, "t", &new_file), RegisterOutcome::Registered);

    resolve(&catalog, "t");
    let table = catalog.binding("t").unwrap();
    let groups = table.parquet.row_groups();
    assert_eq!(groups.len(), 4);
    assert_eq!(groups.iter().map(|rg| rg.num_rows).sum::<i64>(), 5);
}

/// Registering the same path twice (a replayed flush notification) must not
/// double-count its rows. The second handle starts a version behind, so it CAS-
/// conflicts, refreshes, and sees the file already present.
#[test]
fn register_data_file_is_idempotent_per_path() {
    let (dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();

    let new_file = write_ids(dir.path(), "later.parquet", &[40]);
    assert_eq!(register(&catalog, "t", &new_file), RegisterOutcome::Registered);
    assert_eq!(
        register(&catalog, "t", &new_file),
        RegisterOutcome::AlreadyRegistered
    );

    resolve(&catalog, "t");
    assert_eq!(catalog.binding("t").unwrap().parquet.row_groups().len(), 4);
}

/// No table yet (ingest runs before `CREATE TABLE`): there is no handle to
/// register against.
#[test]
fn register_data_file_without_table_is_unresolvable() {
    let dir = TempDir::new().unwrap();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    let _new_file = write_ids(dir.path(), "later.parquet", &[1]);
    assert!(catalog.table_handle("missing").is_none());
}

/// A file outside the table's data directory must not be appended to it.
#[test]
fn register_data_file_refuses_foreign_directory() {
    let (dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();

    let elsewhere = TempDir::new().unwrap();
    let foreign = write_ids(elsewhere.path(), "foreign.parquet", &[1]);
    assert_eq!(
        register(&catalog, "t", &foreign),
        RegisterOutcome::LocationMismatch
    );
    resolve(&catalog, "t");
    assert_eq!(catalog.binding("t").unwrap().parquet.row_groups().len(), 3);
}

/// Compaction's commit: the small files' row groups vanish, the merged file's
/// appear, and indices are renumbered — one atomic version swap.
#[test]
fn replace_data_files_swaps_compacted_inputs_for_merged_output() {
    let (dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();
    let extra = write_ids(dir.path(), "extra.parquet", &[40]);
    assert_eq!(register(&catalog, "t", &extra), RegisterOutcome::Registered);
    resolve(&catalog, "t");
    assert_eq!(catalog.binding("t").unwrap().parquet.row_groups().len(), 4);

    let merged = write_ids(dir.path(), "merged.parquet", &[10, 20, 30, 40]);
    let merged_size = std::fs::metadata(&merged).unwrap().len();
    let removed = vec![opath(dir.path(), "data.parquet"), opath(dir.path(), "extra.parquet")];
    let added = vec![goose::FileRef {
        path: opath(dir.path(), "merged.parquet"),
        size: merged_size,
    }];
    catalog
        .table_handle("t")
        .unwrap()
        .replace_data_files(&removed, &added)
        .unwrap();

    resolve(&catalog, "t");
    let table = catalog.binding("t").unwrap();
    let groups = table.parquet.row_groups();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].num_rows, 4);
}

/// A second catalog over the same persisted root sees another instance's
/// registration at its next bind: `Catalog::table` refreshes from the committed
/// manifest, so cross-process commits surface without any re-`CREATE`.
#[test]
fn other_catalog_instance_sees_registration_at_next_bind() {
    let (data_dir, columns) = three_row_table();
    let db = TempDir::new().unwrap();
    let writer =
        Arc::new(ParquetCatalog::open(db.path().to_str().unwrap(), &dispatcher()).unwrap());
    create_table(&writer, create_request("t", data_dir.path(), columns)).unwrap();

    // The reader opens before the new file exists, at version 1.
    let reader = ParquetCatalog::open(db.path().to_str().unwrap(), &dispatcher()).unwrap();
    assert_eq!(reader.binding("t").unwrap().parquet.row_groups().len(), 3);

    let new_file = write_ids(data_dir.path(), "later.parquet", &[40, 50]);
    assert_eq!(register(&writer, "t", &new_file), RegisterOutcome::Registered);

    // Binding through the trait (what a query does) reloads to version 2.
    assert!(PlannerCatalog::table(&reader, "t").is_some());
    let table = reader.binding("t").unwrap();
    assert_eq!(
        table
            .parquet
            .row_groups()
            .iter()
            .map(|rg| rg.num_rows)
            .sum::<i64>(),
        5
    );
}

/// Restart reads the committed manifest, not the directory: files registered
/// after the `CREATE` survive a reopen.
#[test]
fn reopened_database_restores_registered_files_from_log() {
    let (data_dir, columns) = three_row_table();
    let db = TempDir::new().unwrap();
    {
        let catalog =
            Arc::new(ParquetCatalog::open(db.path().to_str().unwrap(), &dispatcher()).unwrap());
        create_table(&catalog, create_request("t", data_dir.path(), columns)).unwrap();
        let new_file = write_ids(data_dir.path(), "later.parquet", &[40]);
        register(&catalog, "t", &new_file);
    }
    let reopened = ParquetCatalog::open(db.path().to_str().unwrap(), &dispatcher()).unwrap();
    let table = reopened.binding("t").unwrap();
    assert_eq!(table.parquet.row_groups().len(), 4);
}

/// Only committed files exist: after a compaction swap, a leftover input
/// (e.g. a crash before the unlink) is invisible to a reopen — no double-read.
#[test]
fn unlogged_leftover_file_is_invisible_after_swap() {
    let (data_dir, columns) = three_row_table();
    let db = TempDir::new().unwrap();
    let catalog =
        Arc::new(ParquetCatalog::open(db.path().to_str().unwrap(), &dispatcher()).unwrap());
    create_table(&catalog, create_request("t", data_dir.path(), columns)).unwrap();

    // "Compact" data.parquet into merged.parquet but crash before deleting the
    // input: both files are on disk, only merged is in the manifest.
    let merged = write_ids(data_dir.path(), "merged.parquet", &[10, 20, 30]);
    let added = vec![goose::FileRef {
        path: opath(data_dir.path(), "merged.parquet"),
        size: std::fs::metadata(&merged).unwrap().len(),
    }];
    catalog
        .table_handle("t")
        .unwrap()
        .replace_data_files(&[opath(data_dir.path(), "data.parquet")], &added)
        .unwrap();

    let reopened = ParquetCatalog::open(db.path().to_str().unwrap(), &dispatcher()).unwrap();
    let table = reopened.binding("t").unwrap();
    let groups = table.parquet.row_groups();
    assert_eq!(groups.len(), 1, "only the committed merged file is read");
    assert_eq!(groups.iter().map(|rg| rg.num_rows).sum::<i64>(), 3);
}
