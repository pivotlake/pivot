mod common;

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

use catalog::store::ObjectPath;
use catalog::{ParquetCatalog, PartitionEqFilter, TableBinding};
use common::current_parquet;
use planner::Planner;
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
    // The compressed cache is process-global and accumulates a resident region per
    // distinct file scanned across the whole binary; with `PANIC_ON_EVICT` on
    // (the test default), running out of ring slots panics instead of evicting.
    // Size it well above the suite's distinct-file count so adding tests doesn't
    // tip a later one over.
    DISPATCH
        .get_or_init(|| Dispatch::spin_up(1, 128, None))
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
            name: None,
        })),
        right: Box::new(Expression::Constant(constant)),
        compare_type,
        return_type: Type::Boolean,
    })))
}

// Row groups that survive `table`'s pushed-down predicates over the table's
// current files (what `compile` would scan). Pruning is a pure in-memory filter
// — no dispatcher needed.
fn row_group_count(catalog: &ParquetCatalog, name: &str, table: &TableBinding) -> usize {
    table
        .pruned_parquet(&current_parquet(catalog, name))
        .row_groups()
        .len()
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
    assert!(catalog.binding("t").is_some());
    assert!(current_parquet(&catalog, "t").row_groups().is_empty());
}

#[test]
fn create_table_over_a_missing_path_yields_an_empty_table() {
    // A location with no files yields an empty table — the same as a relative or
    // no-path location. The catalog does not stat the path (which only makes
    // sense for a local store; on a bucket an absolute path is just a key).
    let (_dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    let bogus = Path::new("/definitely/not/a/real/path/for/catalog/tests");
    create_table(&catalog, create_request("t", bogus, columns)).unwrap();
    assert!(catalog.binding("t").is_some());
    assert!(current_parquet(&catalog, "t").row_groups().is_empty());
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
    assert_eq!(row_group_count(&catalog, "t", &table), 3);

    table
        .pushdown_filter(col_neq_filter(0, int_constant(20)))
        .unwrap();

    // Row group whose single value is 20 has min == max == 20 and is pruned.
    assert_eq!(row_group_count(&catalog, "t", &table), 2);
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
    assert_eq!(row_group_count(&catalog, "t", &first), 2);

    // A fresh bind starts from the master entry's full row group set.
    let second = catalog.binding("t").unwrap();
    assert_eq!(row_group_count(&catalog, "t", &second), 3);
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
    assert_eq!(row_group_count(&catalog, "t", &table), 0);
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
    assert_eq!(row_group_count(&catalog, "t", &table), 1);
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

    assert_eq!(row_group_count(&catalog, "t", &table), 3);
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

/// Append the file at `path` to table `name` through a cloned-out handle — the
/// table-level API an external writer uses: it writes the bytes into the table's
/// location and CAS-commits the file. The catalog's own copy is not touched (a
/// later `resolve` reconciles it). Recorded by its location-relative name.
fn append(catalog: &ParquetCatalog, name: &str, path: &Path) {
    let bytes = std::fs::read(path).unwrap();
    let relative = ObjectPath::new(path.file_name().unwrap().to_string_lossy());
    catalog
        .table_handle(name)
        .expect("table exists")
        .append_data_file(relative, &bytes, None, None)
        .unwrap()
}

/// Run `sql` through a planner over `catalog` and return the result batches.
fn run_sql(catalog: &Arc<ParquetCatalog>, sql: &str) -> Vec<RecordBatch> {
    let mut planner = Planner::new(catalog.clone() as Arc<dyn PlannerCatalog>);
    planner
        .plan(sql)
        .unwrap()
        .compile(&dispatcher())
        .unwrap()
        .collect()
        .unwrap()
}

/// Flatten a BIGINT column out of the result batches by name.
fn i64_column(batches: &[RecordBatch], name: &str) -> Vec<i64> {
    let idx = batches[0].schema().index_of(name).unwrap();
    common::collect_i64s(batches, idx)
}

/// The result's column names, in order.
fn column_names(batches: &[RecordBatch]) -> Vec<String> {
    batches[0]
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect()
}

/// `metadata('t')` reports one row per row group with its stats, read from the
/// footers (no data scan). `three_row_table` is one file of three single-row
/// groups, two columns each.
#[test]
fn metadata_function_reports_row_group_stats() {
    let (dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();

    let results = run_sql(&catalog, "SELECT * FROM metadata('t')");

    assert_eq!(i64_column(&results, "file_index"), vec![0, 0, 0]);
    assert_eq!(i64_column(&results, "row_group_index"), vec![0, 1, 2]);
    assert_eq!(i64_column(&results, "num_rows"), vec![1, 1, 1]);
    assert_eq!(i64_column(&results, "num_columns"), vec![2, 2, 2]);
    assert!(
        i64_column(&results, "compressed_bytes")
            .iter()
            .all(|&b| b > 0)
    );
}

/// A projected/reordered SELECT over metadata() returns the named columns, not
/// the first N columns of the full schema (DuckDB prunes/reorders the scan's
/// output and references it positionally, so the generated batch is projected).
#[test]
fn metadata_function_honors_column_projection() {
    let (dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();

    let single = run_sql(&catalog, "SELECT num_rows FROM metadata('t')");
    let reordered = run_sql(
        &catalog,
        "SELECT num_columns, file_index FROM metadata('t')",
    );

    assert_eq!(single[0].num_columns(), 1);
    assert_eq!(i64_column(&single, "num_rows"), vec![1, 1, 1]);
    // Assert width and order, not just by-name lookups, so a projection that
    // emitted all six columns or the wrong order would fail here.
    assert_eq!(column_names(&reordered), vec!["num_columns", "file_index"]);
    assert_eq!(i64_column(&reordered, "num_columns"), vec![2, 2, 2]);
    assert_eq!(i64_column(&reordered, "file_index"), vec![0, 0, 0]);
}

/// `metadata()` over a table with no committed files is an empty result that
/// still carries the full schema (an empty batch, not no batch).
#[test]
fn metadata_function_on_empty_table() {
    let (_dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(
        &catalog,
        create_request("t", Path::new("/no/such/dir"), columns),
    )
    .unwrap();

    let results = run_sql(&catalog, "SELECT * FROM metadata('t')");

    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 0);
    assert_eq!(results[0].num_columns(), 6);
}

/// Each file gets its own `file_index`, so the metadata composes with normal SQL
/// to count files and total rows across an appended file.
#[test]
fn metadata_function_numbers_files_distinctly() {
    let (dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();
    let new_file = write_ids(dir.path(), "later.parquet", &[40, 50]);
    append(&catalog, "t", &new_file);

    let results = run_sql(&catalog, "SELECT * FROM metadata('t')");

    assert_eq!(i64_column(&results, "file_index"), vec![0, 0, 0, 1]);
    // row_group_index is per-file, so it resets to 0 for the appended file's
    // single row group rather than continuing the global count.
    assert_eq!(i64_column(&results, "row_group_index"), vec![0, 1, 2, 0]);
    assert_eq!(i64_column(&results, "num_rows").iter().sum::<i64>(), 5);
}

/// A file appended after `CREATE TABLE` becomes visible to new binds, with
/// global row-group indices kept sequential.
#[test]
fn append_data_file_makes_new_file_visible_to_new_binds() {
    let (dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();
    assert_eq!(current_parquet(&catalog, "t").row_groups().len(), 3);

    let new_file = write_ids(dir.path(), "later.parquet", &[40, 50]);
    append(&catalog, "t", &new_file);

    let parquet = current_parquet(&catalog, "t");
    let groups = parquet.row_groups();
    assert_eq!(groups.len(), 4);
    assert_eq!(groups.iter().map(|rg| rg.num_rows).sum::<i64>(), 5);
}

/// Appending the same path twice (a replayed flush notification) must not
/// double-count its rows. The second handle starts a version behind, so it CAS-
/// conflicts, refreshes, and sees the file already present.
#[test]
fn append_data_file_is_idempotent_per_path() {
    let (dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();

    let new_file = write_ids(dir.path(), "later.parquet", &[40]);
    append(&catalog, "t", &new_file);
    // Appending the same file again is a no-op (it must not double-count).
    append(&catalog, "t", &new_file);

    assert_eq!(current_parquet(&catalog, "t").row_groups().len(), 4);
}

/// No table yet (a writer starts before `CREATE TABLE`): there is no handle to
/// append against.
#[test]
fn table_handle_for_a_missing_table_is_none() {
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    assert!(catalog.table_handle("missing").is_none());
}

/// Compaction's commit: the small files' row groups vanish, the merged file's
/// appear, and indices are renumbered — one atomic version swap. And a *second*
/// compacter that picked the same inputs must abort its swap (`Ok(false)`)
/// rather than re-add its output on top, which would double-count the rows.
#[test]
fn replace_data_files_swaps_compacted_inputs_for_merged_output() {
    let (dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();
    let extra = write_ids(dir.path(), "extra.parquet", &[40]);
    append(&catalog, "t", &extra);
    assert_eq!(current_parquet(&catalog, "t").row_groups().len(), 4);

    let merged = write_ids(dir.path(), "merged.parquet", &[10, 20, 30, 40]);
    let merged_size = std::fs::metadata(&merged).unwrap().len();
    let removed = vec![
        ObjectPath::new("data.parquet"),
        ObjectPath::new("extra.parquet"),
    ];
    let added = vec![catalog::ManifestEntry::new(catalog::FileRef {
        path: ObjectPath::new("merged.parquet"),
        size: merged_size,
    })];
    // A losing compacter clones the table out at the version where the inputs
    // are present, before the winning swap lands.
    let mut loser = catalog.table_handle("t").unwrap();
    loser.refresh().unwrap();
    let mut winner = catalog.table_handle("t").unwrap();
    winner.refresh().unwrap();
    assert!(
        winner.replace_data_files(&removed, &added).unwrap(),
        "first swap commits"
    );
    // The loser only discovers the inputs are gone after its CAS conflict +
    // refresh, and aborts — no footer read for its output, no double-count.
    let loser_added = vec![catalog::ManifestEntry::new(catalog::FileRef {
        path: ObjectPath::new("merged-loser.parquet"),
        size: merged_size,
    })];
    assert!(
        !loser.replace_data_files(&removed, &loser_added).unwrap(),
        "second swap aborts: its inputs were already swapped out"
    );

    let parquet = current_parquet(&catalog, "t");
    let groups = parquet.row_groups();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].num_rows, 4);
}

/// A pushed-down predicate prunes against the table's *current* files: a binding
/// resolved before an append, after a refresh, prunes over the new file set —
/// the compile path fetches the latest files from the query context, so a reused
/// (cached) plan's predicates apply to data committed since it was planned.
#[test]
fn pushed_predicate_prunes_latest_files_after_refresh() {
    let (dir, columns) = three_row_table();
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();
    let mut table = catalog.binding("t").unwrap();
    table
        .pushdown_filter(col_eq_filter(0, int_constant(20)))
        .unwrap();
    assert_eq!(row_group_count(&catalog, "t", &table), 1);

    let new_file = write_ids(dir.path(), "later.parquet", &[40, 50]);
    append(&catalog, "t", &new_file);

    // `id = 20` still prunes to the single matching row group, now over 4 files.
    assert_eq!(row_group_count(&catalog, "t", &table), 1);
}

/// A second catalog over the same persisted root sees another instance's
/// append at its next reload: reloading reads the committed manifest, so
/// cross-process commits surface without any re-`CREATE`.
#[test]
fn other_catalog_instance_sees_append_at_next_bind() {
    let (data_dir, columns) = three_row_table();
    let db = TempDir::new().unwrap();
    let writer =
        Arc::new(ParquetCatalog::open(db.path().to_str().unwrap(), &dispatcher()).unwrap());
    create_table(&writer, create_request("t", data_dir.path(), columns)).unwrap();

    // The reader opens before the new file exists, at version 1.
    let reader = ParquetCatalog::open(db.path().to_str().unwrap(), &dispatcher()).unwrap();
    assert_eq!(current_parquet(&reader, "t").row_groups().len(), 3);

    let new_file = write_ids(data_dir.path(), "later.parquet", &[40, 50]);
    append(&writer, "t", &new_file);

    // The reader's next reload picks up version 2 from the committed manifest.
    let rows = current_parquet(&reader, "t")
        .row_groups()
        .iter()
        .map(|rg| rg.num_rows)
        .sum::<i64>();
    assert_eq!(rows, 5);
}

/// Restart reads the committed manifest, not the directory: files appended
/// after the `CREATE` survive a reopen.
#[test]
fn reopened_database_restores_appended_files_from_manifest() {
    let (data_dir, columns) = three_row_table();
    let db = TempDir::new().unwrap();
    {
        let catalog =
            Arc::new(ParquetCatalog::open(db.path().to_str().unwrap(), &dispatcher()).unwrap());
        create_table(&catalog, create_request("t", data_dir.path(), columns)).unwrap();
        let new_file = write_ids(data_dir.path(), "later.parquet", &[40]);
        append(&catalog, "t", &new_file);
    }
    let reopened = ParquetCatalog::open(db.path().to_str().unwrap(), &dispatcher()).unwrap();
    assert_eq!(current_parquet(&reopened, "t").row_groups().len(), 4);
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
    let added = vec![catalog::ManifestEntry::new(catalog::FileRef {
        path: ObjectPath::new("merged.parquet"),
        size: std::fs::metadata(&merged).unwrap().len(),
    })];
    catalog
        .table_handle("t")
        .unwrap()
        .replace_data_files(&[ObjectPath::new("data.parquet")], &added)
        .unwrap();

    let reopened = ParquetCatalog::open(db.path().to_str().unwrap(), &dispatcher()).unwrap();
    let parquet = current_parquet(&reopened, "t");
    let groups = parquet.row_groups();
    assert_eq!(groups.len(), 1, "only the committed merged file is read");
    assert_eq!(groups.iter().map(|rg| rg.num_rows).sum::<i64>(), 3);
}

/// Write `file_name` into `dir` with one row group per value in `ids`, so a file
/// can hold several row groups that all share one `id` (the shape the partition
/// writer emits: one file per partition value, many row groups inside it).
fn write_ids_one_group_each(dir: &Path, file_name: &str, ids: &[i32]) {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batches: Vec<RecordBatch> = ids
        .iter()
        .map(|i| {
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![*i]))])
                .unwrap()
        })
        .collect();
    let props = WriterProperties::builder()
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_max_row_group_row_count(Some(1))
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(dir.join(file_name)).unwrap(),
        schema,
        Some(props),
    )
    .unwrap();
    for batch in &batches {
        writer.write(batch).unwrap();
    }
    writer.close().unwrap();
}

/// A file whose every row group shares one partition value is fully pruned by a
/// filter on the partition column: each row group has min == max == that value,
/// so the existing min/max stats prune drops the whole irrelevant file without
/// any partition-specific read path. This is why partition_by needs no special
/// pruning today — one-partition-per-file makes it fall out of stats pruning.
#[test]
fn filter_on_partition_column_prunes_whole_single_partition_file() {
    let dir = TempDir::new().unwrap();
    write_ids_one_group_each(dir.path(), "part-1.parquet", &[1, 1, 1]);
    write_ids_one_group_each(dir.path(), "part-2.parquet", &[2, 2, 2]);
    let columns = vec![Column {
        name: "id".to_string(),
        col_type: Type::Int32,
    }];
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, create_request("t", dir.path(), columns)).unwrap();

    let mut table = catalog.binding("t").unwrap();
    assert_eq!(row_group_count(&catalog, "t", &table), 6);

    table
        .pushdown_filter(col_eq_filter(0, int_constant(1)))
        .unwrap();

    // `id = 1` excludes the part-2 file's three row groups entirely; only the
    // part-1 file's three survive.
    assert_eq!(row_group_count(&catalog, "t", &table), 3);
}

/// A table partitioned by `name` with one committed file per partition: a
/// one-row-group `keep` file and a three-row-group `drop` file, each tagged with
/// its partition tuple. The distinct group counts let a build's group count name
/// exactly which files it fetched. Files are written *after* `CREATE TABLE` (over
/// an empty dir) so they arrive through the partition-recording append, not as
/// untagged create-time discoveries.
fn table_partitioned_by_name() -> (TempDir, Arc<ParquetCatalog>) {
    let dir = TempDir::new().unwrap();
    let request = CreateTableRequest {
        name: "p".to_string(),
        columns: vec![
            Column {
                name: "id".to_string(),
                col_type: Type::Int32,
            },
            Column {
                name: "name".to_string(),
                col_type: Type::Utf8,
            },
        ],
        options: HashMap::from([
            (
                "path".to_string(),
                dir.path().to_string_lossy().into_owned(),
            ),
            ("partition_by".to_string(), "name".to_string()),
        ]),
        if_not_exists: false,
    };
    let catalog = Arc::new(ParquetCatalog::new(dispatcher()));
    create_table(&catalog, request).unwrap();

    write_ids_one_group_each(dir.path(), "keep.parquet", &[1]);
    write_ids_one_group_each(dir.path(), "drop.parquet", &[2, 2, 2]);
    let mut table = catalog.table_handle("p").unwrap();
    table
        .append_data_file(
            ObjectPath::new("keep.parquet"),
            &std::fs::read(dir.path().join("keep.parquet")).unwrap(),
            Some(serde_json::json!({ "name": "keep" })),
            None,
        )
        .unwrap();
    table
        .append_data_file(
            ObjectPath::new("drop.parquet"),
            &std::fs::read(dir.path().join("drop.parquet")).unwrap(),
            Some(serde_json::json!({ "name": "drop" })),
            None,
        )
        .unwrap();
    (dir, catalog)
}

fn name_eq(value: &str) -> PartitionEqFilter {
    PartitionEqFilter {
        column: "name".to_string(),
        value: serde_json::json!(value),
    }
}

#[test]
fn partition_filter_builds_only_the_matching_partitions_files() {
    let (_dir, catalog) = table_partitioned_by_name();

    let mut table = catalog.table_handle("p").unwrap();
    table.reload_manifest_only().unwrap();
    let kept = table.parquet(&[name_eq("keep")]).unwrap();

    // Only the one-group `keep` file is fetched; the three-group `drop` file is
    // skipped before its footer is read.
    assert_eq!(kept.row_groups().len(), 1);
}

#[test]
fn no_partition_filter_builds_every_partitions_files() {
    let (_dir, catalog) = table_partitioned_by_name();

    let mut table = catalog.table_handle("p").unwrap();
    table.reload_manifest_only().unwrap();
    let all = table.parquet(&[]).unwrap();

    // Without a filter both files are fetched: keep's 1 group + drop's 3.
    assert_eq!(all.row_groups().len(), 4);
}
