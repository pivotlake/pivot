//! A managed table keeps no file's footer: a query planned afresh reads the
//! footers of the files it works on, once, and keeps them with its plan;
//! what the table's metadata already tells is answered without reading any.

mod iceberg_support;

use std::sync::{Mutex, MutexGuard};

use dispatch::io::{OpenFile, open_direct_read};
use iceberg_support::{Database, collect_i64s, parquet_files_in, shared_dispatcher, single_i64};

/// Footer parses are counted, and pages cached, for the whole process, so
/// the tests of this binary run one at a time.
static FOOTER_COUNT: Mutex<()> = Mutex::new(());

fn count_footers_alone() -> MutexGuard<'static, ()> {
    FOOTER_COUNT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A database whose table `t (id BIGINT, name VARCHAR)` holds three files of
/// two rows each, ids 1 to 6.
fn database_with_three_files() -> Database {
    let mut db = Database::new();
    db.execute("CREATE TABLE t (id BIGINT, name VARCHAR)");
    db.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')");
    db.execute("INSERT INTO t VALUES (3, 'c'), (4, 'd')");
    db.execute("INSERT INTO t VALUES (5, 'e'), (6, 'f')");
    db
}

#[test]
fn a_scan_parses_the_footer_of_each_file_it_reads_once() {
    let _alone = count_footers_alone();
    let mut db = database_with_three_files();
    let before = parquet_engine::count_parsed_footers();

    let rows = db.execute("SELECT CAST(sum(id) AS BIGINT) FROM t WHERE id > 2");

    assert_eq!(single_i64(&rows), 18);
    assert_eq!(parquet_engine::count_parsed_footers() - before, 2);
}

#[test]
fn a_filter_on_a_late_column_of_a_wide_table_skips_the_files_it_rules_out() {
    let _alone = count_footers_alone();
    let column_count = 150;
    let declared: Vec<String> = (1..=column_count)
        .map(|column| format!("c{column} BIGINT"))
        .collect();
    let row_of = |value: i64| vec![value.to_string(); column_count].join(", ");
    let mut db = Database::new();
    db.execute(&format!("CREATE TABLE wide ({})", declared.join(", ")));
    db.execute(&format!("INSERT INTO wide VALUES ({})", row_of(1)));
    db.execute(&format!("INSERT INTO wide VALUES ({})", row_of(2)));
    db.execute(&format!("INSERT INTO wide VALUES ({})", row_of(3)));
    let before = parquet_engine::count_parsed_footers();

    let rows = db.execute("SELECT CAST(sum(c1) AS BIGINT) FROM wide WHERE c150 > 2");

    assert_eq!(single_i64(&rows), 3);
    assert_eq!(parquet_engine::count_parsed_footers() - before, 1);
}

#[test]
fn a_table_holds_no_footer_an_earlier_query_read() {
    let _alone = count_footers_alone();
    let mut db = database_with_three_files();
    db.execute("SELECT CAST(sum(id) AS BIGINT) FROM t");
    let before = parquet_engine::count_parsed_footers();

    db.execute("SELECT CAST(sum(id) AS BIGINT) FROM t");

    assert_eq!(parquet_engine::count_parsed_footers() - before, 3);
}

#[test]
fn a_plan_run_again_parses_no_footer() {
    let _alone = count_footers_alone();
    let mut db = database_with_three_files();
    let before = parquet_engine::count_parsed_footers();

    let runs = db.execute_one_plan("SELECT CAST(sum(id) AS BIGINT) FROM t", 2);

    assert_eq!(single_i64(&runs[0]), 21);
    assert_eq!(single_i64(&runs[1]), 21);
    assert_eq!(parquet_engine::count_parsed_footers() - before, 3);
}

#[test]
fn a_late_materializing_query_parses_each_footer_once() {
    let _alone = count_footers_alone();
    let mut db = database_with_three_files();
    let query = "SELECT name FROM t WHERE id > 2 ORDER BY id LIMIT 1";
    let plan = db.explain(query);
    let before = parquet_engine::count_parsed_footers();

    db.execute(query);

    assert!(plan.contains("Materialize"), "{plan}");
    assert_eq!(parquet_engine::count_parsed_footers() - before, 2);
}

#[test]
fn counting_every_row_reads_no_footer() {
    let _alone = count_footers_alone();
    let mut db = database_with_three_files();
    let before = parquet_engine::count_parsed_footers();

    let rows = db.execute("SELECT count(*) FROM t");

    assert_eq!(single_i64(&rows), 6);
    assert_eq!(parquet_engine::count_parsed_footers(), before);
}

#[test]
fn the_extremes_of_a_column_parse_each_footer_once() {
    let _alone = count_footers_alone();
    let mut db = database_with_three_files();
    let before = parquet_engine::count_parsed_footers();

    let rows = db.execute("SELECT min(id), max(id) FROM t");

    assert_eq!(collect_i64s(&rows, 0), [1]);
    assert_eq!(collect_i64s(&rows, 1), [6]);
    assert_eq!(parquet_engine::count_parsed_footers() - before, 3);
}

#[test]
fn the_extremes_of_a_column_decompress_no_page() {
    let _alone = count_footers_alone();
    let mut db = database_with_three_files();
    shared_dispatcher()
        .run_on_worker(|| dispatch::memory_ctx().decompressed_cache().clear())
        .unwrap();

    db.execute("SELECT min(id), max(id) FROM t");

    let no_page_is_cached = shared_dispatcher()
        .run_on_worker(|| dispatch::memory_ctx().decompressed_cache().is_empty())
        .unwrap();
    assert!(no_page_is_cached);
}

#[test]
fn a_commit_keeps_no_footer_of_the_files_it_wrote() {
    let _alone = count_footers_alone();
    let mut db = Database::new();
    db.execute("CREATE TABLE t (id BIGINT, name VARCHAR)");
    db.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')");
    let before = parquet_engine::count_parsed_footers();

    let rows = db.execute("SELECT CAST(sum(id) AS BIGINT) FROM t");

    assert_eq!(single_i64(&rows), 3);
    assert_eq!(parquet_engine::count_parsed_footers() - before, 1);
}

#[test]
fn the_footer_of_a_written_file_is_in_the_ring_before_any_query_reads_it() {
    /// The stretch at a file's end a reader probes for its footer.
    const FOOTER_PROBE_BYTES: usize = 64 * 1024;
    let _alone = count_footers_alone();
    let mut db = Database::new();
    db.execute("CREATE TABLE t (id BIGINT, name VARCHAR)");
    db.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')");
    let table_path = db.table_path("t");
    let file_path = table_path.join(&parquet_files_in(&table_path)[0]);

    let footer_is_cached = shared_dispatcher()
        .run_on_worker(move || {
            let size = std::fs::metadata(&file_path).unwrap().len() as usize;
            let file = OpenFile::Local(open_direct_read(&file_path).unwrap());
            let probed = size.min(FOOTER_PROBE_BYTES);
            dispatch::memory_ctx()
                .compressed_cache()
                .get(&file, size - probed, probed)
                .iter()
                .all(|lookup| lookup.missing().is_none())
        })
        .unwrap();

    assert!(footer_is_cached);
}

#[test]
fn a_compaction_keeps_no_footer_of_the_files_it_merged_or_wrote() {
    let _alone = count_footers_alone();
    let mut db = database_with_three_files();
    db.compact("t", true);
    let before = parquet_engine::count_parsed_footers();

    let rows = db.execute("SELECT CAST(sum(id) AS BIGINT) FROM t");

    assert_eq!(single_i64(&rows), 21);
    assert_eq!(parquet_engine::count_parsed_footers() - before, 1);
}
