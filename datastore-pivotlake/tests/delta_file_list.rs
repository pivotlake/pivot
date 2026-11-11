//! A Delta table's version holds every file of the table with its footer:
//! an open reads them all, a commit made here derives the next version from
//! the one it was made on, and a table of many partitions is scanned and
//! compacted off the one list.

mod iceberg_support;

use datastore_pivotlake::{PivotlakeDatastore, TableFormat};
use iceberg_support::{Database, collect_i64s, shared_dispatcher, single_i64, write_int_parquet};
use planner::catalog::SchemaQualifiedTableName;

/// The directory, under the root of `db`, the files of the table `name` are
/// adopted from.
fn adopted_directory(db: &Database, name: &str) -> std::path::PathBuf {
    db.path().join(format!("{name}_files"))
}

/// Create in `db` the table `name (value BIGINT)` over `file_count` files
/// of one row each, the values 0 to `file_count - 1`.
fn create_table_of_files(db: &mut Database, name: &str, file_count: usize) {
    let adopted = adopted_directory(db, name);
    std::fs::create_dir_all(&adopted).unwrap();
    for file in 0..file_count {
        write_int_parquet(&adopted.join(format!("{file:04}.parquet")), &[file as i64]);
    }
    db.execute(&format!(
        "CREATE TABLE {name} (value BIGINT) WITH (with_pre_existing_parquets = '{name}_files')"
    ));
}

/// The least and the greatest `value` of the rows of the table `name` that
/// `filter` keeps: what a query answers from the footers of the table's
/// files.
fn query_extremes_where(db: &mut Database, name: &str, filter: &str) -> Result<(i64, i64), String> {
    let query = format!("SELECT min(value), max(value) FROM {name} WHERE {filter}");
    let extremes = db.try_execute(&query)?;
    Ok((collect_i64s(&extremes, 0)[0], collect_i64s(&extremes, 1)[0]))
}

/// The least and the greatest `value` of the table `name`.
fn query_extremes(db: &mut Database, name: &str) -> Result<(i64, i64), String> {
    query_extremes_where(db, name, "true")
}

fn reopen(db: Database) -> Database {
    Database::reopen_in_format(db.close(), TableFormat::Delta)
}

#[test]
fn the_first_query_on_a_reopened_database_reads_no_footer() {
    let mut db = Database::in_format(TableFormat::Delta);
    create_table_of_files(&mut db, "t", 3);
    let mut reopened = reopen(db);
    for entry in std::fs::read_dir(adopted_directory(&reopened, "t")).unwrap() {
        std::fs::remove_file(entry.unwrap().path()).unwrap();
    }

    let extremes = query_extremes(&mut reopened, "t");

    assert_eq!(extremes, Ok((0, 2)));
}

#[test]
fn a_database_whose_table_misses_a_data_file_does_not_open() {
    let mut db = Database::in_format(TableFormat::Delta);
    create_table_of_files(&mut db, "t", 2);
    let missing = adopted_directory(&db, "t").join("0001.parquet");
    let directory = db.close();
    std::fs::remove_file(missing).unwrap();

    let reopened = PivotlakeDatastore::open_with_format(
        &directory.path().to_string_lossy(),
        &shared_dispatcher(),
        Some(TableFormat::Delta),
    );

    let failure = reopened.unwrap_err().to_string();
    assert!(failure.contains("No such file"), "{failure}");
}

#[test]
fn a_commit_by_this_process_is_scanned_without_the_log_it_wrote() {
    let mut db = Database::in_format(TableFormat::Delta);
    db.execute("CREATE TABLE t (value BIGINT)");
    db.execute("INSERT INTO t VALUES (1)");
    db.execute("INSERT INTO t VALUES (2)");
    let log = db.table_path("t").join("_delta_log");
    for entry in std::fs::read_dir(&log).unwrap() {
        std::fs::remove_file(entry.unwrap().path()).unwrap();
    }

    let extremes = query_extremes(&mut db, "t");

    assert_eq!(extremes, Ok((1, 2)));
}

#[test]
fn a_table_of_more_partitions_than_fit_any_group_is_scanned_and_compacted_whole() {
    const PARTITIONS: i64 = 150;
    let mut db = Database::in_format(TableFormat::Delta);
    db.execute("CREATE TABLE t (part BIGINT, value BIGINT) WITH (partition_by = 'part')");
    for first_value in [0, PARTITIONS] {
        let rows: Vec<String> = (0..PARTITIONS)
            .map(|part| format!("({part}, {})", first_value + part))
            .collect();
        db.execute(&format!("INSERT INTO t VALUES {}", rows.join(", ")));
    }
    let mut reopened = reopen(db);

    reopened.compact("t", true);

    let files = reopened
        .datastore
        .table_files(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap()
        .unwrap();
    assert_eq!(files.len() as i64, PARTITIONS);
    assert_eq!(
        query_extremes(&mut reopened, "t"),
        Ok((0, 2 * PARTITIONS - 1))
    );
    assert_eq!(
        query_extremes_where(&mut reopened, "t", "part = 149"),
        Ok((149, 299))
    );
    let row_count = reopened.execute("SELECT count(*) FROM t");
    assert_eq!(single_i64(&row_count), 2 * PARTITIONS);
}
