//! A Delta table's files are indexed and their footers read when something
//! first needs them: opening a small database loads what its tables' first
//! queries would read, and a database of more files is left to its queries.

mod iceberg_support;

use datastore_pivotlake::{PRELOADED_DATABASE_MAX_FILES, TableFormat};
use iceberg_support::{Database, collect_i64s, write_int_parquet};

/// The most files a database may hold for an open to preload it.
const MOST_PRELOADED_FILES: usize = PRELOADED_DATABASE_MAX_FILES as usize;

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

/// Delete every data file of the table `name`, so whatever opens one to
/// read its footer from now on fails.
fn delete_data_files(db: &Database, name: &str) {
    for entry in std::fs::read_dir(adopted_directory(db, name)).unwrap() {
        std::fs::remove_file(entry.unwrap().path()).unwrap();
    }
}

/// The least and the greatest `value` of the rows of the table `name` that
/// `filter` keeps: what a query answers from the table's log and its
/// files' footers.
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
fn the_first_query_on_a_reopened_small_database_reads_no_footer() {
    let mut db = Database::in_format(TableFormat::Delta);
    create_table_of_files(&mut db, "t", 3);
    let mut reopened = reopen(db);
    delete_data_files(&reopened, "t");

    let extremes = query_extremes(&mut reopened, "t");

    assert_eq!(extremes, Ok((0, 2)));
}

#[test]
fn the_first_query_on_a_database_of_one_file_too_many_to_preload_reads_its_footers() {
    let mut db = Database::in_format(TableFormat::Delta);
    create_table_of_files(&mut db, "t", MOST_PRELOADED_FILES + 1);
    let mut reopened = reopen(db);

    let read_at_the_first_query = query_extremes(&mut reopened, "t");
    let mut reopened = reopen(reopened);
    delete_data_files(&reopened, "t");
    let unread_at_open = query_extremes(&mut reopened, "t");

    assert_eq!(
        read_at_the_first_query,
        Ok((0, MOST_PRELOADED_FILES as i64))
    );
    assert!(unread_at_open.is_err());
}

#[test]
fn a_table_whose_data_file_is_missing_fails_its_own_query_and_not_the_open() {
    let mut db = Database::in_format(TableFormat::Delta);
    create_table_of_files(&mut db, "broken", 2);
    create_table_of_files(&mut db, "healthy", 2);
    std::fs::remove_file(adopted_directory(&db, "broken").join("0001.parquet")).unwrap();

    let mut reopened = reopen(db);
    let healthy = query_extremes(&mut reopened, "healthy");
    let broken = query_extremes(&mut reopened, "broken");

    assert_eq!(healthy, Ok((0, 1)));
    assert!(broken.unwrap_err().contains("No such file"));
}

#[test]
fn a_table_of_more_partitions_than_are_indexed_together_is_scanned_whole() {
    let mut db = Database::in_format(TableFormat::Delta);
    db.execute("CREATE TABLE t (part BIGINT, value BIGINT) WITH (partition_by = 'part')");
    let rows: Vec<String> = (0..150).map(|part| format!("({part}, {part})")).collect();
    db.execute(&format!("INSERT INTO t VALUES {}", rows.join(", ")));

    let mut reopened = reopen(db);
    let whole = query_extremes(&mut reopened, "t");
    let one_partition = query_extremes_where(&mut reopened, "t", "part = 149");

    assert_eq!(whole, Ok((0, 149)));
    assert_eq!(one_partition, Ok((149, 149)));
}

#[test]
fn a_commit_by_this_process_is_scanned_without_the_log_it_wrote() {
    let mut db = Database::in_format(TableFormat::Delta);
    db.execute("CREATE TABLE t (value BIGINT)");
    db.execute("INSERT INTO t VALUES (1)");
    query_extremes(&mut db, "t").unwrap();
    db.execute("INSERT INTO t VALUES (2)");
    let log = db.table_path("t").join("_delta_log");
    for entry in std::fs::read_dir(&log).unwrap() {
        std::fs::remove_file(entry.unwrap().path()).unwrap();
    }

    let extremes = query_extremes(&mut db, "t");

    assert_eq!(extremes, Ok((1, 2)));
}
