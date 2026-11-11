//! Opening a small database loads what its tables' first queries would
//! read of their manifests and of their files' footers; a database of more
//! files is left to its queries.

mod iceberg_support;

use iceberg_support::{Database, collect_i64s, write_int_parquet};

/// The most files a database may hold for an open to preload it.
const MOST_PRELOADED_FILES: usize = 333;

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

/// Delete the manifest list and the manifests of the table `name` and put
/// in the place of each of its data files a file of its size that holds no
/// footer, so whatever reads a manifest or a footer of the table from now
/// on fails.
fn destroy_manifests_and_footers(db: &Database, name: &str) {
    let metadata = db.table_path(name).join("metadata");
    for entry in std::fs::read_dir(&metadata).unwrap() {
        let path = entry.unwrap().path();
        if path
            .extension()
            .is_some_and(|extension| extension == "avro")
        {
            std::fs::remove_file(path).unwrap();
        }
    }
    for entry in std::fs::read_dir(adopted_directory(db, name)).unwrap() {
        let path = entry.unwrap().path();
        let size = std::fs::metadata(&path).unwrap().len() as usize;
        std::fs::write(path, vec![0; size]).unwrap();
    }
}

/// The least and the greatest `value` of the table `name`: what a query
/// answers from the table's manifests and its files' footers.
fn query_extremes(db: &mut Database, name: &str) -> Result<(i64, i64), String> {
    let extremes = db.try_execute(&format!("SELECT min(value), max(value) FROM {name}"))?;
    Ok((collect_i64s(&extremes, 0)[0], collect_i64s(&extremes, 1)[0]))
}

#[test]
fn the_first_query_on_a_reopened_small_database_reads_no_manifest_and_no_footer() {
    let mut db = Database::new();
    create_table_of_files(&mut db, "t", 3);
    let mut reopened = Database::reopen(db.close());
    destroy_manifests_and_footers(&reopened, "t");

    let extremes = query_extremes(&mut reopened, "t");

    assert_eq!(extremes, Ok((0, 2)));
}

#[test]
fn a_database_of_exactly_the_most_preloaded_files_is_preloaded() {
    let mut db = Database::new();
    create_table_of_files(&mut db, "t", MOST_PRELOADED_FILES);
    let mut reopened = Database::reopen(db.close());
    destroy_manifests_and_footers(&reopened, "t");

    let extremes = query_extremes(&mut reopened, "t");

    assert_eq!(extremes, Ok((0, MOST_PRELOADED_FILES as i64 - 1)));
}

#[test]
fn the_first_query_on_a_database_of_one_file_more_reads_its_manifests_and_footers() {
    let mut db = Database::new();
    create_table_of_files(&mut db, "t", MOST_PRELOADED_FILES + 1);
    let mut reopened = Database::reopen(db.close());

    let read_at_the_first_query = query_extremes(&mut reopened, "t");
    let mut reopened = Database::reopen(reopened.close());
    destroy_manifests_and_footers(&reopened, "t");
    let unread_at_open = query_extremes(&mut reopened, "t");

    assert_eq!(
        read_at_the_first_query,
        Ok((0, MOST_PRELOADED_FILES as i64))
    );
    assert!(unread_at_open.is_err());
}

#[test]
fn tables_that_together_hold_more_than_the_most_preloaded_files_are_not_preloaded() {
    let mut db = Database::new();
    create_table_of_files(&mut db, "first", MOST_PRELOADED_FILES / 2 + 1);
    create_table_of_files(&mut db, "second", MOST_PRELOADED_FILES / 2 + 1);
    let mut reopened = Database::reopen(db.close());
    destroy_manifests_and_footers(&reopened, "first");
    destroy_manifests_and_footers(&reopened, "second");

    let first = query_extremes(&mut reopened, "first");
    let second = query_extremes(&mut reopened, "second");

    assert!(first.is_err());
    assert!(second.is_err());
}

#[test]
fn a_table_whose_data_file_is_missing_fails_its_own_query_and_not_the_open() {
    let mut db = Database::new();
    create_table_of_files(&mut db, "broken", 2);
    create_table_of_files(&mut db, "healthy", 2);
    std::fs::remove_file(adopted_directory(&db, "broken").join("0001.parquet")).unwrap();

    let mut reopened = Database::reopen(db.close());
    destroy_manifests_and_footers(&reopened, "healthy");
    let healthy = query_extremes(&mut reopened, "healthy");
    let broken = query_extremes(&mut reopened, "broken");

    assert_eq!(healthy, Ok((0, 1)));
    assert!(broken.unwrap_err().contains("No such file"));
}
