//! The table format a database's index document records: what a document
//! that records none is read as, the spellings it accepts, and what a
//! rewrite of the document keeps.

mod iceberg_support;

use datastore_pivotlake::TableFormat;
use iceberg_support::Database;
use tempfile::TempDir;

const INDEX_FILE: &str = "_pivot_manifest.json";

/// A database directory holding only the index document `index`.
fn create_directory_with_index(index: &str) -> TempDir {
    let directory = TempDir::new().unwrap();
    std::fs::write(directory.path().join(INDEX_FILE), index).unwrap();
    directory
}

/// The index document of `db` as its directory holds it now.
fn read_index(db: &Database) -> String {
    std::fs::read_to_string(db.path().join(INDEX_FILE)).unwrap()
}

#[test]
fn a_document_without_a_format_is_a_delta_database_and_stays_so_when_rewritten() {
    let directory =
        create_directory_with_index(r#"{"version":3,"schemas":[{"name":"main","table_ids":{}}]}"#);

    let mut db = Database::reopen_in_format(directory, TableFormat::Delta);
    db.execute("CREATE TABLE t (value BIGINT)");

    let rewritten = read_index(&db);
    assert!(db.table_path("t").join("_delta_log").is_dir());
    assert!(rewritten.contains(&db.table_directory("t")), "{rewritten}");
    assert!(!rewritten.contains("table_format"), "{rewritten}");
}

#[test]
fn a_recorded_format_round_trips_and_accepts_the_deltalake_alias() {
    let iceberg = create_directory_with_index(r#"{"version":1,"table_format":"iceberg"}"#);
    let aliased = create_directory_with_index(r#"{"version":1,"table_format":"deltalake"}"#);

    let mut iceberg = Database::reopen_in_format(iceberg, TableFormat::Iceberg);
    let mut aliased = Database::reopen_in_format(aliased, TableFormat::Delta);
    iceberg.execute("CREATE TABLE t (value BIGINT)");
    aliased.execute("CREATE TABLE t (value BIGINT)");

    let rewritten = read_index(&iceberg);
    assert!(iceberg.table_path("t").join("metadata").is_dir());
    assert!(aliased.table_path("t").join("_delta_log").is_dir());
    assert!(
        rewritten.contains(&iceberg.table_directory("t")),
        "{rewritten}"
    );
    assert!(
        rewritten.contains(r#""table_format":"iceberg""#),
        "{rewritten}"
    );
}
