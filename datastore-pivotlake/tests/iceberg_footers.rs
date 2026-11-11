//! What a managed table's metadata and the footers of the files a query
//! works on already tell is answered without reading a page.

mod iceberg_support;

use iceberg_support::{Database, shared_dispatcher};

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
fn the_extremes_of_a_column_decompress_no_page() {
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
