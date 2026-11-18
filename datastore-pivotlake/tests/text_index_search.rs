//! Searching a table's text indexes through SQL: which searches for a token
//! a table's binding takes from above its scan, and that the rows come from
//! the index files a table finds when it is loaded and from those a
//! compaction writes.

mod iceberg_support;

use iceberg_support::{Database, collect_i64s, names_in};
use planner::catalog::{BoundTable, SchemaQualifiedTableName};
use planner::expression::{Expression, Function, HasToken, Ref};
use planner::types::Type;
use std::path::PathBuf;

/// `has_token(message, token)` as it is pushed into a table whose second
/// column is `message`.
fn message_has_token(token: &str) -> Expression {
    Expression::Function(Function::HasToken(HasToken {
        text: Box::new(Expression::Ref(Ref {
            column_idx: 1,
            return_type: Type::Utf8,
            name: None,
        })),
        token: token.to_string(),
    }))
}

/// Whether a binding of the table `name` takes a search of `message` for a
/// token from above its scan.
fn takes_token_search(db: &Database, name: &str) -> bool {
    let mut bound = db
        .datastore
        .clone()
        .begin_transaction()
        .table(
            planner::DEFAULT_DATASTORE_NAME,
            &SchemaQualifiedTableName::in_default_schema(name),
        )
        .unwrap()
        .expect("the table exists");
    bound.pushdown_filter(message_has_token("disk")).unwrap()
}

/// A table `name` with a text index `idx_message` of `message` that keeps
/// letter case, and one file holding `rows`, written after the index.
fn create_indexed_logs(db: &mut Database, name: &str, rows: &str) {
    db.execute(&format!("CREATE TABLE {name} (id BIGINT, message VARCHAR)"));
    db.execute(&format!(
        "CREATE TEXT INDEX idx_message ON {name} (message) WITH (case_sensitive = true)"
    ));
    db.execute(&format!("INSERT INTO {name} VALUES {rows}"));
}

/// The one index file the index `idx_message` of the table `name` has.
fn find_index_file(db: &Database, name: &str) -> PathBuf {
    let index_directory = db.table_path(name).join("indexes").join("idx_message");
    let index_file = names_in(&index_directory)
        .into_iter()
        .find(|entry| entry.ends_with(".textindex"))
        .expect("the table's one file has an index file");
    index_directory.join(index_file)
}

#[test]
fn a_search_for_a_token_is_taken_from_above_the_scan_only_by_an_index_that_keeps_letter_case() {
    let mut db = Database::new();
    db.execute("CREATE TABLE unindexed_logs (id BIGINT, message VARCHAR)");
    db.execute("CREATE TABLE lowered_logs (id BIGINT, message VARCHAR)");
    db.execute("CREATE TABLE exact_logs (id BIGINT, message VARCHAR)");
    db.execute("CREATE TEXT INDEX idx_lowered ON lowered_logs (message)");
    db.execute("CREATE TEXT INDEX idx_exact ON exact_logs (message) WITH (case_sensitive = true)");

    let taken = ["unindexed_logs", "lowered_logs", "exact_logs"]
        .map(|table| takes_token_search(&db, table));

    assert_eq!(taken, [false, false, true]);
}

#[test]
fn a_reopened_table_answers_a_token_search_from_its_index_files() {
    let mut db = Database::new();
    create_indexed_logs(&mut db, "logs", "(1, 'alpha'), (2, 'beta')");
    create_indexed_logs(&mut db, "swapped_logs", "(1, 'beta'), (2, 'alpha')");
    // The index file of `logs` now says what that of `swapped_logs` does, so
    // only an answer read from the index file names row 2. One read off the
    // rows names row 1.
    std::fs::copy(
        find_index_file(&db, "swapped_logs"),
        find_index_file(&db, "logs"),
    )
    .unwrap();
    let mut db = Database::reopen(db.close());

    let found = db.execute("SELECT id FROM logs WHERE has_token(message, 'alpha')");

    assert_eq!(collect_i64s(&found, 0), [2]);
}

#[test]
fn a_compacted_table_finds_tokens_in_the_index_files_of_its_merged_files() {
    let mut db = Database::new();
    create_indexed_logs(
        &mut db,
        "logs",
        "(1, 'Disk full'), (2, 'disk full on sda1')",
    );
    db.execute("INSERT INTO logs VALUES (3, 'network down'), (4, 'sda1 is full')");

    db.compact("logs", true);
    let found = db.execute("SELECT id FROM logs WHERE has_token(message, 'sda1') ORDER BY id");

    assert_eq!(collect_i64s(&found, 0), [2, 4]);
}
