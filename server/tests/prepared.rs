//! End-to-end tests for the Postgres **extended query protocol** (prepared
//! statements). `tokio-postgres`'s typed `prepare`/`execute` drive the real
//! Parse/Bind/Describe/Execute flow and send parameters in **binary**, so these
//! exercise the parameter-binding path the Kafka Connect JDBC sink relies on.
//!
//! Prepared `INSERT ... VALUES ($1, …)` statements gather bound values straight
//! into Arrow columns, while scalar parameters bind through normal expression
//! compilation for result-producing statements.

mod common;

use common::{Conn, conn};
use rstest::rstest;
use tempfile::TempDir;
use tokio_postgres::SimpleQueryMessage;

/// Run `sql` over the simple protocol and decode every data row to text. Used to
/// read back what a prepared INSERT wrote.
async fn select_rows(conn: &Conn, sql: &str) -> Vec<Vec<Option<String>>> {
    conn.simple_query(sql)
        .await
        .unwrap()
        .into_iter()
        .filter_map(|m| match m {
            SimpleQueryMessage::Row(r) => Some(
                (0..r.len())
                    .map(|i| r.get(i).map(|s| s.to_string()))
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .collect()
}

/// `CREATE TABLE name (id BIGINT, name VARCHAR)` backed by a fresh (empty)
/// directory so it is writable. Returns the dir (keep it alive for the test).
async fn create_writable_table(conn: &Conn, name: &str) -> TempDir {
    let dir = TempDir::new().unwrap();
    conn.simple_query(&format!(
        "CREATE TABLE {name} (id BIGINT, name VARCHAR) WITH (path = '{}')",
        dir.path().to_str().unwrap()
    ))
    .await
    .unwrap();
    dir
}

/// A prepared `INSERT ... (cols) VALUES ($1, $2)` binds each row's parameters
/// (in binary) and lands the rows.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn prepared_insert_with_column_list(#[future] conn: Conn) {
    let _dir = create_writable_table(&conn, "prep_collist").await;

    let stmt = conn
        .prepare("INSERT INTO prep_collist (id, name) VALUES ($1, $2)")
        .await
        .unwrap();
    let affected = conn.execute(&stmt, &[&1i64, &"alice"]).await.unwrap();
    conn.execute(&stmt, &[&2i64, &"bob"]).await.unwrap();

    assert_eq!(affected, 1, "each INSERT reports one affected row");
    let rows = select_rows(&conn, "SELECT id, name FROM prep_collist ORDER BY id").await;
    assert_eq!(
        rows,
        vec![
            vec![Some("1".into()), Some("alice".into())],
            vec![Some("2".into()), Some("bob".into())],
        ],
    );
}

/// A positional `VALUES ($1, $2)` with no column list also binds and inserts.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn prepared_insert_positional(#[future] conn: Conn) {
    let _dir = create_writable_table(&conn, "prep_positional").await;

    let stmt = conn
        .prepare("INSERT INTO prep_positional VALUES ($1, $2)")
        .await
        .unwrap();
    conn.execute(&stmt, &[&7i64, &"carol"]).await.unwrap();

    let rows = select_rows(&conn, "SELECT id, name FROM prep_positional").await;
    assert_eq!(rows, vec![vec![Some("7".into()), Some("carol".into())]]);
}

/// A multi-row prepared `VALUES ($1,$2),($3,$4)` gathers one column from
/// parameters [0, 2] and the other from [1, 3] into a single inserted batch.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn prepared_insert_multiple_rows(#[future] conn: Conn) {
    let _dir = create_writable_table(&conn, "prep_multi").await;

    let stmt = conn
        .prepare("INSERT INTO prep_multi (id, name) VALUES ($1, $2), ($3, $4)")
        .await
        .unwrap();
    let affected = conn
        .execute(&stmt, &[&1i64, &"a", &2i64, &"b"])
        .await
        .unwrap();

    assert_eq!(affected, 2);
    let rows = select_rows(&conn, "SELECT id, name FROM prep_multi ORDER BY id").await;
    assert_eq!(
        rows,
        vec![
            vec![Some("1".into()), Some("a".into())],
            vec![Some("2".into()), Some("b".into())],
        ],
    );
}

/// Pointer columns retain each parameter's own declared type, then cast the
/// gathered values to their shared output type.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn prepared_values_gather_mixed_parameter_types(#[future] conn: Conn) {
    use tokio_postgres::types::Type;

    let dir = TempDir::new().unwrap();
    conn.simple_query(&format!(
        "CREATE TABLE prep_mixed (id BIGINT) WITH (path = '{}')",
        dir.path().to_str().unwrap()
    ))
    .await
    .unwrap();
    let statement = conn
        .prepare_typed(
            "INSERT INTO prep_mixed VALUES ($1), ($2)",
            &[Type::INT4, Type::INT8],
        )
        .await
        .unwrap();

    let affected = conn.execute(&statement, &[&1i32, &2i64]).await.unwrap();

    assert_eq!(statement.params(), &[Type::INT4, Type::INT8]);
    assert_eq!(affected, 2);
    assert_eq!(
        select_rows(&conn, "SELECT id FROM prep_mixed ORDER BY id").await,
        vec![vec![Some("1".into())], vec![Some("2".into())]],
    );
}

/// A prepared `SELECT ... WHERE id = $1` binds the scalar parameter (the "single
/// value pointer") and returns only the matching row. DuckDB doesn't type a
/// parameter used only in a comparison, so the client must **declare** its type
/// in Parse, here via `prepare_typed`. `tokio-postgres` requests **binary**
/// result columns, so this also exercises binary result encoding.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn parameterized_select_filters_by_bound_value(#[future] conn: Conn) {
    use tokio_postgres::types::Type;

    let _dir = create_writable_table(&conn, "prep_where").await;
    let insert = conn
        .prepare("INSERT INTO prep_where (id, name) VALUES ($1, $2)")
        .await
        .unwrap();
    conn.execute(&insert, &[&1i64, &"alice"]).await.unwrap();
    conn.execute(&insert, &[&2i64, &"bob"]).await.unwrap();

    let select = conn
        .prepare_typed(
            "SELECT id, name FROM prep_where WHERE id = $1",
            &[Type::INT8],
        )
        .await
        .unwrap();
    let rows = conn.query(&select, &[&2i64]).await.unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<_, i64>(0), 2);
    assert_eq!(rows[0].get::<_, &str>(1), "bob");
}

/// A comparison parameter whose type the client does not declare (plain `query`
/// sends no parameter types, and DuckDB can't resolve one from `WHERE id = $1`)
/// is rejected with a clear error rather than mis-decoded.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn undeclared_comparison_parameter_errors(#[future] conn: Conn) {
    let _dir = create_writable_table(&conn, "prep_undeclared").await;

    let result = conn
        .query("SELECT id FROM prep_undeclared WHERE id = $1", &[&1i64])
        .await;

    assert!(
        result.is_err(),
        "an undeclared comparison parameter must error, not run: {result:?}"
    );
}
