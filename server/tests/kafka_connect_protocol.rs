//! Black-box coverage for the PostgreSQL protocol sequence used by the JDBC
//! sink: explicit transaction, DDL, prepared INSERT batch, then commit.

mod common;

use common::{Conn, conn};
use rstest::rstest;
use tokio_postgres::SimpleQueryMessage;
use tokio_postgres::types::Type;

async fn simple_scalar(conn: &Conn, sql: &str) -> String {
    conn.simple_query(sql)
        .await
        .unwrap()
        .into_iter()
        .find_map(|message| match message {
            SimpleQueryMessage::Row(row) => row.get(0).map(str::to_string),
            _ => None,
        })
        .unwrap()
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn prepared_insert_batch_inside_transaction(#[future] conn: Conn) {
    conn.batch_execute("BEGIN").await.unwrap();
    conn.batch_execute("CREATE TABLE connect_prepared_batch (id BIGINT, name VARCHAR)")
        .await
        .unwrap();
    let insert = conn
        .prepare_typed(
            "INSERT INTO connect_prepared_batch (id, name) VALUES ($1, $2)",
            &[Type::INT8, Type::VARCHAR],
        )
        .await
        .unwrap();

    let first = conn.execute(&insert, &[&1_i64, &"first"]).await.unwrap();
    let second = conn.execute(&insert, &[&2_i64, &"second"]).await.unwrap();
    conn.batch_execute("COMMIT").await.unwrap();

    assert_eq!((first, second), (1, 1));
    let count = simple_scalar(&conn, "SELECT COUNT(*) FROM connect_prepared_batch").await;
    assert_eq!(
        count, "0",
        "the temporary fake INSERT must not persist rows"
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn rollback_ends_failed_transaction(#[future] conn: Conn) {
    conn.batch_execute("BEGIN").await.unwrap();
    let missing = conn
        .prepare_typed(
            "INSERT INTO missing_connect_table VALUES ($1)",
            &[Type::INT4],
        )
        .await
        .unwrap();
    let error = conn.execute(&missing, &[&1_i32]).await.unwrap_err();

    let aborted = conn.simple_query("SELECT 1").await.unwrap_err();
    conn.batch_execute("ROLLBACK").await.unwrap();
    let value = simple_scalar(&conn, "SELECT 1").await;

    assert_eq!(error.code().unwrap().code(), "XX000");
    assert_eq!(aborted.code().unwrap().code(), "25P02");
    assert_eq!(value, "1");
}
