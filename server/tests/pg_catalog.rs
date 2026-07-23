//! The virtual `pg_catalog` system tables a JDBC client's catalog introspection
//! reads. A `CREATE TABLE` must then show up in `pg_class`/`pg_attribute` so the
//! client can describe it.

mod common;

use common::{Conn, conn};
use rstest::rstest;
use tempfile::TempDir;
use tokio_postgres::SimpleQueryMessage;

async fn select_rows(conn: &Conn, sql: &str) -> Result<Vec<Vec<Option<String>>>, String> {
    conn.simple_query(sql)
        .await
        .map(|msgs| {
            msgs.into_iter()
                .filter_map(|m| match m {
                    SimpleQueryMessage::Row(r) => Some(
                        (0..r.len())
                            .map(|i| r.get(i).map(str::to_string))
                            .collect::<Vec<_>>(),
                    ),
                    _ => None,
                })
                .collect()
        })
        .map_err(|e| e.to_string())
}

async fn create_people(conn: &Conn, name: &str) -> TempDir {
    let dir = TempDir::new().unwrap();
    conn.simple_query(&format!(
        "CREATE TABLE {name} (id BIGINT, name VARCHAR) WITH (path = '{}')",
        dir.path().to_str().unwrap()
    ))
    .await
    .unwrap();
    dir
}

/// `pg_class` (unqualified) lists the user's tables.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn unqualified_pg_class_lists_user_tables(#[future] conn: Conn) {
    let _dir = create_people(&conn, "pcu").await;

    let rows = select_rows(&conn, "SELECT relname FROM pg_class WHERE relname = 'pcu'").await;

    assert_eq!(
        rows,
        Ok(vec![vec![Some("pcu".into())]]),
        "unqualified pg_class"
    );
}

/// `pg_catalog.pg_class` (schema-qualified, as the JDBC driver writes it).
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn qualified_pg_catalog_pg_class(#[future] conn: Conn) {
    let _dir = create_people(&conn, "pcq").await;

    let rows = select_rows(
        &conn,
        "SELECT relname FROM pg_catalog.pg_class WHERE relname = 'pcq'",
    )
    .await;

    assert_eq!(
        rows,
        Ok(vec![vec![Some("pcq".into())]]),
        "qualified pg_catalog.pg_class"
    );
}

/// `pg_attribute` lists a table's columns with a `pg_class` join.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn pg_attribute_lists_columns(#[future] conn: Conn) {
    let _dir = create_people(&conn, "pca").await;

    let rows = select_rows(
        &conn,
        "SELECT a.attname FROM pg_attribute a JOIN pg_class c ON a.attrelid = c.oid \
         WHERE c.relname = 'pca' ORDER BY a.attnum",
    )
    .await;

    assert_eq!(
        rows,
        Ok(vec![vec![Some("id".into())], vec![Some("name".into())]]),
        "pg_attribute columns via join"
    );
}
