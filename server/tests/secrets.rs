//! End-to-end secrets tests over the Postgres wire: `CREATE SECRET` /
//! `DROP SECRET` through a real client, listed back via `pivot_secrets()`.
//!
//! The shared server's catalog is process-global, so each test uses uniquely
//! named secrets and filters `pivot_secrets()` by name.

mod common;

use common::{Conn, conn, select_rows};
use rstest::rstest;
use tokio_postgres::{Client, SimpleQueryMessage, error::SqlState};

/// The `pivot_secrets()` rows for one secret name.
async fn secret_rows(client: &Client, name: &str) -> Vec<Vec<Option<String>>> {
    select_rows(
        client,
        &format!(
            "SELECT name, type, provider, persistent, scope, secret_string \
             FROM pivot_secrets() WHERE name = '{name}'"
        ),
    )
    .await
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn create_secret_lists_redacted(#[future] conn: Conn) {
    let tags = conn
        .simple_query(
            "CREATE SECRET wire_secret (TYPE s3, KEY_ID 'AKIAEXAMPLE', SECRET 'hunter2', \
             SESSION_TOKEN 'tok', REGION 'eu-west-1', SCOPE 's3://bkt/pfx')",
        )
        .await
        .unwrap();

    let rows = secret_rows(&conn, "wire_secret").await;

    assert!(matches!(&tags[0], SimpleQueryMessage::CommandComplete(_)));
    assert_eq!(
        rows,
        vec![vec![
            Some("wire_secret".into()),
            Some("s3".into()),
            Some("config".into()),
            Some("t".into()),
            Some("s3://bkt/pfx".into()),
            Some(
                "key_id=AKIAEXAMPLE;region=eu-west-1;secret=redacted;session_token=redacted".into()
            ),
        ]]
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn duplicate_secret_errors_and_or_replace_overwrites(#[future] conn: Conn) {
    conn.simple_query("CREATE SECRET dup_secret (TYPE s3, KEY_ID 'a', SECRET 'b')")
        .await
        .unwrap();

    let err = conn
        .simple_query("CREATE SECRET dup_secret (TYPE s3, KEY_ID 'a', SECRET 'b')")
        .await
        .unwrap_err();
    conn.simple_query(
        "CREATE OR REPLACE SECRET dup_secret (TYPE s3, KEY_ID 'replaced', SECRET 'b')",
    )
    .await
    .unwrap();
    // IF NOT EXISTS leaves the replaced secret untouched.
    conn.simple_query(
        "CREATE SECRET IF NOT EXISTS dup_secret (TYPE s3, KEY_ID 'ignored', SECRET 'b')",
    )
    .await
    .unwrap();

    assert!(
        err.as_db_error()
            .unwrap()
            .message()
            .contains("already exists")
    );
    let rows = secret_rows(&conn, "dup_secret").await;
    assert_eq!(
        rows[0][5],
        Some("key_id=replaced;secret=redacted".to_string())
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn drop_secret_removes_it(#[future] conn: Conn) {
    conn.simple_query("CREATE SECRET drop_me (TYPE s3, KEY_ID 'a', SECRET 'b')")
        .await
        .unwrap();

    conn.simple_query("DROP SECRET drop_me").await.unwrap();

    assert!(secret_rows(&conn, "drop_me").await.is_empty());
    let err = conn.simple_query("DROP SECRET drop_me").await.unwrap_err();
    assert!(
        err.as_db_error()
            .unwrap()
            .message()
            .contains("does not exist")
    );
    conn.simple_query("DROP SECRET IF EXISTS drop_me")
        .await
        .unwrap();
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn temporary_secret_lists_as_not_persistent_with_default_scope(#[future] conn: Conn) {
    conn.simple_query("CREATE TEMPORARY SECRET temp_secret (TYPE s3, KEY_ID 'a', SECRET 'b')")
        .await
        .unwrap();

    let rows = secret_rows(&conn, "temp_secret").await;

    assert_eq!(rows[0][3], Some("f".to_string()));
    // With no SCOPE given, the secret covers every s3 path (DuckDB's default).
    assert_eq!(rows[0][4], Some("s3://;s3n://;s3a://".to_string()));
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn quoted_mixed_case_secret_can_be_dropped(#[future] conn: Conn) {
    // DuckDB lowercases the CREATE name but passes the DROP name verbatim;
    // both must land on the same registry entry.
    conn.simple_query("CREATE SECRET \"CasedSecret\" (TYPE s3, KEY_ID 'a', SECRET 'b')")
        .await
        .unwrap();

    conn.simple_query("DROP SECRET \"CasedSecret\"")
        .await
        .unwrap();

    assert!(secret_rows(&conn, "casedsecret").await.is_empty());
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn null_secret_option_is_rejected(#[future] conn: Conn) {
    let err = conn
        .simple_query("CREATE SECRET null_option (TYPE s3, KEY_ID NULL, SECRET 'b')")
        .await
        .unwrap_err();

    assert!(
        err.as_db_error()
            .unwrap()
            .message()
            .contains("cannot be NULL")
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn unknown_secret_option_is_rejected(#[future] conn: Conn) {
    let err = conn
        .simple_query("CREATE SECRET bad_option (TYPE s3, PASSWORD 'nope')")
        .await
        .unwrap_err();

    let db_err = err.as_db_error().unwrap();
    assert_eq!(db_err.code(), &SqlState::INTERNAL_ERROR);
    assert!(db_err.message().contains("Unknown parameter 'password'"));
}
