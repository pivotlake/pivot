//! End-to-end blackbox tests for the extended query protocol (prepared
//! statements): `tokio-postgres`'s `prepare`/`query`/`execute` all go through
//! Parse/Bind/Describe/Execute, with binary result columns, against the real
//! server wired up by the [`common::conn`] fixture.

mod common;

use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use chrono::{NaiveDate, NaiveDateTime};
use common::{Conn, conn, select_rows};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rstest::rstest;
use tempfile::TempDir;
use tokio_postgres::Client;
use tokio_postgres::types::Type;

/// Write `batch` as a single snappy-compressed parquet file inside a fresh
/// tempdir and return the directory (kept alive by the caller).
fn write_parquet(batch: &RecordBatch) -> TempDir {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("data.parquet");
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
    dir
}

/// Two seed rows: (1, 'one'), (2, 'two').
fn kv_batch() -> RecordBatch {
    let k: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
    let v: ArrayRef = Arc::new(StringArray::from(vec!["one", "two"]));
    RecordBatch::try_from_iter(vec![("k", k), ("v", v)]).unwrap()
}

async fn create_kv_table(client: &Client, table: &str, dir: &Path) {
    client
        .simple_query(&format!(
            "CREATE TABLE {table} (k BIGINT, v VARCHAR) WITH (path = '{}')",
            dir.display()
        ))
        .await
        .unwrap();
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn insert_with_bound_parameters_appends_a_row(#[future] conn: Conn) {
    let dir = write_parquet(&kv_batch());
    create_kv_table(&conn, "ext_insert", dir.path()).await;

    let inserted = conn
        .execute("INSERT INTO ext_insert VALUES ($1, $2)", &[&3i64, &"three"])
        .await
        .unwrap();

    assert_eq!(inserted, 1);
    let rows = select_rows(&conn, "SELECT k, v FROM ext_insert WHERE k = 3").await;
    assert_eq!(rows, vec![vec![Some("3".into()), Some("three".into())]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn select_with_a_bound_parameter_filters_rows(#[future] conn: Conn) {
    let dir = write_parquet(&kv_batch());
    create_kv_table(&conn, "ext_select", dir.path()).await;

    let rows = conn
        .query("SELECT k, v FROM ext_select WHERE k = $1", &[&2i64])
        .await
        .unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<_, i64>(0), 2);
    assert_eq!(rows[0].get::<_, &str>(1), "two");
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn a_prepared_statement_is_reusable_with_new_values(#[future] conn: Conn) {
    let dir = write_parquet(&kv_batch());
    create_kv_table(&conn, "ext_reuse", dir.path()).await;
    let statement = conn
        .prepare("SELECT v FROM ext_reuse WHERE k = $1")
        .await
        .unwrap();

    let first = conn.query(&statement, &[&1i64]).await.unwrap();
    let second = conn.query(&statement, &[&2i64]).await.unwrap();

    assert_eq!(first[0].get::<_, &str>(0), "one");
    assert_eq!(second[0].get::<_, &str>(0), "two");
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn describe_reports_parameter_and_result_types(#[future] conn: Conn) {
    let dir = write_parquet(&kv_batch());
    create_kv_table(&conn, "ext_describe", dir.path()).await;

    let select = conn
        .prepare("SELECT k, v FROM ext_describe WHERE k = $1 AND v = $2")
        .await
        .unwrap();
    let insert = conn
        .prepare("INSERT INTO ext_describe VALUES ($1, $2)")
        .await
        .unwrap();

    assert_eq!(select.params(), &[Type::INT8, Type::TEXT]);
    assert_eq!(select.columns()[0].name(), "k");
    assert_eq!(select.columns()[0].type_(), &Type::INT8);
    assert_eq!(select.columns()[1].name(), "v");
    assert_eq!(select.columns()[1].type_(), &Type::TEXT);
    assert_eq!(insert.params(), &[Type::INT8, Type::TEXT]);
    assert!(insert.columns().is_empty());
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn a_parameter_without_an_inferable_type_is_rejected_at_prepare(#[future] conn: Conn) {
    let error = conn.prepare("SELECT $1").await.unwrap_err();

    let message = error
        .as_db_error()
        .map_or_else(|| error.to_string(), |db| db.message().to_string());
    assert!(
        message.contains("could not be inferred"),
        "unexpected error: {message}"
    );
}

/// A NULL parameter must bind as SQL NULL, not as a value. `$1 IS NULL` is
/// true exactly when it did: the filter passes every row for a NULL binding
/// and none for a non-NULL one. (An INSERT of a NULL is not assertable here:
/// the datastore's insert path stamps a non-nullable write schema, so
/// inserting NULLs is unsupported regardless of protocol.)
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn a_null_parameter_binds_as_sql_null(#[future] conn: Conn) {
    let dir = write_parquet(&kv_batch());
    create_kv_table(&conn, "ext_null", dir.path()).await;
    let statement = conn
        .prepare("SELECT COUNT(*) FROM ext_null WHERE $1::VARCHAR IS NULL")
        .await
        .unwrap();

    let with_null = conn
        .query(&statement, &[&Option::<&str>::None])
        .await
        .unwrap();
    let with_value = conn.query(&statement, &[&"present"]).await.unwrap();

    assert_eq!(with_null[0].get::<_, i64>(0), 2);
    assert_eq!(with_value[0].get::<_, i64>(0), 0);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn an_unparameterized_extended_query_returns_binary_results(#[future] conn: Conn) {
    let dir = write_parquet(&kv_batch());
    create_kv_table(&conn, "ext_no_params", dir.path()).await;

    let rows = conn
        .query("SELECT COUNT(*), MIN(k) FROM ext_no_params", &[])
        .await
        .unwrap();

    assert_eq!(rows[0].get::<_, i64>(0), 2);
    assert_eq!(rows[0].get::<_, i64>(1), 1);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn date_and_timestamp_parameters_bind(#[future] conn: Conn) {
    let dir = write_parquet(&kv_batch());
    create_kv_table(&conn, "ext_temporal", dir.path()).await;
    let date = NaiveDate::from_ymd_opt(2024, 5, 17).unwrap();
    let timestamp =
        NaiveDateTime::parse_from_str("2024-05-17 10:30:00", "%Y-%m-%d %H:%M:%S").unwrap();

    let rows = conn
        .query(
            "SELECT k FROM ext_temporal WHERE k = $1 AND $2 = DATE '2024-05-17' \
             AND $3 = TIMESTAMP '2024-05-17 10:30:00'",
            &[&1i64, &date, &timestamp],
        )
        .await
        .unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<_, i64>(0), 1);
}
