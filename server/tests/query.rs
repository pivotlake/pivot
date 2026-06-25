//! End-to-end blackbox tests: spin up the real `Server` (with a real
//! `dispatch` worker pool and `ParquetCatalog`), connect with a real Postgres
//! client (`tokio-postgres`), and exercise the full
//! `CREATE TABLE` → `SELECT` → wire-encoding flow.
//!
//! The shared server, serial guard, and connected client are all wired up by
//! the [`common::conn`] rstest fixture — tests just take `conn: Conn` and go
//! straight to setup/execute/assert.

mod common;

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use common::{Conn, conn, connect_client, server_port};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rstest::rstest;
use tempfile::TempDir;
use tokio_postgres::{Client, SimpleQueryMessage};

/// Run `sql` and decode every `DataRow` in the response into
/// `Vec<Option<String>>` (text format). Non-row messages (e.g. `RowDescription`,
/// `CommandComplete`) are dropped — tests assert on data only.
async fn select_rows(client: &Client, sql: &str) -> Vec<Vec<Option<String>>> {
    let msgs = client.simple_query(sql).await.unwrap();
    msgs.into_iter()
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

/// Write `batch` as a single parquet file inside a fresh tempdir and return
/// the directory (kept alive by the caller — drop it to clean up).
fn write_parquet(batch: &RecordBatch) -> TempDir {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("data.parquet");
    // The dispatch reader assumes snappy-compressed pages — match that.
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer =
        ArrowWriter::try_new(File::create(&path).unwrap(), batch.schema(), Some(props)).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
    dir
}

/// (id BIGINT, name VARCHAR) batch with three rows.
fn people_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let id: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2, 3]));
    let name: ArrayRef = Arc::new(StringArray::from(vec!["alice", "bob", "carol"]));
    RecordBatch::try_new(schema, vec![id, name]).unwrap()
}

async fn create_people_table(client: &Client, table: &str, dir: &Path) {
    let path = dir.to_str().unwrap();
    client
        .simple_query(&format!(
            "CREATE TABLE {table} (id BIGINT, name VARCHAR) WITH (path = '{path}')"
        ))
        .await
        .unwrap();
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn create_table_and_query(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_filter", dir.path()).await;

    let rows = select_rows(&conn, "SELECT id, name FROM people_filter WHERE id = 2").await;

    assert_eq!(rows, vec![vec![Some("2".into()), Some("bob".into())]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn second_connection_sees_table_created_by_first(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    let reader = connect_client(server_port()).await;
    create_people_table(&conn, "people_shared", dir.path()).await;

    let rows = select_rows(&reader, "SELECT id, name FROM people_shared").await;

    assert_eq!(
        rows,
        vec![
            vec![Some("1".into()), Some("alice".into())],
            vec![Some("2".into()), Some("bob".into())],
            vec![Some("3".into()), Some("carol".into())],
        ],
    );
}

/// Decode a one-row, one-column integer result (e.g. `SELECT drop_cache()`).
async fn select_one_i64(client: &Client, sql: &str) -> i64 {
    let rows = select_rows(client, sql).await;
    assert_eq!(rows.len(), 1, "expected exactly one row from `{sql}`");
    rows[0][0]
        .as_deref()
        .expect("non-null scalar")
        .parse()
        .expect("integer scalar")
}

/// `SELECT drop_cache()` evicts pivot's file cache: after a scan populates it,
/// the first drop reports ≥1 region freed and an immediate second drop reports
/// 0 (nothing left to evict). Serialised against the shared server by the
/// `conn` fixture, so no other query touches the cache in between.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn drop_cache_evicts_file_memory_cache(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_drop_cache", dir.path()).await;

    // Scan the table so its parquet region lands in the file cache.
    let rows = select_rows(&conn, "SELECT id, name FROM people_drop_cache").await;
    assert_eq!(rows.len(), 3);

    // First drop frees the cached region(s); an immediate second drop finds none.
    let first = select_one_i64(&conn, "SELECT drop_cache()").await;
    assert!(first >= 1, "expected ≥1 region evicted, got {first}");
    let second = select_one_i64(&conn, "SELECT drop_cache()").await;
    assert_eq!(second, 0, "second drop should find an empty cache");
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn global_min_max_answers_from_metadata(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_minmax", dir.path()).await;

    let rows = select_rows(&conn, "SELECT MIN(id), MAX(id) FROM people_minmax").await;

    assert_eq!(rows, vec![vec![Some("1".into()), Some("3".into())]]);
}

/// The unfiltered global MIN/MAX short-circuits to parquet row-group stats and
/// never reads a data page, so it leaves the file cache empty. Proven by
/// clearing the cache (of any footer-load residue), running the aggregate, and
/// finding nothing to evict afterwards.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn global_min_max_does_not_scan(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_minmax_noscan", dir.path()).await;

    select_one_i64(&conn, "SELECT drop_cache()").await;
    select_rows(&conn, "SELECT MIN(id), MAX(id) FROM people_minmax_noscan").await;
    let evicted = select_one_i64(&conn, "SELECT drop_cache()").await;

    assert_eq!(evicted, 0, "metadata short-circuit must not scan any pages");
}

/// A WHERE clause excludes rows, so the metadata short-circuit is unsound and
/// must not fire: the aggregate returns the *filtered* extremes (not the whole
/// table's) and actually scans pages (the cache has something to evict after).
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn filtered_min_max_does_not_short_circuit(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_minmax_filtered", dir.path()).await;

    select_one_i64(&conn, "SELECT drop_cache()").await;
    let rows = select_rows(
        &conn,
        "SELECT MIN(id), MAX(id) FROM people_minmax_filtered WHERE id > 1",
    )
    .await;
    let evicted = select_one_i64(&conn, "SELECT drop_cache()").await;

    assert_eq!(rows, vec![vec![Some("2".into()), Some("3".into())]]);
    assert!(
        evicted >= 1,
        "a filtered aggregate must scan, not read stats"
    );
}

/// A global MIN/MAX over a DOUBLE column returns the real float extremes, not
/// the Int64-truncated [3, 9] the metadata short-circuit once produced; that
/// path declines non-integer columns, so the scan-based float aggregate runs.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn global_min_max_over_double_column(#[future] conn: Conn) {
    let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, false)]));
    let f: ArrayRef = Arc::new(Float64Array::from(vec![3.7f64, 9.2, 5.0]));
    let dir = write_parquet(&RecordBatch::try_new(schema, vec![f]).unwrap());
    conn.simple_query(&format!(
        "CREATE TABLE doubles (f DOUBLE) WITH (path = '{}')",
        dir.path().to_str().unwrap()
    ))
    .await
    .unwrap();

    let rows = select_floats(&conn, "SELECT MIN(f), MAX(f) FROM doubles").await;

    assert_eq!(rows, vec![vec![Some(3.7), Some(9.2)]]);
}

/// (g BIGINT, v DOUBLE) batch for the float-aggregate tests. Group 1 holds
/// {1.5, 2.5}; group 2 holds {10.0, 20.0, 30.0}.
fn float_group_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("g", DataType::Int64, false),
        Field::new("v", DataType::Float64, false),
    ]));
    let g: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 1, 2, 2, 2]));
    let v: ArrayRef = Arc::new(Float64Array::from(vec![1.5f64, 2.5, 10.0, 20.0, 30.0]));
    RecordBatch::try_new(schema, vec![g, v]).unwrap()
}

async fn create_floats_table(client: &Client, table: &str, dir: &Path) {
    let path = dir.to_str().unwrap();
    client
        .simple_query(&format!(
            "CREATE TABLE {table} (g BIGINT, v DOUBLE) WITH (path = '{path}')"
        ))
        .await
        .unwrap();
}

/// Run `sql` and decode each cell as an `Option<f64>` (text format, `NULL` →
/// `None`). Float results are compared numerically to avoid wire-format
/// formatting brittleness.
async fn select_floats(client: &Client, sql: &str) -> Vec<Vec<Option<f64>>> {
    select_rows(client, sql)
        .await
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|cell| cell.map(|s| s.parse::<f64>().expect("float scalar")))
                .collect()
        })
        .collect()
}

/// Global SUM/AVG/MIN/MAX over a DOUBLE column, folded in `f64`.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn global_float_aggregates(#[future] conn: Conn) {
    let dir = write_parquet(&float_group_batch());
    create_floats_table(&conn, "floats_global", dir.path()).await;

    let rows = select_floats(
        &conn,
        "SELECT SUM(v), AVG(v), MIN(v), MAX(v) FROM floats_global",
    )
    .await;

    assert_eq!(rows, vec![vec![Some(64.0), Some(12.8), Some(1.5), Some(30.0)]]);
}

/// Grouped SUM/AVG/MIN/MAX over a DOUBLE column.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn grouped_float_aggregates(#[future] conn: Conn) {
    let dir = write_parquet(&float_group_batch());
    create_floats_table(&conn, "floats_grouped", dir.path()).await;

    let rows = select_floats(
        &conn,
        "SELECT g, SUM(v), AVG(v), MIN(v), MAX(v) FROM floats_grouped GROUP BY g ORDER BY g",
    )
    .await;

    assert_eq!(
        rows,
        vec![
            vec![Some(1.0), Some(4.0), Some(2.0), Some(1.5), Some(2.5)],
            vec![Some(2.0), Some(60.0), Some(20.0), Some(10.0), Some(30.0)],
        ],
    );
}

/// A float SUM over zero matching rows is SQL `NULL`, not a fabricated `0.0`.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn global_float_sum_of_no_rows_is_null(#[future] conn: Conn) {
    let dir = write_parquet(&float_group_batch());
    create_floats_table(&conn, "floats_empty", dir.path()).await;

    let rows = select_floats(&conn, "SELECT SUM(v) FROM floats_empty WHERE g = 99").await;

    assert_eq!(rows, vec![vec![None]]);
}

/// An unfiltered global COUNT(*) is answered from the sum of parquet row-group
/// row counts, with no scan.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn global_count_star_answers_from_metadata(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_count", dir.path()).await;

    select_one_i64(&conn, "SELECT drop_cache()").await;
    let count = select_one_i64(&conn, "SELECT COUNT(*) FROM people_count").await;
    let evicted = select_one_i64(&conn, "SELECT drop_cache()").await;

    assert_eq!(count, 3);
    assert_eq!(evicted, 0, "count from metadata must not scan any pages");
}

/// A WHERE clause excludes rows, so the COUNT(*) short-circuit is unsound and
/// must not fire: the count reflects the filter, not the whole table.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn filtered_count_star_is_correct(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_count_filtered", dir.path()).await;

    let count = select_one_i64(
        &conn,
        "SELECT COUNT(*) FROM people_count_filtered WHERE id > 1",
    )
    .await;

    assert_eq!(count, 2);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn query_against_unknown_table_errors(#[future] conn: Conn) {
    let err = conn
        .simple_query("SELECT id FROM does_not_exist")
        .await
        .unwrap_err();

    assert!(
        err.to_string().to_lowercase().contains("does_not_exist")
            || err.to_string().to_lowercase().contains("not found")
            || err.code().is_some(),
        "expected a structured pgwire error, got: {err}",
    );
}
