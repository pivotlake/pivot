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

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
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
async fn insert_returns_affected_row_count(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_insert", dir.path()).await;

    let messages = conn
        .simple_query("INSERT INTO people_insert VALUES (4, 'dave'), (5, 'eve')")
        .await
        .unwrap();

    assert_eq!(messages.len(), 1, "INSERT should not return a data row");
    match &messages[0] {
        SimpleQueryMessage::CommandComplete(rows) => assert_eq!(*rows, 2),
        other => panic!("expected INSERT command completion, got {other:?}"),
    }

    let rows = select_rows(&conn, "SELECT id FROM people_insert ORDER BY id").await;
    assert_eq!(rows.len(), 5);
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

/// `SELECT drop_cache()` evicts pivot's compressed cache: after a scan populates it,
/// the first drop reports ≥1 region freed and an immediate second drop reports
/// 0 (nothing left to evict). Serialised against the shared server by the
/// `conn` fixture, so no other query touches the cache in between.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn drop_cache_evicts_compressed_cache(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_drop_cache", dir.path()).await;

    // Scan the table so its parquet region lands in the compressed cache.
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
/// never reads a data page, so it leaves the compressed cache empty. Proven by
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

/// A global MIN/MAX over a DOUBLE column returns the exact float extremes. (This
/// also guards the old metadata short-circuit, which only fires for
/// integer/temporal columns: a float column declines it and computes from the
/// scan, so the bounds are never cast through Int64 and truncated to [3, 9].)
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn global_min_max_over_double_column(#[future] conn: Conn) {
    let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, false)]));
    let f: ArrayRef = Arc::new(arrow_array::Float64Array::from(vec![3.7f64, 9.2, 5.0]));
    let dir = write_parquet(&RecordBatch::try_new(schema, vec![f]).unwrap());
    conn.simple_query(&format!(
        "CREATE TABLE doubles (f DOUBLE) WITH (path = '{}')",
        dir.path().to_str().unwrap()
    ))
    .await
    .unwrap();

    let rows = select_rows(&conn, "SELECT MIN(f), MAX(f) FROM doubles").await;

    let min: f64 = rows[0][0].as_deref().unwrap().parse().unwrap();
    let max: f64 = rows[0][1].as_deref().unwrap().parse().unwrap();
    assert_eq!(min, 3.7);
    assert_eq!(max, 9.2);
}

/// Grouped SUM/AVG over a DOUBLE value column, with a REAL column alongside, end
/// to end through the SQL server.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn grouped_float_aggregates(#[future] conn: Conn) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("g", DataType::Int64, false),
        Field::new("d", DataType::Float64, false),
        Field::new("r", DataType::Float32, false),
    ]));
    let g: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 1, 2]));
    let d: ArrayRef = Arc::new(arrow_array::Float64Array::from(vec![1.5f64, 2.5, 4.0]));
    let r: ArrayRef = Arc::new(arrow_array::Float32Array::from(vec![1.0f32, 3.0, 5.0]));
    let dir = write_parquet(&RecordBatch::try_new(schema, vec![g, d, r]).unwrap());
    conn.simple_query(&format!(
        "CREATE TABLE fmetrics (g BIGINT, d DOUBLE, r REAL) WITH (path = '{}')",
        dir.path().to_str().unwrap()
    ))
    .await
    .unwrap();

    // Clear any cache residue from prior tests so the grouped aggregate has the
    // whole ring (the shared test server runs with a small buffer pool).
    select_one_i64(&conn, "SELECT drop_cache()").await;
    let rows = select_rows(
        &conn,
        "SELECT g, SUM(d), AVG(d), MIN(r) FROM fmetrics GROUP BY g ORDER BY g",
    )
    .await;

    let parse = |v: &Option<String>| -> f64 { v.as_deref().unwrap().parse().unwrap() };
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0][0].as_deref(), Some("1"));
    assert_eq!(parse(&rows[0][1]), 4.0); // SUM(d) group 1: 1.5 + 2.5
    assert_eq!(parse(&rows[0][2]), 2.0); // AVG(d) group 1
    assert_eq!(parse(&rows[0][3]), 1.0); // MIN(r) group 1
    assert_eq!(parse(&rows[1][1]), 4.0); // SUM(d) group 2
}

/// A DECIMAL column scans, filters against a decimal constant, and renders
/// scale-correct NUMERIC text on the wire. The file is written by arrow-rs,
/// which stores the decimal as FIXED_LEN_BYTE_ARRAY.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn decimal_scan_filter_and_wire_format(#[future] conn: Conn) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "v",
        DataType::Decimal128(10, 2),
        false,
    )]));
    let v: ArrayRef = Arc::new(
        arrow_array::Decimal128Array::from(vec![1234i128, -500, 1050])
            .with_precision_and_scale(10, 2)
            .unwrap(),
    );
    let dir = write_parquet(&RecordBatch::try_new(schema, vec![v]).unwrap());
    conn.simple_query(&format!(
        "CREATE TABLE prices (v DECIMAL(10,2)) WITH (path = '{}')",
        dir.path().to_str().unwrap()
    ))
    .await
    .unwrap();

    let rows = select_rows(&conn, "SELECT v FROM prices WHERE v > 10.00 ORDER BY v").await;

    assert_eq!(
        rows,
        vec![vec![Some("10.50".into())], vec![Some("12.34".into())]]
    );
}

/// Global SUM/MIN/MAX over a DECIMAL column keep the exact fixed-point values
/// and the bound output scale; AVG comes back as a double.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn decimal_aggregates(#[future] conn: Conn) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "v",
        DataType::Decimal128(10, 2),
        false,
    )]));
    let v: ArrayRef = Arc::new(
        arrow_array::Decimal128Array::from(vec![1234i128, -500, 1050])
            .with_precision_and_scale(10, 2)
            .unwrap(),
    );
    let dir = write_parquet(&RecordBatch::try_new(schema, vec![v]).unwrap());
    conn.simple_query(&format!(
        "CREATE TABLE decimal_metrics (v DECIMAL(10,2)) WITH (path = '{}')",
        dir.path().to_str().unwrap()
    ))
    .await
    .unwrap();

    let rows = select_rows(
        &conn,
        "SELECT SUM(v), MIN(v), MAX(v), AVG(v) FROM decimal_metrics",
    )
    .await;

    assert_eq!(rows[0][0].as_deref(), Some("17.84"));
    assert_eq!(rows[0][1].as_deref(), Some("-5.00"));
    assert_eq!(rows[0][2].as_deref(), Some("12.34"));
    let avg: f64 = rows[0][3].as_deref().unwrap().parse().unwrap();
    assert!((avg - 17.84 / 3.0).abs() < 1e-9);
}

/// GROUP BY a DECIMAL key: groups form on the exact fixed-point value and the
/// key column renders at its declared scale.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn group_by_decimal_key(#[future] conn: Conn) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("g", DataType::Decimal128(4, 1), false),
        Field::new("x", DataType::Int64, false),
    ]));
    let g: ArrayRef = Arc::new(
        arrow_array::Decimal128Array::from(vec![15i128, 15, 20])
            .with_precision_and_scale(4, 1)
            .unwrap(),
    );
    let x: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2, 3]));
    let dir = write_parquet(&RecordBatch::try_new(schema, vec![g, x]).unwrap());
    conn.simple_query(&format!(
        "CREATE TABLE decimal_groups (g DECIMAL(4,1), x BIGINT) WITH (path = '{}')",
        dir.path().to_str().unwrap()
    ))
    .await
    .unwrap();

    let rows = select_rows(
        &conn,
        "SELECT g, SUM(x) FROM decimal_groups GROUP BY g ORDER BY g",
    )
    .await;

    assert_eq!(
        rows,
        vec![
            vec![Some("1.5".into()), Some("3".into())],
            vec![Some("2.0".into()), Some("3".into())],
        ]
    );
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
