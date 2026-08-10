//! End-to-end blackbox tests: spin up the real `Server` (with a real
//! `dispatch` worker pool and `DeltaDatastore`), connect with a real Postgres
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

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use common::{
    Conn, conn, connect_client, create_people_table, people_batch, server_port, table_dir,
    write_parquet,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rstest::rstest;
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

/// The server-side message of a failed query. `tokio_postgres::Error` renders
/// as a bare "db error", so assertions on what the server said have to read the
/// `DbError` it carries.
fn extract_db_error_message(error: &tokio_postgres::Error) -> String {
    error
        .as_db_error()
        .map_or_else(|| error.to_string(), |db| db.message().to_string())
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
async fn create_table_if_not_exists_is_a_noop_on_an_existing_table(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_if_not_exists", dir.path()).await;

    conn.simple_query("CREATE TABLE IF NOT EXISTS people_if_not_exists (id BIGINT, name VARCHAR)")
        .await
        .unwrap();

    let rows = select_rows(&conn, "SELECT COUNT(*) FROM people_if_not_exists").await;
    assert_eq!(rows, vec![vec![Some("3".into())]]);
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
async fn insert_stores_a_null_constant_as_a_null(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_null_insert", dir.path()).await;

    conn.simple_query("INSERT INTO people_null_insert VALUES (4, NULL), (NULL, NULL)")
        .await
        .unwrap();

    let typed_null = select_rows(
        &conn,
        "SELECT id, name FROM people_null_insert WHERE id = 4",
    )
    .await;
    assert_eq!(typed_null, vec![vec![Some("4".into()), None]]);
    let all_null = select_rows(
        &conn,
        "SELECT id, name FROM people_null_insert WHERE id IS NULL",
    )
    .await;
    assert_eq!(all_null, vec![vec![None, None]]);
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
        "CREATE TABLE doubles (f DOUBLE) WITH (with_pre_existing_parquets = '{}')",
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
        "CREATE TABLE fmetrics (g BIGINT, d DOUBLE, r REAL) WITH (with_pre_existing_parquets = '{}')",
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
        "CREATE TABLE prices (v DECIMAL(10,2)) WITH (with_pre_existing_parquets = '{}')",
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
        "CREATE TABLE decimal_metrics (v DECIMAL(10,2)) WITH (with_pre_existing_parquets = '{}')",
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
        "CREATE TABLE decimal_groups (g DECIMAL(4,1), x BIGINT) WITH (with_pre_existing_parquets = '{}')",
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

/// Create `table` and fill it with `docs`, one document per row.
///
/// The rows go in through the server, so the files in the table's directory are
/// what our own write path produces: the column is annotated VARIANT in the
/// footer and its shredding is chosen from the documents, which is what a table
/// reading them back needs. A plain Arrow writer cannot stand in here.
async fn write_shredded_variant(conn: &Conn, table: &str, docs: &[String]) {
    conn.simple_query(&format!("CREATE TABLE {table} (j VARIANT)"))
        .await
        .unwrap();
    let values: Vec<String> = docs
        .iter()
        .map(|doc| format!("('{}')", doc.replace('\'', "''")))
        .collect();
    conn.simple_query(&format!("INSERT INTO {table} VALUES {}", values.join(",")))
        .await
        .unwrap();
}

/// The `typed_value` leaves of every Parquet file in `dir`: empty when nothing
/// was shredded, so a caller can tell an opaque document from a laid-out one.
fn shredded_leaves(dir: &Path) -> Vec<String> {
    let mut leaves = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "parquet") {
            continue;
        }
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path).unwrap()).unwrap();
        for column in reader.parquet_schema().columns() {
            let name = column.path().string();
            if name.contains("typed_value") {
                leaves.push(name);
            }
        }
    }
    leaves
}

/// 80 documents carrying `a.x`, 20 carrying nothing but `kind`. The second kind
/// is the one that matters: a row with none of the shredded paths, which the
/// reader can only tell apart by the null mask it rebuilds over `typed_value`.
fn mixed_documents() -> Vec<String> {
    let mut docs: Vec<String> = (0..80)
        .map(|i| format!("{{\"kind\":\"commit\",\"a\":{{\"x\":{i}}}}}"))
        .collect();
    docs.extend((0..20).map(|_| "{\"kind\":\"identity\"}".to_string()));
    docs
}

/// Copying a variant from one table into another has to reassemble it: the
/// source hands over the shape *it* shredded into, while the destination shreds
/// for the rows it gets. Rows that have none of the source's shredded paths are
/// what makes this more than a copy, and they must survive it.
///
/// The copy goes through a select list that computes a column, so the variant
/// travels the path that rebuilds output fields rather than the one that passes
/// them through untouched.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn insert_from_a_shredded_variant_reassembles_and_reshreds(#[future] conn: Conn) {
    write_shredded_variant(&conn, "variant_source", &mixed_documents()).await;
    conn.simple_query("CREATE TABLE variant_copy (n BIGINT, j VARIANT)")
        .await
        .unwrap();

    // The computed `1` matters: a select list of nothing but column references
    // is copied through with its fields intact, while one that computes anything
    // rebuilds them, and that is where a variant column can lose the tag saying
    // it is one.
    conn.simple_query("INSERT INTO variant_copy SELECT 1, j FROM variant_source")
        .await
        .unwrap();

    let kinds = select_rows(
        &conn,
        "SELECT CAST(j->'kind' AS VARCHAR) AS kind, count(*) FROM variant_copy \
         GROUP BY kind ORDER BY kind",
    )
    .await;
    assert_eq!(
        kinds,
        vec![
            vec![Some("commit".into()), Some("80".into())],
            vec![Some("identity".into()), Some("20".into())],
        ],
        "every document survives the copy, including the ones with no shredded path"
    );

    // Restricted to the documents that have the path: an aggregate over the
    // rows that do not would be feeding it NULLs, which is a separate gap.
    let total = select_rows(
        &conn,
        "SELECT sum(CAST(j->'a'->'x' AS BIGINT)) FROM variant_copy \
         WHERE CAST(j->'kind' AS VARCHAR) = 'commit'",
    )
    .await;
    assert_eq!(total, vec![vec![Some("3160".into())]], "0..80 summed");

    assert!(
        !shredded_leaves(&table_dir("variant_copy")).is_empty(),
        "the copy shreds for itself; an unshredded write would answer the same \
         queries while storing every document opaque"
    );
}

/// Dot access is the syntax DuckDB folds into the scan's projection, so this
/// query reaches the reader as a pushed-down path rather than as an expression
/// above the scan. It has to answer exactly what the arrow syntax answers.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn dot_access_pushed_into_the_scan_reads_the_same_values(#[future] conn: Conn) {
    write_shredded_variant(&conn, "variant_dot", &mixed_documents()).await;

    let pushed = select_rows(
        &conn,
        "SELECT CAST(j.a.x AS BIGINT) AS x FROM variant_dot \
         WHERE CAST(j.kind AS VARCHAR) = 'commit' ORDER BY x LIMIT 3",
    )
    .await;
    let above_scan = select_rows(
        &conn,
        "SELECT CAST(j->'a'->'x' AS BIGINT) AS x FROM variant_dot \
         WHERE CAST(j->'kind' AS VARCHAR) = 'commit' ORDER BY x LIMIT 3",
    )
    .await;

    assert_eq!(
        pushed,
        vec![
            vec![Some("0".into())],
            vec![Some("1".into())],
            vec![Some("2".into())]
        ]
    );
    assert_eq!(
        pushed, above_scan,
        "the pushed path answers what the arrow syntax does"
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn cte_read_twice_runs_once_and_feeds_both_readers(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_cte", dir.path()).await;

    // Grouping by a unique column gives one row per id, so the self-join below
    // pairs each row of the CTE with itself: a reader that saw only some of the
    // rows loses whole output rows rather than changing a value. DuckDB
    // materializes this CTE rather than inlining a copy per reference, because
    // it ends in an aggregate.
    let rows = select_rows(
        &conn,
        "WITH totals AS (SELECT id, count(*) AS n FROM people_cte GROUP BY id) \
         SELECT a.id, a.n, b.n FROM totals a, totals b WHERE a.id = b.id ORDER BY a.id",
    )
    .await;

    assert_eq!(
        rows,
        vec![
            vec![Some("1".into()), Some("1".into()), Some("1".into())],
            vec![Some("2".into()), Some("1".into()), Some("1".into())],
            vec![Some("3".into()), Some("1".into()), Some("1".into())],
        ]
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn one_query_reads_two_ctes_including_one_built_from_the_other(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_nested_cte", dir.path()).await;

    // `doubled` is built from `totals`, so its definition sits inside the other
    // CTE's body, and the query reads both. Each is read more than once and
    // ends in an aggregate, which is what makes DuckDB materialize the pair
    // instead of inlining a copy per reference.
    let rows = select_rows(
        &conn,
        "WITH totals AS (SELECT id, count(*) AS n FROM people_nested_cte GROUP BY id), \
              doubled AS (SELECT id, sum(n) * 2 AS n2 FROM totals GROUP BY id) \
         SELECT t1.id, t2.n, d1.n2, d2.n2 \
         FROM totals t1, totals t2, doubled d1, doubled d2 \
         WHERE t1.id = t2.id AND t1.id = d1.id AND t1.id = d2.id ORDER BY t1.id",
    )
    .await;

    assert_eq!(
        rows,
        vec![
            vec![
                Some("1".into()),
                Some("1".into()),
                Some("2".into()),
                Some("2".into())
            ],
            vec![
                Some("2".into()),
                Some("1".into()),
                Some("2".into()),
                Some("2".into())
            ],
            vec![
                Some("3".into()),
                Some("1".into()),
                Some("2".into()),
                Some("2".into())
            ],
        ]
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn table_in_a_created_schema_is_queryable(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    let path = dir.path().to_str().unwrap();
    conn.simple_query("CREATE SCHEMA analytics").await.unwrap();

    conn.simple_query(&format!(
        "CREATE TABLE analytics.people (id BIGINT, name VARCHAR) WITH (with_pre_existing_parquets = '{path}')"
    ))
    .await
    .unwrap();

    let rows = select_rows(&conn, "SELECT name FROM analytics.people WHERE id = 2").await;
    assert_eq!(rows, vec![vec![Some("bob".into())]]);
}

/// A write names its table the same way a read does, so the rows an INSERT
/// produces have to land in the table the qualified name resolves to and come
/// back from it alongside the ones already there.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn insert_into_a_schema_qualified_table(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    let path = dir.path().to_str().unwrap();
    conn.simple_query("CREATE SCHEMA staffing").await.unwrap();
    conn.simple_query(&format!(
        "CREATE TABLE staffing.people (id BIGINT, name VARCHAR) WITH (with_pre_existing_parquets = '{path}')"
    ))
    .await
    .unwrap();

    conn.simple_query("INSERT INTO staffing.people VALUES (4, 'dave')")
        .await
        .unwrap();

    let rows = select_rows(&conn, "SELECT id, name FROM staffing.people ORDER BY id").await;
    assert_eq!(
        rows,
        vec![
            vec![Some("1".into()), Some("alice".into())],
            vec![Some("2".into()), Some("bob".into())],
            vec![Some("3".into()), Some("carol".into())],
            vec![Some("4".into()), Some("dave".into())],
        ],
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn a_schema_qualifies_a_table_name(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    let path = dir.path().to_str().unwrap();
    conn.simple_query("CREATE SCHEMA reporting").await.unwrap();
    conn.simple_query(&format!(
        "CREATE TABLE reporting.staff (id BIGINT, name VARCHAR) WITH (with_pre_existing_parquets = '{path}')"
    ))
    .await
    .unwrap();

    // The same bare name in the default schema is a different table.
    let err = conn.simple_query("SELECT id FROM staff").await.unwrap_err();

    let message = extract_db_error_message(&err);
    assert!(
        message.to_lowercase().contains("staff"),
        "expected an unknown-table error naming `staff`, got: {message}",
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn query_against_unknown_schema_errors(#[future] conn: Conn) {
    let err = conn
        .simple_query("SELECT id FROM no_such_schema.people")
        .await
        .unwrap_err();

    let message = extract_db_error_message(&err);
    assert!(
        message.contains("no_such_schema"),
        "expected the error to name the missing schema, got: {message}",
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn create_table_in_an_unknown_schema_errors(#[future] conn: Conn) {
    let err = conn
        .simple_query("CREATE TABLE absent_schema.t (id BIGINT)")
        .await
        .unwrap_err();

    let message = extract_db_error_message(&err);
    assert!(
        message.contains("absent_schema"),
        "expected the error to name the missing schema, got: {message}",
    );
}

/// The set of data files the Delta log currently considers live: every `add`
/// path across the commits, minus every `remove` path.
fn live_log_files(dir: &std::path::Path) -> std::collections::HashSet<String> {
    let mut live = std::collections::HashSet::new();
    let mut commits: Vec<_> = std::fs::read_dir(dir.join("_delta_log"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|e| e == "json"))
        .collect();
    commits.sort();
    for commit in commits {
        for line in std::fs::read_to_string(&commit).unwrap().lines() {
            let action: serde_json::Value = serde_json::from_str(line).unwrap();
            if let Some(add) = action.get("add") {
                live.insert(add["path"].as_str().unwrap().to_string());
            }
            if let Some(remove) = action.get("remove") {
                live.remove(remove["path"].as_str().unwrap());
            }
        }
    }
    live
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn compact_final_merges_small_insert_files(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE compact_me (id BIGINT)")
        .await
        .unwrap();
    for i in 0..5 {
        conn.simple_query(&format!("INSERT INTO compact_me VALUES ({i})"))
            .await
            .unwrap();
    }
    let dir = table_dir("compact_me");
    assert_eq!(live_log_files(&dir).len(), 5);

    conn.simple_query("COMPACT compact_me FINAL").await.unwrap();

    assert_eq!(live_log_files(&dir).len(), 1);
    let rows = select_rows(&conn, "SELECT COUNT(*), SUM(id) FROM compact_me").await;
    assert_eq!(rows, vec![vec![Some("5".into()), Some("10".into())]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn compact_of_a_missing_table_errors(#[future] conn: Conn) {
    let result = conn.simple_query("COMPACT no_such_table FINAL").await;

    let message = format!("{:?}", result.unwrap_err());
    assert!(message.contains("no table named"), "{message}");
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn compact_accepts_a_fully_qualified_table_name(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE compact_qualified (id BIGINT)")
        .await
        .unwrap();
    for i in 0..4 {
        conn.simple_query(&format!("INSERT INTO compact_qualified VALUES ({i})"))
            .await
            .unwrap();
    }

    // The default datastore's name is also a reserved SQL keyword, so the
    // qualified form needs it quoted.
    conn.simple_query("COMPACT \"default\".main.compact_qualified FINAL")
        .await
        .unwrap();

    assert_eq!(live_log_files(&table_dir("compact_qualified")).len(), 1);
    let rows = select_rows(&conn, "SELECT COUNT(*) FROM compact_qualified").await;
    assert_eq!(rows, vec![vec![Some("4".into())]]);
}
