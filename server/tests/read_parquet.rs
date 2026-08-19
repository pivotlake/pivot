//! End-to-end tests of `read_parquet('<location>')`: reading Parquet files a
//! query names directly, over the real server and a real Postgres client.

mod common;

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use common::{Conn, conn, select_rows};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use rstest::rstest;
use tempfile::TempDir;

/// Write a `(id BIGINT, name VARCHAR)` file of the given rows at `path`.
fn write_people(path: &Path, rows: &[(i64, &str)]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let id: ArrayRef = Arc::new(Int64Array::from(
        rows.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
    ));
    let name: ArrayRef = Arc::new(StringArray::from(
        rows.iter().map(|(_, name)| *name).collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(schema.clone(), vec![id, name]).unwrap();
    // The scan reads snappy-compressed pages.
    let properties = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer =
        ArrowWriter::try_new(File::create(path).unwrap(), schema, Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// A directory holding `people-a.parquet` (ids 1-2) and `people-b.parquet`
/// (id 3), plus `other.parquet` of a different schema that a `people-*` pattern
/// must not pick up.
fn people_dir() -> TempDir {
    let dir = TempDir::new().unwrap();
    write_people(
        &dir.path().join("people-a.parquet"),
        &[(1, "alice"), (2, "bob")],
    );
    write_people(&dir.path().join("people-b.parquet"), &[(3, "carol")]);
    write_other(&dir.path().join("other.parquet"));
    dir
}

/// A `(value BIGINT)` file: a schema deliberately unlike the people files'.
fn write_other(path: &Path) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int64,
        false,
    )]));
    let value: ArrayRef = Arc::new(Int64Array::from(vec![7i64]));
    let batch = RecordBatch::try_new(schema.clone(), vec![value]).unwrap();
    let properties = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer =
        ArrowWriter::try_new(File::create(path).unwrap(), schema, Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// The server-side message of a failed query: `tokio_postgres::Error` renders as
/// a bare "db error", so what the server said is read off the `DbError`.
async fn query_error(conn: &Conn, sql: &str) -> String {
    let error = conn.simple_query(sql).await.unwrap_err();
    error
        .as_db_error()
        .map_or_else(|| error.to_string(), |db| db.message().to_string())
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn reads_one_named_file(#[future] conn: Conn) {
    let dir = people_dir();
    let file = dir.path().join("people-a.parquet");

    let rows = select_rows(
        &conn,
        &format!(
            "SELECT id, name FROM read_parquet('{}') ORDER BY id",
            file.display()
        ),
    )
    .await;

    assert_eq!(
        rows,
        vec![
            vec![Some("1".into()), Some("alice".into())],
            vec![Some("2".into()), Some("bob".into())],
        ]
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn a_pattern_reads_every_file_it_matches(#[future] conn: Conn) {
    let dir = people_dir();

    let rows = select_rows(
        &conn,
        &format!(
            "SELECT name FROM read_parquet('{}/people-*.parquet') ORDER BY id",
            dir.path().display()
        ),
    )
    .await;

    assert_eq!(
        rows,
        vec![
            vec![Some("alice".into())],
            vec![Some("bob".into())],
            vec![Some("carol".into())],
        ]
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn a_filter_over_a_pattern_returns_only_matching_rows(#[future] conn: Conn) {
    let dir = people_dir();

    let rows = select_rows(
        &conn,
        &format!(
            "SELECT id, name FROM read_parquet('{}/people-*.parquet') WHERE id = 3",
            dir.path().display()
        ),
    )
    .await;

    assert_eq!(rows, vec![vec![Some("3".into()), Some("carol".into())]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn an_aggregate_reads_every_matched_file(#[future] conn: Conn) {
    let dir = people_dir();

    let rows = select_rows(
        &conn,
        &format!(
            "SELECT COUNT(*), SUM(id) FROM read_parquet('{}/people-*.parquet')",
            dir.path().display()
        ),
    )
    .await;

    assert_eq!(rows, vec![vec![Some("3".into()), Some("6".into())]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn parquet_scan_reads_the_same_files(#[future] conn: Conn) {
    let dir = people_dir();

    let rows = select_rows(
        &conn,
        &format!(
            "SELECT COUNT(*) FROM parquet_scan('{}/people-*.parquet')",
            dir.path().display()
        ),
    )
    .await;

    assert_eq!(rows, vec![vec![Some("3".into())]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn a_location_on_its_own_reads_as_the_files_it_names(#[future] conn: Conn) {
    let dir = people_dir();

    let rows = select_rows(
        &conn,
        &format!("SELECT id FROM '{}/people-b.parquet'", dir.path().display()),
    )
    .await;

    assert_eq!(rows, vec![vec![Some("3".into())]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn a_location_joins_against_a_stored_table(#[future] conn: Conn) {
    let dir = people_dir();
    conn.simple_query("CREATE TABLE roles (id BIGINT, role VARCHAR)")
        .await
        .unwrap();
    conn.simple_query("INSERT INTO roles VALUES (1, 'admin'), (3, 'guest')")
        .await
        .unwrap();

    let rows = select_rows(
        &conn,
        &format!(
            "SELECT people.name, roles.role
             FROM read_parquet('{}/people-*.parquet') AS people
             JOIN roles ON roles.id = people.id
             ORDER BY people.name",
            dir.path().display()
        ),
    )
    .await;

    assert_eq!(
        rows,
        vec![
            vec![Some("alice".into()), Some("admin".into())],
            vec![Some("carol".into()), Some("guest".into())],
        ]
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn a_file_written_after_the_first_query_is_read_by_the_next(#[future] conn: Conn) {
    let dir = people_dir();
    let pattern = format!("{}/people-*.parquet", dir.path().display());
    let before = select_rows(
        &conn,
        &format!("SELECT COUNT(*) FROM read_parquet('{pattern}')"),
    )
    .await;

    write_people(&dir.path().join("people-c.parquet"), &[(4, "dave")]);
    let after = select_rows(
        &conn,
        &format!("SELECT COUNT(*) FROM read_parquet('{pattern}')"),
    )
    .await;

    assert_eq!(before, vec![vec![Some("3".into())]]);
    assert_eq!(after, vec![vec![Some("4".into())]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn a_pattern_matching_nothing_is_an_error(#[future] conn: Conn) {
    let dir = people_dir();

    let message = query_error(
        &conn,
        &format!(
            "SELECT * FROM read_parquet('{}/missing-*.parquet')",
            dir.path().display()
        ),
    )
    .await;

    assert!(message.contains("no file matches"), "{message}");
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn files_of_different_schemas_are_an_error(#[future] conn: Conn) {
    let dir = people_dir();

    let message = query_error(
        &conn,
        &format!(
            "SELECT * FROM read_parquet('{}/*.parquet')",
            dir.path().display()
        ),
    )
    .await;

    assert!(message.contains("more than one schema"), "{message}");
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn a_pattern_above_the_file_name_is_an_error(#[future] conn: Conn) {
    let dir = people_dir();

    let message = query_error(
        &conn,
        &format!(
            "SELECT * FROM read_parquet('{}/*/people-a.parquet')",
            dir.path().display()
        ),
    )
    .await;

    assert!(message.contains("final path segment"), "{message}");
}

/// Row `row`'s payload in [`banded_dir`], wide enough that a query selecting a
/// few rows out of it is one DuckDB late-materializes.
fn payload_of(row: i64) -> String {
    format!("payload-{row:08}-{}", "x".repeat(200))
}

/// Write a `(band BIGINT, payload VARCHAR)` file whose rows run from `first_row`
/// for `ROWS_PER_FILE`, cut into row groups of `BAND_ROWS` whose `band` is the
/// row group's own number. A `band = k` predicate therefore eliminates every
/// other row group by its statistics.
fn write_banded(path: &Path, first_band: i64) {
    const BAND_ROWS: i64 = 512;
    const BANDS_PER_FILE: i64 = 8;
    let schema = Arc::new(Schema::new(vec![
        Field::new("band", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
        Field::new("note", DataType::Utf8, false),
    ]));
    let rows: Vec<i64> = (0..BAND_ROWS * BANDS_PER_FILE).collect();
    let band: ArrayRef = Arc::new(Int64Array::from(
        rows.iter()
            .map(|row| first_band + row / BAND_ROWS)
            .collect::<Vec<_>>(),
    ));
    let payloads: Vec<String> = rows
        .iter()
        .map(|row| payload_of(first_band * BAND_ROWS + row))
        .collect();
    let payload: ArrayRef = Arc::new(StringArray::from(payloads.clone()));
    let note: ArrayRef = Arc::new(StringArray::from(payloads));
    let batch = RecordBatch::try_new(schema.clone(), vec![band, payload, note]).unwrap();
    let properties = WriterProperties::builder()
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_row_count(Some(BAND_ROWS as usize))
        .build();
    let mut writer =
        ArrowWriter::try_new(File::create(path).unwrap(), schema, Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Two banded files: bands 0-7 in the first, 8-15 in the second.
fn banded_dir() -> TempDir {
    let dir = TempDir::new().unwrap();
    write_banded(&dir.path().join("banded-a.parquet"), 0);
    write_banded(&dir.path().join("banded-b.parquet"), 8);
    dir
}

/// A wide `ORDER BY … LIMIT` over a location is one DuckDB rewrites into a
/// narrow scan plus a late re-read of the surviving rows. Those rows are
/// addressed by position within the *pruned* view the narrow scan read, so a
/// re-read that numbered the row groups differently — across two files, with
/// the `band` predicate having dropped fifteen of the sixteen — would return
/// another band's payloads rather than an error.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn a_late_materialized_query_reads_the_row_groups_its_predicate_left(#[future] conn: Conn) {
    let dir = banded_dir();

    let rows = select_rows(
        &conn,
        &format!(
            "SELECT payload, note FROM read_parquet('{}/banded-*.parquet')
             WHERE band = 11 ORDER BY payload LIMIT 3",
            dir.path().display()
        ),
    )
    .await;

    let expected: Vec<Vec<Option<String>>> = (11 * 512..11 * 512 + 3)
        .map(|row| vec![Some(payload_of(row)), Some(payload_of(row))])
        .collect();
    assert_eq!(rows, expected);
}
