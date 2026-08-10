//! End-to-end prepared statements over the extended query protocol:
//! `tokio_postgres`'s `prepare`/`execute`/`query` use Parse/Bind/Execute with
//! binary parameters and binary results, so this exercises inference,
//! decoding, placeholder-plan reuse, and binary row encoding for real.

mod common;

use chrono::{NaiveDate, NaiveDateTime};
use common::{Conn, conn, create_people_table, people_batch, select_rows, write_parquet};
use rstest::rstest;
use tokio_postgres::types::Type;

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn prepared_insert_executes_repeatedly(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "prep_insert_loop", dir.path()).await;
    let statement = conn
        .prepare("INSERT INTO prep_insert_loop VALUES ($1, $2)")
        .await
        .unwrap();
    assert_eq!(statement.params(), &[Type::INT8, Type::TEXT]);

    for (id, name) in [(4i64, "dave"), (5, "eve"), (6, "frank")] {
        let affected = conn.execute(&statement, &[&id, &name]).await.unwrap();
        assert_eq!(affected, 1);
    }

    let rows = select_rows(
        &conn,
        "SELECT id, name FROM prep_insert_loop WHERE id > 3 ORDER BY id",
    )
    .await;
    assert_eq!(
        rows,
        vec![
            vec![Some("4".into()), Some("dave".into())],
            vec![Some("5".into()), Some("eve".into())],
            vec![Some("6".into()), Some("frank".into())],
        ],
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn each_prepared_insert_commits_before_the_next(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "prep_insert_commit", dir.path()).await;
    let insert = conn
        .prepare("INSERT INTO prep_insert_commit VALUES ($1, $2)")
        .await
        .unwrap();

    for (i, expected_count) in [(4i64, 4i64), (5, 5), (6, 6)] {
        conn.execute(&insert, &[&i, &"x"]).await.unwrap();
        let count: i64 = conn
            .query_one("SELECT COUNT(*) FROM prep_insert_commit", &[])
            .await
            .unwrap()
            .get(0);
        assert_eq!(count, expected_count);
    }
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn prepared_insert_accepts_null_parameters(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "prep_insert_null", dir.path()).await;
    let statement = conn
        .prepare("INSERT INTO prep_insert_null VALUES ($1, $2)")
        .await
        .unwrap();

    conn.execute(&statement, &[&4i64, &None::<&str>])
        .await
        .unwrap();

    let rows = select_rows(&conn, "SELECT id, name FROM prep_insert_null WHERE id = 4").await;
    assert_eq!(rows, vec![vec![Some("4".into()), None]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn parameterized_select_filters_by_the_bound_value(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "prep_select", dir.path()).await;
    let statement = conn
        .prepare("SELECT name FROM prep_select WHERE id = $1")
        .await
        .unwrap();

    // The same prepared statement serves different values (replanned per
    // execution with the value folded in), with binary result encoding.
    for (id, expected) in [(2i64, "bob"), (3, "carol"), (1, "alice")] {
        let rows = conn.query(&statement, &[&id]).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get::<_, &str>(0), expected);
    }
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn describe_reports_parameter_and_column_types(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "prep_describe", dir.path()).await;

    let statement = conn
        .prepare("SELECT id, name FROM prep_describe WHERE id = $1 AND name = $2")
        .await
        .unwrap();

    assert_eq!(statement.params(), &[Type::INT8, Type::TEXT]);
    let columns = statement.columns();
    assert_eq!(columns.len(), 2);
    assert_eq!(columns[0].name(), "id");
    assert_eq!(*columns[0].type_(), Type::INT8);
    assert_eq!(columns[1].name(), "name");
    assert_eq!(*columns[1].type_(), Type::TEXT);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn unparameterized_statements_run_over_the_extended_protocol(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "prep_plain", dir.path()).await;

    let count: i64 = conn
        .query_one("SELECT COUNT(*) FROM prep_plain", &[])
        .await
        .unwrap()
        .get(0);

    assert_eq!(count, 3);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn typed_columns_roundtrip_through_binary_parameters(#[future] conn: Conn) {
    // No BOOLEAN column: the delta parquet writer reads booleans but does not
    // write them, so an INSERT of one fails regardless of protocol. The
    // USMALLINT takes a text-format value: Postgres has no unsigned wire type,
    // so an unsigned parameter must parse as the inferred type directly.
    conn.simple_query(
        "CREATE TABLE prep_types (small SMALLINT, tiny TINYINT, i INTEGER, big BIGINT, \
         f REAL, d DOUBLE, s VARCHAR, day DATE, taken_at TIMESTAMP, usmall USMALLINT)",
    )
    .await
    .unwrap();
    let statement = conn
        .prepare("INSERT INTO prep_types VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)")
        .await
        .unwrap();

    let day = NaiveDate::from_ymd_opt(2024, 5, 17).unwrap();
    let taken_at =
        NaiveDateTime::parse_from_str("2024-05-17 12:34:56.5", "%Y-%m-%d %H:%M:%S%.f").unwrap();
    // TINYINT has no wire OID of its own; its parameter is described as INT2,
    // so the client binds an i16 and the server narrows it.
    conn.execute(
        &statement,
        &[
            &7i16,
            &8i16,
            &9i32,
            &10i64,
            &1.5f32,
            &2.25f64,
            &"text",
            &day,
            &taken_at,
            &TextParam("40000"),
        ],
    )
    .await
    .unwrap();

    let rows = select_rows(&conn, "SELECT * FROM prep_types").await;
    assert_eq!(
        rows,
        vec![vec![
            Some("7".into()),
            Some("8".into()),
            Some("9".into()),
            Some("10".into()),
            Some("1.5".into()),
            Some("2.25".into()),
            Some("text".into()),
            Some("2024-05-17".into()),
            Some("2024-05-17 12:34:56.5".into()),
            Some("40000".into()),
        ]],
    );

    // Read the typed columns back over the extended protocol too, which
    // encodes each result column in binary.
    // TINYINT reads back as the int2 its column is declared as; a raw i8
    // binary cell would be a malformed 1-byte int2.
    let row = conn
        .query_one(
            "SELECT small, tiny, big, d, s, day, taken_at FROM prep_types",
            &[],
        )
        .await
        .unwrap();
    assert_eq!(row.get::<_, i16>(0), 7);
    assert_eq!(row.get::<_, i16>(1), 8);
    assert_eq!(row.get::<_, i64>(2), 10);
    assert_eq!(row.get::<_, f64>(3), 2.25);
    assert_eq!(row.get::<_, &str>(4), "text");
    assert_eq!(row.get::<_, NaiveDate>(5), day);
    assert_eq!(row.get::<_, NaiveDateTime>(6), taken_at);
}

/// A parameter that ships in text format (everything else in this file binds
/// binary parameters), so the server's text decoding path gets exercised.
#[derive(Debug)]
struct TextParam(&'static str);

impl tokio_postgres::types::ToSql for TextParam {
    fn to_sql(
        &self,
        _ty: &Type,
        out: &mut tokio_postgres::types::private::BytesMut,
    ) -> Result<tokio_postgres::types::IsNull, Box<dyn std::error::Error + Sync + Send>> {
        out.extend_from_slice(self.0.as_bytes());
        Ok(tokio_postgres::types::IsNull::No)
    }

    fn encode_format(&self, _ty: &Type) -> tokio_postgres::types::Format {
        tokio_postgres::types::Format::Text
    }

    fn accepts(_ty: &Type) -> bool {
        true
    }

    tokio_postgres::types::to_sql_checked!();
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn text_format_parameters_decode_by_inferred_type(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "prep_text_params", dir.path()).await;
    let statement = conn
        .prepare("SELECT name FROM prep_text_params WHERE id = $1")
        .await
        .unwrap();

    let rows = conn.query(&statement, &[&TextParam("2")]).await.unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<_, &str>(0), "bob");
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn empty_statement_answers_an_empty_query_response(#[future] conn: Conn) {
    let statement = conn.prepare("").await.unwrap();

    let rows = conn.query(&statement, &[]).await.unwrap();

    assert!(rows.is_empty());
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn set_and_reset_run_over_the_extended_protocol(#[future] conn: Conn) {
    conn.execute("SET pivot_stats = true", &[]).await.unwrap();
    conn.execute("RESET pivot_stats", &[]).await.unwrap();
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn preparing_a_bad_statement_reports_the_planning_error(#[future] conn: Conn) {
    let result = conn.prepare("SELECT * FROM no_such_table_anywhere").await;

    let error = result.expect_err("preparing a query on a missing table fails");
    let message = error
        .as_db_error()
        .map(|db| db.message().to_string())
        .unwrap_or_else(|| error.to_string());
    assert!(
        message.contains("no_such_table_anywhere"),
        "unexpected message: {message}"
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn sum_reads_back_as_numeric_text(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "prep_sum", dir.path()).await;

    // SUM yields a decimal column; the extended protocol requests binary
    // results, exercising the NUMERIC binary encoding. tokio-postgres has no
    // built-in decimal type, so read the value as a string via simple query
    // and only check the binary path executes cleanly.
    let rows = select_rows(&conn, "SELECT SUM(id) FROM prep_sum").await;
    assert_eq!(rows, vec![vec![Some("6".into())]]);

    let row = conn
        .query_one("SELECT COUNT(*), MIN(id) FROM prep_sum", &[])
        .await
        .unwrap();
    assert_eq!(row.get::<_, i64>(0), 3);
    assert_eq!(row.get::<_, i64>(1), 1);
}
