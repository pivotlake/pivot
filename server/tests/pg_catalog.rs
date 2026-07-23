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
        .map_err(|e| {
            e.as_db_error()
                .map(|db| db.message().to_string())
                .unwrap_or_else(|| e.to_string())
        })
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

/// The DDL/DML shape the Kafka Connect sink emits: an auto-created table with
/// `NOT NULL` columns, then a `INSERT INTO t (cols) VALUES (…)` naming its
/// columns in table order.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn sink_create_not_null_and_column_list_insert(#[future] conn: Conn) {
    let dir = tempfile::TempDir::new().unwrap();
    conn.simple_query(&format!(
        "CREATE TABLE sink (id BIGINT NOT NULL, name TEXT NOT NULL) WITH (path = '{}')",
        dir.path().to_str().unwrap()
    ))
    .await
    .unwrap();
    conn.simple_query("INSERT INTO sink (id, name) VALUES (1, 'alice')")
        .await
        .unwrap();

    let rows = select_rows(&conn, "SELECT id, name FROM sink").await;
    assert_eq!(rows, Ok(vec![vec![Some("1".into()), Some("alice".into())]]));
}

/// pgjdbc's real getTables query and its `name`-type connection-setup probe
/// both execute (the `::name`/`::regclass` casts DuckDB lacks are normalized).
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn jdbc_gettables_and_name_setup_execute(#[future] conn: Conn) {
    let _dir = create_people(&conn, "gt").await;

    let name_setup = "SELECT length(repeat('1234567890', 1000)::NAME)";
    assert!(
        select_rows(&conn, name_setup).await.is_ok(),
        "name-type setup query"
    );

    let gettables = "SELECT NULL AS TABLE_CAT, n.nspname AS TABLE_SCHEM, c.relname AS TABLE_NAME, \
      CASE n.nspname ~ '^pg_' OR n.nspname = 'information_schema' \
        WHEN true THEN CASE WHEN n.nspname = 'pg_catalog' OR n.nspname = 'information_schema' \
          THEN CASE c.relkind WHEN 'r' THEN 'SYSTEM TABLE' WHEN 'v' THEN 'SYSTEM VIEW' ELSE NULL END \
          ELSE CASE c.relkind WHEN 'r' THEN 'TEMPORARY TABLE' WHEN 'v' THEN 'TEMPORARY VIEW' ELSE NULL END END \
        WHEN false THEN CASE c.relkind WHEN 'r' THEN 'TABLE' WHEN 'v' THEN 'VIEW' WHEN 'm' THEN 'MATERIALIZED VIEW' ELSE NULL END \
        ELSE NULL END AS TABLE_TYPE, d.description AS REMARKS \
      FROM pg_catalog.pg_namespace n, pg_catalog.pg_class c \
        LEFT JOIN pg_catalog.pg_description d ON (c.oid = d.objoid AND d.objsubid = 0 AND d.classoid = 'pg_class'::regclass) \
      WHERE c.relnamespace = n.oid AND c.relname LIKE 'gt' \
        AND (false OR (c.relkind = 'r' AND n.nspname !~ '^pg_' AND n.nspname <> 'information_schema')) \
      ORDER BY TABLE_TYPE, TABLE_SCHEM, TABLE_NAME";
    let rows = select_rows(&conn, gettables).await;
    assert_eq!(
        rows.as_ref()
            .map(|r| r.iter().map(|row| row[2].clone()).collect::<Vec<_>>()),
        Ok(vec![Some("gt".into())]),
        "getTables should list the 'gt' table: {rows:?}"
    );
}

/// pgjdbc's getPrimaryKeys (the `_pg_expandarray` over `pg_index` form) returns
/// empty: pivot tables have no indexes, so a table has no primary keys.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn jdbc_getprimarykeys_returns_empty(#[future] conn: Conn) {
    let _dir = create_people(&conn, "pk").await;

    let getprimarykeys = "SELECT result.TABLE_CAT, result.TABLE_NAME, result.COLUMN_NAME, \
        result.KEY_SEQ, result.PK_NAME FROM ( \
          SELECT NULL AS TABLE_CAT, ct.relname AS TABLE_NAME, a.attname AS COLUMN_NAME, \
            (information_schema._pg_expandarray(i.indkey)).n AS KEY_SEQ, ci.relname AS PK_NAME \
          FROM pg_catalog.pg_class ct JOIN pg_catalog.pg_index i ON (ct.oid = i.indrelid) \
          WHERE ct.relname = 'pk' AND i.indisprimary) result ORDER BY result.KEY_SEQ";

    let rows = select_rows(&conn, getprimarykeys).await;
    assert_eq!(rows, Ok(vec![]), "getPrimaryKeys should be empty: {rows:?}");
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

/// Slices of pgjdbc's real getColumns, up to the whole query, all execute:
/// virtual pg_catalog tables + joins + LEFT joins + the row_number window +
/// pg_get_expr/nullif, and the IN-list that would otherwise materialize.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn jdbc_getcolumns_slices_all_execute(#[future] conn: Conn) {
    let _dir = create_people(&conn, "probe").await;
    let variants: [(&str, &str); 9] = [
        (
            "relkind-eq",
            "SELECT relname FROM pg_class WHERE relkind = 'r'",
        ),
        (
            "in-2vals",
            "SELECT relname FROM pg_class WHERE relkind IN ('r', 'p')",
        ),
        (
            "in-5vals",
            "SELECT relname FROM pg_class WHERE relkind IN ('r', 'p', 'v', 'f', 'm')",
        ),
        (
            "in-5vals-different-col",
            "SELECT relname FROM pg_class WHERE relname IN ('a', 'b', 'c', 'd', 'e')",
        ),
        (
            "inner-joins",
            "SELECT n.nspname, c.relname, a.attname, a.atttypid, t.typtype \
             FROM pg_catalog.pg_namespace n \
               JOIN pg_catalog.pg_class c ON (c.relnamespace = n.oid) \
               JOIN pg_catalog.pg_attribute a ON (a.attrelid = c.oid) \
               JOIN pg_catalog.pg_type t ON (a.atttypid = t.oid) \
             WHERE c.relname = 'probe' AND a.attnum > 0",
        ),
        (
            "left-joins",
            "SELECT a.attname, def.adbin, dsc.description \
             FROM pg_catalog.pg_class c \
               JOIN pg_catalog.pg_attribute a ON (a.attrelid = c.oid) \
               LEFT JOIN pg_catalog.pg_attrdef def ON (a.attrelid = def.adrelid AND a.attnum = def.adnum) \
               LEFT JOIN pg_catalog.pg_description dsc ON (c.oid = dsc.objoid AND a.attnum = dsc.objsubid) \
             WHERE c.relname = 'probe'",
        ),
        (
            "row_number-window",
            "SELECT a.attname, row_number() OVER (PARTITION BY a.attrelid ORDER BY a.attnum) AS attnum \
             FROM pg_catalog.pg_class c JOIN pg_catalog.pg_attribute a ON (a.attrelid = c.oid) \
             WHERE c.relname = 'probe'",
        ),
        (
            "pg_get_expr+nullif",
            "SELECT a.attname, nullif(a.attidentity, '') AS attidentity, \
             pg_catalog.pg_get_expr(def.adbin, def.adrelid) AS adsrc \
             FROM pg_catalog.pg_class c JOIN pg_catalog.pg_attribute a ON (a.attrelid = c.oid) \
             LEFT JOIN pg_catalog.pg_attrdef def ON (a.attrelid = def.adrelid AND a.attnum = def.adnum) \
             WHERE c.relname = 'probe'",
        ),
        (
            "full-getColumns",
            "SELECT * FROM ( \
               SELECT n.nspname, c.relname, a.attname, a.atttypid, \
                 a.attnotnull OR (t.typtype = 'd' AND t.typnotnull) AS attnotnull, \
                 a.atttypmod, a.attlen, t.typtypmod, \
                 row_number() OVER (PARTITION BY a.attrelid ORDER BY a.attnum) AS attnum, \
                 nullif(a.attidentity, '') AS attidentity, nullif(a.attgenerated, '') AS attgenerated, \
                 pg_catalog.pg_get_expr(def.adbin, def.adrelid) AS adsrc, \
                 dsc.description, t.typbasetype, t.typtype \
               FROM pg_catalog.pg_namespace n \
                 JOIN pg_catalog.pg_class c ON (c.relnamespace = n.oid) \
                 JOIN pg_catalog.pg_attribute a ON (a.attrelid = c.oid) \
                 JOIN pg_catalog.pg_type t ON (a.atttypid = t.oid) \
                 LEFT JOIN pg_catalog.pg_attrdef def ON (a.attrelid = def.adrelid AND a.attnum = def.adnum) \
                 LEFT JOIN pg_catalog.pg_description dsc ON (c.oid = dsc.objoid AND a.attnum = dsc.objsubid) \
               WHERE c.relkind in ('r','p','v','f','m') AND a.attnum > 0 AND NOT a.attisdropped \
                 AND c.relname = 'probe' \
               ORDER BY nspname, c.relname, attnum \
             ) c WHERE true",
        ),
    ];

    let mut report = Vec::new();
    for (label, sql) in variants {
        match select_rows(&conn, sql).await {
            Ok(rows) => report.push(format!("{label}: OK ({} rows)", rows.len())),
            Err(e) => report.push(format!("{label}: ERR {e}")),
        }
    }
    let failed: Vec<&String> = report.iter().filter(|r| r.contains("ERR")).collect();
    assert!(
        failed.is_empty(),
        "getColumns feature gaps:\n{}",
        report.join("\n")
    );
}
