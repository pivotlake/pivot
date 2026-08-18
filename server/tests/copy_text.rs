//! Wire-level tests for the PostgreSQL COPY text format, the protocol's
//! default `COPY <table> FROM STDIN`, including end-to-end runs through a
//! real `psql`, the client the format exists for.

mod common;

use common::{Conn, RawConn, command_tag, conn, select_rows, server_port};
use futures::SinkExt;
use rstest::rstest;

/// Run `actions` against a fresh raw connection off the async runtime.
async fn with_raw_conn<T: Send + 'static>(
    actions: impl FnOnce(&mut RawConn) -> T + Send + 'static,
) -> T {
    let port = server_port();
    tokio::task::spawn_blocking(move || {
        let mut raw = RawConn::connect(port);
        actions(&mut raw)
    })
    .await
    .unwrap()
}

/// Run `psql` against the shared server with the given trailing arguments,
/// panicking (with its stderr) unless it exits cleanly.
fn run_psql(arguments: &[&str]) {
    let port = server_port().to_string();
    let output = std::process::Command::new("psql")
        .args(["-X", "-w", "-v", "ON_ERROR_STOP=1"])
        .args(["-h", "127.0.0.1", "-p", &port, "-U", "pivot", "-d", "test"])
        .args(arguments)
        .output()
        .expect("psql runs these tests end to end; install postgresql-client");
    assert!(
        output.status.success(),
        "psql failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn copy_text_is_the_default_format(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_text_default (id BIGINT, name VARCHAR)")
        .await
        .unwrap();

    let messages = with_raw_conn(|raw| {
        raw.query("COPY copy_text_default FROM STDIN");
        let (kind, payload) = raw.read_message();
        assert_eq!(kind, b'G', "expected CopyInResponse");
        assert_eq!(payload[0], 0, "text copies advertise a text payload");
        raw.copy_data(b"1\talice\n2\t\\N\n");
        raw.copy_done();
        raw.read_until_ready()
    })
    .await;

    assert_eq!(command_tag(&messages).as_deref(), Some("COPY 2"));
    let rows = select_rows(&conn, "SELECT id, name FROM copy_text_default ORDER BY id").await;
    assert_eq!(
        rows,
        vec![
            vec![Some("1".into()), Some("alice".into())],
            vec![Some("2".into()), None],
        ]
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn copy_text_roundtrips_every_column_type(#[future] conn: Conn) {
    conn.simple_query(
        "CREATE TABLE copy_text_types (i INTEGER, u UBIGINT, f DOUBLE, \
         d DECIMAL(10,2), day DATE, ts TIMESTAMP, s VARCHAR, j VARIANT)",
    )
    .await
    .unwrap();

    let messages = with_raw_conn(|raw| {
        raw.query("COPY copy_text_types FROM STDIN");
        let (kind, _) = raw.read_message();
        assert_eq!(kind, b'G', "expected CopyInResponse");
        raw.copy_data(
            b"42\t18446744073709551615\t1.5\t123.45\t2024-01-15\t\
              2020-01-01 00:00:00.5\ta\\tb\t{\"kind\":\"commit\"}\n",
        );
        raw.copy_done();
        raw.read_until_ready()
    })
    .await;

    assert_eq!(command_tag(&messages).as_deref(), Some("COPY 1"));
    let rows = select_rows(
        &conn,
        "SELECT i, u, f, d, day, ts, s, CAST(j->'kind' AS VARCHAR) FROM copy_text_types",
    )
    .await;
    assert_eq!(
        rows,
        vec![vec![
            Some("42".into()),
            Some("18446744073709551615".into()),
            Some("1.5".into()),
            Some("123.45".into()),
            Some("2024-01-15".into()),
            Some("2020-01-01 00:00:00.5".into()),
            Some("a\tb".into()),
            Some("commit".into()),
        ]]
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn copy_text_accepts_frames_split_mid_row_and_the_end_marker(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_text_split (id BIGINT, name VARCHAR)")
        .await
        .unwrap();
    let data = b"1\talice\n2\tbob\n3\tcarol\n\\.\n";

    let messages = with_raw_conn(move |raw| {
        raw.query("COPY copy_text_split FROM STDIN");
        let (kind, _) = raw.read_message();
        assert_eq!(kind, b'G', "expected CopyInResponse");
        // One byte per frame forces every row to reassemble across frames.
        for byte in data {
            raw.copy_data(std::slice::from_ref(byte));
        }
        raw.copy_done();
        raw.read_until_ready()
    })
    .await;

    assert_eq!(command_tag(&messages).as_deref(), Some("COPY 3"));
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn copy_text_column_list_null_fills_the_rest(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_text_columns (id BIGINT, name VARCHAR, note VARCHAR)")
        .await
        .unwrap();

    let messages = with_raw_conn(|raw| {
        raw.query("COPY copy_text_columns (name) FROM STDIN");
        let (kind, _) = raw.read_message();
        assert_eq!(kind, b'G', "expected CopyInResponse");
        raw.copy_data(b"dave\n");
        raw.copy_done();
        raw.read_until_ready()
    })
    .await;

    assert_eq!(command_tag(&messages).as_deref(), Some("COPY 1"));
    let rows = select_rows(&conn, "SELECT id, name, note FROM copy_text_columns").await;
    assert_eq!(rows, vec![vec![None, Some("dave".into()), None]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn copy_text_bad_value_reports_and_rolls_back(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_text_bad_value (id BIGINT)")
        .await
        .unwrap();

    let messages = with_raw_conn(|raw| {
        raw.query("COPY copy_text_bad_value FROM STDIN");
        let (kind, _) = raw.read_message();
        assert_eq!(kind, b'G', "expected CopyInResponse");
        raw.copy_data(b"7\nseven\n");
        raw.copy_done();
        raw.read_until_ready()
    })
    .await;

    assert!(messages.iter().any(|(kind, _)| *kind == b'E'));
    let rows = select_rows(&conn, "SELECT COUNT(*) FROM copy_text_bad_value").await;
    assert_eq!(rows, vec![vec![Some("0".into())]], "nothing was committed");
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn copy_text_wrong_field_count_reports_and_rolls_back(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_text_bad_width (id BIGINT, name VARCHAR)")
        .await
        .unwrap();

    let messages = with_raw_conn(|raw| {
        raw.query("COPY copy_text_bad_width FROM STDIN");
        let (kind, _) = raw.read_message();
        assert_eq!(kind, b'G', "expected CopyInResponse");
        raw.copy_data(b"1\talice\tsurplus\n");
        raw.copy_done();
        raw.read_until_ready()
    })
    .await;

    assert!(messages.iter().any(|(kind, _)| *kind == b'E'));
    let rows = select_rows(&conn, "SELECT COUNT(*) FROM copy_text_bad_width").await;
    assert_eq!(rows, vec![vec![Some("0".into())]], "nothing was committed");
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn copy_text_over_the_extended_protocol(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_text_extended (id BIGINT, name VARCHAR)")
        .await
        .unwrap();

    let sink = conn
        .copy_in("COPY copy_text_extended FROM STDIN")
        .await
        .unwrap();
    let mut sink = Box::pin(sink);
    sink.send(bytes::Bytes::from_static(b"1\talice\n"))
        .await
        .unwrap();
    let rows = sink.as_mut().finish().await.unwrap();

    assert_eq!(rows, 1);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn psql_replays_a_dump_style_copy_script(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE psql_replay (id BIGINT, name VARCHAR)")
        .await
        .unwrap();
    let script_dir = tempfile::TempDir::new().unwrap();
    let script = script_dir.path().join("dump.sql");
    std::fs::write(
        &script,
        "COPY psql_replay (id, name) FROM stdin;\n1\talice\n2\t\\N\n\\.\n",
    )
    .unwrap();

    let script = script.to_str().unwrap().to_string();
    tokio::task::spawn_blocking(move || run_psql(&["-f", &script]))
        .await
        .unwrap();

    let rows = select_rows(&conn, "SELECT id, name FROM psql_replay ORDER BY id").await;
    assert_eq!(
        rows,
        vec![
            vec![Some("1".into()), Some("alice".into())],
            vec![Some("2".into()), None],
        ]
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn psql_copies_a_text_file(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE psql_text_file (id BIGINT, name VARCHAR)")
        .await
        .unwrap();
    let data_dir = tempfile::TempDir::new().unwrap();
    let data = data_dir.path().join("rows.tsv");
    std::fs::write(&data, "3\tcarol\n4\tdave\n").unwrap();

    let command = format!("\\copy psql_text_file from '{}'", data.display());
    tokio::task::spawn_blocking(move || run_psql(&["-c", &command]))
        .await
        .unwrap();

    let rows = select_rows(&conn, "SELECT COUNT(*) FROM psql_text_file").await;
    assert_eq!(rows, vec![vec![Some("2".into())]]);
}
