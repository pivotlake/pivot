//! End-to-end tests for live stats: after `SET live_stats_interval_ms`, the
//! server sends notices about a statement while it runs: first its operators,
//! then every worker's per-operator stats.

mod common;

use common::{Conn, RawConn, conn, server_port};
use rstest::rstest;

/// What every live-stats notice's message starts with; the rest is one
/// message as JSON.
const LIVE_STATS_NOTICE_PREFIX: &str = "live_stats: ";

/// Run `statements` in order on one fresh connection and return every message
/// the server sent for the last of them.
async fn run_on_one_connection(statements: &'static [&'static str]) -> Vec<(u8, Vec<u8>)> {
    let port = server_port();
    tokio::task::spawn_blocking(move || {
        let mut raw = RawConn::connect(port);
        let mut last_response = Vec::new();
        for statement in statements {
            raw.query(statement);
            last_response = raw.read_until_ready();
        }
        last_response
    })
    .await
    .unwrap()
}

/// The live-stats messages among `messages`, in the order they were sent: the
/// JSON of each notice whose message field starts with
/// [`LIVE_STATS_NOTICE_PREFIX`]. A notice's payload is a list of fields, each
/// a type byte and a NUL-terminated string; `M` is the message.
fn parse_live_stats(messages: &[(u8, Vec<u8>)]) -> Vec<serde_json::Value> {
    messages
        .iter()
        .filter(|(kind, _)| *kind == b'N')
        .filter_map(|(_, payload)| {
            payload.split(|byte| *byte == 0).find_map(|field| {
                let message = std::str::from_utf8(field.strip_prefix(b"M")?).unwrap();
                let json = message.strip_prefix(LIVE_STATS_NOTICE_PREFIX)?;
                Some(serde_json::from_str(json).unwrap())
            })
        })
        .collect()
}

/// Every custom stat the workers sent in `live_stats`, whichever worker and
/// operator sent it.
fn custom_stats(live_stats: &[serde_json::Value]) -> Vec<serde_json::Value> {
    live_stats
        .iter()
        .filter(|message| message["type"] == "worker")
        .flat_map(|message| message["operators"].as_array().unwrap().clone())
        .flat_map(|operator| operator["custom_stats"].as_array().unwrap().clone())
        .collect()
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn a_scan_with_live_stats_on_names_the_row_group_it_fetched(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE live_stats_scanned (id BIGINT)")
        .await
        .unwrap();
    conn.simple_query("INSERT INTO live_stats_scanned VALUES (1), (2), (3)")
        .await
        .unwrap();

    let response = run_on_one_connection(&[
        "SET live_stats_interval_ms = 1",
        "SELECT id FROM live_stats_scanned",
    ])
    .await;

    let fetched: Vec<serde_json::Value> = custom_stats(&parse_live_stats(&response))
        .into_iter()
        .filter(|stat| stat["name"] == "row_group_read" || stat["name"] == "row_group_from_memory")
        .collect();
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0]["value"]["row_group"], 0);
    assert!(
        fetched[0]["value"]["file"]
            .as_str()
            .unwrap()
            .ends_with(".parquet")
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn the_first_live_stats_notice_describes_the_operators_and_their_inputs(
    #[future] conn: Conn,
) {
    conn.simple_query("CREATE TABLE live_stats_operators (id BIGINT)")
        .await
        .unwrap();
    conn.simple_query("INSERT INTO live_stats_operators VALUES (1)")
        .await
        .unwrap();

    let response = run_on_one_connection(&[
        "SET live_stats_interval_ms = 1",
        "SELECT id FROM live_stats_operators",
    ])
    .await;

    let live_stats = parse_live_stats(&response);
    assert_eq!(live_stats[0]["type"], "operators");
    let operators = live_stats[0]["operators"].as_array().unwrap();
    assert_eq!(operators[0]["name"], "RowGroupFetcher");
    assert_eq!(operators[0]["inputs"], serde_json::json!([]));
    assert_eq!(operators[1]["inputs"], serde_json::json!([0]));
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn every_worker_message_is_followed_by_a_last_one(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE live_stats_finished (id BIGINT)")
        .await
        .unwrap();
    conn.simple_query("INSERT INTO live_stats_finished VALUES (1)")
        .await
        .unwrap();

    let response = run_on_one_connection(&[
        "SET live_stats_interval_ms = 1",
        "SELECT id FROM live_stats_finished",
    ])
    .await;

    let live_stats = parse_live_stats(&response);
    let mut workers_seen: Vec<u64> = live_stats
        .iter()
        .filter(|message| message["type"] == "worker")
        .map(|message| message["worker"].as_u64().unwrap())
        .collect();
    workers_seen.sort_unstable();
    workers_seen.dedup();
    let mut workers_finished: Vec<u64> = live_stats
        .iter()
        .filter(|message| message["finished"] == true)
        .map(|message| message["worker"].as_u64().unwrap())
        .collect();
    workers_finished.sort_unstable();
    assert!(!workers_finished.is_empty());
    assert_eq!(workers_seen, workers_finished);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn an_interval_of_zero_turns_live_stats_off(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE live_stats_off (id BIGINT)")
        .await
        .unwrap();
    conn.simple_query("INSERT INTO live_stats_off VALUES (1)")
        .await
        .unwrap();

    let response = run_on_one_connection(&[
        "SET live_stats_interval_ms = 1",
        "SET live_stats_interval_ms = 0",
        "SELECT id FROM live_stats_off",
    ])
    .await;

    assert!(parse_live_stats(&response).is_empty());
}
