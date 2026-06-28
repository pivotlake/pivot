//! Blackbox end-to-end test for the Kafka ingest path: produce JSON messages to
//! a real broker, start an `Ingestor` with a Kafka source, and read the rows
//! back through the catalog - the whole `Ingestor::start` → consume → decode →
//! append → query flow, including the offset commit.
//!
//! Gated on a running broker, so it is `#[ignore]`d by default. Run it with:
//!
//! ```bash
//! KAFKA_BROKERS=127.0.0.1:9092 cargo test -p ingest --test kafka_e2e -- --ignored
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use catalog::ParquetCatalog;
use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch};
use ingest::{AutoOffsetReset, IngestConfig, Ingestor, KafkaConfig, KafkaFormat};
use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};

/// A 64 MiB file-cache ring, as the in-crate sink tests use.
const RING_BUFFERS: usize = 64 * 1024 * 1024 / BUFFER_SIZE;

fn brokers() -> String {
    std::env::var("KAFKA_BROKERS").unwrap_or_else(|_| "127.0.0.1:9092".to_string())
}

/// A topic name unique to this run, so repeated runs don't share offsets.
fn unique_topic() -> String {
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("pivot_kafka_e2e_{}_{}", std::process::id(), nanos)
}

/// `CREATE TABLE events (user_id BIGINT, action VARCHAR, _offset BIGINT)` - the
/// destination the sink appends to. `_offset` is a Kafka virtual column, so the
/// consumer fills it from each message's offset.
fn create_events_table(catalog: &ParquetCatalog, d: &DataFlowDispatcher, dir: &std::path::Path) {
    use planner::catalog::{Catalog as _, Column, CreateTableRequest};
    use planner::types::Type;
    let request = CreateTableRequest {
        name: "events".to_string(),
        columns: vec![
            Column {
                name: "user_id".to_string(),
                col_type: Type::Int64,
            },
            Column {
                name: "action".to_string(),
                col_type: Type::Utf8,
            },
            Column {
                name: "_offset".to_string(),
                col_type: Type::Int64,
            },
        ],
        options: HashMap::from([("path".to_string(), dir.to_str().unwrap().to_string())]),
        if_not_exists: false,
    };
    catalog
        .create_table(request, d)
        .unwrap()
        .execute()
        .collect()
        .unwrap();
}

/// Total rows visible in `events` at a fresh bind.
fn table_rows(catalog: &ParquetCatalog) -> i64 {
    let mut table = catalog.table_handle("events").expect("table exists");
    table.refresh().expect("manifest reload");
    table
        .parquet(&[])
        .expect("build scan view")
        .row_groups()
        .iter()
        .map(|rg| rg.num_rows)
        .sum()
}

/// Poll [`table_rows`] until it reaches `expected` or `timeout` elapses.
async fn await_rows(catalog: &ParquetCatalog, expected: i64, timeout: Duration) -> i64 {
    let start = Instant::now();
    loop {
        let rows = table_rows(catalog);
        if rows >= expected || start.elapsed() > timeout {
            return rows;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// JSON messages produced to a topic become queryable rows after the consumer
/// flushes them (the flush timer drives it; `flush_rows` is effectively off).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires a running Kafka broker (set KAFKA_BROKERS)"]
async fn json_messages_become_queryable_rows() {
    let topic = unique_topic();
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let data = tempfile::tempdir().unwrap();
    let catalog = Arc::new(ParquetCatalog::new(dispatch.dispatcher().clone()));
    create_events_table(&catalog, dispatch.dispatcher(), data.path());

    let producer: FutureProducer = ClientConfig::new()
        .set("bootstrap.servers", brokers())
        .create()
        .expect("create producer");
    for i in 0..5 {
        let payload = serde_json::json!({ "user_id": i, "action": "buy" }).to_string();
        let key = format!("k{i}");
        producer
            .send(
                FutureRecord::to(&topic).payload(&payload).key(&key),
                Duration::from_secs(5),
            )
            .await
            .expect("produce message");
    }

    let config = KafkaConfig {
        brokers: brokers(),
        topics: vec![topic.clone()],
        group_id: format!("{topic}-group"),
        table: "events".to_string(),
        format: KafkaFormat::Json,
        schema_registry_url: None,
        flush_rows: usize::MAX, // never auto-flush on rows; the timer drains.
        flush_interval: Duration::from_millis(250),
        num_consumers: 1,
        auto_offset_reset: AutoOffsetReset::Earliest,
        properties: HashMap::new(),
        skip_broken_messages: 0,
        dead_letter_table: None,
    };
    let ingestor = Ingestor::start(
        vec![IngestConfig::Kafka(config)],
        dispatch.dispatcher().clone(),
        catalog.clone(),
        0, // no bundled compacter
        4,
    )
    .unwrap();

    let rows = await_rows(&catalog, 5, Duration::from_secs(30)).await;
    ingestor.shutdown().await;

    assert_eq!(rows, 5, "all five produced messages should land as rows");

    dispatch.exit();
}
