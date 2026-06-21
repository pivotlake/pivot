//! Blackbox end-to-end test for the OTLP ingest path: start a real `Ingestor`
//! (gRPC receiver + flush timers, over a live `dispatch` pool and catalog),
//! send telemetry over the wire with a `tonic` OTLP client, and read the rows
//! back through the catalog — the whole `Ingestor::start` → gRPC → sink → append
//! → query flow the server runs. (The in-crate unit tests drive `ParquetSink`
//! directly; this exercises the public lifecycle the way an OTLP exporter does.)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use catalog::ParquetCatalog;
use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch};
use ingest::{IngestConfig, Ingestor, OtelConfig, Signal};
use opentelemetry_proto::tonic::collector::logs::v1::ExportLogsServiceRequest;
use opentelemetry_proto::tonic::collector::logs::v1::logs_service_client::LogsServiceClient;
use opentelemetry_proto::tonic::common::v1::{AnyValue, KeyValue, any_value::Value};
use opentelemetry_proto::tonic::logs::v1::{LogRecord, ResourceLogs, ScopeLogs};
use opentelemetry_proto::tonic::resource::v1::Resource;
use tonic::transport::{Channel, Endpoint};

/// A 64 MiB file-cache ring, as the in-crate sink tests use.
const RING_BUFFERS: usize = 64 * 1024 * 1024 / BUFFER_SIZE;

// --- helpers ---------------------------------------------------------------

/// A free loopback port for the gRPC receiver to bind to.
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// An OTLP logs request of `n` records, each `severity_number = 9`,
/// `service.name = "svc"`.
fn log_request(n: usize) -> ExportLogsServiceRequest {
    log_request_for("svc", n)
}

/// An OTLP logs request of `n` records for `service`, severities/timestamps
/// descending in record order (so a `sort_by = Timestamp` has work to do), with
/// `service.name = service` as a resource attribute.
fn log_request_for(service: &str, n: usize) -> ExportLogsServiceRequest {
    let records = (0..n)
        .map(|i| LogRecord {
            // Descending timestamps so the writer's sort reorders them.
            time_unix_nano: (1_000 - i) as u64,
            severity_number: 9,
            severity_text: "INFO".into(),
            body: Some(AnyValue {
                value: Some(Value::StringValue(format!("hello {i}"))),
            }),
            ..Default::default()
        })
        .collect();
    ExportLogsServiceRequest {
        resource_logs: vec![ResourceLogs {
            resource: Some(Resource {
                attributes: vec![KeyValue {
                    key: "service.name".into(),
                    value: Some(AnyValue {
                        value: Some(Value::StringValue(service.into())),
                    }),
                }],
                ..Default::default()
            }),
            scope_logs: vec![ScopeLogs {
                log_records: records,
                ..Default::default()
            }],
            ..Default::default()
        }],
    }
}

/// `CREATE TABLE otel_logs (...) WITH (path = dir)` — the sink only ever appends
/// to an existing table, so it must be created (with a location) up front.
fn create_otel_logs(catalog: &ParquetCatalog, d: &DataFlowDispatcher, dir: &std::path::Path) {
    use planner::catalog::{Catalog as _, Column, CreateTableRequest};
    let request = CreateTableRequest {
        name: "otel_logs".to_string(),
        columns: vec![Column {
            name: "Timestamp".to_string(),
            col_type: planner::types::Type::Int64,
        }],
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

/// `CREATE TABLE otel_logs (ServiceName, Timestamp) WITH (path, partition_by =
/// ServiceName, sort_by = Timestamp)`. The partition/sort columns must be
/// declared (they're validated against the declared columns), and they match the
/// default logs mapping the sink writes.
fn create_partitioned_otel_logs(
    catalog: &ParquetCatalog,
    d: &DataFlowDispatcher,
    dir: &std::path::Path,
) {
    use planner::catalog::{Catalog as _, Column, CreateTableRequest};
    use planner::types::Type;
    let request = CreateTableRequest {
        name: "otel_logs".to_string(),
        columns: vec![
            Column {
                name: "ServiceName".to_string(),
                col_type: Type::Utf8,
            },
            Column {
                name: "Timestamp".to_string(),
                col_type: Type::Int64,
            },
        ],
        options: HashMap::from([
            ("path".to_string(), dir.to_str().unwrap().to_string()),
            ("partition_by".to_string(), "ServiceName".to_string()),
            ("sort_by".to_string(), "Timestamp".to_string()),
        ]),
        if_not_exists: false,
    };
    catalog
        .create_table(request, d)
        .unwrap()
        .execute()
        .collect()
        .unwrap();
}

/// Start an `Ingestor` with one OTLP receiver on `port`, logs enabled with the
/// default mapping. `flush_interval` controls the per-sink flush timer; row- and
/// shutdown-flushing aside, only the timer drains the buffer.
fn start_logs_ingestor(
    port: u16,
    dispatch: &Dispatch,
    catalog: &Arc<ParquetCatalog>,
    flush_interval: Duration,
) -> Ingestor {
    let mut cfg = OtelConfig::new(format!("127.0.0.1:{port}").parse().unwrap());
    cfg.enable_default(Signal::Logs).unwrap();
    cfg.flush_rows = usize::MAX; // never auto-flush on row count; the timer/shutdown drains.
    cfg.flush_interval = flush_interval;
    Ingestor::start(
        vec![IngestConfig::Otel(cfg)],
        dispatch.dispatcher().clone(),
        catalog.clone(),
        0, // no bundled compacter
    )
    .unwrap()
}

/// Connect an OTLP logs client to `port`, retrying until the gRPC server is up.
async fn logs_client(port: u16) -> LogsServiceClient<Channel> {
    let url = format!("http://127.0.0.1:{port}");
    for _ in 0..100 {
        if let Ok(channel) = Endpoint::from_shared(url.clone()).unwrap().connect().await {
            return LogsServiceClient::new(channel);
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("otel ingest never accepted a connection on {url}");
}

/// Total rows visible in `otel_logs` at a fresh bind (refreshes from the latest
/// committed manifest, then sums the row groups).
fn table_rows(catalog: &ParquetCatalog) -> i64 {
    let mut table = catalog.table_handle("otel_logs").expect("table exists");
    table.refresh().expect("manifest reload");
    table
        .parquet()
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
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

// --- tests -----------------------------------------------------------------

/// Logs sent over gRPC are drained by `shutdown` and become queryable rows.
#[tokio::test(flavor = "multi_thread")]
async fn logs_sent_over_grpc_are_queryable_after_shutdown() {
    let port = free_port();
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let data = tempfile::tempdir().unwrap();
    let catalog = Arc::new(ParquetCatalog::new(dispatch.dispatcher().clone()));
    create_otel_logs(&catalog, dispatch.dispatcher(), data.path());
    let ingestor = start_logs_ingestor(port, &dispatch, &catalog, Duration::from_secs(3600));

    logs_client(port)
        .await
        .export(log_request(5))
        .await
        .unwrap();
    ingestor.shutdown().await;

    assert_eq!(table_rows(&catalog), 5);

    dispatch.exit();
}

/// The per-sink flush timer lands the rows on its own cadence — queryable
/// without waiting for shutdown.
#[tokio::test(flavor = "multi_thread")]
async fn flush_timer_makes_logs_queryable_without_shutdown() {
    let port = free_port();
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let data = tempfile::tempdir().unwrap();
    let catalog = Arc::new(ParquetCatalog::new(dispatch.dispatcher().clone()));
    create_otel_logs(&catalog, dispatch.dispatcher(), data.path());
    let ingestor = start_logs_ingestor(port, &dispatch, &catalog, Duration::from_millis(250));

    logs_client(port)
        .await
        .export(log_request(3))
        .await
        .unwrap();
    let rows = await_rows(&catalog, 3, Duration::from_secs(5)).await;

    assert_eq!(rows, 3);

    ingestor.shutdown().await;
    dispatch.exit();
}

/// A partitioned + sorted table: logs from two services flushed together land in
/// one file per service, each tagged with its partition tuple in the manifest.
#[tokio::test(flavor = "multi_thread")]
async fn partitioned_logs_land_one_file_per_service() {
    let port = free_port();
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let data = tempfile::tempdir().unwrap();
    let catalog = Arc::new(ParquetCatalog::new(dispatch.dispatcher().clone()));
    create_partitioned_otel_logs(&catalog, dispatch.dispatcher(), data.path());
    let ingestor = start_logs_ingestor(port, &dispatch, &catalog, Duration::from_secs(3600));

    let mut client = logs_client(port).await;
    client.export(log_request_for("svc-a", 2)).await.unwrap();
    client.export(log_request_for("svc-b", 3)).await.unwrap();
    ingestor.shutdown().await;

    // One file per service, each tagged with its partition tuple; 5 rows total.
    let mut table = catalog.table_handle("otel_logs").unwrap();
    table.refresh().unwrap();
    let mut services: Vec<String> = table
        .file_partitions()
        .into_iter()
        .map(|(_, p)| {
            p.expect("file has a partition tuple")["ServiceName"]
                .as_str()
                .expect("ServiceName string")
                .to_string()
        })
        .collect();
    services.sort();

    assert_eq!(services, vec!["svc-a", "svc-b"]);
    assert_eq!(table_rows(&catalog), 5);

    // Each file records its sort-key (Timestamp) bounds, min <= max.
    for (_, bounds) in table.file_sort_bounds() {
        let b = bounds.expect("file has sort bounds");
        let min = b.min["Timestamp"].as_i64().expect("min Timestamp");
        let max = b.max["Timestamp"].as_i64().expect("max Timestamp");
        assert!(min <= max, "sort bounds: {min} <= {max}");
    }

    // The written files carry real Parquet statistics for the sort column, so a
    // strict external reader (arrow-rs, our DuckDB-readability proxy) sees them —
    // i.e. row-group pruning works on pivot-written files. `Timestamp` is column
    // 0 of the default logs schema.
    for path in std::fs::read_dir(data.path())
        .unwrap()
        .flatten()
        .map(|e| e.path())
    {
        if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
            continue;
        }
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
            std::fs::File::open(&path).unwrap(),
        )
        .unwrap();
        let meta = reader.metadata();
        for rg in 0..meta.num_row_groups() {
            assert!(
                meta.row_group(rg).column(0).statistics().is_some(),
                "Timestamp column has footer statistics in {path:?}"
            );
        }
    }

    dispatch.exit();
}
