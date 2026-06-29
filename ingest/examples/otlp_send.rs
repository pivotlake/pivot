//! Send synthetic OTLP log records to a running pivotdb-server's OTLP/gRPC
//! receiver - a quick way to make the dashboard's data-flow graph come alive.
//!
//! Run a server with a logs receiver and the table it feeds, then point this at
//! it:
//!
//! ```text
//! # 1. create the destination table (once), e.g. in the SQL console:
//! #    CREATE TABLE otel_logs (Timestamp BIGINT) WITH (path = 'otel_logs')
//! # 2. run the server with a receiver + the dashboard:
//! #    pivotdb-server --http-bind 127.0.0.1:8080 --path ./db \
//! #                   --otel 'addr=127.0.0.1:4317,logs'
//! # 3. generate traffic (from the `ingest` crate dir):
//! cargo run --example otlp_send                 # defaults below
//! cargo run --example otlp_send -- 127.0.0.1:4317 200 25000 300
//! ```
//!
//! Args (all optional, in order):
//!   addr         gRPC address of the receiver        (default 127.0.0.1:4317)
//!   batches      how many export calls to send        (default 60)
//!   per_batch    log records per call                 (default 20000)
//!   interval_ms  pause between calls                  (default 500)

use opentelemetry_proto::tonic::collector::logs::v1::ExportLogsServiceRequest;
use opentelemetry_proto::tonic::collector::logs::v1::logs_service_client::LogsServiceClient;
use opentelemetry_proto::tonic::common::v1::{AnyValue, KeyValue, any_value::Value};
use opentelemetry_proto::tonic::logs::v1::{LogRecord, ResourceLogs, ScopeLogs};
use opentelemetry_proto::tonic::resource::v1::Resource;
use std::time::Duration;
use tonic::transport::Endpoint;

/// One export request of `n` log records, tagged with `service.name`.
fn log_request(service: &str, batch: usize, n: usize) -> ExportLogsServiceRequest {
    let records = (0..n)
        .map(|i| LogRecord {
            time_unix_nano: (1_000_000 + batch * 1000 + i) as u64,
            severity_number: 9,
            severity_text: "INFO".into(),
            body: Some(AnyValue {
                value: Some(Value::StringValue(format!("log message {batch}-{i}"))),
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

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let addr = args
        .first()
        .cloned()
        .unwrap_or_else(|| "127.0.0.1:4317".to_string());
    let batches: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let per: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let interval_ms: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(500);

    let url = format!("http://{addr}");
    let channel = Endpoint::from_shared(url.clone())
        .expect("invalid address")
        .connect()
        .await
        .unwrap_or_else(|e| panic!("could not connect to {url}: {e}"));
    let mut client = LogsServiceClient::new(channel);

    println!("sending {batches} batches × {per} records to {url} (every {interval_ms}ms)");
    let mut total = 0;
    for b in 0..batches {
        client
            .export(log_request("demo-service", b, per))
            .await
            .expect("export failed");
        total += per;
        println!("batch {:>3}/{batches}  ({total} records total)", b + 1);
        tokio::time::sleep(Duration::from_millis(interval_ms)).await;
    }
    println!("done - sent {total} records");
}
