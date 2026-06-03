//! `ingest` receives telemetry from the outside world and sinks it into Parquet
//! files, running *inside* the pivotdb server.
//!
//! The first (and currently only) source is an **OTLP-over-gRPC** receiver,
//! modelled on the standalone `parquet_sink`. The twist for pivotdb: the
//! CPU-heavy part — encoding and compressing each Parquet file — is shipped to a
//! [`dispatch`] worker via
//! [`DataFlowDispatcher::run_on_worker`](dispatch::DataFlowDispatcher::run_on_worker)
//! instead of running on the async runtime, so the heavy lifting lands on the
//! same thread-per-core pool that executes queries.
//!
//! # Shape
//!
//! - [`ParquetSink`](sink::ParquetSink) buffers Arrow batches for one stream and
//!   flushes one Parquet file per drain.
//! - The [`otel`] module turns each OTLP signal (logs / traces / metrics) into
//!   batches and feeds a sink. Which signals run is configuration, not code:
//!   see [`OtelConfig`].
//! - [`Ingestor`] is the lifecycle handle the server holds: [`Ingestor::start`]
//!   launches every configured source; [`Ingestor::shutdown`] stops the
//!   receivers and flushes whatever is still buffered **before** the dispatch
//!   workers are torn down (the final flush needs them alive).
//!
//! # Querying the output
//!
//! Each sink writes flat Parquet files into one directory, so the stream is
//! queryable with `CREATE TABLE <name> (...) WITH (path = '<dir>')`. Files
//! flushed after the `CREATE TABLE` are not visible until the table is
//! (re)created — the catalog snapshots a directory at creation time.

mod otel;
mod sink;

use std::sync::Arc;

use dispatch::DataFlowDispatcher;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tracing::{error, info, warn};

pub use otel::{DEFAULT_OTLP_ADDR, OtelConfig};
pub use sink::SinkDestination;

use crate::otel::OtelServer;
use crate::sink::ParquetSink;

/// One configured ingest source. Add variants here as sources land (e.g. a
/// Postgres logical-replication source); the server selects which to start.
#[derive(Debug, Clone)]
pub enum IngestConfig {
    /// An OTLP-over-gRPC receiver.
    Otel(OtelConfig),
}

/// Lifecycle handle for all running ingest sources.
///
/// Holds the spawned gRPC server task(s), the per-sink flush timers, and the
/// sinks themselves so a clean [`shutdown`](Self::shutdown) can drain them.
pub struct Ingestor {
    shutdown_tx: watch::Sender<bool>,
    tasks: Vec<JoinHandle<()>>,
    sinks: Vec<Arc<ParquetSink>>,
}

impl Ingestor {
    /// Start every configured source. Output directories are created eagerly,
    /// so a bad path is reported here; the gRPC bind happens asynchronously and
    /// any bind error is logged from the server task.
    ///
    /// Returns an `Ingestor` even when `configs` is empty — its
    /// [`shutdown`](Self::shutdown) is then a no-op.
    pub fn start(
        configs: Vec<IngestConfig>,
        dispatcher: DataFlowDispatcher,
    ) -> std::io::Result<Self> {
        let (shutdown_tx, _) = watch::channel(false);
        let mut tasks = Vec::new();
        let mut sinks = Vec::new();

        for config in configs {
            match config {
                IngestConfig::Otel(cfg) => {
                    if !cfg.any_signal_enabled() {
                        warn!(
                            addr = %cfg.addr,
                            "otel ingest configured with no signals enabled (no logs/traces/metrics dir); skipping"
                        );
                        continue;
                    }
                    let server = OtelServer::build(&cfg, &dispatcher)?;
                    info!(
                        addr = %server.addr,
                        logs = cfg.logs_dir.is_some(),
                        traces = cfg.traces_dir.is_some(),
                        metrics = cfg.metrics_dir.is_some(),
                        flush_rows = cfg.flush_rows,
                        "starting otel ingest"
                    );

                    // Per-sink flush timers so low-traffic streams still land
                    // files on a cadence.
                    for sink in &server.sinks {
                        sinks.push(sink.clone());
                        tasks.push(spawn_flush_timer(
                            sink.clone(),
                            cfg.flush_interval,
                            shutdown_tx.subscribe(),
                        ));
                    }

                    let addr = server.addr;
                    let router = server.router;
                    let mut shutdown_rx = shutdown_tx.subscribe();
                    tasks.push(tokio::spawn(async move {
                        let shutdown = async move {
                            let _ = shutdown_rx.wait_for(|stop| *stop).await;
                        };
                        if let Err(e) = router.serve_with_shutdown(addr, shutdown).await {
                            error!(%addr, error = %e, "otel gRPC server stopped with error");
                        }
                    }));
                }
            }
        }

        Ok(Self {
            shutdown_tx,
            tasks,
            sinks,
        })
    }

    /// Stop the receivers and flush every sink one last time, then return.
    ///
    /// Must be awaited **before** the dispatch workers shut down: the final
    /// flush encodes on a worker via `run_on_worker`. Order is: stop accepting
    /// (signal + join the server/timer tasks), then drain.
    pub async fn shutdown(self) {
        let _ = self.shutdown_tx.send(true);
        for task in self.tasks {
            let _ = task.await;
        }
        for sink in &self.sinks {
            sink.flush_now().await;
        }
        info!(sinks = self.sinks.len(), "ingest drained and stopped");
    }

    /// Stop the receivers without a final flush. Used on the fatal path (a
    /// dispatch worker died), where a `run_on_worker` flush would hang waiting
    /// on a dead worker.
    pub fn abort(self) {
        let _ = self.shutdown_tx.send(true);
        for task in self.tasks {
            task.abort();
        }
    }
}

/// Spawn a timer that flushes `sink` every `interval`, stopping when the
/// shutdown watch flips to `true`.
fn spawn_flush_timer(
    sink: Arc<ParquetSink>,
    interval: std::time::Duration,
    mut shutdown_rx: watch::Receiver<bool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
        // The first `tick()` completes immediately; flushing an empty buffer is
        // a no-op, so it's harmless.
        loop {
            tokio::select! {
                _ = tick.tick() => sink.flush_now().await,
                res = shutdown_rx.changed() => {
                    if res.is_err() || *shutdown_rx.borrow() {
                        break;
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dispatch::{BUFFER_SIZE, Dispatch, ParquetTable};
    use opentelemetry_proto::tonic::collector::logs::v1::ExportLogsServiceRequest;
    use opentelemetry_proto::tonic::common::v1::{AnyValue, KeyValue, any_value::Value};
    use opentelemetry_proto::tonic::logs::v1::{LogRecord, ResourceLogs, ScopeLogs};
    use opentelemetry_proto::tonic::resource::v1::Resource;

    fn log_request(n: usize) -> ExportLogsServiceRequest {
        let records = (0..n)
            .map(|i| LogRecord {
                time_unix_nano: 1_000 + i as u64,
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
                            value: Some(Value::StringValue("svc".into())),
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

    /// End to end on a real dispatch pool: append a logs batch, flush it to a
    /// Parquet file on a worker, then read the file back via
    /// `ParquetTable::from_directory` (also on a worker) and check the rows.
    #[test]
    fn flush_writes_readable_parquet() {
        let dispatch = Dispatch::spin_up(1, 64 * 1024 * 1024 / BUFFER_SIZE);
        let dispatcher = dispatch.dispatcher().clone();
        let dir = tempfile::tempdir().unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let dest = SinkDestination::Local(dir.path().to_path_buf());
            let sink = ParquetSink::new("otel_logs", &dest, 1_000, dispatcher).unwrap();
            let batch = otel::convert_logs_for_test(log_request(3)).unwrap();
            sink.append(batch).await;
            sink.flush_now().await;
        });

        // Exactly one file in the directory (the `.inflight` staging dir is a
        // directory, so it is not counted).
        let files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .filter(|e| e.path().is_file())
            .collect();
        assert_eq!(files.len(), 1, "expected exactly one parquet file");

        // Read it back through the engine to prove pivot can query it.
        let dir_path = dir.path().to_path_buf();
        let table = dispatch
            .dispatcher()
            .run_on_worker(move || ParquetTable::from_directory(&dir_path))
            .unwrap()
            .unwrap();
        let rows: i64 = table.row_groups().iter().map(|rg| rg.num_rows).sum();
        assert_eq!(rows, 3);

        dispatch.exit();
    }
}
