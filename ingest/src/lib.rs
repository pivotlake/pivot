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
//! - `ParquetSink` buffers Arrow batches for one stream and
//!   flushes one Parquet file per drain.
//! - The `otel` module turns each OTLP signal (logs / traces / metrics) into
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
mod parquet_writing;
mod sink;

use std::sync::Arc;

use dispatch::DataFlowDispatcher;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tracing::{error, info, warn};

pub use otel::{ConfigError, DEFAULT_OTLP_ADDR, OtelConfig, Signal};
pub use sink::SinkDestination;

use crate::otel::OtelServer;
use crate::sink::Flushable;

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
    sinks: Vec<Arc<dyn Flushable>>,
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
                        logs = cfg.logs.is_some(),
                        traces = cfg.traces.is_some(),
                        metrics = cfg.metrics.is_some(),
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
            sink.flush().await;
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
    sink: Arc<dyn Flushable>,
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
                _ = tick.tick() => sink.flush().await,
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
    use crate::sink::ParquetSink;
    use arrow_array::{Array, Int32Array, RecordBatch, StringViewArray};
    use catalog::parquet::{ParquetTable, table_input};
    use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch, Projection};
    use opentelemetry_proto::tonic::collector::logs::v1::ExportLogsServiceRequest;
    use opentelemetry_proto::tonic::common::v1::{AnyValue, KeyValue, any_value::Value};
    use opentelemetry_proto::tonic::logs::v1::{LogRecord, ResourceLogs, ScopeLogs};
    use opentelemetry_proto::tonic::resource::v1::Resource;
    use std::path::Path;

    const RING_BUFFERS: usize = 64 * 1024 * 1024 / BUFFER_SIZE;
    /// Column indices in the `otel_logs` schema, for read-back assertions.
    const SEVERITY_NUMBER: usize = 4;
    const SERVICE_NAME: usize = 6;
    const NUM_LOG_COLUMNS: usize = 11;

    /// An OTLP logs request of `n` records, all `severity_number = 9`,
    /// `service.name = "svc"`.
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

    /// Flush `requests` (one batch each) through a sink as a single flush.
    fn write_logs(
        dispatcher: &DataFlowDispatcher,
        dir: &Path,
        requests: &[ExportLogsServiceRequest],
    ) {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let dest = SinkDestination::Local(dir.to_path_buf());
            // usize::MAX threshold: appends never auto-flush, so one flush_now
            // writes all `requests` as a single multi-row-group file.
            let sink =
                ParquetSink::new("otel_logs", &dest, usize::MAX, dispatcher.clone()).unwrap();
            for request in requests {
                sink.append(otel::logs_item_for_test(request.clone())).await;
            }
            sink.flush_now().await;
        });
    }

    /// Load the written directory back into a `ParquetTable` (on a worker).
    fn read_table(dispatch: &Dispatch, dir: &Path) -> Arc<ParquetTable> {
        let dir = dir.to_path_buf();
        Arc::new(
            dispatch
                .dispatcher()
                .run_on_worker(move || ParquetTable::from_directory(&dir))
                .unwrap()
                .unwrap(),
        )
    }

    fn parquet_file_count(dir: &Path) -> usize {
        std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .filter(|e| e.path().is_file())
            .count()
    }

    fn i32_column(batches: &[RecordBatch], col: usize) -> Vec<i32> {
        batches
            .iter()
            .flat_map(|b| {
                let a = b.column(col).as_any().downcast_ref::<Int32Array>().unwrap();
                (0..a.len()).map(|i| a.value(i)).collect::<Vec<_>>()
            })
            .collect()
    }

    fn str_column(batches: &[RecordBatch], col: usize) -> Vec<String> {
        batches
            .iter()
            .flat_map(|b| {
                let a = b
                    .column(col)
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .unwrap();
                (0..a.len())
                    .map(|i| a.value(i).to_string())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// A written file decodes back to the original values through the engine's
    /// scan (proves the PLAIN int and byte-array encodings are correct).
    #[test]
    fn written_pages_decode_to_original_values() {
        let dispatch = Dispatch::spin_up(1, RING_BUFFERS);
        let dir = tempfile::tempdir().unwrap();

        write_logs(dispatch.dispatcher(), dir.path(), &[log_request(3)]);
        let table = read_table(&dispatch, dir.path());
        let batches = table_input(
            dispatch.dispatcher(),
            &table,
            Projection::all(NUM_LOG_COLUMNS),
            false,
        )
        .collect()
        .unwrap();

        assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
        assert!(
            i32_column(&batches, SEVERITY_NUMBER)
                .iter()
                .all(|&v| v == 9)
        );
        assert!(
            str_column(&batches, SERVICE_NAME)
                .iter()
                .all(|v| v == "svc")
        );

        dispatch.exit();
    }

    /// Page jobs encoded in parallel across the pool are stitched into one file
    /// with a single row group spanning all the flush's rows.
    #[test]
    fn parallel_pages_assemble_into_one_file() {
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(4);
        let dispatch = Dispatch::spin_up(workers, RING_BUFFERS);
        let dir = tempfile::tempdir().unwrap();

        write_logs(dispatch.dispatcher(), dir.path(), &vec![log_request(2); 5]);
        let table = read_table(&dispatch, dir.path());

        assert_eq!(parquet_file_count(dir.path()), 1);
        assert_eq!(table.row_groups().len(), 1);
        assert_eq!(table.row_groups()[0].num_rows, 10);

        dispatch.exit();
    }
}
