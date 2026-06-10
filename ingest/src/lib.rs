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
//!   flushes one Parquet file per drain, registering it with the catalog so
//!   the rows are queryable at once.
//! - The `otel` module turns each OTLP signal (logs / traces / metrics) into
//!   batches and feeds a sink. Which signals run is configuration, not code:
//!   see [`OtelConfig`].
//! - The `compact` module merges a table's small files into target-sized
//!   ones (one scan→encode dataflow on the dispatch pool) and swaps them in
//!   one table-log commit. It only watches the log, so the bundled background
//!   task is a convenience — the same compacter can run in a separate
//!   process over the same database root.
//! - [`Ingestor`] is the lifecycle handle the server holds: [`Ingestor::start`]
//!   launches every configured source; [`Ingestor::shutdown`] stops the
//!   receivers and flushes whatever is still buffered **before** the dispatch
//!   workers are torn down (the final flush needs them alive).
//!
//! # Querying the output
//!
//! Each sink writes flat Parquet files into one directory, so the stream is
//! queryable with `CREATE TABLE <name> (...) WITH (path = '<dir>')`, where
//! `<name>` is the sink's name (e.g. `otel_logs`) and `<dir>` its (absolute)
//! destination. When the server hands [`Ingestor::start`] its catalog, each
//! flushed file is *registered* with that table, so rows become queryable as
//! soon as they land — no re-`CREATE` needed. Files flushed before the
//! `CREATE TABLE` are picked up by the create itself (it snapshots the
//! directory).

mod compact;
mod otel;
mod parquet_writing;
mod sink;

use std::sync::Arc;

use dispatch::DataFlowDispatcher;
use goose::ParquetCatalog;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tracing::{error, info, warn};

pub use compact::{Compacter, DEFAULT_COMPACT_BYTES, DEFAULT_COMPACT_POLL};
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
    /// `compact_bytes` sizes the bundled, catalog-wide [`Compacter`] (`0`
    /// skips it — e.g. when a dedicated compacter process owns the job).
    pub fn start(
        configs: Vec<IngestConfig>,
        dispatcher: DataFlowDispatcher,
        catalog: Arc<ParquetCatalog>,
        compact_bytes: u64,
    ) -> std::io::Result<Self> {
        let (shutdown_tx, _) = watch::channel(false);
        let mut tasks = Vec::new();
        let mut sinks = Vec::new();

        // The bundled compacter: one catalog-wide maintenance loop, joined on
        // shutdown like the flush timers (an in-flight merge encodes on the
        // dispatch workers, so it must finish before they are torn down). It
        // knows nothing of the sources below — the same loop can run in a
        // separate process instead.
        if compact_bytes > 0 {
            let compacter = Arc::new(Compacter::new(
                compact_bytes,
                DEFAULT_COMPACT_POLL,
                catalog.clone(),
            ));
            tasks.push(tokio::spawn(compacter.run(shutdown_tx.subscribe())));
        }

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
                    let server = OtelServer::build(&cfg, &dispatcher, &catalog)?;
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
    use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch, Projection};
    use goose::parquet::{ParquetTable, table_input};
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

    /// Flush `requests` (one batch each) through a sink as a single flush,
    /// registering each file with `catalog`.
    fn write_logs(
        dispatcher: &DataFlowDispatcher,
        dir: &Path,
        requests: &[ExportLogsServiceRequest],
        catalog: Arc<goose::ParquetCatalog>,
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
                ParquetSink::new("otel_logs", &dest, usize::MAX, dispatcher.clone(), catalog)
                    .unwrap();
            for request in requests {
                sink.append(otel::logs_item_for_test(request.clone())).await;
            }
            sink.flush_now().await;
        });
    }

    /// Flush each request as its **own** Parquet file through one sink (one
    /// flush per request), registering each with `catalog`.
    fn flush_each(
        dispatcher: &DataFlowDispatcher,
        dir: &Path,
        requests: &[ExportLogsServiceRequest],
        catalog: Arc<goose::ParquetCatalog>,
    ) {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let dest = SinkDestination::Local(dir.to_path_buf());
            let sink =
                ParquetSink::new("otel_logs", &dest, usize::MAX, dispatcher.clone(), catalog)
                    .unwrap();
            for request in requests {
                sink.append(otel::logs_item_for_test(request.clone())).await;
                sink.flush_now().await;
            }
        });
    }

    /// `CREATE TABLE otel_logs (...) WITH (path = dir)` against `catalog`, the
    /// way the server would run it. The declared columns don't matter to these
    /// tests (they assert on the Parquet row groups, not the logical schema).
    fn create_catalog_table(
        catalog: &Arc<goose::ParquetCatalog>,
        dispatcher: &DataFlowDispatcher,
        dir: &Path,
    ) {
        create_table(catalog, dispatcher, "otel_logs", Some(dir));
    }

    /// `CREATE TABLE <name>` against `catalog` the way the server would run
    /// it: with an explicit absolute `path` option, or — when `dir` is `None`
    /// — at `<name>` under the database root (a store-relative table).
    fn create_table(
        catalog: &Arc<goose::ParquetCatalog>,
        dispatcher: &DataFlowDispatcher,
        name: &str,
        dir: Option<&Path>,
    ) {
        use planner::catalog::{Catalog as _, Column, CreateTableRequest};
        let options = match dir {
            Some(dir) => std::collections::HashMap::from([(
                "path".to_string(),
                dir.to_str().unwrap().to_string(),
            )]),
            None => std::collections::HashMap::new(),
        };
        let request = CreateTableRequest {
            name: name.to_string(),
            columns: vec![Column {
                name: "Timestamp".to_string(),
                col_type: planner::types::Type::Int64,
            }],
            options,
            if_not_exists: false,
        };
        catalog
            .create_table(request, dispatcher)
            .unwrap()
            .execute()
            .collect()
            .unwrap();
    }

    /// Load the written directory back into a `ParquetTable`. Drives the
    /// metadata-fetch dataflow from this (coordinator) thread; the footer
    /// reads themselves land on the workers.
    fn read_table(dispatch: &Dispatch, dir: &Path) -> Arc<ParquetTable> {
        Arc::new(ParquetTable::from_directory(dispatch.dispatcher(), dir).unwrap())
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

        write_logs(
            dispatch.dispatcher(),
            dir.path(),
            &[log_request(3)],
            Arc::new(goose::ParquetCatalog::new(dispatch.dispatcher().clone())),
        );
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

        write_logs(
            dispatch.dispatcher(),
            dir.path(),
            &vec![log_request(2); 5],
            Arc::new(goose::ParquetCatalog::new(dispatch.dispatcher().clone())),
        );
        let table = read_table(&dispatch, dir.path());

        assert_eq!(parquet_file_count(dir.path()), 1);
        assert_eq!(table.row_groups().len(), 1);
        assert_eq!(table.row_groups()[0].num_rows, 10);

        dispatch.exit();
    }

    /// A flush that lands after `CREATE TABLE` is registered with the catalog,
    /// so its rows are visible to new binds without re-creating the table.
    #[test]
    fn flushed_file_is_registered_with_catalog_table() {
        let dispatch = Dispatch::spin_up(2, RING_BUFFERS);
        let dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(goose::ParquetCatalog::new(dispatch.dispatcher().clone()));
        create_catalog_table(&catalog, dispatch.dispatcher(), dir.path());
        assert!(
            catalog
                .parquet_table("otel_logs")
                .unwrap()
                .parquet
                .row_groups()
                .is_empty()
        );

        flush_each(
            dispatch.dispatcher(),
            dir.path(),
            &[log_request(3), log_request(2)],
            catalog.clone(),
        );

        let table = catalog.parquet_table("otel_logs").unwrap();
        let groups = table.parquet.row_groups();
        assert_eq!(
            groups.iter().map(|rg| rg.num_rows).sum::<i64>(),
            5,
            "both flushed files' rows should be registered"
        );
        assert!(
            groups
                .iter()
                .enumerate()
                .all(|(i, rg)| rg.global_row_group_idx == i)
        );

        dispatch.exit();
    }

    /// End-to-end compaction: several registered small files merge into one
    /// (a single scan→encode dataflow on the dispatch pool), the catalog swaps
    /// to the merged file in one log commit, the inputs are deleted, and the
    /// merged file decodes back to all the original rows.
    #[test]
    fn compaction_merges_registered_files_and_swaps_catalog() {
        // 4× the usual test ring: the fused scan→encode dataflow keeps decode
        // and encode in flight together, so decompressed pages (a full ring
        // buffer each, however small the page) queue while workers encode.
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS);
        let dir = tempfile::tempdir().unwrap();
        let catalog = Arc::new(goose::ParquetCatalog::new(dispatch.dispatcher().clone()));
        create_catalog_table(&catalog, dispatch.dispatcher(), dir.path());

        flush_each(
            dispatch.dispatcher(),
            dir.path(),
            &[log_request(3), log_request(2), log_request(4)],
            catalog.clone(),
        );
        assert_eq!(parquet_file_count(dir.path()), 3);

        // Target = the three files' combined size: each is smaller than it
        // (candidate) and together they reach it (trigger).
        let total: u64 = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .filter(|e| e.path().is_file())
            .map(|e| e.metadata().unwrap().len())
            .sum();
        let compacter = Compacter::new(total, std::time::Duration::from_secs(1), catalog.clone());
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(compacter.compact_all());

        // One merged file replaced the three inputs, on disk and in the
        // catalog, and indices stayed sequential.
        assert_eq!(parquet_file_count(dir.path()), 1);
        let table = catalog.parquet_table("otel_logs").unwrap();
        let groups = table.parquet.row_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].num_rows, 9);
        assert_eq!(groups[0].global_row_group_idx, 0);
        assert!(groups[0].file_name.contains("compacted"));

        // The merged file's contents round-trip through the engine's scan.
        let reread = read_table(&dispatch, dir.path());
        let batches = table_input(
            dispatch.dispatcher(),
            &reread,
            Projection::all(NUM_LOG_COLUMNS),
            false,
        )
        .collect()
        .unwrap();
        assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 9);
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
    /// Compaction is location-agnostic: a **store-relative** table (data under
    /// the database root, resolved through the store — the same path a remote
    /// `s3://` root takes) compacts through the exact same code, with writes
    /// and deletes going through the table's store handle.
    #[test]
    fn compaction_works_on_store_relative_tables() {
        use arrow_array::Int64Array;
        use arrow_schema::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;

        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS);
        let db = tempfile::tempdir().unwrap();
        let table_dir = db.path().join("events");
        std::fs::create_dir_all(&table_dir).unwrap();

        // Two small files under the table's prefix inside the database root.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "Timestamp",
            DataType::Int64,
            false,
        )]));
        for (name, values) in [("a.parquet", vec![1i64, 2]), ("b.parquet", vec![3i64])] {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from(values)) as _],
            )
            .unwrap();
            let file = std::fs::File::create(table_dir.join(name)).unwrap();
            // SNAPPY, like every pivot-written file — the engine's decompressor
            // expects it.
            let props = parquet::file::properties::WriterProperties::builder()
                .set_compression(parquet::basic::Compression::SNAPPY)
                .build();
            let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        let catalog = Arc::new(
            goose::ParquetCatalog::open(db.path().to_str().unwrap(), dispatch.dispatcher())
                .unwrap(),
        );
        // No `path` option: the table lives at `events` under the root.
        create_table(&catalog, dispatch.dispatcher(), "events", None);

        let total: u64 = catalog
            .table_files("events")
            .unwrap()
            .iter()
            .map(|f| f.size)
            .sum();
        let compacter = Compacter::new(total, std::time::Duration::from_secs(1), catalog.clone());
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(compacter.compact_all());

        // One merged object replaced the two inputs, in the store and in the
        // catalog.
        let files = catalog.table_files("events").unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].name.contains("compacted"));
        let on_disk: Vec<_> = std::fs::read_dir(&table_dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".parquet"))
            .collect();
        assert_eq!(on_disk, vec![files[0].name.clone()]);

        let table = catalog.parquet_table("events").unwrap();
        assert_eq!(
            table
                .parquet
                .row_groups()
                .iter()
                .map(|rg| rg.num_rows)
                .sum::<i64>(),
            3
        );

        dispatch.exit();
    }

    /// The compacter can live in a **separate process**: it holds nothing but
    /// a catalog handle, and each poll round reloads the table from its log.
    /// Here the "server" registers flushes through one catalog instance while
    /// the compacter works through a second instance over the same database
    /// root — and the server sees the swap at its next bind.
    #[test]
    fn compacter_in_another_process_compacts_the_servers_flushes() {
        let dispatch = Dispatch::spin_up(2, 4 * RING_BUFFERS);
        let db = tempfile::tempdir().unwrap();
        let data_dir = tempfile::tempdir().unwrap();

        // "Server" process: creates the table and registers three flushes.
        let server_catalog = Arc::new(
            goose::ParquetCatalog::open(db.path().to_str().unwrap(), dispatch.dispatcher())
                .unwrap(),
        );
        create_catalog_table(&server_catalog, dispatch.dispatcher(), data_dir.path());
        flush_each(
            dispatch.dispatcher(),
            data_dir.path(),
            &[log_request(3), log_request(2), log_request(4)],
            server_catalog.clone(),
        );

        // "Compacter" process: a separate catalog over the same root. Its
        // poll round reloads the table from the log before scanning.
        let compacter_catalog = Arc::new(
            goose::ParquetCatalog::open(db.path().to_str().unwrap(), dispatch.dispatcher())
                .unwrap(),
        );
        let total: u64 = server_catalog
            .table_files("otel_logs")
            .unwrap()
            .iter()
            .map(|f| f.size)
            .sum();
        let compacter = Compacter::new(total, std::time::Duration::from_secs(1), compacter_catalog);
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(compacter.compact_all());

        // The server's next bind reloads to the compacted version.
        assert!(planner::catalog::Catalog::table(&*server_catalog, "otel_logs").is_some());
        let table = server_catalog.parquet_table("otel_logs").unwrap();
        let groups = table.parquet.row_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].num_rows, 9);
        assert!(groups[0].file_name.contains("compacted"));
        assert_eq!(parquet_file_count(data_dir.path()), 1);

        dispatch.exit();
    }
}
