//! OTLP-over-gRPC receiver.
//!
//! A single gRPC endpoint exposes whichever of the three OTLP services
//! (logs / traces / metrics) the configuration enables — a signal is enabled by
//! giving it a [`SignalSetup`] (the catalog table to append to plus a column
//! mapping). A service
//! does no real work on the receive path: it pairs its payload with the
//! signal's [`CompiledMapping`] (see [`mapping`]) into
//! a buffered item and hands it to that signal's [`ParquetSink`]. Flattening to
//! Arrow and Parquet encoding are deferred to flush, where they run on the
//! dispatch worker pool. Nothing is hardcoded to a single signal or schema:
//! enable any subset, with any columns, via [`OtelConfig`].

mod config;
mod convert;
mod defaults;
mod mapping;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use dispatch::DataFlowDispatcher;
use goose::ParquetCatalog;
use opentelemetry_proto::tonic::collector::logs::v1::logs_service_server::{
    LogsService, LogsServiceServer,
};
use opentelemetry_proto::tonic::collector::logs::v1::{
    ExportLogsServiceRequest, ExportLogsServiceResponse,
};
use opentelemetry_proto::tonic::collector::metrics::v1::metrics_service_server::{
    MetricsService, MetricsServiceServer,
};
use opentelemetry_proto::tonic::collector::metrics::v1::{
    ExportMetricsServiceRequest, ExportMetricsServiceResponse,
};
use opentelemetry_proto::tonic::collector::trace::v1::trace_service_server::{
    TraceService, TraceServiceServer,
};
use opentelemetry_proto::tonic::collector::trace::v1::{
    ExportTraceServiceRequest, ExportTraceServiceResponse,
};
use tonic::codec::CompressionEncoding;
use tonic::transport::Server;
use tonic::transport::server::Router;
use tonic::{Request, Response, Status};

use crate::parquet_writing::ToRecordBatch;
use crate::sink::{Flushable, ParquetSink};
use convert::CompiledMapping;
use mapping::{LogsItem, MetricsItem, TracesItem};

pub use config::ConfigError;
pub use mapping::Signal;

/// The catalog table a signal writes to when none is configured.
pub(crate) fn default_table(signal: Signal) -> &'static str {
    match signal {
        Signal::Logs => "otel_logs",
        Signal::Traces => "otel_traces",
        Signal::Metrics => "otel_metrics",
    }
}

/// Default OTLP/gRPC port (the OpenTelemetry collector's `otlp` receiver
/// default).
pub const DEFAULT_OTLP_ADDR: &str = "0.0.0.0:4317";

/// One enabled signal: which catalog table it appends to and how to flatten it
/// into columns.
#[derive(Debug, Clone)]
pub(crate) struct SignalSetup {
    pub(crate) table: String,
    pub(crate) mapping: Arc<CompiledMapping>,
}

/// Configuration for an OTLP/gRPC receiver.
///
/// A signal is ingested only if it has a `SignalSetup`; the others are not
/// registered on the gRPC server at all, so the collector gets an
/// `Unimplemented` for anything not enabled. Build one with
/// [`OtelConfig::from_toml`] or by calling [`OtelConfig::enable_default`].
#[derive(Debug, Clone)]
pub struct OtelConfig {
    /// Address to bind the gRPC server to.
    pub addr: SocketAddr,
    /// Flush a Parquet file once a sink buffers this many rows.
    pub flush_rows: usize,
    /// Also flush every sink on this interval, so low-traffic streams still
    /// land files promptly.
    pub flush_interval: Duration,
    /// Largest gRPC message we accept; collectors batch aggressively and blow
    /// past the 4 MiB default.
    pub max_decoding_message_size: usize,
    /// Logs setup (`None` disables the logs service).
    pub(crate) logs: Option<SignalSetup>,
    /// Traces setup (`None` disables the trace service).
    pub(crate) traces: Option<SignalSetup>,
    /// Metrics setup (`None` disables the metrics service).
    pub(crate) metrics: Option<SignalSetup>,
}

impl OtelConfig {
    /// A config with sensible defaults and no signals enabled yet. Enable at
    /// least one with [`enable_default`](Self::enable_default) (or use
    /// [`from_toml`](Self::from_toml)).
    pub fn new(addr: SocketAddr) -> Self {
        Self {
            addr,
            flush_rows: 50_000,
            flush_interval: Duration::from_secs(10),
            max_decoding_message_size: 256 * 1024 * 1024,
            logs: None,
            traces: None,
            metrics: None,
        }
    }

    /// Whether any signal is enabled. A config with none enabled is a no-op.
    pub fn any_signal_enabled(&self) -> bool {
        self.logs.is_some() || self.traces.is_some() || self.metrics.is_some()
    }
}

struct LogsSinkService {
    sink: Arc<ParquetSink<LogsItem>>,
    mapping: Arc<CompiledMapping>,
}

#[tonic::async_trait]
impl LogsService for LogsSinkService {
    async fn export(
        &self,
        request: Request<ExportLogsServiceRequest>,
    ) -> Result<Response<ExportLogsServiceResponse>, Status> {
        let item = LogsItem {
            req: request.into_inner(),
            mapping: self.mapping.clone(),
        };
        self.sink.append(item).await;
        Ok(Response::new(ExportLogsServiceResponse::default()))
    }
}

struct TraceSinkService {
    sink: Arc<ParquetSink<TracesItem>>,
    mapping: Arc<CompiledMapping>,
}

#[tonic::async_trait]
impl TraceService for TraceSinkService {
    async fn export(
        &self,
        request: Request<ExportTraceServiceRequest>,
    ) -> Result<Response<ExportTraceServiceResponse>, Status> {
        let item = TracesItem {
            req: request.into_inner(),
            mapping: self.mapping.clone(),
        };
        self.sink.append(item).await;
        Ok(Response::new(ExportTraceServiceResponse::default()))
    }
}

struct MetricsSinkService {
    sink: Arc<ParquetSink<MetricsItem>>,
    mapping: Arc<CompiledMapping>,
}

#[tonic::async_trait]
impl MetricsService for MetricsSinkService {
    async fn export(
        &self,
        request: Request<ExportMetricsServiceRequest>,
    ) -> Result<Response<ExportMetricsServiceResponse>, Status> {
        let item = MetricsItem {
            req: request.into_inner(),
            mapping: self.mapping.clone(),
        };
        self.sink.append(item).await;
        Ok(Response::new(ExportMetricsServiceResponse::default()))
    }
}

/// Test-only shim so crate tests can build a buffered logs item (with the
/// default mapping) without exposing the private modules.
#[cfg(test)]
pub(crate) fn logs_item_for_test(req: ExportLogsServiceRequest) -> LogsItem {
    let mapping = Arc::new(
        CompiledMapping::compile(Signal::Logs, &defaults::columns_for(Signal::Logs)).unwrap(),
    );
    LogsItem { req, mapping }
}

/// A built-but-not-yet-running OTLP server: the configured sinks (as type-erased
/// [`Flushable`]s for the lifecycle paths) plus the tonic router wired with the
/// enabled services.
pub(crate) struct OtelServer {
    pub addr: SocketAddr,
    pub sinks: Vec<Arc<dyn Flushable>>,
    pub router: Router,
}

/// Create a sink for one signal, register it for flushing, and return the typed
/// handle the gRPC service appends to. Fails fast if the signal's catalog table
/// doesn't exist — the sink only writes into a table, never creates one.
fn build_sink<T: ToRecordBatch>(
    setup: &SignalSetup,
    cfg: &OtelConfig,
    dispatcher: &DataFlowDispatcher,
    catalog: &Arc<ParquetCatalog>,
    sinks: &mut Vec<Arc<dyn Flushable>>,
) -> std::io::Result<Arc<ParquetSink<T>>> {
    if !catalog.contains_table(&setup.table) {
        return Err(std::io::Error::other(format!(
            "ingest table `{}` does not exist — create it before starting ingest",
            setup.table
        )));
    }
    let sink = Arc::new(ParquetSink::new(
        &setup.table,
        cfg.flush_rows,
        dispatcher.clone(),
        catalog.clone(),
    ));
    sinks.push(sink.clone());
    Ok(sink)
}

/// Wrap a generated `*ServiceServer<T>` with the compression / size limits
/// every OTLP endpoint needs: accept gzip (exporters compress by default) and
/// allow large batches. A macro, not a fn, because the three concrete server
/// types share no common trait for these builder methods.
macro_rules! tune {
    ($server:expr, $cfg:expr) => {
        $server
            .accept_compressed(CompressionEncoding::Gzip)
            .send_compressed(CompressionEncoding::Gzip)
            .max_decoding_message_size($cfg.max_decoding_message_size)
    };
}

impl OtelServer {
    /// Build the sinks and gRPC router for `cfg`. Fails if an enabled signal's
    /// catalog table doesn't exist — each sink appends to (and never creates)
    /// the table named after it.
    pub(crate) fn build(
        cfg: &OtelConfig,
        dispatcher: &DataFlowDispatcher,
        catalog: &Arc<ParquetCatalog>,
    ) -> std::io::Result<Self> {
        let mut sinks = Vec::new();

        // For each enabled signal, build its sink (registered for flushing) and
        // wire the service. `tune!` stays at each use site so the concrete
        // generated server type is preserved.
        let logs = match &cfg.logs {
            Some(setup) => {
                let sink = build_sink(setup, cfg, dispatcher, catalog, &mut sinks)?;
                let svc = LogsSinkService {
                    sink,
                    mapping: setup.mapping.clone(),
                };
                Some(tune!(LogsServiceServer::new(svc), cfg))
            }
            None => None,
        };
        let traces = match &cfg.traces {
            Some(setup) => {
                let sink = build_sink(setup, cfg, dispatcher, catalog, &mut sinks)?;
                let svc = TraceSinkService {
                    sink,
                    mapping: setup.mapping.clone(),
                };
                Some(tune!(TraceServiceServer::new(svc), cfg))
            }
            None => None,
        };
        let metrics = match &cfg.metrics {
            Some(setup) => {
                let sink = build_sink(setup, cfg, dispatcher, catalog, &mut sinks)?;
                let svc = MetricsSinkService {
                    sink,
                    mapping: setup.mapping.clone(),
                };
                Some(tune!(MetricsServiceServer::new(svc), cfg))
            }
            None => None,
        };

        let router = Server::builder()
            .add_optional_service(logs)
            .add_optional_service(traces)
            .add_optional_service(metrics);

        Ok(Self {
            addr: cfg.addr,
            sinks,
            router,
        })
    }
}
