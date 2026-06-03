//! OTLP-over-gRPC receiver.
//!
//! A single gRPC endpoint exposes whichever of the three OTLP services
//! (logs / traces / metrics) the configuration enables — a signal is enabled by
//! giving it an output directory. Each service flattens its protocol payload
//! into a [`RecordBatch`](arrow_array::RecordBatch) (see [`convert`]) and hands
//! it to that signal's [`ParquetSink`]. Nothing is hardcoded to a single
//! signal: enable any subset via [`OtelConfig`].

mod convert;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

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
use tonic::{Request, Response, Status};

use crate::sink::{ParquetSink, SinkDestination};

/// Default OTLP/gRPC port (the OpenTelemetry collector's `otlp` receiver
/// default).
pub const DEFAULT_OTLP_ADDR: &str = "0.0.0.0:4317";

/// Configuration for an OTLP/gRPC receiver.
///
/// A signal is ingested only if it has a directory configured; the others are
/// not registered on the gRPC server at all, so the collector gets an
/// `Unimplemented` for anything not enabled.
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
    /// Destination for logs (`None` disables the logs service).
    pub logs_dir: Option<SinkDestination>,
    /// Destination for traces (`None` disables the trace service).
    pub traces_dir: Option<SinkDestination>,
    /// Destination for metrics (`None` disables the metrics service).
    pub metrics_dir: Option<SinkDestination>,
}

impl OtelConfig {
    /// A config with sensible defaults and no signals enabled yet. Set at least
    /// one of `logs_dir` / `traces_dir` / `metrics_dir`.
    pub fn new(addr: SocketAddr) -> Self {
        Self {
            addr,
            flush_rows: 50_000,
            flush_interval: Duration::from_secs(10),
            max_decoding_message_size: 256 * 1024 * 1024,
            logs_dir: None,
            traces_dir: None,
            metrics_dir: None,
        }
    }

    /// Whether any signal is enabled. A config with none enabled is a no-op.
    pub fn any_signal_enabled(&self) -> bool {
        self.logs_dir.is_some() || self.traces_dir.is_some() || self.metrics_dir.is_some()
    }
}

// ---------------------------------------------------------------------------
// gRPC services — one per signal, each forwarding to its sink.
// ---------------------------------------------------------------------------

struct LogsSinkService {
    sink: Arc<ParquetSink>,
}

#[tonic::async_trait]
impl LogsService for LogsSinkService {
    async fn export(
        &self,
        request: Request<ExportLogsServiceRequest>,
    ) -> Result<Response<ExportLogsServiceResponse>, Status> {
        match convert::build_logs_batch(request.into_inner()) {
            Ok(Some(batch)) => self.sink.append(batch).await,
            Ok(None) => {}
            Err(e) => return Err(Status::internal(format!("building logs batch: {e}"))),
        }
        Ok(Response::new(ExportLogsServiceResponse::default()))
    }
}

struct TraceSinkService {
    sink: Arc<ParquetSink>,
}

#[tonic::async_trait]
impl TraceService for TraceSinkService {
    async fn export(
        &self,
        request: Request<ExportTraceServiceRequest>,
    ) -> Result<Response<ExportTraceServiceResponse>, Status> {
        match convert::build_traces_batch(request.into_inner()) {
            Ok(Some(batch)) => self.sink.append(batch).await,
            Ok(None) => {}
            Err(e) => return Err(Status::internal(format!("building traces batch: {e}"))),
        }
        Ok(Response::new(ExportTraceServiceResponse::default()))
    }
}

struct MetricsSinkService {
    sink: Arc<ParquetSink>,
}

#[tonic::async_trait]
impl MetricsService for MetricsSinkService {
    async fn export(
        &self,
        request: Request<ExportMetricsServiceRequest>,
    ) -> Result<Response<ExportMetricsServiceResponse>, Status> {
        match convert::build_metrics_batch(request.into_inner()) {
            Ok(Some(batch)) => self.sink.append(batch).await,
            Ok(None) => {}
            Err(e) => return Err(Status::internal(format!("building metrics batch: {e}"))),
        }
        Ok(Response::new(ExportMetricsServiceResponse::default()))
    }
}

pub(crate) use builder::OtelServer;

/// Test-only shim so crate tests can build a logs batch without exposing the
/// private `convert` module.
#[cfg(test)]
pub(crate) fn convert_logs_for_test(
    req: ExportLogsServiceRequest,
) -> Option<arrow_array::RecordBatch> {
    convert::build_logs_batch(req).unwrap()
}

mod builder {
    use super::*;
    use dispatch::DataFlowDispatcher;
    use tonic::transport::Server;
    use tonic::transport::server::Router;

    /// A built-but-not-yet-running OTLP server: the configured sinks plus the
    /// tonic router wired with the enabled services.
    pub(crate) struct OtelServer {
        pub addr: SocketAddr,
        pub sinks: Vec<Arc<ParquetSink>>,
        pub router: Router,
    }

    /// Wrap a generated `*ServiceServer<T>` with the compression / size limits
    /// every OTLP endpoint needs: accept gzip (exporters compress by default)
    /// and allow large batches.
    macro_rules! tune {
        ($server:expr, $cfg:expr) => {
            $server
                .accept_compressed(CompressionEncoding::Gzip)
                .send_compressed(CompressionEncoding::Gzip)
                .max_decoding_message_size($cfg.max_decoding_message_size)
        };
    }

    impl OtelServer {
        /// Build the sinks and gRPC router for `cfg`. Creates each enabled
        /// signal's output directory.
        pub(crate) fn build(
            cfg: &OtelConfig,
            dispatcher: &DataFlowDispatcher,
        ) -> std::io::Result<Self> {
            let mut sinks = Vec::new();

            let logs = match &cfg.logs_dir {
                Some(dest) => {
                    let sink = Arc::new(ParquetSink::new(
                        "otel_logs",
                        dest,
                        cfg.flush_rows,
                        dispatcher.clone(),
                    )?);
                    sinks.push(sink.clone());
                    Some(tune!(
                        LogsServiceServer::new(LogsSinkService { sink }),
                        cfg
                    ))
                }
                None => None,
            };

            let traces = match &cfg.traces_dir {
                Some(dest) => {
                    let sink = Arc::new(ParquetSink::new(
                        "otel_traces",
                        dest,
                        cfg.flush_rows,
                        dispatcher.clone(),
                    )?);
                    sinks.push(sink.clone());
                    Some(tune!(
                        TraceServiceServer::new(TraceSinkService { sink }),
                        cfg
                    ))
                }
                None => None,
            };

            let metrics = match &cfg.metrics_dir {
                Some(dest) => {
                    let sink = Arc::new(ParquetSink::new(
                        "otel_metrics",
                        dest,
                        cfg.flush_rows,
                        dispatcher.clone(),
                    )?);
                    sinks.push(sink.clone());
                    Some(tune!(
                        MetricsServiceServer::new(MetricsSinkService { sink }),
                        cfg
                    ))
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
}
