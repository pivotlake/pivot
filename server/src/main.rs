//! Binary entry point for the pivotdb postgres-wire server.
//!
//! This internally plans & compiles queries using the `planner` crate (internally based on DuckDB's,
//! planner) with the `ParquetCatalog` and runs queries on `dispatch`

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use catalog::ParquetCatalog;
use clap::Parser;
use dispatch::{BUFFER_SIZE, Dispatch};
use ingest::{IngestConfig, OtelConfig, Signal, SinkDestination};
use server::{Error, Server};
use tracing::info;

/// Postgres-wire-compatible server in front of pivotdb's dispatch engine.
#[derive(Parser, Debug)]
#[command(name = "pivot", about, version)]
struct Args {
    /// TCP socket to bind to.
    #[arg(long, default_value = "127.0.0.1:5432")]
    bind: SocketAddr,

    /// Number of dispatch worker threads. Defaults to the number of cores.
    #[arg(long)]
    workers: Option<usize>,

    /// Start an OTLP/gRPC ingest receiver with the built-in default column
    /// layout. Repeatable — pass `--otel` once per receiver (each needs a
    /// distinct `addr` and output dirs).
    ///
    /// The value is a comma-separated `key=value` spec. Keys: `addr`, `logs`,
    /// `traces`, `metrics` (destination per signal — at least one enables the
    /// receiver), `flush_rows`, `flush_secs`. A destination is a local dir or an
    /// object-store URL (`gs://bucket/prefix`, `s3://bucket/prefix`; object
    /// storage is write-only — read it elsewhere). Examples:
    ///
    ///   --otel 'addr=0.0.0.0:4317,logs=./otel/logs,traces=./otel/traces'
    ///   --otel 'addr=0.0.0.0:4318,logs=gs://my-bucket/otel/logs'
    ///
    /// To choose the columns (and where each comes from), use `--otel-config`.
    #[arg(long = "otel", value_name = "SPEC")]
    otel: Vec<OtelSpec>,

    /// Start an OTLP/gRPC ingest receiver from a TOML file that defines its
    /// signals and per-column mapping. Repeatable. See `ingest::otel::config`
    /// for the file format.
    #[arg(long = "otel-config", value_name = "PATH")]
    otel_config: Vec<OtelFileSpec>,
}

impl Args {
    /// Translate the ingest flags into the configs the server starts. Returns
    /// an empty vec when no ingest is requested.
    fn ingests(&self) -> Vec<IngestConfig> {
        let inline = self.otel.iter().map(|spec| spec.0.clone());
        let from_file = self.otel_config.iter().map(|spec| spec.0.clone());
        inline.chain(from_file).map(IngestConfig::Otel).collect()
    }
}

/// One `--otel-config` receiver, parsed from a TOML file path.
#[derive(Clone, Debug)]
struct OtelFileSpec(OtelConfig);

impl std::str::FromStr for OtelFileSpec {
    type Err = String;

    fn from_str(path: &str) -> Result<Self, Self::Err> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("reading `{path}`: {e}"))?;
        OtelConfig::from_toml(&text)
            .map(OtelFileSpec)
            .map_err(|e| format!("`{path}`: {e}"))
    }
}

/// One `--otel` receiver, parsed from a `key=value,...` spec string.
#[derive(Clone, Debug)]
struct OtelSpec(OtelConfig);

impl std::str::FromStr for OtelSpec {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut addr: Option<SocketAddr> = None;
        let mut logs = None;
        let mut traces = None;
        let mut metrics = None;
        let mut flush_rows = None;
        let mut flush_secs = None;

        for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (key, value) = part
                .split_once('=')
                .ok_or_else(|| format!("expected `key=value`, got `{part}`"))?;
            let value = value.trim();
            match key.trim() {
                "addr" => addr = Some(value.parse().map_err(|e| format!("addr `{value}`: {e}"))?),
                "logs" => logs = Some(SinkDestination::parse(value)),
                "traces" => traces = Some(SinkDestination::parse(value)),
                "metrics" => metrics = Some(SinkDestination::parse(value)),
                "flush_rows" => {
                    flush_rows = Some(
                        value
                            .parse()
                            .map_err(|e| format!("flush_rows `{value}`: {e}"))?,
                    )
                }
                "flush_secs" => {
                    flush_secs = Some(
                        value
                            .parse()
                            .map_err(|e| format!("flush_secs `{value}`: {e}"))?,
                    )
                }
                other => return Err(format!("unknown key `{other}` in --otel spec")),
            }
        }

        let addr = match addr {
            Some(a) => a,
            None => ingest::DEFAULT_OTLP_ADDR.parse().unwrap(),
        };
        let mut cfg = OtelConfig::new(addr);
        if let Some(rows) = flush_rows {
            cfg.flush_rows = rows;
        }
        if let Some(secs) = flush_secs {
            cfg.flush_interval = Duration::from_secs(secs);
        }
        // The inline spec uses the built-in default column mapping for each
        // enabled signal; `--otel-config` is the route to custom columns.
        for (signal, dest) in [
            (Signal::Logs, logs),
            (Signal::Traces, traces),
            (Signal::Metrics, metrics),
        ] {
            if let Some(dest) = dest {
                cfg.enable_default(signal, dest)
                    .map_err(|e| e.to_string())?;
            }
        }
        Ok(OtelSpec(cfg))
    }
}

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

/// Returns the total physical memory of the machine in bytes.
pub fn get_total_memory() -> usize {
    sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
    )
    .total_memory() as usize
}

fn main() -> Result<(), Error> {
    init_tracing();
    let args = Args::parse();

    let workers = args.workers.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    info!(workers, "initialising dispatch");
    let dispatch = Dispatch::spin_up(workers, get_total_memory() / 2 / BUFFER_SIZE);

    let ingests = args.ingests();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    rt.block_on(async move {
        let catalog = Arc::new(ParquetCatalog::new());
        let server = Server::new(args.bind, dispatch, catalog, ingests);
        let shutdown = Box::pin(async {
            let _ = tokio::signal::ctrl_c().await;
        });
        server.serve(shutdown).await
    })
}
