//! The server's log: the config's `log` filter over a `tracing` subscriber,
//! installed once at startup.

use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::MakeWriter;

/// The filter when the config sets none: everything at `info` and above.
pub const DEFAULT_FILTER: &str = "info";

/// Targets whose `INFO` output is noise for anyone reading the log.
const QUIET_TARGETS: &str = "delta_kernel=warn,delta_kernel_default_engine=warn";

/// The config's `log` is not a `tracing` filter.
#[derive(Debug, thiserror::Error)]
#[error("`log` filter `{filter}` is not valid: {source}")]
pub struct FilterError {
    filter: String,
    #[source]
    source: tracing_subscriber::filter::ParseError,
}

/// Install the process's subscriber: what `filter` selects, on top of the
/// quiet targets, written through `writer`, coloured when `ansi`. Once per
/// process, before the workers start, so their startup lines are caught.
pub fn install<W>(filter: &str, writer: W, ansi: bool) -> Result<(), FilterError>
where
    W: for<'a> MakeWriter<'a> + Send + Sync + 'static,
{
    let env_filter =
        EnvFilter::try_new(format!("{QUIET_TARGETS},{filter}")).map_err(|source| FilterError {
            filter: filter.to_string(),
            source,
        })?;
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(writer)
        .with_ansi(ansi)
        .init();
    Ok(())
}
