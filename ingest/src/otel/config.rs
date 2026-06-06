//! TOML configuration for an OTLP receiver, and the constructors that turn it
//! into an [`OtelConfig`].
//!
//! A receiver file looks like:
//!
//! ```toml
//! addr = "0.0.0.0:4317"
//! flush_rows = 50000
//! flush_secs = 10
//!
//! [logs]
//! destination = "./otel/logs"   # local dir, or gs:// / s3:// (write-only)
//! columns = [
//!   { name = "Timestamp",          field      = "time_unix_nano" },
//!   { name = "ServiceName",        attr       = "resource:service.name" },
//!   { name = "ResourceAttributes", attrs_json = "resource" },
//! ]
//!
//! # Omit `columns` to use the built-in default mapping for that signal.
//! # Omit a whole signal table ([traces] / [metrics]) to disable it.
//! ```
//!
//! Each column carries exactly one of `field`, `attr` (`"scope:key"`), or
//! `attrs_json` (`"scope"`), where a scope is `resource` / `scope` / `record`.

use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;

use super::convert::{AttrScope, ColumnSpec, CompiledMapping, Source, UnknownField};
use super::mapping::Signal;
use super::{DEFAULT_OTLP_ADDR, OtelConfig, SignalSetup};
use crate::sink::SinkDestination;

/// An error building an [`OtelConfig`] from TOML.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("parsing receiver config: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("invalid addr `{addr}`: {source}")]
    Addr {
        addr: String,
        source: std::net::AddrParseError,
    },
    #[error("column `{name}`: {reason}")]
    Column { name: String, reason: String },
    #[error(transparent)]
    UnknownField(#[from] UnknownField),
}

impl OtelConfig {
    /// Build a receiver config from a TOML document.
    pub fn from_toml(toml_str: &str) -> Result<Self, ConfigError> {
        let raw: ReceiverToml = toml::from_str(toml_str)?;

        let addr = match raw.addr {
            Some(a) => a
                .parse()
                .map_err(|source| ConfigError::Addr { addr: a, source })?,
            None => DEFAULT_OTLP_ADDR.parse().expect("valid default addr"),
        };

        let mut cfg = OtelConfig::new(addr);
        if let Some(rows) = raw.flush_rows {
            cfg.flush_rows = rows;
        }
        if let Some(secs) = raw.flush_secs {
            cfg.flush_interval = Duration::from_secs(secs);
        }
        if let Some(size) = raw.max_decoding_message_size {
            cfg.max_decoding_message_size = size;
        }
        cfg.logs = raw.logs.map(|s| s.into_setup(Signal::Logs)).transpose()?;
        cfg.traces = raw
            .traces
            .map(|s| s.into_setup(Signal::Traces))
            .transpose()?;
        cfg.metrics = raw
            .metrics
            .map(|s| s.into_setup(Signal::Metrics))
            .transpose()?;
        Ok(cfg)
    }

    /// Enable `signal` writing to `destination` with the built-in default
    /// column mapping. Used by the inline `--otel` CLI spec and anywhere a
    /// caller wants the out-of-the-box layout.
    pub fn enable_default(
        &mut self,
        signal: Signal,
        destination: SinkDestination,
    ) -> Result<&mut Self, ConfigError> {
        let setup =
            SignalSetup::compile(signal, destination, super::defaults::columns_for(signal))?;
        *self.signal_slot(signal) = Some(setup);
        Ok(self)
    }

    fn signal_slot(&mut self, signal: Signal) -> &mut Option<SignalSetup> {
        match signal {
            Signal::Logs => &mut self.logs,
            Signal::Traces => &mut self.traces,
            Signal::Metrics => &mut self.metrics,
        }
    }
}

impl SignalSetup {
    /// Compile `columns` for `signal` and pair them with a destination.
    fn compile(
        signal: Signal,
        destination: SinkDestination,
        columns: Vec<ColumnSpec>,
    ) -> Result<Self, UnknownField> {
        Ok(SignalSetup {
            destination,
            mapping: Arc::new(CompiledMapping::compile(signal, &columns)?),
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiverToml {
    addr: Option<String>,
    flush_rows: Option<usize>,
    flush_secs: Option<u64>,
    max_decoding_message_size: Option<usize>,
    logs: Option<SignalToml>,
    traces: Option<SignalToml>,
    metrics: Option<SignalToml>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SignalToml {
    destination: String,
    #[serde(default)]
    columns: Vec<ColumnToml>,
}

impl SignalToml {
    fn into_setup(self, signal: Signal) -> Result<SignalSetup, ConfigError> {
        let columns = if self.columns.is_empty() {
            super::defaults::columns_for(signal)
        } else {
            self.columns
                .into_iter()
                .map(ColumnToml::into_spec)
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(SignalSetup::compile(
            signal,
            SinkDestination::parse(&self.destination),
            columns,
        )?)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ColumnToml {
    name: String,
    field: Option<String>,
    attr: Option<String>,
    attrs_json: Option<String>,
}

impl ColumnToml {
    fn into_spec(self) -> Result<ColumnSpec, ConfigError> {
        let err = |reason: &str| ConfigError::Column {
            name: self.name.clone(),
            reason: reason.to_string(),
        };
        let source = match (self.field, self.attr, self.attrs_json) {
            (Some(field), None, None) => Source::Field(field),
            (None, Some(attr), None) => {
                let (scope, key) = attr.split_once(':').ok_or_else(|| {
                    err("`attr` must be `scope:key`, e.g. `resource:service.name`")
                })?;
                Source::Attr {
                    scope: parse_scope(scope).ok_or_else(|| err(&unknown_scope(scope)))?,
                    key: key.to_string(),
                }
            }
            (None, None, Some(scope)) => {
                Source::AttrsJson(parse_scope(&scope).ok_or_else(|| err(&unknown_scope(&scope)))?)
            }
            _ => return Err(err("set exactly one of `field`, `attr`, `attrs_json`")),
        };
        Ok(ColumnSpec {
            name: self.name,
            source,
        })
    }
}

fn parse_scope(token: &str) -> Option<AttrScope> {
    match token {
        "resource" => Some(AttrScope::Resource),
        "scope" => Some(AttrScope::Scope),
        "record" => Some(AttrScope::Record),
        _ => None,
    }
}

fn unknown_scope(token: &str) -> String {
    format!("unknown attribute scope `{token}` (expected resource / scope / record)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::DataType;

    #[test]
    fn custom_columns_compile_to_expected_schema() {
        let toml = r#"
            addr = "0.0.0.0:4319"
            flush_rows = 100

            [logs]
            destination = "./out/logs"
            columns = [
              { name = "Ts",      field      = "time_unix_nano" },
              { name = "Svc",     attr       = "resource:service.name" },
              { name = "Method",  attr       = "record:http.method" },
              { name = "ResAttr", attrs_json = "resource" },
            ]
        "#;

        let cfg = OtelConfig::from_toml(toml).unwrap();

        assert_eq!(cfg.flush_rows, 100);
        let setup = cfg.logs.expect("logs enabled");
        let schema = setup.mapping.schema();
        let got: Vec<(&str, &DataType)> = schema
            .fields()
            .iter()
            .map(|f| (f.name().as_str(), f.data_type()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("Ts", &DataType::Int64),
                ("Svc", &DataType::Utf8),
                ("Method", &DataType::Utf8),
                ("ResAttr", &DataType::Utf8),
            ]
        );
    }

    #[test]
    fn empty_columns_fall_back_to_defaults() {
        let toml = r#"
            [traces]
            destination = "./out/traces"
        "#;

        let cfg = OtelConfig::from_toml(toml).unwrap();

        let setup = cfg.traces.expect("traces enabled");
        // Same width as the default trace mapping.
        assert_eq!(setup.mapping.schema().fields().len(), 13);
        assert!(cfg.logs.is_none() && cfg.metrics.is_none());
    }

    #[test]
    fn unknown_field_is_rejected() {
        let toml = r#"
            [logs]
            destination = "./out/logs"
            columns = [ { name = "X", field = "not_a_field" } ]
        "#;

        let err = OtelConfig::from_toml(toml).unwrap_err();

        assert!(matches!(err, ConfigError::UnknownField(_)), "got {err:?}");
    }

    #[test]
    fn ambiguous_source_is_rejected() {
        let toml = r#"
            [logs]
            destination = "./out/logs"
            columns = [ { name = "X", field = "body", attr = "resource:k" } ]
        "#;

        let err = OtelConfig::from_toml(toml).unwrap_err();

        assert!(matches!(err, ConfigError::Column { .. }), "got {err:?}");
    }
}
