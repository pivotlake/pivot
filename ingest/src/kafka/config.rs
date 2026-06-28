//! Configuration for a Kafka ingest source, built from a `--kafka` inline spec
//! or a `--kafka-config` TOML file.
//!
//! A TOML receiver looks like:
//!
//! ```toml
//! brokers = "localhost:9092"
//! topics = ["events"]
//! group_id = "pivot-events"
//! table = "events"                       # destination catalog table (must exist)
//! format = "json"                        # json | avro | protobuf
//! schema_registry_url = "http://localhost:8081"   # required for avro/protobuf
//! flush_rows = 1000000
//! flush_secs = 5
//! num_consumers = 1
//! auto_offset_reset = "earliest"         # earliest | latest (new groups only)
//! skip_broken_messages = 0
//! dead_letter_table = "events_errors"    # optional; poison messages land here
//!
//! [properties]                            # raw librdkafka passthrough (SASL/SSL)
//! "security.protocol" = "SASL_SSL"
//! "sasl.mechanisms"   = "PLAIN"
//! ```

use std::collections::HashMap;
use std::time::Duration;

use serde::Deserialize;

use super::decode::Format;

/// Flush a block once a consumer buffers this many rows (the size trigger).
pub const DEFAULT_FLUSH_ROWS: usize = 1_000_000;
/// Also flush on this cadence so a low-traffic topic still lands files (the
/// latency trigger).
pub const DEFAULT_FLUSH_SECS: u64 = 5;
/// Consumer tasks per source, sharing the group (Kafka assigns partitions).
pub const DEFAULT_NUM_CONSUMERS: usize = 1;

/// Where a brand-new consumer group starts when it has no committed offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AutoOffsetReset {
    #[default]
    Earliest,
    Latest,
}

impl AutoOffsetReset {
    /// The librdkafka `auto.offset.reset` value.
    pub(crate) fn as_librdkafka(self) -> &'static str {
        match self {
            AutoOffsetReset::Earliest => "earliest",
            AutoOffsetReset::Latest => "latest",
        }
    }
}

impl std::str::FromStr for AutoOffsetReset {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "earliest" | "beginning" => Ok(AutoOffsetReset::Earliest),
            "latest" | "end" => Ok(AutoOffsetReset::Latest),
            other => Err(format!(
                "unknown auto_offset_reset `{other}` (expected earliest/latest)"
            )),
        }
    }
}

/// An error building a [`KafkaConfig`].
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("parsing kafka config: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("{0}")]
    Field(String),
    #[error("`{key}`: {reason}")]
    Spec { key: String, reason: String },
}

/// Configuration for one Kafka ingest source.
#[derive(Debug, Clone)]
pub struct KafkaConfig {
    /// Comma-separated bootstrap servers (`host:port,host:port`).
    pub brokers: String,
    /// Topics to subscribe to (all feed the one destination table).
    pub topics: Vec<String>,
    /// Consumer group id; replicas sharing it split the partitions.
    pub group_id: String,
    /// Destination catalog table (must already exist); its schema drives decode.
    pub table: String,
    /// Message value format.
    pub format: Format,
    /// Confluent Schema Registry URL; required for avro/protobuf.
    pub schema_registry_url: Option<String>,
    /// Flush a block once this many rows buffer.
    pub flush_rows: usize,
    /// Also flush on this cadence.
    pub flush_interval: Duration,
    /// Consumer tasks to run for this source.
    pub num_consumers: usize,
    /// Where a new group starts.
    pub auto_offset_reset: AutoOffsetReset,
    /// Raw librdkafka properties (SASL/SSL etc.), applied verbatim.
    pub properties: HashMap<String, String>,
    /// Tolerate up to this many undecodable messages before stopping the
    /// consumer (when no dead-letter table is set).
    pub skip_broken_messages: u64,
    /// Optional table that undecodable messages are written to instead of being
    /// skipped.
    pub dead_letter_table: Option<String>,
}

impl KafkaConfig {
    /// Validate required fields and registry presence for registry-backed
    /// formats.
    fn validate(&self) -> Result<(), ConfigError> {
        let require = |cond: bool, msg: &str| {
            cond.then_some(())
                .ok_or_else(|| ConfigError::Field(msg.to_string()))
        };
        require(!self.brokers.trim().is_empty(), "`brokers` is required")?;
        require(!self.topics.is_empty(), "at least one `topic` is required")?;
        require(
            self.topics.iter().all(|t| !t.trim().is_empty()),
            "topic names must not be empty",
        )?;
        require(!self.group_id.trim().is_empty(), "`group_id` is required")?;
        require(!self.table.trim().is_empty(), "`table` is required")?;
        require(self.num_consumers >= 1, "`num_consumers` must be >= 1")?;
        require(self.flush_rows >= 1, "`flush_rows` must be >= 1")?;
        require(
            !self.flush_interval.is_zero(),
            "`flush_secs` must be >= 1 (a zero flush interval is rejected)",
        )?;
        if self.format.needs_registry() {
            require(
                self.schema_registry_url.is_some(),
                "`schema_registry_url` is required for avro/protobuf",
            )?;
        }
        Ok(())
    }

    /// Build from a TOML document.
    pub fn from_toml(toml_str: &str) -> Result<Self, ConfigError> {
        let raw: KafkaToml = toml::from_str(toml_str)?;
        let config = raw.into_config()?;
        config.validate()?;
        Ok(config)
    }

    /// Build from a `key=value,...` inline spec (the `--kafka` flag). Multiple
    /// topics are `;`-separated (`topics=a;b`); raw librdkafka properties go
    /// through `--kafka-config` only.
    pub fn from_spec(spec: &str) -> Result<Self, ConfigError> {
        let mut brokers = None;
        let mut topics = Vec::new();
        let mut group_id = None;
        let mut table = None;
        let mut format = Format::Json;
        let mut schema_registry_url = None;
        let mut flush_rows = DEFAULT_FLUSH_ROWS;
        let mut flush_secs = DEFAULT_FLUSH_SECS;
        let mut num_consumers = DEFAULT_NUM_CONSUMERS;
        let mut auto_offset_reset = AutoOffsetReset::default();
        let mut skip_broken_messages = 0u64;
        let mut dead_letter_table = None;

        let spec_err = |key: &str, reason: String| ConfigError::Spec {
            key: key.to_string(),
            reason,
        };

        for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (key, value) = part
                .split_once('=')
                .ok_or_else(|| spec_err(part, "expected `key=value`".into()))?;
            let (key, value) = (key.trim(), value.trim());
            match key {
                "brokers" => brokers = Some(value.to_string()),
                "topic" | "topics" => topics.extend(
                    value
                        .split(';')
                        .map(str::trim)
                        .filter(|t| !t.is_empty())
                        .map(str::to_string),
                ),
                "group_id" | "group" => group_id = Some(value.to_string()),
                "table" => table = Some(value.to_string()),
                "format" => format = value.parse().map_err(|e| spec_err(key, e))?,
                "schema_registry_url" | "registry" => schema_registry_url = Some(value.to_string()),
                "flush_rows" => {
                    flush_rows = value.parse().map_err(|e| spec_err(key, format!("{e}")))?
                }
                "flush_secs" => {
                    flush_secs = value.parse().map_err(|e| spec_err(key, format!("{e}")))?
                }
                "num_consumers" | "consumers" => {
                    num_consumers = value.parse().map_err(|e| spec_err(key, format!("{e}")))?
                }
                "auto_offset_reset" | "offset_reset" => {
                    auto_offset_reset = value.parse().map_err(|e| spec_err(key, e))?
                }
                "skip_broken_messages" | "skip_broken" => {
                    skip_broken_messages =
                        value.parse().map_err(|e| spec_err(key, format!("{e}")))?
                }
                "dead_letter_table" | "dlq" => dead_letter_table = Some(value.to_string()),
                other => return Err(spec_err(other, "unknown key".into())),
            }
        }

        let config = KafkaConfig {
            brokers: brokers.unwrap_or_default(),
            topics,
            group_id: group_id.unwrap_or_default(),
            table: table.unwrap_or_default(),
            format,
            schema_registry_url,
            flush_rows,
            flush_interval: Duration::from_secs(flush_secs),
            num_consumers,
            auto_offset_reset,
            properties: HashMap::new(),
            skip_broken_messages,
            dead_letter_table,
        };
        config.validate()?;
        Ok(config)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KafkaToml {
    brokers: String,
    topics: Vec<String>,
    group_id: String,
    table: String,
    format: Option<String>,
    schema_registry_url: Option<String>,
    flush_rows: Option<usize>,
    flush_secs: Option<u64>,
    num_consumers: Option<usize>,
    auto_offset_reset: Option<String>,
    skip_broken_messages: Option<u64>,
    dead_letter_table: Option<String>,
    #[serde(default)]
    properties: HashMap<String, String>,
}

impl KafkaToml {
    fn into_config(self) -> Result<KafkaConfig, ConfigError> {
        let format = match self.format {
            Some(f) => f.parse().map_err(ConfigError::Field)?,
            None => Format::default(),
        };
        let auto_offset_reset = match self.auto_offset_reset {
            Some(o) => o.parse().map_err(ConfigError::Field)?,
            None => AutoOffsetReset::default(),
        };
        Ok(KafkaConfig {
            brokers: self.brokers,
            topics: self.topics,
            group_id: self.group_id,
            table: self.table,
            format,
            schema_registry_url: self.schema_registry_url,
            flush_rows: self.flush_rows.unwrap_or(DEFAULT_FLUSH_ROWS),
            flush_interval: Duration::from_secs(self.flush_secs.unwrap_or(DEFAULT_FLUSH_SECS)),
            num_consumers: self.num_consumers.unwrap_or(DEFAULT_NUM_CONSUMERS),
            auto_offset_reset,
            properties: self.properties,
            skip_broken_messages: self.skip_broken_messages.unwrap_or(0),
            dead_letter_table: self.dead_letter_table,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = "brokers=localhost:9092,topics=events,group_id=g,table=events";

    #[test]
    fn a_minimal_json_spec_parses() {
        let config = KafkaConfig::from_spec(BASE).unwrap();

        assert_eq!(config.topics, vec!["events".to_string()]);
        assert_eq!(config.format, Format::Json);
        assert_eq!(config.flush_rows, DEFAULT_FLUSH_ROWS);
    }

    #[test]
    fn a_trailing_topic_separator_is_ignored() {
        let config = KafkaConfig::from_spec("brokers=b,topics=a;b;,group_id=g,table=t").unwrap();

        assert_eq!(config.topics, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn invalid_specs_are_rejected() {
        // Zero flush interval would panic `tokio::time::interval`.
        assert!(KafkaConfig::from_spec(&format!("{BASE},flush_secs=0")).is_err());
        // Zero flush size would flush one file per message.
        assert!(KafkaConfig::from_spec(&format!("{BASE},flush_rows=0")).is_err());
        // No topics left after filtering the empty one.
        assert!(KafkaConfig::from_spec("brokers=b,topics=,group_id=g,table=t").is_err());
        // avro/protobuf need a registry.
        assert!(KafkaConfig::from_spec(&format!("{BASE},format=avro")).is_err());
    }
}
