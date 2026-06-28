//! Kafka ingest source.
//!
//! Models ClickHouse's classic Kafka table engine: an in-process consumer reads
//! a topic, accumulates a block, writes it to the destination catalog table, and
//! commits the Kafka offset - reusing the same encode/partition/commit/compact
//! pipeline as the OTLP source ([`encode_and_append`](crate::write)). The one
//! source-specific concern is the **offset durability coupling**, which lives in
//! [`consumer`].
//!
//! Records are decoded into the destination table's schema (JSONEachRow style):
//! the table's declared columns are the schema, JSON/Avro/Protobuf values match
//! fields by name, and a table may declare virtual columns (`_topic`,
//! `_partition`, `_offset`, `_timestamp`, `_key`) to capture Kafka metadata. See
//! [`decode`].

mod config;
mod consumer;
mod decode;

use std::sync::Arc;

use catalog::ParquetCatalog;
use dispatch::DataFlowDispatcher;
use rdkafka::ClientConfig;
use rdkafka::consumer::{Consumer, StreamConsumer};
use tracing::info;

pub use config::{
    AutoOffsetReset, ConfigError, DEFAULT_FLUSH_ROWS, DEFAULT_FLUSH_SECS, DEFAULT_NUM_CONSUMERS,
    KafkaConfig,
};
pub use decode::Format;

use consumer::KafkaConsumer;
use decode::SchemaRegistry;

/// The consumer tasks for one configured Kafka source, built but not yet spawned.
/// [`Ingestor::start`](crate::Ingestor) spawns each with the shutdown watch; each
/// task self-flushes its final block on shutdown, so no separate `Flushable`
/// registration is needed.
pub(crate) struct KafkaSource {
    pub(crate) consumers: Vec<KafkaConsumer>,
}

impl KafkaSource {
    /// Build the decoder, destination schema, and `num_consumers` Kafka consumers
    /// for `config`. Fails if the destination (or dead-letter) table doesn't
    /// exist, a column type can't be written, or a consumer can't be created.
    pub(crate) fn build(
        config: &KafkaConfig,
        dispatcher: &DataFlowDispatcher,
        catalog: &Arc<ParquetCatalog>,
    ) -> std::io::Result<Self> {
        let schema = table_schema(catalog, &config.table)?;

        let dead_letter = match &config.dead_letter_table {
            Some(table) => Some((table.clone(), table_schema(catalog, table)?)),
            None => None,
        };

        let registry = config
            .schema_registry_url
            .as_ref()
            .map(|url| Arc::new(SchemaRegistry::new(url.clone())));
        let decoder = decode::build_decoder(config.format, registry);

        let config = Arc::new(config.clone());
        let topics: Vec<&str> = config.topics.iter().map(String::as_str).collect();

        let mut consumers = Vec::with_capacity(config.num_consumers);
        for _ in 0..config.num_consumers {
            let stream_consumer = build_stream_consumer(&config)?;
            stream_consumer
                .subscribe(&topics)
                .map_err(|e| std::io::Error::other(format!("subscribing to topics: {e}")))?;
            consumers.push(KafkaConsumer::new(
                config.clone(),
                Arc::new(stream_consumer),
                decoder.clone(),
                catalog.clone(),
                dispatcher.clone(),
                schema.clone(),
                dead_letter.clone(),
            ));
        }

        info!(
            group = %config.group_id,
            table = %config.table,
            topics = ?config.topics,
            format = ?config.format,
            num_consumers = config.num_consumers,
            "starting kafka ingest"
        );
        Ok(Self { consumers })
    }
}

/// Build one librdkafka `StreamConsumer` with auto-commit off (we commit after
/// the catalog append) and the configured offset reset / raw properties.
fn build_stream_consumer(config: &KafkaConfig) -> std::io::Result<StreamConsumer> {
    let mut client = ClientConfig::new();
    client
        .set("bootstrap.servers", &config.brokers)
        .set("group.id", &config.group_id)
        .set("enable.auto.commit", "false")
        .set(
            "auto.offset.reset",
            config.auto_offset_reset.as_librdkafka(),
        )
        .set("enable.partition.eof", "false");
    for (key, value) in &config.properties {
        client.set(key, value);
    }
    client
        .create()
        .map_err(|e| std::io::Error::other(format!("creating kafka consumer: {e}")))
}

/// Derive the Arrow schema a source decodes records into from its destination
/// table's declared columns. Errors if the table doesn't exist or a column has a
/// type Kafka ingest can't write.
fn table_schema(
    catalog: &Arc<ParquetCatalog>,
    table: &str,
) -> std::io::Result<arrow_schema::SchemaRef> {
    let handle = catalog.table_handle(table).ok_or_else(|| {
        std::io::Error::other(format!(
            "ingest table `{table}` does not exist - create it before starting ingest"
        ))
    })?;
    decode::table_schema(&handle.columns())
        .map_err(|e| std::io::Error::other(format!("table `{table}`: {e}")))
}
