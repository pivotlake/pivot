//! One Kafka consumer task: poll messages, decode them into the destination
//! table's schema, buffer a block, and flush it to Parquet, committing Kafka
//! offsets *only after* the block is durably appended to the catalog
//! (at-least-once).
//!
//! The offset coupling is the whole point. librdkafka auto-commit is off; we
//! track, per partition, the first uncommitted offset (`block_start`) and the
//! last consumed offset. A flush distinguishes the failure it hit:
//!
//! - on success, commit `last + 1`;
//! - on a *transient* write failure (a catalog/store I/O error), `seek` back to
//!   `block_start` and re-consume the block;
//! - on a *permanent* failure (the destination table is gone, or an encode bug),
//!   stop the consumer loudly rather than spin re-consuming a block that can
//!   never land.
//!
//! Poison messages are caught at *both* decode and encode time: a value that
//! decodes but whose types don't fit the table schema is validated per row
//! before it can poison the whole block, then routed to the dead-letter table or
//! counted against `skip_broken_messages` (which, once exceeded, stops the
//! consumer). No rows are dropped silently; duplicates are possible after a
//! transient failure, which is the at-least-once contract.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use std::sync::Arc;

use arrow_schema::SchemaRef;
use catalog::ParquetCatalog;
use dispatch::DataFlowDispatcher;
use rdkafka::consumer::{CommitMode, Consumer, StreamConsumer};
use rdkafka::error::KafkaError;
use rdkafka::message::Message;
use rdkafka::topic_partition_list::{Offset, TopicPartitionList};
use serde_json::{Value, json};
use tokio::sync::watch;
use tokio::time::MissedTickBehavior;
use tracing::{error, info, warn};

use super::KafkaConfig;
use super::decode::{self, DecodeError, KafkaBatch, MessageDecoder};
use crate::write::{WriteError, encode_and_append};

/// Virtual-column field names a table may declare to capture Kafka metadata,
/// mirroring ClickHouse's `_topic` / `_partition` / `_offset` / `_timestamp`
/// (seconds) / `_timestamp_ms` (milliseconds) / `_key`.
const VIRTUAL_TOPIC: &str = "_topic";
const VIRTUAL_PARTITION: &str = "_partition";
const VIRTUAL_OFFSET: &str = "_offset";
const VIRTUAL_TIMESTAMP: &str = "_timestamp";
const VIRTUAL_TIMESTAMP_MS: &str = "_timestamp_ms";
const VIRTUAL_KEY: &str = "_key";

/// Which virtual columns the destination schema declares, so we only populate
/// the ones the table actually wants.
struct VirtualFields {
    topic: bool,
    partition: bool,
    offset: bool,
    timestamp: bool,
    timestamp_ms: bool,
    key: bool,
}

impl VirtualFields {
    fn from_schema(schema: &SchemaRef) -> Self {
        let has = |name: &str| schema.field_with_name(name).is_ok();
        Self {
            topic: has(VIRTUAL_TOPIC),
            partition: has(VIRTUAL_PARTITION),
            offset: has(VIRTUAL_OFFSET),
            timestamp: has(VIRTUAL_TIMESTAMP),
            timestamp_ms: has(VIRTUAL_TIMESTAMP_MS),
            key: has(VIRTUAL_KEY),
        }
    }
}

/// The owned copy of a message's data we keep after releasing the borrowed
/// rdkafka message (so the consumer handle is free for the next `recv`).
struct OwnedRecord {
    topic: String,
    partition: i32,
    offset: i64,
    key: Option<String>,
    payload: Vec<u8>,
    timestamp_millis: Option<i64>,
}

/// Per-partition offset bookkeeping for the in-flight block.
#[derive(Clone, Copy)]
struct PartitionProgress {
    /// First offset of the current (uncommitted) block: the seek target on a
    /// transient flush failure.
    block_start: i64,
    /// Last offset consumed into the current block.
    last_consumed: i64,
}

/// One consumer's mutable state and its share of a source's configuration.
pub(crate) struct KafkaConsumer {
    config: Arc<KafkaConfig>,
    consumer: Arc<StreamConsumer>,
    decoder: Arc<dyn MessageDecoder>,
    catalog: Arc<ParquetCatalog>,
    dispatcher: DataFlowDispatcher,
    schema: SchemaRef,
    virtual_fields: VirtualFields,
    /// `(table, schema)` for the dead-letter table, if configured.
    dead_letter: Option<(String, SchemaRef)>,
    rows: Vec<Value>,
    dead_rows: Vec<Value>,
    progress: HashMap<(String, i32), PartitionProgress>,
    broken_seen: u64,
    /// Set when the consumer must stop (poison budget exhausted, a permanent
    /// write failure, or a failed rewind). The run loop breaks and skips the
    /// final commit so a restart reprocesses from the last committed offset.
    fatal: bool,
}

impl KafkaConsumer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        config: Arc<KafkaConfig>,
        consumer: Arc<StreamConsumer>,
        decoder: Arc<dyn MessageDecoder>,
        catalog: Arc<ParquetCatalog>,
        dispatcher: DataFlowDispatcher,
        schema: SchemaRef,
        dead_letter: Option<(String, SchemaRef)>,
    ) -> Self {
        let virtual_fields = VirtualFields::from_schema(&schema);
        Self {
            config,
            consumer,
            decoder,
            catalog,
            dispatcher,
            schema,
            virtual_fields,
            dead_letter,
            rows: Vec::new(),
            dead_rows: Vec::new(),
            progress: HashMap::new(),
            broken_seen: 0,
            fatal: false,
        }
    }

    /// Run until the shutdown watch flips (or a fatal condition stops the
    /// consumer), flushing a final block on a clean exit.
    pub(crate) async fn run(mut self, mut shutdown: watch::Receiver<bool>) {
        let consumer = Arc::clone(&self.consumer);
        // Idle backstop: drives a flush when no messages arrive. Under load the
        // per-message latency check below is what bounds flush latency, since a
        // fair `select!` could otherwise keep choosing `recv`.
        let mut tick = tokio::time::interval(self.config.flush_interval);
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let mut last_flush = Instant::now();

        loop {
            tokio::select! {
                changed = shutdown.changed() => {
                    if changed.is_err() || *shutdown.borrow() {
                        break;
                    }
                }
                received = consumer.recv() => {
                    match received {
                        Ok(message) => {
                            let record = OwnedRecord::extract(&message);
                            drop(message);
                            self.ingest(record).await;
                            if self.fatal {
                                break;
                            }
                            let buffered = self.rows.len() + self.dead_rows.len();
                            if buffered >= self.config.flush_rows
                                || last_flush.elapsed() >= self.config.flush_interval
                            {
                                if !self.flush().await {
                                    break;
                                }
                                last_flush = Instant::now();
                            }
                        }
                        Err(e) => warn!(group = %self.config.group_id, error = %e, "kafka recv error"),
                    }
                }
                _ = tick.tick() => {
                    if !self.flush().await {
                        break;
                    }
                    last_flush = Instant::now();
                }
            }
        }

        // A fatal stop leaves the block uncommitted on purpose (reprocess on
        // restart); only a clean shutdown drains the final block.
        if !self.fatal {
            self.flush().await;
        }
        info!(group = %self.config.group_id, "kafka consumer stopped");
    }

    /// Decode and buffer one message, advancing that partition's progress. A
    /// value that decodes but doesn't fit the table schema is treated as a poison
    /// message (caught here so it can't fail the whole block at flush).
    async fn ingest(&mut self, record: OwnedRecord) {
        self.advance(&record);
        match self.decoder.decode(&record.payload) {
            Ok(mut value) => {
                self.inject_virtuals(&mut value, &record);
                if let Err(e) = decode::validate_row(&self.schema, &value) {
                    self.handle_broken(&record, &DecodeError::SchemaMismatch(e.to_string()));
                } else {
                    self.rows.push(value);
                }
            }
            Err(e) => self.handle_broken(&record, &e),
        }
    }

    /// Route an undecodable / non-fitting message to the dead-letter table, or
    /// count it against `skip_broken_messages`. Once the budget is exceeded with
    /// no dead-letter table, mark the consumer fatal so it stops rather than
    /// silently dropping records.
    fn handle_broken(&mut self, record: &OwnedRecord, err: &DecodeError) {
        if let Some((_, dead_schema)) = &self.dead_letter {
            let mut row = json!({
                VIRTUAL_TOPIC: record.topic,
                VIRTUAL_PARTITION: record.partition,
                VIRTUAL_OFFSET: record.offset,
                VIRTUAL_KEY: record.key,
                "_raw": String::from_utf8_lossy(&record.payload),
                "_error": err.to_string(),
            });
            // Keep only the fields the dead-letter table declares (arrow-json
            // would ignore extras, but trimming keeps the rows clean).
            retain_schema_fields(&mut row, dead_schema);
            self.dead_rows.push(row);
            return;
        }
        self.broken_seen += 1;
        warn!(
            group = %self.config.group_id,
            topic = %record.topic,
            partition = record.partition,
            offset = record.offset,
            error = %err,
            "undecodable kafka message"
        );
        if self.broken_seen > self.config.skip_broken_messages {
            error!(
                group = %self.config.group_id,
                skip_broken_messages = self.config.skip_broken_messages,
                "exceeded skip_broken_messages; stopping consumer (set a dead_letter_table or fix the producer)"
            );
            self.fatal = true;
        }
    }

    /// Flush the buffered block: write the good rows and any dead-letter rows,
    /// then commit offsets. Returns `false` when the consumer must stop.
    async fn flush(&mut self) -> bool {
        if self.progress.is_empty() {
            return true; // nothing consumed since the last flush
        }

        let rows = std::mem::take(&mut self.rows);
        let dead_rows = std::mem::take(&mut self.dead_rows);

        let table = self.config.table.clone();
        if let Err(e) = self.write_block(&table, self.schema.clone(), rows).await {
            return self.handle_write_error(&table, e).await;
        }
        if let Some((dlq_table, dlq_schema)) = self.dead_letter.clone() {
            if let Err(e) = self.write_block(&dlq_table, dlq_schema, dead_rows).await {
                return self.handle_write_error(&dlq_table, e).await;
            }
        }

        self.commit().await;
        true
    }

    /// Encode `rows` against `schema` and append them to `table`. A no-op when
    /// `rows` is empty.
    async fn write_block(
        &self,
        table: &str,
        schema: SchemaRef,
        rows: Vec<Value>,
    ) -> Result<(), WriteError> {
        if rows.is_empty() {
            return Ok(());
        }
        let batch = KafkaBatch { rows, schema };
        encode_and_append(&self.catalog, table, &self.dispatcher, vec![batch]).await
    }

    /// Decide what a write failure means: a catalog/store I/O error is transient
    /// (rewind and re-consume the block); a missing table or encode failure is
    /// permanent (stop the consumer). Returns `false` to stop.
    async fn handle_write_error(&mut self, table: &str, err: WriteError) -> bool {
        match err {
            WriteError::Append(e) => {
                warn!(group = %self.config.group_id, table, error = %e, "kafka flush failed (transient); rewinding block");
                self.rewind().await
            }
            other => {
                error!(group = %self.config.group_id, table, error = %other, "kafka flush failed permanently; stopping consumer");
                self.fatal = true;
                false
            }
        }
    }

    /// Commit `last_consumed + 1` for each partition of the just-written block,
    /// then roll each partition's block forward.
    async fn commit(&mut self) {
        let mut tpl = TopicPartitionList::new();
        for ((topic, partition), progress) in &self.progress {
            let _ = tpl.add_partition_offset(
                topic,
                *partition,
                Offset::Offset(progress.last_consumed + 1),
            );
        }
        if let Err(e) = self.consumer.commit(&tpl, CommitMode::Async) {
            warn!(group = %self.config.group_id, error = %e, "committing kafka offsets failed");
        }
        // Next block starts after what we just committed.
        for progress in self.progress.values_mut() {
            progress.block_start = progress.last_consumed + 1;
        }
    }

    /// Seek every partition back to its block start so the failed block is
    /// re-consumed. The seeks block on the broker, so they run on a blocking
    /// thread. Returns `true` to keep running; a failed seek can't be recovered
    /// in-process (the position has advanced), so it stops the consumer to avoid
    /// dropping the block (a restart resumes from the last committed offset).
    async fn rewind(&mut self) -> bool {
        let consumer = Arc::clone(&self.consumer);
        let seeks: Vec<(String, i32, i64)> = self
            .progress
            .iter()
            .map(|((topic, partition), progress)| (topic.clone(), *partition, progress.block_start))
            .collect();
        let result = tokio::task::spawn_blocking(move || -> Result<(), KafkaError> {
            for (topic, partition, offset) in seeks {
                consumer.seek(
                    &topic,
                    partition,
                    Offset::Offset(offset),
                    Duration::from_secs(5),
                )?;
            }
            Ok(())
        })
        .await;

        match result {
            Ok(Ok(())) => {
                self.rows.clear();
                self.dead_rows.clear();
                self.broken_seen = 0;
                self.progress.clear();
                true
            }
            Ok(Err(e)) => {
                error!(group = %self.config.group_id, error = %e, "seeking back after a failed flush failed; stopping consumer (reprocess on restart)");
                self.fatal = true;
                false
            }
            Err(e) => {
                error!(group = %self.config.group_id, error = %e, "seek task panicked; stopping consumer");
                self.fatal = true;
                false
            }
        }
    }

    /// Note a consumed message against its partition's progress.
    fn advance(&mut self, record: &OwnedRecord) {
        let key = (record.topic.clone(), record.partition);
        self.progress
            .entry(key)
            .and_modify(|p| p.last_consumed = record.offset)
            .or_insert(PartitionProgress {
                block_start: record.offset,
                last_consumed: record.offset,
            });
    }

    /// Populate whichever virtual columns the table declares.
    fn inject_virtuals(&self, value: &mut Value, record: &OwnedRecord) {
        let Some(object) = value.as_object_mut() else {
            return;
        };
        let v = &self.virtual_fields;
        if v.topic {
            object.insert(VIRTUAL_TOPIC.into(), json!(record.topic));
        }
        if v.partition {
            object.insert(VIRTUAL_PARTITION.into(), json!(record.partition));
        }
        if v.offset {
            object.insert(VIRTUAL_OFFSET.into(), json!(record.offset));
        }
        if v.timestamp {
            // ClickHouse `_timestamp` is seconds; `_timestamp_ms` carries millis.
            let seconds = record.timestamp_millis.map(|ms| ms / 1000);
            object.insert(VIRTUAL_TIMESTAMP.into(), json!(seconds));
        }
        if v.timestamp_ms {
            object.insert(VIRTUAL_TIMESTAMP_MS.into(), json!(record.timestamp_millis));
        }
        if v.key {
            object.insert(VIRTUAL_KEY.into(), json!(record.key));
        }
    }
}

impl OwnedRecord {
    fn extract<M: Message>(message: &M) -> Self {
        Self {
            topic: message.topic().to_string(),
            partition: message.partition(),
            offset: message.offset(),
            key: message
                .key()
                .map(|k| String::from_utf8_lossy(k).into_owned()),
            payload: message.payload().unwrap_or(&[]).to_vec(),
            timestamp_millis: message.timestamp().to_millis(),
        }
    }
}

/// Drop every field of `value` not in `schema` (used to trim dead-letter rows).
fn retain_schema_fields(value: &mut Value, schema: &SchemaRef) {
    if let Some(object) = value.as_object_mut() {
        object.retain(|name, _| schema.field_with_name(name).is_ok());
    }
}
