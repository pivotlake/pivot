# ingest

Receives telemetry from the outside world and sinks it into Parquet files,
running **inside the pivotdb server** rather than as a standalone process.

It is modelled on the standalone `parquet_sink` (in `pigivm/`), with one key
difference: the CPU-heavy part — encoding and Snappy-compressing the Parquet
data — runs on the **dispatch worker pool**, parallelized at the **page** level
(the mirror of dispatch's page-parallel reader), instead of on the async
runtime. The heavy lifting lands on the same thread-per-core pool that executes
queries, and the server's tokio threads stay free to keep accepting telemetry.

The Parquet encoding is hand-rolled (no upstream `parquet` crate): it reuses the
shared **`thriftparquet`** crate — the same Thrift compact-protocol codec and
metadata structures the reader is built on — plus the in-house snappy. A flush
forms one row group and cuts each column into **~1 MiB data pages** (the usual
Parquet page size). A page is just a list of zero-copy slices into the batch
arrays — it may span a batch boundary — so cutting pages copies nothing. The
pages fan out across the workers via a work-stealing source, so each is
PLAIN-encoded + compressed on whatever worker steals it (the single copy of the
data, done in parallel), then the encoded pages are stitched into one file
(offsets + footer) — a cheap serial step. One flush → one file; parallelism
scales with the flush's size (more data → more pages), just like the reader
parallelizes over the pages a file already contains.

> The footer is fully populated (column type/encodings/codec/sizes, row-group
> and file metadata, string columns marked UTF8), so the output is spec-compliant
> Parquet — readable by pivot's reader and by strict engines (arrow-rs, DuckDB).

## Shape

```
OTLP/gRPC  ──►  flatten to RecordBatch  ──►  ParquetSink (buffer)
                                                  │  rows ≥ flush_rows, or timer
                                                  ▼
                                       run_on_worker(encode + write)
                                                  │
                                                  ▼
                                       <dir>/<name>-<ts>-<seq>.parquet
```

- **`ParquetSink`** buffers Arrow `RecordBatch`es for one stream and flushes one
  Parquet file per drain. Files are written into a `.inflight` staging
  subdirectory and atomically renamed into place, so a concurrent reader never
  sees a half-written file.
- The **`otel`** module turns each OTLP signal (logs / traces / metrics) into
  batches and feeds a sink. Which signals run is **configuration, not code** —
  enable a signal by giving it an output directory.
- **`Ingestor`** is the lifecycle handle the server holds. `Ingestor::start`
  launches every configured source; `Ingestor::shutdown` stops the receivers and
  flushes whatever is still buffered **before** the dispatch workers are torn
  down (the final flush needs them alive).

## Running (via the server)

Each `--otel` flag starts one OTLP receiver; it is **repeatable**, so several
receivers can run at once (each needs a distinct `addr` and destinations). A
signal is enabled by giving it a destination; omit it and that signal's gRPC
service returns `Unimplemented`.

```bash
pivotdb-server \
  --otel 'addr=0.0.0.0:4317,logs=./otel/logs,traces=./otel/traces,metrics=./otel/metrics' \
  --otel 'addr=0.0.0.0:4318,logs=gs://my-bucket/otel/logs'
```

Spec keys: `addr`, `logs`, `traces`, `metrics`, `flush_rows`, `flush_secs`.

OTLP exporters gzip by default and batch aggressively; the receiver accepts gzip
and allows large messages (256 MiB) to match.

### Destinations: local vs object storage

A destination is either a **local directory** or an **object-store URL**:

| Destination            | Backend            | Queryable by pivot? |
|------------------------|--------------------|---------------------|
| `./otel/logs`, `file://…` | local filesystem | **yes** — `CREATE TABLE … WITH (path='…')` |
| `gs://bucket/prefix`   | Google Cloud Storage | no — read elsewhere (e.g. DuckDB) |
| `s3://bucket/prefix`   | Amazon S3          | no — read elsewhere |

Object storage is **write-only export**: pivot's reader only opens local
directories, so a `gs://`/`s3://` stream is meant to be read by something else
(DuckDB `read_parquet`, a warehouse, etc.). Credentials come from the
environment — `GOOGLE_APPLICATION_CREDENTIALS` / workload identity for GCS,
`AWS_*` / instance role for S3.

The work is split to match each part's nature: the **encode** (CPU) runs on a
dispatch worker; the **write/upload** (I/O) runs on the async runtime, so a
network round-trip to GCS/S3 never blocks a pinned worker.

## Querying the output

Each sink writes flat Parquet files into one directory, so a stream is queryable
with a `CREATE TABLE` pointing at it:

```sql
CREATE TABLE otel_logs (
    Timestamp          BIGINT,   -- unix nanos
    ObservedTimestamp  BIGINT,
    TraceId            VARCHAR,
    SpanId             VARCHAR,
    SeverityNumber     INTEGER,
    SeverityText       VARCHAR,
    ServiceName        VARCHAR,
    Body               VARCHAR,
    ResourceAttributes VARCHAR,  -- JSON
    LogAttributes      VARCHAR,  -- JSON
    ScopeName          VARCHAR
) WITH (path = './otel/logs');
```

> The catalog snapshots a directory's files at `CREATE TABLE` time, so files
> flushed *after* the table is created are not visible until it is (re)created.

### Schemas

Column types are chosen for what pivot's Parquet reader round-trips: timestamps
and durations are `BIGINT` (unix nanos), strings are `VARCHAR`, and attribute
maps are JSON strings. Pull fields out with a `json_extract`-style function.

| Signal  | File prefix    | Columns |
|---------|----------------|---------|
| logs    | `otel_logs`    | Timestamp, ObservedTimestamp, TraceId, SpanId, SeverityNumber, SeverityText, ServiceName, Body, ResourceAttributes, LogAttributes, ScopeName |
| traces  | `otel_traces`  | Timestamp, Duration, TraceId, SpanId, ParentSpanId, SpanName, SpanKind, ServiceName, StatusCode, StatusMessage, ResourceAttributes, SpanAttributes, ScopeName |
| metrics | `otel_metrics` | Timestamp, MetricName, MetricType, MetricUnit, ServiceName, Value, Count, ResourceAttributes, MetricAttributes, ScopeName |

Metrics are flattened one row per data point: gauge/sum data points carry their
numeric `Value`; histogram data points carry their `sum`/`count`. Exponential
histograms and summaries are skipped for now.

## Kafka

A second source consumes Kafka topics into a catalog table, modelled on
ClickHouse's classic **Kafka table engine**: an in-process consumer reads a
topic, accumulates a block, writes it to the destination table, and commits the
Kafka offset, reusing the same encode/partition/commit/compact pipeline as the
OTLP source. The only source-specific concern is the **offset durability
coupling** (`kafka/consumer.rs`).

```
StreamConsumer -> decode to JSON value -> buffer block -> encode_and_append
   (rdkafka)        (json|avro|protobuf)   (rows / time)   (shared write path)
                                                                | success
                                                                v
                                                       commit Kafka offsets
```

### Delivery semantics (at-least-once)

librdkafka auto-commit is **off**. A partition's offset is committed only after
its block is durably appended to the catalog. A flush distinguishes its failure:
a **transient** write error (catalog/store I/O) seeks the block start and
re-consumes, while a **permanent** one (the destination table was dropped, or a
failed seek) stops the consumer rather than spin on a block that can never land
(a restart then resumes from the last committed offset). No rows are dropped, but
**duplicates are possible** after a transient failure (there is no idempotent
dedup yet, see future work).

### Configuration

Each `--kafka` flag starts one consumer (repeatable). The value is a
comma-separated `key=value` spec; for SASL/SSL and other raw librdkafka settings
use a `--kafka-config` TOML file with a `[properties]` table.

```bash
pivotdb-server \
  --kafka 'brokers=localhost:9092,topics=events,group_id=pivot,table=events'
```

Spec keys: `brokers`, `topics` (`;`-separated), `group_id`, `table`, `format`
(`json`|`avro`|`protobuf`), `registry` (schema-registry URL, required for
avro/protobuf), `flush_rows` (default 1,000,000), `flush_secs` (default 5),
`num_consumers`, `auto_offset_reset` (`earliest`|`latest`), `skip_broken`, `dlq`
(dead-letter table). See `kafka/config.rs` for the TOML form.

### Schema (the table is the schema)

The destination table's declared columns **are** the decode schema (the
JSONEachRow model): JSON/Avro/Protobuf values match columns by name, a missing
field is null, and unknown fields are ignored. Column types are restricted to
what pivot's writer round-trips: `INTEGER`, `BIGINT`, `DOUBLE`, `VARCHAR`. A
value that decodes but whose type doesn't fit its column (e.g. a JSON `true` for
a `BIGINT`) is treated as a poison message (below), not a silent null.

```sql
CREATE TABLE events (
    user_id      BIGINT,
    action       VARCHAR,
    amount       DOUBLE,
    _partition   INTEGER,   -- optional Kafka virtual columns, populated when declared
    _offset      BIGINT,
    _timestamp   BIGINT,    -- message time in seconds (ClickHouse _timestamp)
    _timestamp_ms BIGINT,   -- message time in milliseconds
    _topic       VARCHAR,
    _key         VARCHAR
) WITH (path = './kafka/events');
```

Avro and Protobuf values are Confluent-framed (magic byte + schema id); the
writer schema is fetched from the schema registry and cached by id, then decoded
to a JSON object so all three formats share one Arrow-construction path. Protobuf
fields keep their `snake_case` names (so they match snake_case columns).

### Errors / dead-letter

A poison message (undecodable, or decoded but not fitting the schema) is routed
to the `dead_letter_table` if configured (`_topic`, `_partition`, `_offset`,
`_key`, `_raw`, `_error`). Otherwise it counts against `skip_broken_messages`,
and once that budget is exceeded the consumer **stops** (loudly) rather than
silently dropping records past it; the default `skip_broken_messages = 0` stops
on the first poison message.

## Notes / future work

- Kafka is **at-least-once**; an idempotent dedup token keyed off
  `(topic, partition, offset)` (à la ClickHouse `insert_deduplication_token`)
  would make it effectively-once. Avro union rendering and Protobuf schemas with
  imports are current decode limitations. The consumer could run as a separate
  process (like the compacter, and like ClickPipes) to scale ingestion apart from
  queries.
- `run_on_worker` currently always runs on worker 0, so flushes serialise there.
  Fine for typical ingest volumes; could round-robin if it becomes a bottleneck.
- Object-store output is write-only (pivot reads only local dirs). Reading
  `gs://`/`s3://` directly from the engine would close that gap.
- Azure (`az://`) is a one-line addition (`object_store` already supports it);
  only GCS and S3 are wired today.
- New sources (e.g. a Postgres logical-replication source) slot in as another
  `IngestConfig` variant feeding the same `ParquetSink`.
