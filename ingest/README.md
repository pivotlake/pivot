# ingest

Receives telemetry from the outside world and sinks it into Parquet files,
running **inside the pivotdb server** rather than as a standalone process.

It is modelled on the standalone `parquet_sink` (in `pigivm/`), with one key
difference: the CPU-heavy part — encoding and Snappy-compressing each Parquet
file — is shipped to a **dispatch worker** via
`DataFlowDispatcher::run_on_worker` instead of running on the async runtime. The
heavy lifting lands on the same thread-per-core pool that executes queries, and
the server's tokio threads stay free to keep accepting telemetry.

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

## Notes / future work

- `run_on_worker` currently always runs on worker 0, so flushes serialise there.
  Fine for typical ingest volumes; could round-robin if it becomes a bottleneck.
- Object-store output is write-only (pivot reads only local dirs). Reading
  `gs://`/`s3://` directly from the engine would close that gap.
- Azure (`az://`) is a one-line addition (`object_store` already supports it);
  only GCS and S3 are wired today.
- New sources (e.g. a Postgres logical-replication source) slot in as another
  `IngestConfig` variant feeding the same `ParquetSink`.
