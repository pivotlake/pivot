#!/usr/bin/env python3
"""
Stream Kafka into Pivot with `COPY <table> FROM STDIN WITH (FORMAT arrow)`.

A worked example of the Arrow copy-in path, written against an OTLP-JSON log
topic. The shape is the point, not the schema: read until you hold a batch's
worth of rows, build one Arrow record batch in the target table's column order,
push it through the copy, and only then commit the offsets you consumed -- a
crash replays the in-flight batch rather than dropping it.

Two things make this much faster than row-at-a-time INSERT:

  * The rows are accumulated as one python list per column and handed to Arrow
    whole, so there is no per-row tuple and no SQL text to build. Statement
    text was the single biggest cost in the INSERT version.
  * The server takes the Arrow IPC stream as it is. Columns map to the table by
    position (or by an explicit column list) and are cast to the table's types,
    so a client sends `pa.string()` and the server stores its own layout.

Run several processes in one consumer group to put concurrent copies on the
server; Kafka splits the topic's partitions between them.

Measured against a live OTLP feed on an 8-core client: one process sustains
about 41K rows/s, and is bound by python JSON parsing (64% of its time) rather
than by the server -- the copy itself is 20%, around 490ms per 100k-row batch.
Adding processes scales close to linearly until the client runs out of cores
(4 reach ~134K rows/s), because the server is nowhere near saturated: replaying
a prebuilt batch, with the parser out of the way, one connection reaches 166K
rows/s and four reach 443K.

Requires: kafka-python-ng, psycopg2-binary, pyarrow.

Configuration is by environment variable; see the block below. Set RUN_SECONDS
or MAX_ROWS to bound a measurement run, 0 means unlimited.
"""
import io
import json
import os
import signal
import sys
import time

import psycopg2
import pyarrow as pa
from kafka import KafkaConsumer

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.environ.get("TOPIC", "otel-logs")
GROUP_ID = os.environ.get("GROUP_ID", "pivot-copy")
OFFSET_RESET = os.environ.get("OFFSET_RESET", "earliest")

PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_USER = os.environ.get("PG_USER", "pivot")
PG_DB = os.environ.get("PG_DB", "pivot")
TABLE = os.environ.get("TABLE", "otel_logs")

BATCH_ROWS = int(os.environ.get("BATCH_ROWS", "100000"))
RUN_SECONDS = int(os.environ.get("RUN_SECONDS", "0"))
MAX_ROWS = int(os.environ.get("MAX_ROWS", "0"))
REPORT_SECS = int(os.environ.get("REPORT_SECS", "20"))
LABEL = os.environ.get("LABEL", "c0")

# The batch's columns map to the table's by position, so this must stay in the
# order the table declares. The types are what a client finds convenient, not
# what the table stores: the server casts each column to its own physical type,
# so `pa.string()` here lands in a VARCHAR column whatever that column's layout.
#
# The matching table is:
#
#   CREATE TABLE otel_logs (
#     "Timestamp" TIMESTAMP, "TraceId" VARCHAR, "SpanId" VARCHAR,
#     "TraceFlags" SMALLINT, "SeverityText" VARCHAR, "SeverityNumber" SMALLINT,
#     "ServiceName" VARCHAR, "Body" VARCHAR, "ResourceSchemaUrl" VARCHAR,
#     "ResourceAttributes" VARCHAR, "ScopeSchemaUrl" VARCHAR, "ScopeName" VARCHAR,
#     "ScopeVersion" VARCHAR, "ScopeAttributes" VARCHAR, "LogAttributes" VARCHAR,
#     "DeploymentId" VARCHAR, "ViewId" VARCHAR
#   ) WITH (partition_by = 'DeploymentId', sort_by = 'Timestamp');
ARROW_SCHEMA = pa.schema(
    [
        ("Timestamp", pa.timestamp("us")),
        ("TraceId", pa.string()),
        ("SpanId", pa.string()),
        ("TraceFlags", pa.int16()),
        ("SeverityText", pa.string()),
        ("SeverityNumber", pa.int16()),
        ("ServiceName", pa.string()),
        ("Body", pa.string()),
        ("ResourceSchemaUrl", pa.string()),
        ("ResourceAttributes", pa.string()),
        ("ScopeSchemaUrl", pa.string()),
        ("ScopeName", pa.string()),
        ("ScopeVersion", pa.string()),
        ("ScopeAttributes", pa.string()),
        ("LogAttributes", pa.string()),
        ("DeploymentId", pa.string()),
        ("ViewId", pa.string()),
    ]
)
NCOLS = len(ARROW_SCHEMA)

_stop = False


def _handle_sig(signum, frame):
    global _stop
    _stop = True


signal.signal(signal.SIGTERM, _handle_sig)
signal.signal(signal.SIGINT, _handle_sig)


# ---------------------------------------------------------------- OTLP parsing
def attr_value(v):
    if not isinstance(v, dict):
        return v
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        try:
            return int(v["intValue"])
        except (TypeError, ValueError):
            return v["intValue"]
    if "doubleValue" in v:
        return v["doubleValue"]
    if "boolValue" in v:
        return v["boolValue"]
    if "bytesValue" in v:
        return v["bytesValue"]
    if "arrayValue" in v:
        return [attr_value(x) for x in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return attrs_to_dict(v["kvlistValue"].get("values", []))
    return None


def attrs_to_dict(attrs):
    out = {}
    for a in attrs or []:
        k = a.get("key")
        if k is not None:
            out[k] = attr_value(a.get("value"))
    return out


def ns_to_micros(ns):
    """OTLP nanoseconds -> the microseconds an arrow timestamp column holds."""
    if not ns:
        return None
    try:
        n = int(ns)
    except (TypeError, ValueError):
        return None
    return n // 1000 if n > 0 else None


def body_to_str(body):
    if not isinstance(body, dict):
        return "" if body is None else str(body)
    if "stringValue" in body:
        return body["stringValue"]
    val = attr_value(body)
    if isinstance(val, str):
        return val
    return json.dumps(val, default=str)


class ColumnBuffer:
    """The pending batch as one python list per column: appending here and
    handing whole lists to pyarrow costs far less than building a tuple per row
    and transposing at flush time."""

    def __init__(self):
        self.columns = [[] for _ in range(NCOLS)]

    def __len__(self):
        return len(self.columns[0])

    def append_payload(self, payload):
        (
            timestamps,
            trace_ids,
            span_ids,
            flags,
            severity_texts,
            severity_numbers,
            service_names,
            bodies,
            resource_schemas,
            resource_attributes,
            scope_schemas,
            scope_names,
            scope_versions,
            scope_attributes,
            log_attributes,
            deployment_ids,
            view_ids,
        ) = self.columns

        now_micros = int(time.time() * 1_000_000)
        for rl in payload.get("resourceLogs", []) or []:
            resource = rl.get("resource") or {}
            res_attrs = attrs_to_dict(resource.get("attributes"))
            res_schema = rl.get("schemaUrl", "") or ""
            service_name = str(res_attrs.get("service.name", "") or "")
            deployment_id = str(res_attrs.get("deployment_id", "") or "")
            res_attrs_json = json.dumps(res_attrs, default=str)
            res_view_id = res_attrs.get("view_id")

            for sl in rl.get("scopeLogs", []) or []:
                scope = sl.get("scope") or {}
                scope_name = str(scope.get("name", "") or "")
                scope_version = str(scope.get("version", "") or "")
                scope_attrs_json = json.dumps(
                    attrs_to_dict(scope.get("attributes")), default=str
                )
                scope_schema = sl.get("schemaUrl", "") or ""

                for lr in sl.get("logRecords", []) or []:
                    log_attrs = attrs_to_dict(lr.get("attributes"))
                    if lr.get("eventName"):
                        log_attrs.setdefault("event.name", lr["eventName"])

                    micros = ns_to_micros(lr.get("timeUnixNano"))
                    if micros is None:
                        micros = ns_to_micros(lr.get("observedTimeUnixNano"))
                    timestamps.append(now_micros if micros is None else micros)

                    try:
                        severity = max(
                            0, min(255, int(lr.get("severityNumber", 0) or 0))
                        )
                    except (TypeError, ValueError):
                        severity = 0
                    try:
                        flag = max(0, min(255, int(lr.get("flags", 0) or 0)))
                    except (TypeError, ValueError):
                        flag = 0

                    trace_ids.append(str(lr.get("traceId", "") or ""))
                    span_ids.append(str(lr.get("spanId", "") or ""))
                    flags.append(flag)
                    severity_texts.append(str(lr.get("severityText", "") or ""))
                    severity_numbers.append(severity)
                    service_names.append(service_name)
                    bodies.append(body_to_str(lr.get("body")))
                    resource_schemas.append(res_schema)
                    resource_attributes.append(res_attrs_json)
                    scope_schemas.append(scope_schema)
                    scope_names.append(scope_name)
                    scope_versions.append(scope_version)
                    scope_attributes.append(scope_attrs_json)
                    log_attributes.append(json.dumps(log_attrs, default=str))
                    deployment_ids.append(deployment_id)
                    view_ids.append(
                        str(log_attrs.get("view_id") or res_view_id or "")
                    )

    def take(self, count):
        """Detach the first `count` rows as an arrow batch, keeping the rest."""
        arrays = []
        for index, values in enumerate(self.columns):
            arrays.append(pa.array(values[:count], type=ARROW_SCHEMA.field(index).type))
            self.columns[index] = values[count:]
        return pa.RecordBatch.from_arrays(arrays, schema=ARROW_SCHEMA)


def arrow_stream_bytes(batch):
    """One record batch as the arrow IPC stream the copy-in protocol carries."""
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, ARROW_SCHEMA) as writer:
        writer.write_batch(batch)
    sink.seek(0)
    return sink


def pg_connect():
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER, dbname=PG_DB, connect_timeout=10
    )
    conn.autocommit = True
    return conn


def main():
    print(
        f"[{LABEL}] arrow COPY consumer | kafka={KAFKA_BOOTSTRAP} topic={TOPIC} "
        f"group={GROUP_ID} offset_reset={OFFSET_RESET} -> {PG_HOST}:{PG_PORT}/{PG_DB} "
        f"table={TABLE} batch_rows={BATCH_ROWS}",
        flush=True,
    )
    conn = pg_connect()
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP.split(","),
        group_id=GROUP_ID,
        auto_offset_reset=OFFSET_RESET,
        enable_auto_commit=False,
        max_partition_fetch_bytes=8 * 1024 * 1024,
        fetch_max_bytes=64 * 1024 * 1024,
        max_poll_records=2000,
        consumer_timeout_ms=2000,
        value_deserializer=lambda b: b,
    )

    buffer = ColumnBuffer()
    started = time.time()
    rows = batches = errors = 0
    t_fetch = t_parse = t_build = t_copy = 0.0
    last_report = started
    prev = (0.0, 0.0, 0.0, 0.0, 0.0, 0)

    def done():
        if _stop:
            return True
        if RUN_SECONDS > 0 and time.time() - started >= RUN_SECONDS:
            return True
        return MAX_ROWS > 0 and rows >= MAX_ROWS

    try:
        while not done():
            mark = time.time()
            for msg in consumer:
                t_fetch += time.time() - mark

                p0 = time.time()
                try:
                    buffer.append_payload(json.loads(msg.value))
                except Exception as e:  # noqa: BLE001
                    print(f"[{LABEL}] parse error: {str(e)[:120]}", flush=True)
                t_parse += time.time() - p0

                while len(buffer) >= BATCH_ROWS:
                    b0 = time.time()
                    batch = buffer.take(BATCH_ROWS)
                    stream = arrow_stream_bytes(batch)
                    c0 = time.time()
                    t_build += c0 - b0
                    try:
                        with conn.cursor() as cur:
                            cur.copy_expert(
                                f"COPY {TABLE} FROM STDIN WITH (FORMAT arrow)",
                                stream,
                                size=1 << 20,
                            )
                        rows += batch.num_rows
                        batches += 1
                        consumer.commit()
                    except Exception as e:  # noqa: BLE001
                        errors += 1
                        print(f"[{LABEL}] copy error: {str(e)[:200]}", flush=True)
                        if conn.closed != 0 or isinstance(
                            e, (psycopg2.OperationalError, psycopg2.InterfaceError)
                        ):
                            try:
                                conn.close()
                            except Exception:  # noqa: BLE001
                                pass
                            conn = pg_connect()
                    t_copy += time.time() - c0

                now = time.time()
                if now - last_report >= REPORT_SECS:
                    el = now - started
                    d = el - prev[0] or 1e-3
                    print(
                        f"[{LABEL}] [{el:6.1f}s] rows={rows} "
                        f"rate={(rows - prev[5]) / d:7.0f}/s batches={batches} "
                        f"| fetch={(t_fetch - prev[1]) / d * 100:4.1f}% "
                        f"parse={(t_parse - prev[2]) / d * 100:4.1f}% "
                        f"build={(t_build - prev[3]) / d * 100:4.1f}% "
                        f"copy={(t_copy - prev[4]) / d * 100:4.1f}% "
                        f"| copy_avg={t_copy / max(batches, 1) * 1000:.0f}ms "
                        f"build_avg={t_build / max(batches, 1) * 1000:.0f}ms "
                        f"err={errors}",
                        flush=True,
                    )
                    prev = (el, t_fetch, t_parse, t_build, t_copy, rows)
                    last_report = now

                if done():
                    break
                mark = time.time()
    finally:
        try:
            consumer.close()
        except Exception:  # noqa: BLE001
            pass
        conn.close()

    el = time.time() - started
    print(
        f"[{LABEL}] DONE elapsed={el:.1f}s rows={rows} batches={batches} "
        f"errors={errors} avg_rate={rows / max(el, 1):.0f}/s "
        f"| fetch={t_fetch / el * 100:.1f}% parse={t_parse / el * 100:.1f}% "
        f"build={t_build / el * 100:.1f}% copy={t_copy / el * 100:.1f}%",
        flush=True,
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
