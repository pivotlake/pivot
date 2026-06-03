//! Flatten OTLP protobuf payloads into Arrow [`RecordBatch`]es.
//!
//! Each signal (logs / traces / metrics) has a fixed, denormalised schema:
//! resource/scope context is copied onto every row and attribute maps are
//! serialised to JSON strings (query them later with `json_extract`-style
//! functions). Column names mirror HyperDX's ClickHouse `otel_*` tables so the
//! Parquet output ports over with minimal churn.
//!
//! Types are chosen for what pivot's Parquet reader supports and round-trips:
//! timestamps/durations are `Int64` unix-nanos (the reader maps every INT64
//! back to `Int64`, ignoring timestamp logical annotations), strings are
//! `Utf8` (read back as `Utf8View`), and numeric values are `Int32` / `Float64`.

use std::sync::{Arc, LazyLock};

use arrow_array::{Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use opentelemetry_proto::tonic::collector::logs::v1::ExportLogsServiceRequest;
use opentelemetry_proto::tonic::collector::metrics::v1::ExportMetricsServiceRequest;
use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use opentelemetry_proto::tonic::common::v1::{AnyValue, KeyValue, any_value::Value};
use opentelemetry_proto::tonic::metrics::v1::metric::Data;
use opentelemetry_proto::tonic::metrics::v1::number_data_point::Value as NumberValue;
use serde_json::{Map, Value as Json};

// ---------------------------------------------------------------------------
// Shared attribute / value helpers (shared with the standalone parquet_sink).
// ---------------------------------------------------------------------------

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn lookup_str(attrs: &[KeyValue], key: &str) -> Option<String> {
    attrs.iter().find(|kv| kv.key == key).and_then(|kv| {
        kv.value.as_ref().and_then(|v| match &v.value {
            Some(Value::StringValue(s)) => Some(s.clone()),
            _ => None,
        })
    })
}

fn attrs_to_json(attrs: &[KeyValue]) -> String {
    let mut map = Map::new();
    for kv in attrs {
        let v = kv.value.as_ref().map(any_value_to_json).unwrap_or(Json::Null);
        map.insert(kv.key.clone(), v);
    }
    Json::Object(map).to_string()
}

/// Render an `AnyValue` as a plain string (used for the log body).
fn any_value_to_string(v: &AnyValue) -> String {
    match &v.value {
        Some(Value::StringValue(s)) => s.clone(),
        Some(other) => any_value_to_json_inner(other).to_string(),
        None => String::new(),
    }
}

fn any_value_to_json(v: &AnyValue) -> Json {
    match &v.value {
        Some(inner) => any_value_to_json_inner(inner),
        None => Json::Null,
    }
}

fn any_value_to_json_inner(v: &Value) -> Json {
    match v {
        Value::StringValue(s) => Json::String(s.clone()),
        Value::BoolValue(b) => Json::Bool(*b),
        Value::IntValue(i) => Json::Number((*i).into()),
        Value::DoubleValue(d) => serde_json::Number::from_f64(*d)
            .map(Json::Number)
            .unwrap_or(Json::Null),
        Value::BytesValue(b) => Json::String(hex(b)),
        Value::ArrayValue(arr) => Json::Array(arr.values.iter().map(any_value_to_json).collect()),
        Value::KvlistValue(kv) => {
            let mut map = Map::new();
            for entry in &kv.values {
                let val = entry.value.as_ref().map(any_value_to_json).unwrap_or(Json::Null);
                map.insert(entry.key.clone(), val);
            }
            Json::Object(map)
        }
    }
}

fn str_col(values: impl Iterator<Item = String>) -> Arc<StringArray> {
    Arc::new(StringArray::from_iter_values(values))
}

// ---------------------------------------------------------------------------
// Logs
// ---------------------------------------------------------------------------

static LOGS_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("Timestamp", DataType::Int64, false),
        Field::new("ObservedTimestamp", DataType::Int64, false),
        Field::new("TraceId", DataType::Utf8, false),
        Field::new("SpanId", DataType::Utf8, false),
        Field::new("SeverityNumber", DataType::Int32, false),
        Field::new("SeverityText", DataType::Utf8, false),
        Field::new("ServiceName", DataType::Utf8, false),
        Field::new("Body", DataType::Utf8, false),
        Field::new("ResourceAttributes", DataType::Utf8, false),
        Field::new("LogAttributes", DataType::Utf8, false),
        Field::new("ScopeName", DataType::Utf8, false),
    ]))
});

/// Arrow schema of the `otel_logs` Parquet files this sink writes.
pub fn logs_schema() -> SchemaRef {
    LOGS_SCHEMA.clone()
}

struct LogRow {
    timestamp: i64,
    observed_timestamp: i64,
    trace_id: String,
    span_id: String,
    severity_number: i32,
    severity_text: String,
    service_name: String,
    body: String,
    resource_attributes: String,
    log_attributes: String,
    scope_name: String,
}

/// Flatten an OTLP logs export into a single `RecordBatch`, or `None` if the
/// request carried no records.
pub fn build_logs_batch(req: ExportLogsServiceRequest) -> Result<Option<RecordBatch>, ArrowError> {
    let mut rows = Vec::new();
    for rl in req.resource_logs {
        let res_attrs = rl
            .resource
            .as_ref()
            .map(|r| r.attributes.as_slice())
            .unwrap_or(&[]);
        let service_name = lookup_str(res_attrs, "service.name").unwrap_or_default();
        let res_attrs_json = attrs_to_json(res_attrs);
        for sl in rl.scope_logs {
            let scope_name = sl.scope.as_ref().map(|s| s.name.clone()).unwrap_or_default();
            for lr in sl.log_records {
                rows.push(LogRow {
                    timestamp: lr.time_unix_nano as i64,
                    observed_timestamp: lr.observed_time_unix_nano as i64,
                    trace_id: hex(&lr.trace_id),
                    span_id: hex(&lr.span_id),
                    severity_number: lr.severity_number,
                    severity_text: lr.severity_text.clone(),
                    service_name: service_name.clone(),
                    body: lr.body.as_ref().map(any_value_to_string).unwrap_or_default(),
                    resource_attributes: res_attrs_json.clone(),
                    log_attributes: attrs_to_json(&lr.attributes),
                    scope_name: scope_name.clone(),
                });
            }
        }
    }
    if rows.is_empty() {
        return Ok(None);
    }
    let batch = RecordBatch::try_new(
        logs_schema(),
        vec![
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.timestamp))),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|r| r.observed_timestamp),
            )),
            str_col(rows.iter().map(|r| r.trace_id.clone())),
            str_col(rows.iter().map(|r| r.span_id.clone())),
            Arc::new(Int32Array::from_iter_values(
                rows.iter().map(|r| r.severity_number),
            )),
            str_col(rows.iter().map(|r| r.severity_text.clone())),
            str_col(rows.iter().map(|r| r.service_name.clone())),
            str_col(rows.iter().map(|r| r.body.clone())),
            str_col(rows.iter().map(|r| r.resource_attributes.clone())),
            str_col(rows.iter().map(|r| r.log_attributes.clone())),
            str_col(rows.iter().map(|r| r.scope_name.clone())),
        ],
    )?;
    Ok(Some(batch))
}

// ---------------------------------------------------------------------------
// Traces
// ---------------------------------------------------------------------------

static TRACES_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("Timestamp", DataType::Int64, false),
        Field::new("Duration", DataType::Int64, false),
        Field::new("TraceId", DataType::Utf8, false),
        Field::new("SpanId", DataType::Utf8, false),
        Field::new("ParentSpanId", DataType::Utf8, false),
        Field::new("SpanName", DataType::Utf8, false),
        Field::new("SpanKind", DataType::Utf8, false),
        Field::new("ServiceName", DataType::Utf8, false),
        Field::new("StatusCode", DataType::Int32, false),
        Field::new("StatusMessage", DataType::Utf8, false),
        Field::new("ResourceAttributes", DataType::Utf8, false),
        Field::new("SpanAttributes", DataType::Utf8, false),
        Field::new("ScopeName", DataType::Utf8, false),
    ]))
});

/// Arrow schema of the `otel_traces` Parquet files this sink writes.
pub fn traces_schema() -> SchemaRef {
    TRACES_SCHEMA.clone()
}

fn span_kind_str(kind: i32) -> &'static str {
    match kind {
        1 => "Internal",
        2 => "Server",
        3 => "Client",
        4 => "Producer",
        5 => "Consumer",
        _ => "Unspecified",
    }
}

struct SpanRow {
    timestamp: i64,
    duration: i64,
    trace_id: String,
    span_id: String,
    parent_span_id: String,
    span_name: String,
    span_kind: String,
    service_name: String,
    status_code: i32,
    status_message: String,
    resource_attributes: String,
    span_attributes: String,
    scope_name: String,
}

/// Flatten an OTLP trace export into a single `RecordBatch`, or `None` if the
/// request carried no spans.
pub fn build_traces_batch(
    req: ExportTraceServiceRequest,
) -> Result<Option<RecordBatch>, ArrowError> {
    let mut rows = Vec::new();
    for rs in req.resource_spans {
        let res_attrs = rs
            .resource
            .as_ref()
            .map(|r| r.attributes.as_slice())
            .unwrap_or(&[]);
        let service_name = lookup_str(res_attrs, "service.name").unwrap_or_default();
        let res_attrs_json = attrs_to_json(res_attrs);
        for ss in rs.scope_spans {
            let scope_name = ss.scope.as_ref().map(|s| s.name.clone()).unwrap_or_default();
            for span in ss.spans {
                let (status_code, status_message) = span
                    .status
                    .as_ref()
                    .map(|s| (s.code, s.message.clone()))
                    .unwrap_or((0, String::new()));
                rows.push(SpanRow {
                    timestamp: span.start_time_unix_nano as i64,
                    duration: span
                        .end_time_unix_nano
                        .saturating_sub(span.start_time_unix_nano)
                        as i64,
                    trace_id: hex(&span.trace_id),
                    span_id: hex(&span.span_id),
                    parent_span_id: hex(&span.parent_span_id),
                    span_name: span.name.clone(),
                    span_kind: span_kind_str(span.kind).to_string(),
                    service_name: service_name.clone(),
                    status_code,
                    status_message,
                    resource_attributes: res_attrs_json.clone(),
                    span_attributes: attrs_to_json(&span.attributes),
                    scope_name: scope_name.clone(),
                });
            }
        }
    }
    if rows.is_empty() {
        return Ok(None);
    }
    let batch = RecordBatch::try_new(
        traces_schema(),
        vec![
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.timestamp))),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.duration))),
            str_col(rows.iter().map(|r| r.trace_id.clone())),
            str_col(rows.iter().map(|r| r.span_id.clone())),
            str_col(rows.iter().map(|r| r.parent_span_id.clone())),
            str_col(rows.iter().map(|r| r.span_name.clone())),
            str_col(rows.iter().map(|r| r.span_kind.clone())),
            str_col(rows.iter().map(|r| r.service_name.clone())),
            Arc::new(Int32Array::from_iter_values(
                rows.iter().map(|r| r.status_code),
            )),
            str_col(rows.iter().map(|r| r.status_message.clone())),
            str_col(rows.iter().map(|r| r.resource_attributes.clone())),
            str_col(rows.iter().map(|r| r.span_attributes.clone())),
            str_col(rows.iter().map(|r| r.scope_name.clone())),
        ],
    )?;
    Ok(Some(batch))
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

static METRICS_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("Timestamp", DataType::Int64, false),
        Field::new("MetricName", DataType::Utf8, false),
        Field::new("MetricType", DataType::Utf8, false),
        Field::new("MetricUnit", DataType::Utf8, false),
        Field::new("ServiceName", DataType::Utf8, false),
        Field::new("Value", DataType::Float64, false),
        Field::new("Count", DataType::Int64, false),
        Field::new("ResourceAttributes", DataType::Utf8, false),
        Field::new("MetricAttributes", DataType::Utf8, false),
        Field::new("ScopeName", DataType::Utf8, false),
    ]))
});

/// Arrow schema of the `otel_metrics` Parquet files this sink writes.
pub fn metrics_schema() -> SchemaRef {
    METRICS_SCHEMA.clone()
}

struct MetricRow {
    timestamp: i64,
    metric_name: String,
    metric_type: String,
    metric_unit: String,
    service_name: String,
    value: f64,
    count: i64,
    resource_attributes: String,
    metric_attributes: String,
    scope_name: String,
}

fn number_value(v: Option<&NumberValue>) -> f64 {
    match v {
        Some(NumberValue::AsDouble(d)) => *d,
        Some(NumberValue::AsInt(i)) => *i as f64,
        None => 0.0,
    }
}

/// Flatten an OTLP metrics export into a single `RecordBatch`, one row per data
/// point. Gauge and Sum data points carry their numeric value; Histogram data
/// points carry their `sum`/`count`. Exponential histograms and summaries are
/// skipped for now. Returns `None` if nothing flattenable was present.
pub fn build_metrics_batch(
    req: ExportMetricsServiceRequest,
) -> Result<Option<RecordBatch>, ArrowError> {
    let mut rows = Vec::new();
    for rm in req.resource_metrics {
        let res_attrs = rm
            .resource
            .as_ref()
            .map(|r| r.attributes.as_slice())
            .unwrap_or(&[]);
        let service_name = lookup_str(res_attrs, "service.name").unwrap_or_default();
        let res_attrs_json = attrs_to_json(res_attrs);
        for sm in rm.scope_metrics {
            let scope_name = sm.scope.as_ref().map(|s| s.name.clone()).unwrap_or_default();
            for metric in sm.metrics {
                let mut push = |timestamp: u64,
                                metric_type: &str,
                                value: f64,
                                count: i64,
                                attrs: &[KeyValue]| {
                    rows.push(MetricRow {
                        timestamp: timestamp as i64,
                        metric_name: metric.name.clone(),
                        metric_type: metric_type.to_string(),
                        metric_unit: metric.unit.clone(),
                        service_name: service_name.clone(),
                        value,
                        count,
                        resource_attributes: res_attrs_json.clone(),
                        metric_attributes: attrs_to_json(attrs),
                        scope_name: scope_name.clone(),
                    });
                };
                match metric.data {
                    Some(Data::Gauge(g)) => {
                        for dp in g.data_points {
                            push(dp.time_unix_nano, "gauge", number_value(dp.value.as_ref()), 0, &dp.attributes);
                        }
                    }
                    Some(Data::Sum(s)) => {
                        for dp in s.data_points {
                            push(dp.time_unix_nano, "sum", number_value(dp.value.as_ref()), 0, &dp.attributes);
                        }
                    }
                    Some(Data::Histogram(h)) => {
                        for dp in h.data_points {
                            push(
                                dp.time_unix_nano,
                                "histogram",
                                dp.sum.unwrap_or(0.0),
                                dp.count as i64,
                                &dp.attributes,
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    if rows.is_empty() {
        return Ok(None);
    }
    let batch = RecordBatch::try_new(
        metrics_schema(),
        vec![
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.timestamp))),
            str_col(rows.iter().map(|r| r.metric_name.clone())),
            str_col(rows.iter().map(|r| r.metric_type.clone())),
            str_col(rows.iter().map(|r| r.metric_unit.clone())),
            str_col(rows.iter().map(|r| r.service_name.clone())),
            Arc::new(Float64Array::from_iter_values(rows.iter().map(|r| r.value))),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.count))),
            str_col(rows.iter().map(|r| r.resource_attributes.clone())),
            str_col(rows.iter().map(|r| r.metric_attributes.clone())),
            str_col(rows.iter().map(|r| r.scope_name.clone())),
        ],
    )?;
    Ok(Some(batch))
}
