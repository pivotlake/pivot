//! The per-signal half of OTLP → Arrow conversion: the (genuinely
//! signal-specific) `resource → scope → record` walks, each signal's
//! well-known-field set, and the two small tables those fields live in.
//!
//! Everything generic — schema derivation, attribute / JSON sources, row
//! accumulation — lives in [`convert`](super::convert). A signal contributes
//! exactly two things here, kept side by side so they can't drift:
//!
//! 1. [`field_type`]: maps a field name to its Arrow [`DataType`] (used to
//!    derive the schema and validate the config).
//! 2. A [`Fields`] impl: extracts that field's value as a [`Cell`] from a
//!    record (used at conversion time).
//!
//! A consistency test asserts every name in (1) resolves to the matching
//! variant in (2).

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, DataType};
use opentelemetry_proto::tonic::collector::logs::v1::ExportLogsServiceRequest;
use opentelemetry_proto::tonic::collector::metrics::v1::ExportMetricsServiceRequest;
use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use opentelemetry_proto::tonic::common::v1::{InstrumentationScope, KeyValue};
use opentelemetry_proto::tonic::logs::v1::LogRecord;
use opentelemetry_proto::tonic::metrics::v1::Metric;
use opentelemetry_proto::tonic::metrics::v1::metric::Data;
use opentelemetry_proto::tonic::metrics::v1::number_data_point::Value as NumberValue;
use opentelemetry_proto::tonic::resource::v1::Resource;
use opentelemetry_proto::tonic::trace::v1::Span;

use super::convert::{Cell, CompiledMapping, Fields, RowView, any_value_to_string, hex};
use crate::parquet_writing::ToRecordBatch;

/// The three OTLP signals. Selects which field table / walk applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    Logs,
    Traces,
    Metrics,
}

/// The Arrow type of a signal's well-known field, or `None` if the signal has
/// no such field. The `scope_name` field is handled generically (always `Utf8`)
/// and is intentionally absent here. Keep in sync with the `Fields` impls below.
pub fn field_type(signal: Signal, name: &str) -> Option<DataType> {
    use DataType::{Float64, Int32, Int64, Utf8};
    let ty = match signal {
        Signal::Logs => match name {
            "time_unix_nano" | "observed_time_unix_nano" => Int64,
            "severity_number" => Int32,
            "trace_id" | "span_id" | "severity_text" | "body" => Utf8,
            _ => return None,
        },
        Signal::Traces => match name {
            "start_time_unix_nano" | "duration" => Int64,
            "status_code" => Int32,
            "trace_id" | "span_id" | "parent_span_id" | "name" | "kind" | "status_message" => Utf8,
            _ => return None,
        },
        Signal::Metrics => match name {
            "time_unix_nano" | "count" => Int64,
            "value" => Float64,
            "name" | "type" | "unit" => Utf8,
            _ => return None,
        },
    };
    Some(ty)
}

fn resource_attrs(resource: Option<&Resource>) -> &[KeyValue] {
    resource.map(|r| r.attributes.as_slice()).unwrap_or(&[])
}

fn scope_parts(scope: Option<&InstrumentationScope>) -> (&str, &[KeyValue]) {
    match scope {
        Some(s) => (s.name.as_str(), s.attributes.as_slice()),
        None => ("", &[]),
    }
}

struct LogFields<'a> {
    lr: &'a LogRecord,
}

impl Fields for LogFields<'_> {
    fn get(&self, name: &str) -> Option<Cell> {
        Some(match name {
            "time_unix_nano" => Cell::I64(self.lr.time_unix_nano as i64),
            "observed_time_unix_nano" => Cell::I64(self.lr.observed_time_unix_nano as i64),
            "trace_id" => Cell::Str(hex(&self.lr.trace_id)),
            "span_id" => Cell::Str(hex(&self.lr.span_id)),
            "severity_number" => Cell::I32(self.lr.severity_number),
            "severity_text" => Cell::Str(self.lr.severity_text.clone()),
            "body" => Cell::Str(
                self.lr
                    .body
                    .as_ref()
                    .map(any_value_to_string)
                    .unwrap_or_default(),
            ),
            _ => return None,
        })
    }
}

/// Flatten an OTLP logs export into one `RecordBatch`, or `None` if it carried
/// no records.
pub(super) fn build_logs_batch(
    req: ExportLogsServiceRequest,
    mapping: &CompiledMapping,
) -> Result<Option<RecordBatch>, ArrowError> {
    let mut rows = mapping.builder();
    for rl in req.resource_logs {
        let resource_attrs = resource_attrs(rl.resource.as_ref());
        for sl in rl.scope_logs {
            let (scope_name, scope_attrs) = scope_parts(sl.scope.as_ref());
            for lr in &sl.log_records {
                rows.push(&RowView {
                    resource_attrs,
                    scope_attrs,
                    record_attrs: &lr.attributes,
                    scope_name,
                    fields: &LogFields { lr },
                });
            }
        }
    }
    rows.finish()
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

struct SpanFields<'a> {
    span: &'a Span,
}

impl Fields for SpanFields<'_> {
    fn get(&self, name: &str) -> Option<Cell> {
        let s = self.span;
        Some(match name {
            "start_time_unix_nano" => Cell::I64(s.start_time_unix_nano as i64),
            "duration" => {
                Cell::I64(s.end_time_unix_nano.saturating_sub(s.start_time_unix_nano) as i64)
            }
            "trace_id" => Cell::Str(hex(&s.trace_id)),
            "span_id" => Cell::Str(hex(&s.span_id)),
            "parent_span_id" => Cell::Str(hex(&s.parent_span_id)),
            "name" => Cell::Str(s.name.clone()),
            "kind" => Cell::Str(span_kind_str(s.kind).to_string()),
            "status_code" => Cell::I32(s.status.as_ref().map(|st| st.code).unwrap_or(0)),
            "status_message" => Cell::Str(
                s.status
                    .as_ref()
                    .map(|st| st.message.clone())
                    .unwrap_or_default(),
            ),
            _ => return None,
        })
    }
}

/// Flatten an OTLP trace export into one `RecordBatch`, or `None` if it carried
/// no spans.
pub(super) fn build_traces_batch(
    req: ExportTraceServiceRequest,
    mapping: &CompiledMapping,
) -> Result<Option<RecordBatch>, ArrowError> {
    let mut rows = mapping.builder();
    for rs in req.resource_spans {
        let resource_attrs = resource_attrs(rs.resource.as_ref());
        for ss in rs.scope_spans {
            let (scope_name, scope_attrs) = scope_parts(ss.scope.as_ref());
            for span in &ss.spans {
                rows.push(&RowView {
                    resource_attrs,
                    scope_attrs,
                    record_attrs: &span.attributes,
                    scope_name,
                    fields: &SpanFields { span },
                });
            }
        }
    }
    rows.finish()
}

fn number_value(v: Option<&NumberValue>) -> f64 {
    match v {
        Some(NumberValue::AsDouble(d)) => *d,
        Some(NumberValue::AsInt(i)) => *i as f64,
        None => 0.0,
    }
}

/// One metric data point, flattened. `metric_type` / `value` / `count` are
/// computed per data point (they aren't fields of the proto `Metric`), so the
/// walk fills them in; `name` / `unit` come straight off the metric.
struct MetricFields<'a> {
    metric: &'a Metric,
    metric_type: &'static str,
    time_unix_nano: u64,
    value: f64,
    count: i64,
}

impl Fields for MetricFields<'_> {
    fn get(&self, name: &str) -> Option<Cell> {
        Some(match name {
            "time_unix_nano" => Cell::I64(self.time_unix_nano as i64),
            "name" => Cell::Str(self.metric.name.clone()),
            "type" => Cell::Str(self.metric_type.to_string()),
            "unit" => Cell::Str(self.metric.unit.clone()),
            "value" => Cell::F64(self.value),
            "count" => Cell::I64(self.count),
            _ => return None,
        })
    }
}

/// Flatten an OTLP metrics export into one `RecordBatch`, one row per data
/// point. Gauge / Sum data points carry their numeric value; Histogram data
/// points carry their `sum` / `count`. Exponential histograms and summaries are
/// skipped. `None` if nothing flattenable was present.
pub(super) fn build_metrics_batch(
    req: ExportMetricsServiceRequest,
    mapping: &CompiledMapping,
) -> Result<Option<RecordBatch>, ArrowError> {
    let mut rows = mapping.builder();
    for rm in req.resource_metrics {
        let resource_attrs = resource_attrs(rm.resource.as_ref());
        for sm in rm.scope_metrics {
            let (scope_name, scope_attrs) = scope_parts(sm.scope.as_ref());
            for metric in &sm.metrics {
                let mut emit = |time, metric_type, value, count, record_attrs: &[KeyValue]| {
                    rows.push(&RowView {
                        resource_attrs,
                        scope_attrs,
                        record_attrs,
                        scope_name,
                        fields: &MetricFields {
                            metric,
                            metric_type,
                            time_unix_nano: time,
                            value,
                            count,
                        },
                    });
                };
                match &metric.data {
                    Some(Data::Gauge(g)) => {
                        for dp in &g.data_points {
                            emit(
                                dp.time_unix_nano,
                                "gauge",
                                number_value(dp.value.as_ref()),
                                0,
                                &dp.attributes,
                            );
                        }
                    }
                    Some(Data::Sum(s)) => {
                        for dp in &s.data_points {
                            emit(
                                dp.time_unix_nano,
                                "sum",
                                number_value(dp.value.as_ref()),
                                0,
                                &dp.attributes,
                            );
                        }
                    }
                    Some(Data::Histogram(h)) => {
                        for dp in &h.data_points {
                            emit(
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
    rows.finish()
}

pub(crate) struct LogsItem {
    pub(crate) req: ExportLogsServiceRequest,
    pub(crate) mapping: Arc<CompiledMapping>,
}

impl ToRecordBatch for LogsItem {
    fn num_rows(&self) -> usize {
        self.req
            .resource_logs
            .iter()
            .flat_map(|rl| &rl.scope_logs)
            .map(|sl| sl.log_records.len())
            .sum()
    }
    fn to_record_batch(self) -> Result<Option<RecordBatch>, ArrowError> {
        build_logs_batch(self.req, &self.mapping)
    }
}

pub(crate) struct TracesItem {
    pub(crate) req: ExportTraceServiceRequest,
    pub(crate) mapping: Arc<CompiledMapping>,
}

impl ToRecordBatch for TracesItem {
    fn num_rows(&self) -> usize {
        self.req
            .resource_spans
            .iter()
            .flat_map(|rs| &rs.scope_spans)
            .map(|ss| ss.spans.len())
            .sum()
    }
    fn to_record_batch(self) -> Result<Option<RecordBatch>, ArrowError> {
        build_traces_batch(self.req, &self.mapping)
    }
}

pub(crate) struct MetricsItem {
    pub(crate) req: ExportMetricsServiceRequest,
    pub(crate) mapping: Arc<CompiledMapping>,
}

impl ToRecordBatch for MetricsItem {
    fn num_rows(&self) -> usize {
        self.req
            .resource_metrics
            .iter()
            .flat_map(|rm| &rm.scope_metrics)
            .flat_map(|sm| &sm.metrics)
            .map(metric_point_count)
            .sum()
    }
    fn to_record_batch(self) -> Result<Option<RecordBatch>, ArrowError> {
        build_metrics_batch(self.req, &self.mapping)
    }
}

/// Data points a metric contributes — matching the kinds `build_metrics_batch`
/// flattens (others contribute none).
fn metric_point_count(metric: &Metric) -> usize {
    match &metric.data {
        Some(Data::Gauge(g)) => g.data_points.len(),
        Some(Data::Sum(s)) => s.data_points.len(),
        Some(Data::Histogram(h)) => h.data_points.len(),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::otel::defaults;

    /// Every field named in a signal's `field_type` table must resolve in its
    /// `Fields` impl to a `Cell` of the matching variant — the one place the two
    /// per-signal tables could drift. We probe each field name against an empty
    /// record (values don't matter, only the variant) and compare.
    fn variant_matches(cell: &Cell, dt: &DataType) -> bool {
        matches!(
            (cell, dt),
            (Cell::I64(_), DataType::Int64)
                | (Cell::I32(_), DataType::Int32)
                | (Cell::F64(_), DataType::Float64)
                | (Cell::Str(_), DataType::Utf8)
        )
    }

    #[test]
    fn field_tables_agree_with_extractors() {
        let log = LogRecord::default();
        let log_fields = LogFields { lr: &log };
        let span = Span::default();
        let span_fields = SpanFields { span: &span };
        let metric = Metric::default();
        let metric_fields = MetricFields {
            metric: &metric,
            metric_type: "gauge",
            time_unix_nano: 0,
            value: 0.0,
            count: 0,
        };

        let cases: [(Signal, &dyn Fields); 3] = [
            (Signal::Logs, &log_fields),
            (Signal::Traces, &span_fields),
            (Signal::Metrics, &metric_fields),
        ];
        for (signal, fields) in cases {
            // Drive the names off the default column lists, which between them
            // exercise every field the signal exposes.
            for col in defaults::columns_for(signal) {
                if let super::super::convert::Source::Field(name) = &col.source {
                    if name == super::super::convert::SCOPE_NAME_FIELD {
                        continue;
                    }
                    let dt = field_type(signal, name).unwrap_or_else(|| {
                        panic!("{signal:?} field `{name}` missing from field_type")
                    });
                    let cell = fields.get(name).unwrap_or_else(|| {
                        panic!("{signal:?} field `{name}` missing from Fields::get")
                    });
                    assert!(
                        variant_matches(&cell, &dt),
                        "{signal:?} field `{name}`: Cell variant disagrees with field_type {dt:?}"
                    );
                }
            }
        }
    }
}
