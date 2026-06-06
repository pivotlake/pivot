//! Built-in default column mappings — the denormalised, HyperDX-flavoured
//! layout pivot shipped before schemas were configurable. A signal whose config
//! omits `columns` uses these, so the out-of-the-box behaviour is unchanged.
//! Column names mirror HyperDX's ClickHouse `otel_*` tables.
//!
//! These are plain data (`Vec<ColumnSpec>`), not a hardcoded schema: they go
//! through the same [`CompiledMapping::compile`](super::convert::CompiledMapping::compile)
//! path as any user-supplied list. The column **order** here is load-bearing
//! for the read-back tests, which assert by column index.

use super::convert::{AttrScope, ColumnSpec, SCOPE_NAME_FIELD, Source};
use super::mapping::Signal;

/// The default columns for a signal.
pub fn columns_for(signal: Signal) -> Vec<ColumnSpec> {
    match signal {
        Signal::Logs => logs(),
        Signal::Traces => traces(),
        Signal::Metrics => metrics(),
    }
}

fn field(name: &str, field: &str) -> ColumnSpec {
    ColumnSpec {
        name: name.to_string(),
        source: Source::Field(field.to_string()),
    }
}

fn attr(name: &str, scope: AttrScope, key: &str) -> ColumnSpec {
    ColumnSpec {
        name: name.to_string(),
        source: Source::Attr {
            scope,
            key: key.to_string(),
        },
    }
}

fn attrs_json(name: &str, scope: AttrScope) -> ColumnSpec {
    ColumnSpec {
        name: name.to_string(),
        source: Source::AttrsJson(scope),
    }
}

fn scope_name(name: &str) -> ColumnSpec {
    field(name, SCOPE_NAME_FIELD)
}

fn logs() -> Vec<ColumnSpec> {
    vec![
        field("Timestamp", "time_unix_nano"),
        field("ObservedTimestamp", "observed_time_unix_nano"),
        field("TraceId", "trace_id"),
        field("SpanId", "span_id"),
        field("SeverityNumber", "severity_number"),
        field("SeverityText", "severity_text"),
        attr("ServiceName", AttrScope::Resource, "service.name"),
        field("Body", "body"),
        attrs_json("ResourceAttributes", AttrScope::Resource),
        attrs_json("LogAttributes", AttrScope::Record),
        scope_name("ScopeName"),
    ]
}

fn traces() -> Vec<ColumnSpec> {
    vec![
        field("Timestamp", "start_time_unix_nano"),
        field("Duration", "duration"),
        field("TraceId", "trace_id"),
        field("SpanId", "span_id"),
        field("ParentSpanId", "parent_span_id"),
        field("SpanName", "name"),
        field("SpanKind", "kind"),
        attr("ServiceName", AttrScope::Resource, "service.name"),
        field("StatusCode", "status_code"),
        field("StatusMessage", "status_message"),
        attrs_json("ResourceAttributes", AttrScope::Resource),
        attrs_json("SpanAttributes", AttrScope::Record),
        scope_name("ScopeName"),
    ]
}

fn metrics() -> Vec<ColumnSpec> {
    vec![
        field("Timestamp", "time_unix_nano"),
        field("MetricName", "name"),
        field("MetricType", "type"),
        field("MetricUnit", "unit"),
        attr("ServiceName", AttrScope::Resource, "service.name"),
        field("Value", "value"),
        field("Count", "count"),
        attrs_json("ResourceAttributes", AttrScope::Resource),
        attrs_json("MetricAttributes", AttrScope::Record),
        scope_name("ScopeName"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::otel::convert::CompiledMapping;
    use arrow_schema::DataType;

    /// Pin the default schemas (names + types, in order) so a refactor can't
    /// silently change the out-of-the-box Parquet layout.
    #[test]
    fn default_schemas_match_expected() {
        let expect = |signal: Signal, want: &[(&str, DataType)]| {
            let mapping = CompiledMapping::compile(signal, &columns_for(signal)).unwrap();
            let schema = mapping.schema();
            let got: Vec<(String, DataType)> = schema
                .fields()
                .iter()
                .map(|f| (f.name().clone(), f.data_type().clone()))
                .collect();
            let want: Vec<(String, DataType)> = want
                .iter()
                .map(|(n, t)| (n.to_string(), t.clone()))
                .collect();
            assert_eq!(got, want, "{signal:?} default schema drifted");
        };

        use DataType::{Float64, Int32, Int64, Utf8};
        expect(
            Signal::Logs,
            &[
                ("Timestamp", Int64),
                ("ObservedTimestamp", Int64),
                ("TraceId", Utf8),
                ("SpanId", Utf8),
                ("SeverityNumber", Int32),
                ("SeverityText", Utf8),
                ("ServiceName", Utf8),
                ("Body", Utf8),
                ("ResourceAttributes", Utf8),
                ("LogAttributes", Utf8),
                ("ScopeName", Utf8),
            ],
        );
        expect(
            Signal::Traces,
            &[
                ("Timestamp", Int64),
                ("Duration", Int64),
                ("TraceId", Utf8),
                ("SpanId", Utf8),
                ("ParentSpanId", Utf8),
                ("SpanName", Utf8),
                ("SpanKind", Utf8),
                ("ServiceName", Utf8),
                ("StatusCode", Int32),
                ("StatusMessage", Utf8),
                ("ResourceAttributes", Utf8),
                ("SpanAttributes", Utf8),
                ("ScopeName", Utf8),
            ],
        );
        expect(
            Signal::Metrics,
            &[
                ("Timestamp", Int64),
                ("MetricName", Utf8),
                ("MetricType", Utf8),
                ("MetricUnit", Utf8),
                ("ServiceName", Utf8),
                ("Value", Float64),
                ("Count", Int64),
                ("ResourceAttributes", Utf8),
                ("MetricAttributes", Utf8),
                ("ScopeName", Utf8),
            ],
        );
    }
}
