//! The signal-agnostic core of OTLP → Arrow conversion.
//!
//! A signal's schema is not hardcoded: it is *derived* from a user-supplied
//! list of [`ColumnSpec`]s (see [`config`](super::config) /
//! [`defaults`](super::defaults)). Each column names where its value comes from
//! via a [`Source`] — a well-known record field, an attribute looked up by key,
//! or a whole attribute map serialised to JSON. [`CompiledMapping::compile`]
//! turns that list into an Arrow [`SchemaRef`] plus the resolved per-column
//! plan; the per-signal walks in [`mapping`](super::mapping) then feed rows
//! through a [`RowBuilder`].
//!
//! Types are chosen for what pivot's Parquet reader supports and round-trips:
//! timestamps/durations are `Int64` unix-nanos (the reader maps every INT64
//! back to `Int64`, ignoring timestamp logical annotations), strings are `Utf8`
//! (read back as `Utf8View`), and numeric values are `Int32` / `Float64`.
//! Columns are non-nullable; a missing attribute or absent optional field
//! resolves to `""` / `0`.

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use opentelemetry_proto::tonic::common::v1::{AnyValue, KeyValue, any_value::Value};
use serde_json::{Map, Value as Json};

use super::mapping::{Signal, field_type};

/// Which attribute map a column reads from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttrScope {
    /// `resource.attributes` (per resource, copied onto every row).
    Resource,
    /// The instrumentation scope's `attributes`.
    Scope,
    /// The record's own attributes (log record / span / metric data point).
    Record,
}

/// Where one output column's value comes from in an OTLP record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// A well-known field of the record, by canonical name (the set is
    /// signal-specific; see [`field_type`]). The special name `scope_name`
    /// resolves to the instrumentation scope name for every signal.
    Field(String),
    /// One attribute looked up by key; missing keys resolve to `""`. Always
    /// `Utf8`.
    Attr { scope: AttrScope, key: String },
    /// A scope's whole attribute map, serialised to a JSON object string.
    /// Always `Utf8`.
    AttrsJson(AttrScope),
}

/// One output column: its name and where its value comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnSpec {
    pub name: String,
    pub source: Source,
}

/// The canonical field name that resolves to the instrumentation scope name,
/// shared by every signal (so it lives here, not in a per-signal table).
pub const SCOPE_NAME_FIELD: &str = "scope_name";

/// A column spec named a field its signal doesn't have.
#[derive(Debug, thiserror::Error)]
#[error("signal {signal:?} has no well-known field `{field}`")]
pub struct UnknownField {
    pub signal: Signal,
    pub field: String,
}

/// A signal's column list compiled against its well-known-field table: the
/// derived Arrow schema and each column's `(source, type)`. Immutable and cheap
/// to share behind an `Arc` across gRPC calls.
#[derive(Debug)]
pub struct CompiledMapping {
    schema: SchemaRef,
    columns: Vec<(Source, DataType)>,
}

impl CompiledMapping {
    /// Derive the schema from `columns` for `signal`. Errors if a [`Source::Field`]
    /// names a field the signal doesn't have.
    pub fn compile(signal: Signal, columns: &[ColumnSpec]) -> Result<Self, UnknownField> {
        let mut fields = Vec::with_capacity(columns.len());
        let mut resolved = Vec::with_capacity(columns.len());
        for col in columns {
            let dt = column_type(signal, &col.source)?;
            fields.push(Field::new(&col.name, dt.clone(), false));
            resolved.push((col.source.clone(), dt));
        }
        Ok(Self {
            schema: Arc::new(Schema::new(fields)),
            columns: resolved,
        })
    }

    /// The Arrow schema of the Parquet files written for this signal.
    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Start accumulating a batch. The per-signal walk pushes one [`RowView`]
    /// per row, then [`RowBuilder::finish`] produces the `RecordBatch`.
    pub fn builder(&self) -> RowBuilder<'_> {
        RowBuilder {
            mapping: self,
            accums: self
                .columns
                .iter()
                .map(|(_, dt)| ColumnAccum::for_type(dt))
                .collect(),
            rows: 0,
        }
    }
}

/// The Arrow type a source resolves to. `Attr`/`AttrsJson` are always `Utf8`;
/// `scope_name` is `Utf8` for every signal; other fields come from the signal's
/// table.
fn column_type(signal: Signal, source: &Source) -> Result<DataType, UnknownField> {
    Ok(match source {
        Source::Field(name) if name == SCOPE_NAME_FIELD => DataType::Utf8,
        Source::Field(name) => field_type(signal, name).ok_or_else(|| UnknownField {
            signal,
            field: name.clone(),
        })?,
        Source::Attr { .. } | Source::AttrsJson(_) => DataType::Utf8,
    })
}

/// Accumulates rows for one batch against a [`CompiledMapping`].
pub struct RowBuilder<'m> {
    mapping: &'m CompiledMapping,
    accums: Vec<ColumnAccum>,
    rows: usize,
}

impl RowBuilder<'_> {
    /// Resolve every column for one record and append.
    pub fn push(&mut self, view: &RowView) {
        for (i, (source, dt)) in self.mapping.columns.iter().enumerate() {
            self.accums[i].push(resolve(source, view, dt));
        }
        self.rows += 1;
    }

    /// Finish the batch, or `None` if no rows were pushed.
    pub fn finish(self) -> Result<Option<RecordBatch>, ArrowError> {
        if self.rows == 0 {
            return Ok(None);
        }
        let columns = self.accums.into_iter().map(ColumnAccum::finish).collect();
        RecordBatch::try_new(self.mapping.schema(), columns).map(Some)
    }
}

/// Per-row context shared by all signals: the three attribute scopes, the
/// instrumentation scope name, and a signal-specific accessor for well-known
/// fields.
pub struct RowView<'a> {
    pub resource_attrs: &'a [KeyValue],
    pub scope_attrs: &'a [KeyValue],
    pub record_attrs: &'a [KeyValue],
    pub scope_name: &'a str,
    pub fields: &'a dyn Fields,
}

impl RowView<'_> {
    fn attrs(&self, scope: AttrScope) -> &[KeyValue] {
        match scope {
            AttrScope::Resource => self.resource_attrs,
            AttrScope::Scope => self.scope_attrs,
            AttrScope::Record => self.record_attrs,
        }
    }
}

/// A signal's well-known fields. One small `match` per signal, returning the
/// field as a typed [`Cell`] (its variant is what fixes the column's Arrow
/// type), or `None` for an unknown name. `scope_name` is handled generically by
/// [`resolve`] and is not part of this trait.
pub trait Fields {
    fn get(&self, name: &str) -> Option<Cell>;
}

/// A resolved cell value. The variant determines the column's Arrow type and
/// must match the `DataType` derived for that column at compile time.
pub enum Cell {
    I64(i64),
    I32(i32),
    F64(f64),
    Str(String),
}

fn resolve(source: &Source, view: &RowView, dt: &DataType) -> Cell {
    match source {
        Source::Field(name) if name == SCOPE_NAME_FIELD => Cell::Str(view.scope_name.to_string()),
        Source::Field(name) => view.fields.get(name).unwrap_or_else(|| default_cell(dt)),
        // The one string-valued attribute under `key`, or "" if absent / non-string.
        Source::Attr { scope, key } => {
            let value = view
                .attrs(*scope)
                .iter()
                .find(|kv| kv.key == *key)
                .and_then(|kv| match &kv.value.as_ref()?.value {
                    Some(Value::StringValue(s)) => Some(s.clone()),
                    _ => None,
                });
            Cell::Str(value.unwrap_or_default())
        }
        // The whole attribute map as a JSON object string.
        Source::AttrsJson(scope) => {
            let mut map = Map::new();
            for kv in view.attrs(*scope) {
                let v = kv
                    .value
                    .as_ref()
                    .map(any_value_to_json)
                    .unwrap_or(Json::Null);
                map.insert(kv.key.clone(), v);
            }
            Cell::Str(Json::Object(map).to_string())
        }
    }
}

/// The value for a column whose field is absent on this record — the
/// non-nullable default for its type (`0` / `""`).
fn default_cell(dt: &DataType) -> Cell {
    match dt {
        DataType::Int64 => Cell::I64(0),
        DataType::Int32 => Cell::I32(0),
        DataType::Float64 => Cell::F64(0.0),
        _ => Cell::Str(String::new()),
    }
}

/// A per-column typed value buffer. Mirrors the `DataType` set [`field_type`]
/// and the attribute sources can produce.
enum ColumnAccum {
    I64(Vec<i64>),
    I32(Vec<i32>),
    F64(Vec<f64>),
    Str(Vec<String>),
}

impl ColumnAccum {
    fn for_type(dt: &DataType) -> Self {
        match dt {
            DataType::Int64 => ColumnAccum::I64(Vec::new()),
            DataType::Int32 => ColumnAccum::I32(Vec::new()),
            DataType::Float64 => ColumnAccum::F64(Vec::new()),
            _ => ColumnAccum::Str(Vec::new()),
        }
    }

    fn push(&mut self, cell: Cell) {
        match (self, cell) {
            (ColumnAccum::I64(v), Cell::I64(x)) => v.push(x),
            (ColumnAccum::I32(v), Cell::I32(x)) => v.push(x),
            (ColumnAccum::F64(v), Cell::F64(x)) => v.push(x),
            (ColumnAccum::Str(v), Cell::Str(x)) => v.push(x),
            // The column's `Cell` variant is fixed at compile time (a column's
            // `DataType` drives both `for_type` here and `default_cell`), and a
            // signal's `Fields::get` agrees with its `field_type` table — a
            // consistency test guards that. So this is genuinely unreachable.
            _ => unreachable!("cell type disagrees with the column type fixed at compile time"),
        }
    }

    fn finish(self) -> ArrayRef {
        match self {
            ColumnAccum::I64(v) => Arc::new(Int64Array::from(v)),
            ColumnAccum::I32(v) => Arc::new(Int32Array::from(v)),
            ColumnAccum::F64(v) => Arc::new(Float64Array::from(v)),
            ColumnAccum::Str(v) => Arc::new(StringArray::from(v)),
        }
    }
}

/// Lowercase hex encoding (trace/span ids).
pub(super) fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Render an `AnyValue` as a plain string (used for the log body).
pub(super) fn any_value_to_string(v: &AnyValue) -> String {
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
                let val = entry
                    .value
                    .as_ref()
                    .map(any_value_to_json)
                    .unwrap_or(Json::Null);
                map.insert(entry.key.clone(), val);
            }
            Json::Object(map)
        }
    }
}
