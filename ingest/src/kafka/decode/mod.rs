//! Turning a Kafka message payload into a table row.
//!
//! Every supported format converges on a single intermediate, a
//! [`serde_json::Value`] object (one row, field-by-name), so there is **one**
//! Arrow-construction path: [`KafkaBatch`] feeds the buffered rows through
//! `arrow-json`'s [`Decoder`](arrow_json::reader::Decoder) against the
//! destination table's schema (missing fields → null, extra fields ignored),
//! exactly the JSONEachRow model. The format-specific work is only "bytes →
//! `Value`":
//!
//! - [`json`] parses the payload directly.
//! - [`avro`] / [`protobuf`] strip the Confluent wire-format frame
//!   ([`confluent`]), fetch + cache the writer schema by id from a
//!   [`registry`](registry::SchemaRegistry), and decode the datum to a `Value`.
//!
//! The schema the rows are decoded into is derived from the catalog table's
//! declared columns ([`table_schema`]); only the column types pivot's Parquet
//! writer round-trips are allowed.

mod avro;
mod confluent;
mod json;
mod protobuf;
mod registry;

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_json::ReaderBuilder;
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use planner::catalog::Column;
use planner::types::Type;
use serde_json::Value;

use crate::parquet_writing::ToRecordBatch;

pub(crate) use registry::SchemaRegistry;

/// The wire format of a topic's message values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Format {
    /// Plain JSON object per message (no registry needed).
    #[default]
    Json,
    /// Confluent-framed Avro (magic byte + schema id), writer schema from the
    /// registry.
    Avro,
    /// Confluent-framed Protobuf (magic byte + schema id + message-index),
    /// `.proto` schema from the registry.
    Protobuf,
}

impl Format {
    /// Whether this format needs a schema registry to decode.
    pub fn needs_registry(self) -> bool {
        matches!(self, Format::Avro | Format::Protobuf)
    }
}

impl std::str::FromStr for Format {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "json" => Ok(Format::Json),
            "avro" => Ok(Format::Avro),
            "protobuf" | "proto" => Ok(Format::Protobuf),
            other => Err(format!(
                "unknown format `{other}` (expected json/avro/protobuf)"
            )),
        }
    }
}

/// A decode failure for one message. The consumer routes it to the dead-letter
/// table or counts it against `skip_broken_messages`; it never panics the loop.
#[derive(Debug, thiserror::Error)]
pub(crate) enum DecodeError {
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("not a JSON object (a Kafka row must decode to an object)")]
    NotAnObject,
    #[error("confluent framing: {0}")]
    Framing(String),
    #[error("schema registry: {0}")]
    Registry(String),
    #[error("avro decode: {0}")]
    Avro(String),
    #[error("protobuf decode: {0}")]
    Protobuf(String),
    #[error("does not fit the table schema: {0}")]
    SchemaMismatch(String),
}

/// Deriving a table's Arrow schema for decoding failed.
#[derive(Debug, thiserror::Error)]
pub(crate) enum SchemaError {
    #[error(
        "column `{column}` has type {col_type} which Kafka ingest can't write; use INTEGER, BIGINT, DOUBLE, or VARCHAR"
    )]
    UnsupportedColumnType { column: String, col_type: Type },
}

/// The Arrow type a Kafka column decodes into, restricted to what pivot's
/// Parquet writer round-trips (`Int32`/`Int64`/`Float64`/`Utf8`). Decoded
/// fields are nullable so a message missing one yields null rather than failing.
fn arrow_type(col: &Column) -> Result<DataType, SchemaError> {
    Ok(match &col.col_type {
        Type::Int32 => DataType::Int32,
        Type::Int64 => DataType::Int64,
        Type::Float64 => DataType::Float64,
        Type::Utf8 => DataType::Utf8,
        other => {
            return Err(SchemaError::UnsupportedColumnType {
                column: col.name.clone(),
                col_type: other.clone(),
            });
        }
    })
}

/// Derive the Arrow schema records are decoded into from a table's declared
/// columns (`CatalogTable::columns`). Every field is nullable.
pub(crate) fn table_schema(columns: &[Column]) -> Result<SchemaRef, SchemaError> {
    let fields = columns
        .iter()
        .map(|c| Ok(Field::new(&c.name, arrow_type(c)?, true)))
        .collect::<Result<Vec<_>, SchemaError>>()?;
    Ok(Arc::new(Schema::new(fields)))
}

/// Decodes one Kafka message value into a JSON object row. Shared behind an
/// `Arc` across a topic's consumer tasks (the Avro/Protobuf impls cache schemas
/// internally and are `Send + Sync`).
pub(crate) trait MessageDecoder: Send + Sync {
    fn decode(&self, payload: &[u8]) -> Result<Value, DecodeError>;
}

/// Build the decoder for `format`. Avro/Protobuf share one `SchemaRegistry`
/// (its id→schema cache); JSON ignores it.
pub(crate) fn build_decoder(
    format: Format,
    registry: Option<Arc<SchemaRegistry>>,
) -> Arc<dyn MessageDecoder> {
    match format {
        Format::Json => Arc::new(json::JsonDecoder),
        Format::Avro => Arc::new(avro::AvroDecoder::new(registry.expect("registry required"))),
        Format::Protobuf => Arc::new(protobuf::ProtobufDecoder::new(
            registry.expect("registry required"),
        )),
    }
}

/// A flush's buffered rows plus the table schema to decode them into. Conversion
/// to Arrow runs in the write pipeline's first stage on a dispatch worker, so
/// the consumer task only buffers the cheap, already-format-decoded `Value`s.
pub(crate) struct KafkaBatch {
    pub(crate) rows: Vec<Value>,
    pub(crate) schema: SchemaRef,
}

impl ToRecordBatch for KafkaBatch {
    fn num_rows(&self) -> usize {
        self.rows.len()
    }

    fn to_record_batch(self) -> Result<Option<RecordBatch>, ArrowError> {
        decode_rows(&self.schema, &self.rows)
    }
}

/// Decode JSON object `rows` into one `RecordBatch` against `schema` (the single
/// arrow-json construction path): fields match by name, a missing field is null
/// (every field is nullable), and fields not in the schema are ignored. `None`
/// when `rows` is empty.
fn decode_rows(schema: &SchemaRef, rows: &[Value]) -> Result<Option<RecordBatch>, ArrowError> {
    if rows.is_empty() {
        return Ok(None);
    }
    let mut decoder = ReaderBuilder::new(schema.clone()).build_decoder()?;
    decoder.serialize(rows)?;
    decoder.flush()
}

/// Check that one decoded value fits `schema` before it joins the block, so a
/// type-mismatched-but-valid value (e.g. a bool for a `BIGINT` column) is caught
/// as a poison message here rather than failing the whole block at flush time.
pub(crate) fn validate_row(schema: &SchemaRef, value: &Value) -> Result<(), ArrowError> {
    decode_rows(schema, std::slice::from_ref(value)).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Float64Array, Int64Array, StringArray};
    use serde_json::json;

    fn column(name: &str, col_type: Type) -> Column {
        Column {
            name: name.to_string(),
            col_type,
        }
    }

    #[test]
    fn json_rows_decode_into_the_table_schema_with_nulls_for_missing_fields() {
        let columns = vec![
            column("user_id", Type::Int64),
            column("action", Type::Utf8),
            column("amount", Type::Float64),
        ];
        let schema = table_schema(&columns).unwrap();
        let rows = vec![
            json!({"user_id": 1, "action": "buy", "amount": 9.5, "ignored": true}),
            json!({"user_id": 2, "action": "sell"}),
        ];

        let batch = KafkaBatch { rows, schema }
            .to_record_batch()
            .unwrap()
            .unwrap();

        assert_eq!(batch.num_rows(), 2);
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ids.values(), &[1, 2]);
        let actions = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(actions.value(0), "buy");
        let amounts = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(amounts.value(0), 9.5);
        assert!(amounts.is_null(1), "a missing field decodes to null");
    }

    #[test]
    fn unsupported_column_type_is_rejected() {
        let columns = vec![column("ts", Type::Timestamp)];

        let error = table_schema(&columns).unwrap_err();

        assert!(matches!(error, SchemaError::UnsupportedColumnType { .. }));
    }

    #[test]
    fn validate_row_accepts_a_fitting_value_and_rejects_a_type_mismatch() {
        let schema = table_schema(&[column("user_id", Type::Int64)]).unwrap();

        assert!(validate_row(&schema, &json!({"user_id": 7})).is_ok());
        assert!(
            validate_row(&schema, &json!({"user_id": true})).is_err(),
            "a bool for a BIGINT column must be caught as poison, not silently nulled"
        );
    }
}
