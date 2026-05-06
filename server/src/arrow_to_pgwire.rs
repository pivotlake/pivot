//! Encode an Arrow [`RecordBatch`] into pgwire `FieldInfo` schemas and
//! `DataRow` streams.
//!
//! Only **text format** is emitted (pgwire's [`FieldFormat::Text`]). It is the
//! universally supported representation, requires no per-type binary encoder,
//! and matches what `psql` prints. Each Arrow array kind has a dedicated
//! `encode` arm that pushes its native Rust type into [`DataRowEncoder`];
//! anything we can't represent precisely is shipped as `text` so at least
//! the value arrives.

use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    RecordBatch, StringArray, StringViewArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, SchemaRef};

use pgwire::api::Type;
use pgwire::api::results::{DataRowEncoder, FieldFormat, FieldInfo};
use pgwire::error::PgWireResult;
use pgwire::messages::data::DataRow;

/// Build a pgwire row schema from an Arrow [`SchemaRef`]. Each Arrow column
/// becomes one [`FieldInfo`] with the closest matching Postgres [`Type`].
pub fn build_field_info(schema: &SchemaRef) -> Arc<Vec<FieldInfo>> {
    let fields = schema
        .fields()
        .iter()
        .map(|f| {
            FieldInfo::new(
                f.name().clone(),
                None,
                None,
                pg_type_for_arrow(f.data_type()),
                FieldFormat::Text,
            )
        })
        .collect();
    Arc::new(fields)
}

/// Encode every row in `batch` as a [`DataRow`] using `fields` as the schema.
pub fn encode_batch(
    batch: &RecordBatch,
    fields: Arc<Vec<FieldInfo>>,
) -> Vec<PgWireResult<DataRow>> {
    let mut encoder = DataRowEncoder::new(fields);
    let mut out = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        for col in 0..batch.num_columns() {
            encode_cell(&mut encoder, batch.column(col).as_ref(), row);
        }
        out.push(Ok(encoder.take_row()));
    }
    out
}

/// Encode a single cell. We always feed the encoder a typed Rust value (or
/// `Option::None` for SQL NULL) so pgwire's `ToSqlText` impl handles the
/// formatting — no manual `to_string()` round-trips.
fn encode_cell(encoder: &mut DataRowEncoder, arr: &dyn Array, row: usize) {
    if arr.is_null(row) {
        // Type doesn't matter for null encoding — the encoder writes -1 length.
        let _ = encoder.encode_field::<Option<&str>>(&None);
        return;
    }
    let _ = match arr.data_type() {
        DataType::Boolean => encoder.encode_field(
            &arr.as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(row),
        ),
        DataType::Int8 => {
            encoder.encode_field(&arr.as_any().downcast_ref::<Int8Array>().unwrap().value(row))
        }
        DataType::Int16 => encoder.encode_field(
            &arr.as_any()
                .downcast_ref::<Int16Array>()
                .unwrap()
                .value(row),
        ),
        DataType::Int32 => encoder.encode_field(
            &arr.as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(row),
        ),
        DataType::Int64 => encoder.encode_field(
            &arr.as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(row),
        ),
        // Postgres has no unsigned types; widen to the next signed type so
        // values fit. UInt64 is reported as TEXT (see [`pg_type_for_arrow`]).
        DataType::UInt8 => encoder.encode_field(
            &(arr
                .as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap()
                .value(row) as i16),
        ),
        DataType::UInt16 => encoder.encode_field(
            &(arr
                .as_any()
                .downcast_ref::<UInt16Array>()
                .unwrap()
                .value(row) as i32),
        ),
        DataType::UInt32 => encoder.encode_field(
            &(arr
                .as_any()
                .downcast_ref::<UInt32Array>()
                .unwrap()
                .value(row) as i64),
        ),
        DataType::UInt64 => {
            let v = arr
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(row);
            encoder.encode_field(&v.to_string())
        }
        DataType::Float32 => encoder.encode_field(
            &arr.as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(row),
        ),
        DataType::Float64 => encoder.encode_field(
            &arr.as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(row),
        ),
        DataType::Utf8 => encoder.encode_field(
            &arr.as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(row),
        ),
        DataType::Utf8View => encoder.encode_field(
            &arr.as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap()
                .value(row),
        ),
        // Best-effort fallback: stringify and ship as text.
        _ => encoder.encode_field(&format!("{:?}", arr.slice(row, 1))),
    };
}

/// Map an Arrow [`DataType`] to its closest Postgres [`Type`]. Values we can't
/// represent precisely fall back to [`Type::TEXT`].
fn pg_type_for_arrow(dt: &DataType) -> Type {
    match dt {
        DataType::Boolean => Type::BOOL,
        DataType::Int8 | DataType::Int16 => Type::INT2,
        DataType::Int32 => Type::INT4,
        DataType::Int64 => Type::INT8,
        DataType::UInt8 => Type::INT2,
        DataType::UInt16 => Type::INT4,
        DataType::UInt32 => Type::INT8,
        DataType::UInt64 => Type::TEXT,
        DataType::Float32 => Type::FLOAT4,
        DataType::Float64 => Type::FLOAT8,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Type::TEXT,
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => Type::BYTEA,
        _ => Type::TEXT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::ArrayRef;
    use arrow_schema::{Field, Schema};

    fn schema(fields: Vec<(&str, DataType)>) -> SchemaRef {
        Arc::new(Schema::new(
            fields
                .into_iter()
                .map(|(n, dt)| Field::new(n, dt, true))
                .collect::<Vec<_>>(),
        ))
    }

    fn batch(fields: Vec<(&str, DataType)>, columns: Vec<ArrayRef>) -> RecordBatch {
        RecordBatch::try_new(schema(fields), columns).unwrap()
    }

    /// Decode a text-format `DataRow`'s wire bytes into `Vec<Option<String>>`.
    /// Each field is a big-endian i32 length followed by that many UTF-8 bytes;
    /// length -1 means SQL NULL.
    fn decode_text_row(row: &DataRow) -> Vec<Option<String>> {
        let bytes: &[u8] = &row.data;
        let mut out = Vec::with_capacity(row.field_count as usize);
        let mut i = 0;
        for _ in 0..row.field_count {
            let len = i32::from_be_bytes(bytes[i..i + 4].try_into().unwrap());
            i += 4;
            if len < 0 {
                out.push(None);
            } else {
                let n = len as usize;
                out.push(Some(String::from_utf8(bytes[i..i + n].to_vec()).unwrap()));
                i += n;
            }
        }
        out
    }

    fn rows(batch: &RecordBatch) -> Vec<Vec<Option<String>>> {
        let fields = build_field_info(&batch.schema());
        encode_batch(batch, fields)
            .into_iter()
            .map(|r| decode_text_row(&r.unwrap()))
            .collect()
    }

    #[test]
    fn build_field_info_maps_names_and_types() {
        let schema = schema(vec![
            ("flag", DataType::Boolean),
            ("small", DataType::Int16),
            ("big", DataType::Int64),
            ("u64", DataType::UInt64),
            ("f", DataType::Float64),
            ("s", DataType::Utf8),
        ]);

        let fields = build_field_info(&schema);

        assert_eq!(fields.len(), 6);
        assert_eq!(fields[0].name(), "flag");
        assert_eq!(fields[0].datatype(), &Type::BOOL);
        assert_eq!(fields[0].format(), FieldFormat::Text);
        assert_eq!(fields[1].datatype(), &Type::INT2);
        assert_eq!(fields[2].datatype(), &Type::INT8);
        assert_eq!(fields[3].datatype(), &Type::TEXT);
        assert_eq!(fields[4].datatype(), &Type::FLOAT8);
        assert_eq!(fields[5].datatype(), &Type::TEXT);
    }

    #[test]
    fn build_field_info_widens_unsigned_types() {
        let schema = schema(vec![
            ("a", DataType::UInt8),
            ("b", DataType::UInt16),
            ("c", DataType::UInt32),
        ]);

        let fields = build_field_info(&schema);

        assert_eq!(fields[0].datatype(), &Type::INT2);
        assert_eq!(fields[1].datatype(), &Type::INT4);
        assert_eq!(fields[2].datatype(), &Type::INT8);
    }

    #[test]
    fn encode_batch_row_count_matches_input() {
        let col: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
        let b = batch(vec![("v", DataType::Int32)], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded.len(), 4);
    }

    #[test]
    fn encode_batch_empty_produces_no_rows() {
        let col: ArrayRef = Arc::new(Int32Array::from(Vec::<i32>::new()));
        let b = batch(vec![("v", DataType::Int32)], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded.len(), 0);
    }

    #[test]
    fn encode_batch_encodes_integers_as_text() {
        let col: ArrayRef = Arc::new(Int32Array::from(vec![0, -7, 12345]));
        let b = batch(vec![("v", DataType::Int32)], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded[0], vec![Some("0".to_string())]);
        assert_eq!(decoded[1], vec![Some("-7".to_string())]);
        assert_eq!(decoded[2], vec![Some("12345".to_string())]);
    }

    #[test]
    fn encode_batch_encodes_booleans_as_t_and_f() {
        let col: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
        let b = batch(vec![("v", DataType::Boolean)], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded[0], vec![Some("t".to_string())]);
        assert_eq!(decoded[1], vec![Some("f".to_string())]);
    }

    #[test]
    fn encode_batch_encodes_strings_verbatim() {
        let col: ArrayRef = Arc::new(StringArray::from(vec!["hello", "", "héllo"]));
        let b = batch(vec![("v", DataType::Utf8)], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded[0], vec![Some("hello".to_string())]);
        assert_eq!(decoded[1], vec![Some("".to_string())]);
        assert_eq!(decoded[2], vec![Some("héllo".to_string())]);
    }

    #[test]
    fn encode_batch_encodes_floats() {
        let col: ArrayRef = Arc::new(Float64Array::from(vec![1.5, -0.25]));
        let b = batch(vec![("v", DataType::Float64)], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded[0], vec![Some("1.5".to_string())]);
        assert_eq!(decoded[1], vec![Some("-0.25".to_string())]);
    }

    #[test]
    fn encode_batch_encodes_nulls() {
        let col: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]));
        let b = batch(vec![("v", DataType::Int32)], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded[0], vec![Some("1".to_string())]);
        assert_eq!(decoded[1], vec![None]);
        assert_eq!(decoded[2], vec![Some("3".to_string())]);
    }

    #[test]
    fn encode_batch_encodes_uint64_as_text() {
        let col: ArrayRef = Arc::new(UInt64Array::from(vec![u64::MAX, 0]));
        let b = batch(vec![("v", DataType::UInt64)], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded[0], vec![Some(u64::MAX.to_string())]);
        assert_eq!(decoded[1], vec![Some("0".to_string())]);
    }

    #[test]
    fn encode_batch_handles_multiple_columns_per_row() {
        let id: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
        let name: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let flag: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
        let b = batch(
            vec![
                ("id", DataType::Int32),
                ("name", DataType::Utf8),
                ("flag", DataType::Boolean),
            ],
            vec![id, name, flag],
        );

        let decoded = rows(&b);

        assert_eq!(
            decoded[0],
            vec![
                Some("1".to_string()),
                Some("a".to_string()),
                Some("t".to_string()),
            ],
        );
        assert_eq!(
            decoded[1],
            vec![
                Some("2".to_string()),
                Some("b".to_string()),
                Some("f".to_string()),
            ],
        );
    }
}
