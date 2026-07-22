//! Encode an Arrow [`RecordBatch`] into pgwire `FieldInfo` schemas and
//! `DataRow` streams.
//!
//! The simple query protocol always emits **text format** (pgwire's
//! [`FieldFormat::Text`]) - the universally supported representation `psql`
//! prints. The extended protocol lets the client request **binary** per result
//! column, so the schema and encoder take the portal's [`Format`]. Each Arrow
//! array kind has a dedicated `encode` arm that pushes its native Rust type
//! into [`DataRowEncoder`], whose `ToSql`/`ToSqlText` impls handle both wire
//! formats; anything we can't represent precisely is shipped as `text` so at
//! least the value arrives (and [`binary_encodable`] lets callers reject a
//! binary request for such a column up front instead of corrupting it).

use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, RecordBatch, StringArray, StringViewArray,
    TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, SchemaRef};

use pgwire::api::Type;
use pgwire::api::portal::Format;
use pgwire::api::results::{DataRowEncoder, FieldFormat, FieldInfo};
use pgwire::messages::data::DataRow;

/// Build a pgwire row schema from an Arrow [`SchemaRef`]. Each Arrow column
/// becomes one [`FieldInfo`] with the closest matching Postgres [`Type`], in
/// the wire format the client requested for it.
pub fn build_field_info(schema: &SchemaRef, format: &Format) -> Arc<Vec<FieldInfo>> {
    let fields = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(idx, f)| {
            FieldInfo::new(
                f.name().clone(),
                None,
                None,
                pg_type_for_arrow(f.data_type()),
                format.format_for(idx),
            )
        })
        .collect();
    Arc::new(fields)
}

/// Whether a column of this Arrow type can be encoded in Postgres **binary**
/// format. The types whose `encode_cell` arm feeds a Rust value with a real
/// binary `ToSql` for the advertised OID qualify; the ones shipped via the
/// debug fallback do not, and a binary request for them is rejected up front
/// (a conservatively clean error) rather than sent corrupted.
pub fn binary_encodable(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            // UInt64 is advertised as TEXT, whose binary form is the same
            // UTF-8 bytes as its text form.
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::Utf8View
            | DataType::Date32
            | DataType::Timestamp(_, _)
            // Scale-0 decimals (integer SUM results) go out as real binary
            // NUMERIC via [`PgNumeric`]; other scales never leave the executor.
            | DataType::Decimal128(_, 0)
    )
}

pub struct PGRowBatch {
    pub rows: Vec<DataRow>,
    pub fields: Arc<Vec<FieldInfo>>,
}

impl PGRowBatch {
    /// Encode a batch's rows in the given per-column wire format. The caller
    /// has already validated binary-format columns with [`binary_encodable`].
    pub fn encode(batch: RecordBatch, format: &Format) -> Self {
        let fields = build_field_info(&batch.schema(), format);
        // The per-column format is fixed for the whole batch; resolve it once
        // instead of per cell in the row loop.
        let formats: Vec<FieldFormat> = (0..batch.num_columns())
            .map(|col| format.format_for(col))
            .collect();
        let mut encoder = DataRowEncoder::new(fields.clone());
        let mut rows = Vec::with_capacity(batch.num_rows());
        for row in 0..batch.num_rows() {
            for col in 0..batch.num_columns() {
                encode_cell(&mut encoder, batch.column(col).as_ref(), row, formats[col]);
            }
            rows.push(encoder.take_row());
        }
        Self { rows, fields }
    }
}

/// A scale-0 `Decimal128` value (the integer SUM aggregate's output) with real
/// Postgres NUMERIC encodings: the text form is the plain integer, the binary
/// form is Postgres's base-10000 digit representation.
struct PgNumeric(i128);

impl std::fmt::Debug for PgNumeric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PgNumeric({})", self.0)
    }
}

impl pgwire::types::ToSqlText for PgNumeric {
    fn to_sql_text(
        &self,
        _ty: &Type,
        out: &mut bytes::BytesMut,
        _format_options: &pgwire::types::format::FormatOptions,
    ) -> std::result::Result<postgres_types::IsNull, Box<dyn std::error::Error + Sync + Send>> {
        use std::fmt::Write;
        write!(out, "{}", self.0)?;
        Ok(postgres_types::IsNull::No)
    }
}

impl postgres_types::ToSql for PgNumeric {
    fn to_sql(
        &self,
        _ty: &Type,
        out: &mut bytes::BytesMut,
    ) -> std::result::Result<postgres_types::IsNull, Box<dyn std::error::Error + Sync + Send>> {
        use bytes::BufMut;
        // Postgres NUMERIC binary format: i16 digit count, i16 weight (the
        // base-10000 exponent of the most significant digit), u16 sign, u16
        // display scale, then the base-10000 digits most significant first.
        let mut magnitude = self.0.unsigned_abs();
        let mut digits: Vec<u16> = Vec::new();
        while magnitude != 0 {
            digits.push((magnitude % 10_000) as u16);
            magnitude /= 10_000;
        }
        digits.reverse();
        const NUMERIC_NEG: u16 = 0x4000;
        let sign = if self.0 < 0 { NUMERIC_NEG } else { 0 };
        out.put_i16(digits.len() as i16);
        // Zero is encoded with no digits and weight 0.
        out.put_i16((digits.len() as i16 - 1).max(0));
        out.put_u16(sign);
        out.put_u16(0);
        for digit in digits {
            out.put_u16(digit);
        }
        Ok(postgres_types::IsNull::No)
    }

    fn accepts(ty: &Type) -> bool {
        *ty == Type::NUMERIC
    }

    postgres_types::to_sql_checked!();
}

/// Generates [`pg_type_for_arrow`] (schema: arrow type -> Postgres OID) and
/// [`encode_cell`] (data: one cell -> pgwire field) from one table, so the OID a
/// column advertises and the value its rows carry are declared together and
/// can't drift.
///
/// Two sections, because Postgres has neither unsigned nor 1-byte integers:
///
/// - `direct`: value encoded straight from `array.value(row)`, OID one-to-one.
///   Row shape `(arrow DataType, arrow array type, Postgres Type)`.
/// - `widened`: an int without a same-width Postgres twin, widened to the next
///   signed type on the wire. Row adds the cast target:
///   `(DataType, array type, Postgres Type, signed)`.
///
/// The arms that fit neither shape stay spelled out: `UInt64` (no signed type
/// fits, so it ships as TEXT via `to_string`), `Decimal128` (NUMERIC via
/// [`PgNumeric`] / `value_as_string`), and the advertise-only
/// `Binary`/`LargeUtf8` that fall through to the best-effort text fallback in
/// `encode_cell`.
macro_rules! arrow_pg_types {
    (
        direct: [ $( ($dt:ident, $array:ty, $pg:ident) ),+ $(,)? ],
        widened: [ $( ($udt:ident, $uarray:ty, $upg:ident, $signed:ty) ),+ $(,)? ] $(,)?
    ) => {
        /// Map an Arrow [`DataType`] to its closest Postgres [`Type`]. Values we
        /// can't represent precisely fall back to [`Type::TEXT`].
        pub(crate) fn pg_type_for_arrow(dt: &DataType) -> Type {
            match dt {
                $( DataType::$dt => Type::$pg, )+
                $( DataType::$udt => Type::$upg, )+
                // UInt64 can't fit any signed type, so it ships as TEXT.
                DataType::UInt64 => Type::TEXT,
                DataType::Decimal128(_, _) => Type::NUMERIC,
                // Temporal types reinterpreted from the executor's int columns at
                // the output boundary (see `encode_cell`).
                DataType::Date32 => Type::DATE,
                DataType::Timestamp(_, _) => Type::TIMESTAMP,
                // Advertised but encoded via the text fallback below.
                DataType::LargeUtf8 => Type::TEXT,
                DataType::Binary | DataType::LargeBinary | DataType::BinaryView => Type::BYTEA,
                _ => Type::TEXT,
            }
        }

        /// Encode a single cell. We always feed the encoder a typed Rust value
        /// (or `Option::None` for SQL NULL) so pgwire's `ToSqlText` impl handles
        /// the formatting, with no manual `to_string()` round-trips. `format` is
        /// the column's negotiated wire format; only the timestamp arm needs it
        /// (its text form differs from pgwire's chrono rendering).
        fn encode_cell(encoder: &mut DataRowEncoder, arr: &dyn Array, row: usize, format: FieldFormat) {
            if arr.is_null(row) {
                // Type is irrelevant for null encoding: the encoder writes -1 length.
                let _ = encoder.encode_field::<Option<&str>>(&None);
                return;
            }
            let _ = match arr.data_type() {
                $( DataType::$dt => encoder
                    .encode_field(&arr.as_any().downcast_ref::<$array>().unwrap().value(row)), )+
                // Unsigned ints widen to the next signed type so the value fits.
                $( DataType::$udt => encoder
                    .encode_field(&(arr.as_any().downcast_ref::<$uarray>().unwrap().value(row) as $signed)), )+
                DataType::UInt64 => encoder
                    .encode_field(&arr.as_any().downcast_ref::<UInt64Array>().unwrap().value(row).to_string()),
                // Scale-0 Decimal128 (the integer SUM aggregate output) goes out
                // as a real NUMERIC in both wire forms via [`PgNumeric`].
                DataType::Decimal128(_, 0) => encoder.encode_field(
                    &PgNumeric(arr.as_any().downcast_ref::<Decimal128Array>().unwrap().value(row)),
                ),
                // Other scales render with the scale applied, text form only.
                DataType::Decimal128(_, _) => encoder.encode_field(
                    &arr.as_any().downcast_ref::<Decimal128Array>().unwrap().value_as_string(row),
                ),
                // Temporal columns are fed as chrono civil values, whose pgwire
                // impls render the ISO text form and the real Postgres binary
                // form (pgwire's `pg-type-chrono` feature). The executor only
                // ever produces second-granularity timestamps. A text-format
                // timestamp keeps the plain second-resolution ISO string:
                // pgwire's chrono text rendering appends a fractional part
                // (`.000000`) that Postgres (and psql users) never see.
                DataType::Date32 => encoder.encode_field(
                    &arr.as_any().downcast_ref::<Date32Array>().unwrap().value_as_date(row),
                ),
                DataType::Timestamp(_, _) if format == FieldFormat::Binary => encoder.encode_field(
                    &arr.as_any().downcast_ref::<TimestampSecondArray>().unwrap()
                        .value_as_datetime(row),
                ),
                DataType::Timestamp(_, _) => encoder.encode_field(
                    &arr.as_any().downcast_ref::<TimestampSecondArray>().unwrap()
                        .value_as_datetime(row).map(|t| t.to_string()),
                ),
                // Best-effort fallback: stringify and ship as text.
                _ => encoder.encode_field(&format!("{:?}", arr.slice(row, 1))),
            };
        }
    };
}

arrow_pg_types! {
    direct: [
        (Boolean,  BooleanArray,    BOOL),
        (Int16,    Int16Array,      INT2),
        (Int32,    Int32Array,      INT4),
        (Int64,    Int64Array,      INT8),
        (Float32,  Float32Array,    FLOAT4),
        (Float64,  Float64Array,    FLOAT8),
        (Utf8,     StringArray,     TEXT),
        (Utf8View, StringViewArray, TEXT),
    ],
    widened: [
        // Int8 widens too: Postgres has no 1-byte integer, and a raw `i8`'s
        // binary ToSql is the 1-byte CHAR form, which would corrupt a column
        // advertised as INT2.
        (Int8,   Int8Array,   INT2, i16),
        (UInt8,  UInt8Array,  INT2, i16),
        (UInt16, UInt16Array, INT4, i32),
        (UInt32, UInt32Array, INT8, i64),
    ],
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
        let pg = PGRowBatch::encode(batch.clone(), &Format::UnifiedText);
        pg.rows.iter().map(decode_text_row).collect()
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

        let fields = build_field_info(&schema, &Format::UnifiedText);

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

        let fields = build_field_info(&schema, &Format::UnifiedText);

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
    fn encode_batch_encodes_decimal128_as_integer() {
        let col: ArrayRef = Arc::new(
            Decimal128Array::from(vec![15_i128, 9_223_372_036_854_775_807_i128 * 3])
                .with_precision_and_scale(38, 0)
                .unwrap(),
        );
        let b = batch(vec![("sum", DataType::Decimal128(38, 0))], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded[0], vec![Some("15".to_string())]);
        assert_eq!(decoded[1], vec![Some("27670116110564327421".to_string())]);
    }

    /// Decode a binary-format `DataRow`'s single NUMERIC field into its wire
    /// components: (digit count, weight, sign, dscale, base-10000 digits).
    fn decode_binary_numeric(row: &DataRow) -> (i16, i16, u16, u16, Vec<u16>) {
        let bytes: &[u8] = &row.data;
        let len = i32::from_be_bytes(bytes[0..4].try_into().unwrap());
        assert!(len >= 8, "a numeric payload has at least its header");
        let field = &bytes[4..4 + len as usize];
        let ndigits = i16::from_be_bytes(field[0..2].try_into().unwrap());
        let weight = i16::from_be_bytes(field[2..4].try_into().unwrap());
        let sign = u16::from_be_bytes(field[4..6].try_into().unwrap());
        let dscale = u16::from_be_bytes(field[6..8].try_into().unwrap());
        let digits = field[8..]
            .chunks(2)
            .map(|pair| u16::from_be_bytes(pair.try_into().unwrap()))
            .collect();
        (ndigits, weight, sign, dscale, digits)
    }

    #[test]
    fn decimal128_binary_encodes_postgres_numeric() {
        let col: ArrayRef = Arc::new(
            Decimal128Array::from(vec![123_456_789_i128, -15, 0])
                .with_precision_and_scale(38, 0)
                .unwrap(),
        );
        let b = batch(vec![("sum", DataType::Decimal128(38, 0))], vec![col]);

        let pg = PGRowBatch::encode(b, &Format::UnifiedBinary);

        // 123456789 = 1 * 10000^2 + 2345 * 10000 + 6789.
        assert_eq!(
            decode_binary_numeric(&pg.rows[0]),
            (3, 2, 0, 0, vec![1, 2345, 6789])
        );
        assert_eq!(
            decode_binary_numeric(&pg.rows[1]),
            (1, 0, 0x4000, 0, vec![15])
        );
        assert_eq!(decode_binary_numeric(&pg.rows[2]), (0, 0, 0, 0, vec![]));
    }

    #[test]
    fn int8_binary_widens_to_int2() {
        let col: ArrayRef = Arc::new(Int8Array::from(vec![-3i8]));
        let b = batch(vec![("v", DataType::Int8)], vec![col]);

        let pg = PGRowBatch::encode(b, &Format::UnifiedBinary);

        // 4-byte length prefix (2), then the big-endian i16 payload.
        let bytes: &[u8] = &pg.rows[0].data;
        assert_eq!(i32::from_be_bytes(bytes[0..4].try_into().unwrap()), 2);
        assert_eq!(i16::from_be_bytes(bytes[4..6].try_into().unwrap()), -3);
    }

    #[test]
    fn decimal128_maps_to_numeric() {
        let fields = build_field_info(
            &schema(vec![("s", DataType::Decimal128(38, 0))]),
            &Format::UnifiedText,
        );
        assert_eq!(fields[0].datatype(), &Type::NUMERIC);
    }

    #[test]
    fn date32_maps_to_date_and_renders_iso() {
        let col: ArrayRef = Arc::new(Date32Array::from(vec![0, 7]));
        let b = batch(vec![("d", DataType::Date32)], vec![col]);

        let fields = build_field_info(&b.schema(), &Format::UnifiedText);
        let decoded = rows(&b);

        assert_eq!(fields[0].datatype(), &Type::DATE);
        assert_eq!(decoded[0], vec![Some("1970-01-01".to_string())]);
        assert_eq!(decoded[1], vec![Some("1970-01-08".to_string())]);
    }

    #[test]
    fn timestamp_second_maps_to_timestamp_and_renders_iso() {
        let col: ArrayRef = Arc::new(TimestampSecondArray::from(vec![0, 90]));
        let b = batch(
            vec![(
                "t",
                DataType::Timestamp(arrow_schema::TimeUnit::Second, None),
            )],
            vec![col],
        );

        let fields = build_field_info(&b.schema(), &Format::UnifiedText);
        let decoded = rows(&b);

        assert_eq!(fields[0].datatype(), &Type::TIMESTAMP);
        assert_eq!(decoded[0], vec![Some("1970-01-01 00:00:00".to_string())]);
        assert_eq!(decoded[1], vec![Some("1970-01-01 00:01:30".to_string())]);
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
