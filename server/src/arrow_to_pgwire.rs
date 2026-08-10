//! Encode an Arrow [`RecordBatch`] into pgwire `FieldInfo` schemas and
//! `DataRow` streams.
//!
//! The simple protocol always emits **text format** (pgwire's
//! [`FieldFormat::Text`]), the universally supported representation `psql`
//! prints. The extended protocol lets a client request binary result columns
//! in its Bind; [`encode_batch`] honors the requested per-column format, and
//! pgwire's [`DataRowEncoder`] picks the text or binary codec from each
//! field's declared format. Each Arrow array kind has a dedicated `encode`
//! arm that pushes its native Rust type into the encoder; a type with no
//! faithful binary form errors when binary is requested rather than shipping
//! bytes the client would misread.

use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, Date32Array, Decimal64Array, Decimal128Array, Float32Array, Float64Array,
    Int8Array, Int16Array, Int32Array, Int64Array, RecordBatch, StringArray, StringViewArray,
    TimestampMicrosecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, SchemaRef};

use pgwire::api::Type;
use pgwire::api::portal::Format;
use pgwire::api::results::{DataRowEncoder, FieldFormat, FieldInfo};
use pgwire::error::PgWireResult;
use pgwire::messages::data::DataRow;

/// Build a pgwire row schema from an Arrow [`SchemaRef`] with the client's
/// requested per-column result format (from its Bind message) applied. Each
/// Arrow column becomes one [`FieldInfo`] with the closest matching Postgres
/// [`Type`]. Errors when the client sent fewer individual format codes than
/// the result has columns.
pub fn build_field_info_with_format(
    schema: &SchemaRef,
    format: &Format,
) -> Result<Arc<Vec<FieldInfo>>, String> {
    if let Format::Individual(codes) = format
        && codes.len() < schema.fields().len()
    {
        return Err(format!(
            "the result has {} columns but only {} format codes were bound",
            schema.fields().len(),
            codes.len()
        ));
    }
    let fields = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| {
            FieldInfo::new(
                f.name().clone(),
                None,
                None,
                pg_type_for_arrow(f.data_type()),
                format.format_for(i),
            )
        })
        .collect();
    Ok(Arc::new(fields))
}

pub struct PGRowBatch {
    pub rows: Vec<DataRow>,
    pub fields: Arc<Vec<FieldInfo>>,
}

/// Per-worker memo for [`encode_batch`]'s derived pgwire schema: the batches
/// of one query share their arrow schema, so the `FieldInfo`s are built once
/// and revalidated per batch by schema pointer identity alone.
pub type FieldInfoCache = Option<(SchemaRef, Arc<Vec<FieldInfo>>)>;

/// Encode a batch's rows with the client's requested per-column format. A
/// column whose type has no faithful binary encoding errors when binary was
/// requested for it.
pub fn encode_batch(
    batch: &RecordBatch,
    format: &Format,
    cache: &mut FieldInfoCache,
) -> Result<PGRowBatch, String> {
    let fields = match cache {
        Some((schema, fields)) if Arc::ptr_eq(schema, batch.schema_ref()) => fields.clone(),
        _ => {
            let fields = build_field_info_with_format(&batch.schema(), format)?;
            *cache = Some((batch.schema(), fields.clone()));
            fields
        }
    };
    let mut encoder = DataRowEncoder::new(fields.clone());
    let mut rows = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        for col in 0..batch.num_columns() {
            let field_format = fields[col].format();
            encode_cell(&mut encoder, batch.column(col).as_ref(), row, field_format)
                .map_err(|e| format!("encoding column {}: {e}", fields[col].name()))?;
        }
        rows.push(encoder.take_row());
    }
    Ok(PGRowBatch { rows, fields })
}

/// Generates [`pg_type_for_arrow`] (schema: arrow type -> Postgres OID) and
/// [`encode_cell`] (data: one cell -> pgwire field) from one table, so the OID a
/// column advertises and the value its rows carry are declared together and
/// can't drift.
///
/// Two sections, because Postgres has no unsigned types:
///
/// - `direct`: value encoded straight from `array.value(row)`, OID one-to-one.
///   Row shape `(arrow DataType, arrow array type, Postgres Type)`.
/// - `widened`: an unsigned int that widens to the next signed type on the wire.
///   Row adds the signed cast target: `(DataType, array type, Postgres Type, signed)`.
///
/// The arms that fit neither shape stay spelled out: `UInt64` (no signed type
/// fits, so it ships as TEXT via `to_string`), `Decimal128` (NUMERIC via
/// `value_as_string`), and the advertise-only `Binary`/`LargeUtf8` that fall
/// through to the best-effort text fallback in `encode_cell`.
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
                DataType::Decimal64(_, _) | DataType::Decimal128(_, _) => Type::NUMERIC,
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
        /// (or `Option::None` for SQL NULL); the field's declared format picks
        /// pgwire's `ToSqlText` or `ToSql` (binary) codec for it.
        fn encode_cell(
            encoder: &mut DataRowEncoder,
            arr: &dyn Array,
            row: usize,
            format: FieldFormat,
        ) -> PgWireResult<()> {
            if arr.is_null(row) {
                // Type is irrelevant for null encoding: the encoder writes -1 length.
                return encoder.encode_field::<Option<&str>>(&None);
            }
            match arr.data_type() {
                $( DataType::$dt => encoder
                    .encode_field(&arr.as_any().downcast_ref::<$array>().unwrap().value(row)), )+
                // Unsigned ints widen to the next signed type so the value fits.
                $( DataType::$udt => encoder
                    .encode_field(&(arr.as_any().downcast_ref::<$uarray>().unwrap().value(row) as $signed)), )+
                DataType::UInt64 => encoder
                    .encode_field(&arr.as_any().downcast_ref::<UInt64Array>().unwrap().value(row).to_string()),
                // Decimals in either carrier width (Decimal128 is e.g. the SUM
                // aggregate output) ride [`PgNumeric`], whose text codec
                // renders the scale applied (scale 0 yields a plain integer
                // like "12345") and whose binary codec writes the NUMERIC
                // wire format.
                DataType::Decimal64(_, s) => encoder.encode_field(&PgNumeric {
                    unscaled: arr.as_any().downcast_ref::<Decimal64Array>().unwrap().value(row)
                        as i128,
                    scale: *s,
                }),
                DataType::Decimal128(_, s) => encoder.encode_field(&PgNumeric {
                    unscaled: arr.as_any().downcast_ref::<Decimal128Array>().unwrap().value(row),
                    scale: *s,
                }),
                // Temporal columns: in text, render the ISO string Postgres
                // prints (a timestamp's fraction goes through
                // [`trim_fraction`]; pgwire's own `ToSqlText` for chrono
                // values fixes the fraction at six digits, so every whole
                // second would arrive as `.000000`). In binary, hand the
                // encoder the chrono value itself, which pgwire encodes in
                // the DATE/TIMESTAMP wire format.
                DataType::Date32 => {
                    let date = arr.as_any().downcast_ref::<Date32Array>().unwrap()
                        .value_as_date(row);
                    match format {
                        FieldFormat::Text => encoder.encode_field(&date.map(|d| d.to_string())),
                        FieldFormat::Binary => encoder.encode_field(&date),
                    }
                }
                DataType::Timestamp(_, _) => {
                    let timestamp = arr.as_any().downcast_ref::<TimestampMicrosecondArray>()
                        .unwrap().value_as_datetime(row);
                    match format {
                        FieldFormat::Text => encoder
                            .encode_field(&timestamp.map(|t| trim_fraction(t.to_string()))),
                        FieldFormat::Binary => encoder.encode_field(&timestamp),
                    }
                }
                // Best-effort fallback: stringify and ship as text. In binary
                // the encoder rejects the string against the declared type,
                // which is right: there is no faithful binary form to send.
                _ => encoder.encode_field(&format!("{:?}", arr.slice(row, 1))),
            }
        }
    };
}

/// A decimal value as Postgres's NUMERIC: an unscaled integer plus its scale.
/// The text codec renders the digits with the point applied; the binary codec
/// writes the NUMERIC wire format (base-10000 digit groups).
#[derive(Debug)]
struct PgNumeric {
    unscaled: i128,
    scale: i8,
}

impl PgNumeric {
    fn render(&self) -> String {
        use arrow_array::types::{Decimal128Type, DecimalType};
        // Arrow's decimal formatter applies the scale; the precision argument
        // only bounds it, so the widest carrier's maximum always fits.
        Decimal128Type::format_decimal(self.unscaled, Decimal128Type::MAX_PRECISION, self.scale)
    }
}

/// Render a binary-format NUMERIC (base-10000 digit groups) as a plain
/// decimal literal. The decode counterpart of [`PgNumeric`]'s binary codec,
/// kept beside it so the two directions of the wire format evolve together;
/// the parameter-decoding layer parses the rendered digits through the
/// target type like any text input.
pub(crate) fn numeric_binary_to_string(bytes: &[u8]) -> Result<String, String> {
    if bytes.len() < 8 {
        return Err("binary numeric is truncated".to_string());
    }
    let read_i16 = |at: usize| i16::from_be_bytes([bytes[at], bytes[at + 1]]);
    let ndigits = read_i16(0);
    let weight = read_i16(2);
    let sign = u16::from_be_bytes([bytes[4], bytes[5]]);
    let dscale = read_i16(6);
    if ndigits < 0 || dscale < 0 {
        return Err("binary numeric is malformed".to_string());
    }
    if bytes.len() != 8 + ndigits as usize * 2 {
        return Err("binary numeric length does not match its digit count".to_string());
    }
    let negative = match sign {
        0x0000 => false,
        0x4000 => true,
        _ => return Err("NaN/Infinity numeric parameters are not supported".to_string()),
    };

    // Digit group i (base 10000) has weight `weight - i`: it counts for
    // 10000^(weight-i). Render everything at or above weight 0 as the integer
    // part and the rest as the fraction, then trim to dscale.
    let group = |i: i16| -> u16 {
        if i < 0 || i >= ndigits {
            return 0;
        }
        u16::from_be_bytes([bytes[8 + i as usize * 2], bytes[9 + i as usize * 2]])
    };
    let mut int_part = String::new();
    for i in 0..=weight.max(-1) {
        let rendered = group(i);
        if int_part.is_empty() {
            if rendered != 0 {
                int_part = rendered.to_string();
            }
        } else {
            int_part.push_str(&format!("{rendered:04}"));
        }
    }
    if int_part.is_empty() {
        int_part.push('0');
    }
    let mut frac_part = String::new();
    let mut i = weight + 1;
    while frac_part.len() < dscale as usize {
        frac_part.push_str(&format!("{:04}", group(i)));
        i += 1;
    }
    frac_part.truncate(dscale as usize);

    let sign = if negative { "-" } else { "" };
    if frac_part.is_empty() {
        Ok(format!("{sign}{int_part}"))
    } else {
        Ok(format!("{sign}{int_part}.{frac_part}"))
    }
}

impl pgwire::types::ToSqlText for PgNumeric {
    fn to_sql_text(
        &self,
        _ty: &Type,
        out: &mut bytes::BytesMut,
        _format_options: &pgwire::types::format::FormatOptions,
    ) -> Result<postgres_types::IsNull, Box<dyn std::error::Error + Sync + Send>> {
        use bytes::BufMut;
        out.put_slice(self.render().as_bytes());
        Ok(postgres_types::IsNull::No)
    }
}

impl postgres_types::ToSql for PgNumeric {
    fn to_sql(
        &self,
        _ty: &Type,
        out: &mut bytes::BytesMut,
    ) -> Result<postgres_types::IsNull, Box<dyn std::error::Error + Sync + Send>> {
        use bytes::BufMut;
        if self.scale < 0 {
            return Err(format!("negative NUMERIC scale {} is not supported", self.scale).into());
        }
        let negative = self.unscaled < 0;
        let mut abs = self.unscaled.unsigned_abs();

        // Align the fractional digits to whole base-10000 groups: with the
        // value multiplied so the fraction spans exactly `frac_groups`
        // groups, the group boundary falls on the decimal point.
        let frac_groups = (self.scale as usize).div_ceil(4);
        for _ in 0..(frac_groups * 4 - self.scale as usize) {
            abs = abs
                .checked_mul(10)
                .ok_or("NUMERIC value overflows its binary encoding")?;
        }
        let mut groups = Vec::new();
        while abs > 0 {
            groups.push((abs % 10_000) as u16);
            abs /= 10_000;
        }
        groups.reverse();
        // `groups` ends with the fractional groups; the first group's weight
        // is its base-10000 position relative to the decimal point.
        let weight = groups.len() as i16 - frac_groups as i16 - 1;
        out.put_i16(groups.len() as i16);
        out.put_i16(if groups.is_empty() { 0 } else { weight });
        out.put_u16(if negative { 0x4000 } else { 0x0000 });
        out.put_i16(self.scale as i16);
        for group in groups {
            out.put_u16(group);
        }
        Ok(postgres_types::IsNull::No)
    }

    fn accepts(ty: &Type) -> bool {
        *ty == Type::NUMERIC
    }

    postgres_types::to_sql_checked!();
}

/// Drop the trailing zeros from a rendered timestamp's fractional second, and
/// the fraction itself when nothing is left, which is how a Postgres server
/// prints one: `00:00:00.5`, not `00:00:00.500`. Chrono pads the fraction to
/// three, six or nine digits, so its rendering needs this to read the same as a
/// real server's.
fn trim_fraction(rendered: String) -> String {
    let Some((instant, fraction)) = rendered.split_once('.') else {
        return rendered;
    };
    let fraction = fraction.trim_end_matches('0');
    if fraction.is_empty() {
        instant.to_string()
    } else {
        format!("{instant}.{fraction}")
    }
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
        // Int8 must widen: its column is declared int2, and postgres-types
        // encodes a raw i8 as the 1-byte "char" type, which a binary-format
        // int2 reader rejects as a short buffer.
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

    /// The text-format schema the simple protocol advertises.
    fn build_field_info(schema: &SchemaRef) -> Arc<Vec<FieldInfo>> {
        build_field_info_with_format(schema, &Format::UnifiedText).unwrap()
    }

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
        let pg = encode_batch(batch, &Format::UnifiedText, &mut None).unwrap();
        pg.rows.iter().map(decode_text_row).collect()
    }

    /// The binary NUMERIC codec round-trips through its decode counterpart.
    #[test]
    fn binary_numeric_encoding_round_trips() {
        use postgres_types::ToSql;
        for (unscaled, scale, rendered) in [
            (1234567i128, 2, "12345.67"),
            (-250, 2, "-2.50"),
            (5, 1, "0.5"),
            (0, 0, "0"),
            (12345, 0, "12345"),
        ] {
            let numeric = PgNumeric { unscaled, scale };
            let mut out = bytes::BytesMut::new();
            numeric.to_sql(&Type::NUMERIC, &mut out).unwrap();

            assert_eq!(numeric_binary_to_string(&out).unwrap(), rendered);
            assert_eq!(numeric.render(), rendered);
        }
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

    #[test]
    fn decimal128_maps_to_numeric() {
        let fields = build_field_info(&schema(vec![("s", DataType::Decimal128(38, 0))]));
        assert_eq!(fields[0].datatype(), &Type::NUMERIC);
    }

    #[test]
    fn encode_batch_encodes_decimal64_with_scale_applied() {
        let col: ArrayRef = Arc::new(
            Decimal64Array::from(vec![12345_i64, -250])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        );
        let b = batch(vec![("v", DataType::Decimal64(10, 2))], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded[0], vec![Some("123.45".to_string())]);
        assert_eq!(decoded[1], vec![Some("-2.50".to_string())]);
    }

    #[test]
    fn decimal64_maps_to_numeric() {
        let fields = build_field_info(&schema(vec![("v", DataType::Decimal64(10, 2))]));
        assert_eq!(fields[0].datatype(), &Type::NUMERIC);
    }

    #[test]
    fn date32_maps_to_date_and_renders_iso() {
        let col: ArrayRef = Arc::new(Date32Array::from(vec![0, 7]));
        let b = batch(vec![("d", DataType::Date32)], vec![col]);

        let fields = build_field_info(&b.schema());
        let decoded = rows(&b);

        assert_eq!(fields[0].datatype(), &Type::DATE);
        assert_eq!(decoded[0], vec![Some("1970-01-01".to_string())]);
        assert_eq!(decoded[1], vec![Some("1970-01-08".to_string())]);
    }

    /// A fractional second renders with its trailing zeros dropped, and a
    /// pre-epoch instant keeps the fraction of the second it falls in. Every
    /// expectation here is what a PostgreSQL 16 server prints for the same
    /// value.
    #[test]
    fn fractional_timestamps_render_as_a_postgres_server_prints_them() {
        let col: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            1_577_836_800_500_000,
            1_577_836_800_120_000,
            1_577_836_800_000_001,
            -1_500_000,
        ]));
        let b = batch(
            vec![(
                "t",
                DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
            )],
            vec![col],
        );

        let decoded = rows(&b);

        assert_eq!(decoded[0], vec![Some("2020-01-01 00:00:00.5".to_string())]);
        assert_eq!(decoded[1], vec![Some("2020-01-01 00:00:00.12".to_string())]);
        assert_eq!(
            decoded[2],
            vec![Some("2020-01-01 00:00:00.000001".to_string())]
        );
        assert_eq!(decoded[3], vec![Some("1969-12-31 23:59:58.5".to_string())]);
    }

    #[test]
    fn timestamp_maps_to_timestamp_and_renders_iso() {
        let col: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            0,
            90_000_000,
            1_700_000_000_123_456,
        ]));
        let b = batch(
            vec![(
                "t",
                DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
            )],
            vec![col],
        );

        let fields = build_field_info(&b.schema());
        let decoded = rows(&b);

        assert_eq!(fields[0].datatype(), &Type::TIMESTAMP);
        assert_eq!(decoded[0], vec![Some("1970-01-01 00:00:00".to_string())]);
        assert_eq!(decoded[1], vec![Some("1970-01-01 00:01:30".to_string())]);
        // A sub-second part renders with the timestamp and a whole second
        // renders without one, which is how Postgres prints them.
        assert_eq!(
            decoded[2],
            vec![Some("2023-11-14 22:13:20.123456".to_string())]
        );
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
