//! The one mapping between Parquet's on-disk types and the executor's arrow
//! [`DataType`].
//!
//! The read path ([`parquet_to_arrow`], used when loading a file's footer) and
//! the write path ([`arrow_to_parquet_physical`], used by `ingest` when stamping
//! a footer) are inverses kept side by side, so a type added to one is added to
//! the other instead of the two drifting apart.
//!
//! ```text
//!   arrow DataType    parquet physical   annotation              direction
//!   --------------    ----------------   ----------------------  ---------
//!   Boolean           BOOLEAN            -                       read only
//!   Int8              INT32              Integer{ 8, signed}     read only
//!   UInt8             INT32              Integer{ 8, unsigned}   read only
//!   Int16             INT32              Integer{16, signed}     read only
//!   UInt16            INT32              Integer{16, unsigned}   read only
//!   Int32             INT32              -                       read+write
//!   UInt32            INT32              Integer{32, unsigned}   read only
//!   Int64             INT64              -                       read+write
//!   Float32           FLOAT              -                       read+write
//!   Float64           DOUBLE             -                       read+write
//!   Utf8 / Utf8View   BYTE_ARRAY         converted UTF8 / String read+write
//!   Date32            INT32              Date                    read only
//!   Timestamp(Second) INT64              Timestamp{..}           read only
//!   BinaryView        BYTE_ARRAY         unannotated             read+write
//! ```
//!
//! The "read only" rows resolve files written elsewhere; pivot's own writer only
//! emits the column set its encoder supports (Int32/Int64/Float32/Float64/
//! strings/binary), so [`arrow_to_parquet_physical`] errors on the rest.

use arrow_schema::{DataType, TimeUnit};

use super::table::{Error, Result};
use super::thrift::footer::LogicalType;
use super::thrift::general::Type;

// Parquet physical type ids, named off the same thrift enum the writer emits,
// so read and write reference one definition rather than bare integers.
const BOOLEAN: i32 = Type::BOOLEAN as i32;
const INT32: i32 = Type::INT32 as i32;
const INT64: i32 = Type::INT64 as i32;
const FLOAT: i32 = Type::FLOAT as i32;
const DOUBLE: i32 = Type::DOUBLE as i32;
const BYTE_ARRAY: i32 = Type::BYTE_ARRAY as i32;

// Parquet `ConvertedType` ids: the legacy width/temporal annotation, read
// alongside the modern `LogicalType`.
const CONVERTED_UTF8: i32 = 0;
const CONVERTED_UINT_8: i32 = 11;
const CONVERTED_UINT_16: i32 = 12;
const CONVERTED_UINT_32: i32 = 13;
const CONVERTED_INT_8: i32 = 15;
const CONVERTED_INT_16: i32 = 16;
const CONVERTED_INT_32: i32 = 17;
/// Legacy temporal annotations (superseded by `LogicalType::Date`/`Timestamp`).
const CONVERTED_DATE: i32 = 6;
const CONVERTED_TIMESTAMP_MILLIS: i32 = 9;
const CONVERTED_TIMESTAMP_MICROS: i32 = 10;

/// Read path: a Parquet leaf's physical type (plus optional logical/converted
/// annotations) -> the arrow [`DataType`] the executor decodes it into.
///
/// A BYTE_ARRAY is a string only when the file says so (a UTF8/String
/// annotation); an unannotated one is binary, exactly as the spec defines it.
/// Files in the wild do store text without the annotation, but that fact isn't
/// in the file: it comes from the table's declared schema, and
/// `apply_declared_types` (in `table.rs`) retypes such columns to string when
/// the table declares them VARCHAR.
pub fn parquet_to_arrow(
    physical_type: Option<i32>,
    converted_type: Option<i32>,
    logical_type: Option<&LogicalType>,
) -> Result<DataType> {
    let pt = physical_type.ok_or_else(|| {
        Error::UnsupportedType("leaf schema element missing physical type".to_string())
    })?;

    match pt {
        BOOLEAN => Ok(DataType::Boolean),
        INT32 => int32_arrow(converted_type, logical_type),
        INT64 => int64_arrow(converted_type, logical_type),
        FLOAT => Ok(DataType::Float32),
        DOUBLE => Ok(DataType::Float64),
        BYTE_ARRAY => {
            let is_string = converted_type == Some(CONVERTED_UTF8)
                || matches!(logical_type, Some(LogicalType::String));
            if is_string {
                Ok(DataType::Utf8View)
            } else {
                Ok(DataType::BinaryView)
            }
        }
        _ => Err(Error::UnsupportedType(format!(
            "Parquet physical type {pt}"
        ))),
    }
}

/// The arrow type of an INT32 leaf, resolved from its annotation: a `DATE` is
/// `Date32` (an INT32 day count), the integer annotations give the narrow widths,
/// and a bare INT32 is `Int32`. The modern `LogicalType` wins; the legacy
/// `ConvertedType` is the fallback. Errors on an annotation we don't support
/// (e.g. a DECIMAL/TIME width) rather than silently reading raw `Int32`.
fn int32_arrow(
    converted_type: Option<i32>,
    logical_type: Option<&LogicalType>,
) -> Result<DataType> {
    match logical_type {
        Some(LogicalType::Date) => Ok(DataType::Date32),
        Some(LogicalType::Integer {
            bit_width,
            is_signed,
        }) => match (*bit_width, *is_signed) {
            (8, true) => Ok(DataType::Int8),
            (8, false) => Ok(DataType::UInt8),
            (16, true) => Ok(DataType::Int16),
            (16, false) => Ok(DataType::UInt16),
            (32, true) => Ok(DataType::Int32),
            (32, false) => Ok(DataType::UInt32),
            _ => Err(Error::UnsupportedType(format!(
                "INT32 column with integer width {bit_width} (signed: {is_signed})"
            ))),
        },
        // No (recognized) LogicalType: fall back to the legacy ConvertedType.
        _ => match converted_type {
            Some(CONVERTED_DATE) => Ok(DataType::Date32),
            Some(CONVERTED_INT_8) => Ok(DataType::Int8),
            Some(CONVERTED_UINT_8) => Ok(DataType::UInt8),
            Some(CONVERTED_INT_16) => Ok(DataType::Int16),
            Some(CONVERTED_UINT_16) => Ok(DataType::UInt16),
            Some(CONVERTED_INT_32) => Ok(DataType::Int32),
            Some(CONVERTED_UINT_32) => Ok(DataType::UInt32),
            // A bare INT32 is a plain signed 32-bit int.
            None => Ok(DataType::Int32),
            Some(other) => Err(Error::UnsupportedType(format!(
                "INT32 converted type {other}"
            ))),
        },
    }
}

/// The arrow type of an INT64 leaf: a `TIMESTAMP` is `Timestamp(Second)` (pivot
/// stores timestamps as seconds, so a sub-second file unit is read as seconds);
/// otherwise a plain `Int64`.
fn int64_arrow(
    converted_type: Option<i32>,
    logical_type: Option<&LogicalType>,
) -> Result<DataType> {
    match logical_type {
        Some(LogicalType::Timestamp { .. }) => Ok(DataType::Timestamp(TimeUnit::Second, None)),
        _ => match converted_type {
            Some(CONVERTED_TIMESTAMP_MILLIS | CONVERTED_TIMESTAMP_MICROS) => {
                Ok(DataType::Timestamp(TimeUnit::Second, None))
            }
            _ => Ok(DataType::Int64),
        },
    }
}

/// Write path: the Parquet physical type id for an arrow column. The inverse of
/// [`parquet_to_arrow`] over the writable subset; errors on a type the encoder
/// doesn't emit.
///
/// A `BinaryView` writes as an unannotated BYTE_ARRAY — the spec's own
/// definition of binary, and what a variant's `metadata`/`value` leaves are.
/// Leaving off the UTF8 annotation is what keeps [`parquet_to_arrow`] reading it
/// back as binary rather than as text.
pub fn arrow_to_parquet_physical(data_type: &DataType) -> Result<i32> {
    Ok(match data_type {
        DataType::Int32 => INT32,
        DataType::Int64 => INT64,
        DataType::Float32 => FLOAT,
        DataType::Float64 => DOUBLE,
        DataType::Utf8 | DataType::Utf8View | DataType::BinaryView => BYTE_ARRAY,
        other => {
            return Err(Error::UnsupportedType(format!("arrow type {other:?}")));
        }
    })
}
