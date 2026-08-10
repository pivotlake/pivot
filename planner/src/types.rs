//! Pivot's type system: the column types the planner understands, plus
//! conversions to and from DuckDB's [`LogicalTypeId`] and parsing of constant
//! values into arrow [`Scalar`]s.
//!
//! The supported [`Type`] set is deliberately small — it mirrors what the
//! executor ([`dispatch`]) currently knows how to build columns for. If
//! DuckDB hands us a logical type outside this set, [`type_from_logical`]
//! returns [`Error::UnsupportedLogicalType`] and short-circuits the rest of
//! plan conversion, so unsupported shapes never reach
//! [`compile`](crate::compile).
//!
//! The `Type <-> BoundLogicalType <-> DataType` conversions are plain
//! exhaustive matches: the compiler forces the infallible directions to cover
//! a new type, and the round-trip tests at the bottom of this module hold the
//! fallible directions to the same standard.

use arrow_array::{
    ArrayRef, BooleanArray, Decimal64Array, Decimal128Array, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, Scalar, StringViewArray, TimestampMicrosecondArray,
    UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, TimeUnit};
use duckdb_planner::duckdb_bridge::duckdb_types::LogicalTypeId;
use duckdb_planner::{BoundLogicalType, ExtraTypeInfo, ScalarValue};
use std::fmt;
use std::sync::Arc;
use thiserror::Error;

/// A column type Pivot can plan and execute against.
///
/// Each variant maps 1-1 to an arrow array kind in the executor
/// ([`physical_arrow_type`]) and bidirectionally to a DuckDB
/// [`BoundLogicalType`] ([`type_from_logical`] / [`logical_from_type`]).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Type {
    Boolean,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    /// DuckDB `HUGEINT` — the result type of `SUM` over integers. Pivot's
    /// executor emits `SUM` as a `Decimal128(38, 0)` column matching this
    /// width, so large sums (e.g. `SUM(user_id)`) stay exact.
    Int128,
    /// DuckDB `REAL`/`FLOAT` — a single-precision float column.
    Float32,
    /// DuckDB `DOUBLE` — a double-precision float column and the result type of `AVG`.
    Float64,
    /// DuckDB `DECIMAL(precision, scale)` — a fixed-point number carried as a
    /// column of unscaled integers, so the numeric value is
    /// `raw / 10^scale`. Precision up to [`MAX_DECIMAL64_PRECISION`] rides a
    /// `Decimal64` (i64) column, wider precision a `Decimal128` (i128) one;
    /// the cap is 38 (what 128 bits hold). Division does not produce it: `/`
    /// only has `REAL` and `DOUBLE` overloads, so decimal operands bind to
    /// the `DOUBLE` one.
    Decimal {
        precision: u8,
        scale: i8,
    },
    Utf8,
    /// DuckDB `DATE` — days since the epoch. The source parquet stores it as a
    /// small integer (days), so the executor sees an integer column; the type
    /// exists so `DATE` columns/constants survive plan translation and compare.
    Date,
    /// DuckDB `TIMESTAMP` — the type of a timestamp column and the result of
    /// `date_trunc`. It counts microseconds since the epoch, DuckDB's own
    /// resolution, so a value crosses the two engines unscaled. The executor
    /// sees that count in an `Int64`-shaped timestamp column.
    Timestamp,
    /// A Parquet `variant` (semi-structured / JSON) column, presented to DuckDB
    /// as its native `VARIANT` type: `d.age`, `d->'age'`, and casts all bind
    /// natively. The executor sees the Arrow struct of leaves the file stores.
    Variant,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Type::Boolean => "Boolean",
            Type::Int8 => "Int8",
            Type::Int16 => "Int16",
            Type::Int32 => "Int32",
            Type::Int64 => "Int64",
            Type::UInt8 => "UInt8",
            Type::UInt16 => "UInt16",
            Type::UInt32 => "UInt32",
            Type::UInt64 => "UInt64",
            Type::Int128 => "Int128",
            Type::Float32 => "Float32",
            Type::Float64 => "Float64",
            Type::Decimal { precision, scale } => {
                return write!(f, "Decimal({precision},{scale})");
            }
            Type::Utf8 => "Utf8",
            Type::Date => "Date",
            Type::Timestamp => "Timestamp",
            Type::Variant => "Variant",
        };
        f.write_str(name)
    }
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("Unsupported logical type: {0:?}")]
    UnsupportedLogicalType(LogicalTypeId),
    #[error("Unsupported scalar constant: {0}")]
    UnsupportedScalarConstant(ScalarValue),
    #[error("Unsupported decimal width/scale: DECIMAL({width},{scale})")]
    UnsupportedDecimal { width: u8, scale: u8 },
    #[error("{0}")]
    Arrow(#[from] arrow_schema::ArrowError),
}

/// The widest decimal Pivot supports: what a 128-bit unscaled integer holds.
/// DuckDB enforces the same cap, so any wider type is rejected at its binder.
pub const MAX_DECIMAL_PRECISION: u8 = 38;

/// The widest decimal carried as a `Decimal64` (i64) column; anything wider
/// rides `Decimal128`. 18 digits is also DuckDB's own int64 storage boundary,
/// and its binder casts every operand of a mixed-width operation to the wide
/// type, so kernels never see a `Decimal64`/`Decimal128` mix.
pub const MAX_DECIMAL64_PRECISION: u8 = 18;

/// DuckDB [`BoundLogicalType`] -> Pivot [`Type`] (fallible, returns `Err` for
/// unmapped DuckDB types, including an id whose [`ExtraTypeInfo`] payload
/// does not match it).
pub fn type_from_logical(bound: BoundLogicalType) -> Result<Type, Error> {
    match (&bound.id, bound.extra) {
        (LogicalTypeId::BOOLEAN, ExtraTypeInfo::None) => Ok(Type::Boolean),
        (LogicalTypeId::TINYINT, ExtraTypeInfo::None) => Ok(Type::Int8),
        (LogicalTypeId::SMALLINT, ExtraTypeInfo::None) => Ok(Type::Int16),
        (LogicalTypeId::INTEGER, ExtraTypeInfo::None) => Ok(Type::Int32),
        (LogicalTypeId::BIGINT, ExtraTypeInfo::None) => Ok(Type::Int64),
        (LogicalTypeId::UTINYINT, ExtraTypeInfo::None) => Ok(Type::UInt8),
        (LogicalTypeId::USMALLINT, ExtraTypeInfo::None) => Ok(Type::UInt16),
        (LogicalTypeId::UINTEGER, ExtraTypeInfo::None) => Ok(Type::UInt32),
        (LogicalTypeId::UBIGINT, ExtraTypeInfo::None) => Ok(Type::UInt64),
        (LogicalTypeId::HUGEINT, ExtraTypeInfo::None) => Ok(Type::Int128),
        (LogicalTypeId::FLOAT, ExtraTypeInfo::None) => Ok(Type::Float32),
        (LogicalTypeId::DOUBLE, ExtraTypeInfo::None) => Ok(Type::Float64),
        (LogicalTypeId::DECIMAL, ExtraTypeInfo::Decimal { width, scale }) => {
            if width == 0 || width > MAX_DECIMAL_PRECISION || scale > width {
                return Err(Error::UnsupportedDecimal { width, scale });
            }
            Ok(Type::Decimal {
                precision: width,
                scale: scale as i8,
            })
        }
        (LogicalTypeId::VARCHAR, ExtraTypeInfo::None) => Ok(Type::Utf8),
        (LogicalTypeId::DATE, ExtraTypeInfo::None) => Ok(Type::Date),
        (LogicalTypeId::TIMESTAMP, ExtraTypeInfo::None) => Ok(Type::Timestamp),
        (LogicalTypeId::VARIANT, ExtraTypeInfo::None) => Ok(Type::Variant),
        _ => Err(Error::UnsupportedLogicalType(bound.id.clone())),
    }
}

/// Pivot [`Type`] -> DuckDB [`BoundLogicalType`] (infallible, every Pivot
/// type has a DuckDB counterpart), the inverse of [`type_from_logical`].
pub fn logical_from_type(pivot_type: &Type) -> BoundLogicalType {
    match pivot_type {
        Type::Boolean => BoundLogicalType::plain(LogicalTypeId::BOOLEAN),
        Type::Int8 => BoundLogicalType::plain(LogicalTypeId::TINYINT),
        Type::Int16 => BoundLogicalType::plain(LogicalTypeId::SMALLINT),
        Type::Int32 => BoundLogicalType::plain(LogicalTypeId::INTEGER),
        Type::Int64 => BoundLogicalType::plain(LogicalTypeId::BIGINT),
        Type::UInt8 => BoundLogicalType::plain(LogicalTypeId::UTINYINT),
        Type::UInt16 => BoundLogicalType::plain(LogicalTypeId::USMALLINT),
        Type::UInt32 => BoundLogicalType::plain(LogicalTypeId::UINTEGER),
        Type::UInt64 => BoundLogicalType::plain(LogicalTypeId::UBIGINT),
        Type::Int128 => BoundLogicalType::plain(LogicalTypeId::HUGEINT),
        Type::Float32 => BoundLogicalType::plain(LogicalTypeId::FLOAT),
        Type::Float64 => BoundLogicalType::plain(LogicalTypeId::DOUBLE),
        Type::Decimal { precision, scale } => BoundLogicalType {
            id: LogicalTypeId::DECIMAL,
            extra: ExtraTypeInfo::Decimal {
                width: *precision,
                scale: *scale as u8,
            },
        },
        Type::Utf8 => BoundLogicalType::plain(LogicalTypeId::VARCHAR),
        Type::Date => BoundLogicalType::plain(LogicalTypeId::DATE),
        Type::Timestamp => BoundLogicalType::plain(LogicalTypeId::TIMESTAMP),
        Type::Variant => BoundLogicalType::plain(LogicalTypeId::VARIANT),
    }
}

/// The arrow [`DataType`] a column of this `Type` carries.
///
/// This is where the "logical vs physical" facts live, once: `Date` is
/// `Date32`, `Timestamp` is `Timestamp(Microsecond)`, and the wide aggregate type
/// `Int128` lands on `Decimal128(38, 0)`. This is the single arrow type a
/// value carries everywhere; only the group-by drops a temporal column to its
/// backing int (`temporal_to_int`) to hash/encode it, restoring the type on
/// its output.
pub fn physical_arrow_type(pivot_type: &Type) -> DataType {
    match pivot_type {
        Type::Boolean => DataType::Boolean,
        Type::Int8 => DataType::Int8,
        Type::Int16 => DataType::Int16,
        Type::Int32 => DataType::Int32,
        Type::Int64 => DataType::Int64,
        Type::UInt8 => DataType::UInt8,
        Type::UInt16 => DataType::UInt16,
        Type::UInt32 => DataType::UInt32,
        Type::UInt64 => DataType::UInt64,
        Type::Int128 => DataType::Decimal128(38, 0),
        Type::Float32 => DataType::Float32,
        Type::Float64 => DataType::Float64,
        Type::Decimal { precision, scale } => {
            if *precision <= MAX_DECIMAL64_PRECISION {
                DataType::Decimal64(*precision, *scale)
            } else {
                DataType::Decimal128(*precision, *scale)
            }
        }
        Type::Utf8 => DataType::Utf8View,
        Type::Date => DataType::Date32,
        Type::Timestamp => DataType::Timestamp(TimeUnit::Microsecond, None),
        Type::Variant => variant_struct_type(),
    }
}

/// Maps an Arrow type back to a Pivot type for constants, the inverse of
/// [`physical_arrow_type`] up to one deliberate aliasing: `Decimal128(38, 0)`
/// resolves to `Int128` (the SUM/HUGEINT accumulator type, which arrived
/// first), and every other `Decimal64`/`Decimal128` shape to `Decimal`.
pub fn type_from_physical(data_type: &DataType) -> Option<Type> {
    match data_type {
        DataType::Boolean => Some(Type::Boolean),
        DataType::Int8 => Some(Type::Int8),
        DataType::Int16 => Some(Type::Int16),
        DataType::Int32 => Some(Type::Int32),
        DataType::Int64 => Some(Type::Int64),
        DataType::UInt8 => Some(Type::UInt8),
        DataType::UInt16 => Some(Type::UInt16),
        DataType::UInt32 => Some(Type::UInt32),
        DataType::UInt64 => Some(Type::UInt64),
        DataType::Decimal128(38, 0) => Some(Type::Int128),
        DataType::Float32 => Some(Type::Float32),
        DataType::Float64 => Some(Type::Float64),
        DataType::Decimal128(precision, scale) | DataType::Decimal64(precision, scale) => {
            Some(Type::Decimal {
                precision: *precision,
                scale: *scale,
            })
        }
        DataType::Utf8View => Some(Type::Utf8),
        DataType::Date32 => Some(Type::Date),
        DataType::Timestamp(TimeUnit::Microsecond, None) => Some(Type::Timestamp),
        other if *other == variant_struct_type() => Some(Type::Variant),
        _ => None,
    }
}

/// The nominal arrow shape of an unshredded variant column: the spec's two
/// binary leaves. A file may carry additional shredded `typed_value` leaves, so
/// a scanned batch's actual struct can be wider; everything that touches a
/// variant (the `->` extraction) reads any layout, and this shape only serves
/// as the column's pre-scan placeholder type.
fn variant_struct_type() -> DataType {
    DataType::Struct(arrow_schema::Fields::from(vec![
        arrow_schema::Field::new("metadata", DataType::BinaryView, false),
        arrow_schema::Field::new("value", DataType::BinaryView, true),
    ]))
}

/// Convert a typed DuckDB [`ScalarValue`] into an arrow [`Scalar<ArrayRef>`]
/// suitable for use as a constant in the executor.
pub fn build_scalar_value(value: ScalarValue) -> Result<Scalar<ArrayRef>, Error> {
    let array: ArrayRef = match value {
        ScalarValue::Boolean(v) => Arc::new(BooleanArray::new_scalar(v).into_inner()),
        ScalarValue::Int8(v) => Arc::new(Int8Array::new_scalar(v).into_inner()),
        ScalarValue::Int16(v) => Arc::new(Int16Array::new_scalar(v).into_inner()),
        ScalarValue::Int32(v) => Arc::new(Int32Array::new_scalar(v).into_inner()),
        ScalarValue::Int64(v) => Arc::new(Int64Array::new_scalar(v).into_inner()),
        ScalarValue::UInt8(v) => Arc::new(UInt8Array::new_scalar(v).into_inner()),
        ScalarValue::UInt16(v) => Arc::new(UInt16Array::new_scalar(v).into_inner()),
        ScalarValue::UInt32(v) => Arc::new(UInt32Array::new_scalar(v).into_inner()),
        ScalarValue::UInt64(v) => Arc::new(UInt64Array::new_scalar(v).into_inner()),
        ScalarValue::Float32(v) => Arc::new(Float32Array::new_scalar(v).into_inner()),
        ScalarValue::Float64(v) => Arc::new(Float64Array::new_scalar(v).into_inner()),
        // A HUGEINT constant lands on Int128's physical type, Decimal128(38, 0).
        ScalarValue::Int128(v) => Arc::new(
            Decimal128Array::new_scalar(v)
                .into_inner()
                .with_precision_and_scale(38, 0)?,
        ),
        // A DECIMAL constant lands on the declared type's carrier: `Decimal64`
        // up to 18 digits (where the unscaled integer is guaranteed to fit),
        // `Decimal128` beyond.
        ScalarValue::Decimal {
            value,
            width,
            scale,
        } if width <= MAX_DECIMAL64_PRECISION => Arc::new(
            Decimal64Array::new_scalar(
                i64::try_from(value).expect("an 18-digit decimal constant fits in i64"),
            )
            .into_inner()
            .with_precision_and_scale(width, scale as i8)?,
        ),
        ScalarValue::Decimal {
            value,
            width,
            scale,
        } => Arc::new(
            Decimal128Array::new_scalar(value)
                .into_inner()
                .with_precision_and_scale(width, scale as i8)?,
        ),
        ScalarValue::Utf8(v) => Arc::new(StringViewArray::new_scalar(v).into_inner()),
        // A VARIANT constant reaches the bridge as the text it was built from:
        // DuckDB's variant-to-VARCHAR gives back the raw string, not its JSON
        // rendering, so parse that document into pivot's variant here. This is
        // how `'{...}'::VARIANT` -- an INSERT of a JSON document -- loads. A
        // malformed document is an error, not a panic: constant folding runs on
        // the compile path, which has an error channel.
        ScalarValue::Variant(text) => {
            let json: ArrayRef = Arc::new(arrow_array::StringArray::from(vec![text]));
            crate::expression::json_to_canonical_variant(&json)?
        }
        // DuckDB's DATE is days since the epoch, the same as arrow `Date32`.
        // Comparisons coerce both sides to a common numeric type, so this lines up
        // with the integer day-count the parquet stores for a `DATE` column.
        ScalarValue::Date(days) => {
            Arc::new(arrow_array::Date32Array::new_scalar(days).into_inner())
        }
        // DuckDB's TIMESTAMP is microseconds and so is pivot's, so the count
        // crosses unscaled. A `date ± interval` constant lowers to a TIMESTAMP
        // compared against `CAST(date AS TIMESTAMP)`, which casts the date
        // column to the same resolution.
        ScalarValue::Timestamp(micros) => {
            Arc::new(TimestampMicrosecondArray::new_scalar(micros).into_inner())
        }
        // A NULL constant carries the type the binder gave it, so it becomes a
        // one-element null array of that type's physical shape. An untyped NULL
        // (DuckDB's `SQLNULL`, what a bare `NULL` binds to outside a typed
        // context) has no shape to build and is rejected by `type_from_logical`.
        ScalarValue::Null(logical_type) => {
            arrow_array::new_null_array(&physical_arrow_type(&type_from_logical(logical_type)?), 1)
        }
        // INTERVAL never appears as a query constant we materialise (it is
        // consumed by interval arithmetic), and types the bridge doesn't decode
        // arrive as `Other`.
        other => return Err(Error::UnsupportedScalarConstant(other)),
    };

    Ok(Scalar::new(array))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One value of every [`Type`] variant. Extend this when adding a type;
    /// the round-trip tests below then cover its conversions.
    fn every_type() -> Vec<Type> {
        vec![
            Type::Boolean,
            Type::Int8,
            Type::Int16,
            Type::Int32,
            Type::Int64,
            Type::UInt8,
            Type::UInt16,
            Type::UInt32,
            Type::UInt64,
            Type::Int128,
            Type::Float32,
            Type::Float64,
            Type::Decimal {
                precision: 10,
                scale: 2,
            },
            Type::Decimal {
                precision: 20,
                scale: 2,
            },
            Type::Utf8,
            Type::Date,
            Type::Timestamp,
            Type::Variant,
        ]
    }

    #[test]
    fn every_type_round_trips_through_duckdb() {
        for pivot_type in every_type() {
            let bound = logical_from_type(&pivot_type);

            let back = type_from_logical(bound).unwrap();

            assert_eq!(back, pivot_type);
        }
    }

    #[test]
    fn every_type_round_trips_through_arrow() {
        // Int128 and Decimal(38, 0) share Decimal128(38, 0), which resolves
        // to Int128; every other type comes back as itself.
        for pivot_type in every_type() {
            let arrow = physical_arrow_type(&pivot_type);

            let back = type_from_physical(&arrow);

            assert_eq!(back, Some(pivot_type));
        }
    }

    #[test]
    fn a_mismatched_extra_type_info_is_rejected() {
        let bound = BoundLogicalType {
            id: LogicalTypeId::BIGINT,
            extra: ExtraTypeInfo::Decimal {
                width: 10,
                scale: 2,
            },
        };

        let result = type_from_logical(bound);

        assert!(matches!(result, Err(Error::UnsupportedLogicalType(_))));
    }
}
