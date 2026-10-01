//! Iceberg values as Arrow, and back: a manifest's literals laid out as an
//! array of a column's physical type, so a predicate's constant compares
//! against every file in one kernel, and a predicate's constant as the Iceberg
//! datum a partition transform projects.

use std::sync::Arc;

use arrow_array::builder::{PrimitiveBuilder, StringViewBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    ArrowPrimitiveType, Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type,
    Int32Type, Int64Type, TimestampMicrosecondType,
};
use arrow_array::{Array, ArrayRef, BooleanArray, Datum as ArrowDatum, Scalar, new_null_array};
use arrow_schema::{DataType, TimeUnit};
use iceberg::spec::{Datum, PrimitiveLiteral, PrimitiveType};

/// `literals` as an array of the requested physical Arrow type: one slot per
/// literal, null where there is none or where it is not a literal that type
/// holds. Incompatible or unsupported values yield unknown bounds. Integer and floating
/// point promotions preserve older manifests' statistics.
pub(crate) fn build_array<'a>(
    data_type: &DataType,
    literals: impl Iterator<Item = Option<&'a PrimitiveLiteral>>,
) -> ArrayRef {
    match data_type {
        DataType::Boolean => Arc::new(BooleanArray::from_iter(literals.map(
            |literal| match literal {
                Some(PrimitiveLiteral::Boolean(value)) => Some(*value),
                _ => None,
            },
        ))),
        DataType::Int32 => primitive_array::<Int32Type>(
            data_type.clone(),
            literals.map(|literal| match literal {
                Some(PrimitiveLiteral::Int(value)) => Some(*value),
                _ => None,
            }),
        ),
        DataType::Date32 => primitive_array::<Date32Type>(
            data_type.clone(),
            literals.map(|literal| match literal {
                Some(PrimitiveLiteral::Int(days)) => Some(*days),
                _ => None,
            }),
        ),
        DataType::Int64 => primitive_array::<Int64Type>(
            data_type.clone(),
            literals.map(|literal| match literal {
                Some(PrimitiveLiteral::Long(value)) => Some(*value),
                Some(PrimitiveLiteral::Int(value)) => Some(i64::from(*value)),
                _ => None,
            }),
        ),
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            primitive_array::<TimestampMicrosecondType>(
                data_type.clone(),
                literals.map(|literal| match literal {
                    Some(PrimitiveLiteral::Long(micros)) => Some(*micros),
                    _ => None,
                }),
            )
        }
        DataType::Float32 => primitive_array::<Float32Type>(
            data_type.clone(),
            literals.map(|literal| match literal {
                Some(PrimitiveLiteral::Float(value)) => Some(value.0),
                _ => None,
            }),
        ),
        DataType::Float64 => primitive_array::<Float64Type>(
            data_type.clone(),
            literals.map(|literal| match literal {
                Some(PrimitiveLiteral::Double(value)) => Some(value.0),
                Some(PrimitiveLiteral::Float(value)) => Some(f64::from(value.0)),
                _ => None,
            }),
        ),
        DataType::Decimal64(..) => primitive_array::<Decimal64Type>(
            data_type.clone(),
            literals.map(|literal| match literal {
                Some(PrimitiveLiteral::Int128(unscaled)) => i64::try_from(*unscaled).ok(),
                _ => None,
            }),
        ),
        DataType::Decimal128(..) => primitive_array::<Decimal128Type>(
            data_type.clone(),
            literals.map(|literal| match literal {
                Some(PrimitiveLiteral::Int128(unscaled)) => Some(*unscaled),
                _ => None,
            }),
        ),
        DataType::Utf8View => {
            let mut builder = StringViewBuilder::new();
            for literal in literals {
                builder.append_option(match literal {
                    Some(PrimitiveLiteral::String(value)) => Some(value.as_str()),
                    _ => None,
                });
            }
            Arc::new(builder.finish())
        }
        _ => new_null_array(data_type, literals.count()),
    }
}

/// `values` as an array of `data_type`, a primitive type of `T`'s width.
fn primitive_array<T: ArrowPrimitiveType>(
    data_type: DataType,
    values: impl Iterator<Item = Option<T::Native>>,
) -> ArrayRef {
    let mut builder = PrimitiveBuilder::<T>::new().with_data_type(data_type);
    for value in values {
        builder.append_option(value);
    }
    Arc::new(builder.finish())
}

/// `literal` as a one-value scalar of the requested physical Arrow type, or
/// `None` when that type does not hold it.
pub(crate) fn build_scalar(
    data_type: &DataType,
    literal: &PrimitiveLiteral,
) -> Option<Scalar<ArrayRef>> {
    let array = build_array(data_type, [Some(literal)].into_iter());
    array.is_valid(0).then(|| Scalar::new(array))
}

/// `constant`, a predicate's one-value scalar, as the Iceberg datum of the
/// same value: the form a partition transform projects. `None` when it is
/// NULL or of a type Iceberg has no primitive for.
pub(crate) fn to_datum(constant: &Scalar<ArrayRef>) -> Option<Datum> {
    let (array, _) = constant.get();
    if array.is_null(0) {
        return None;
    }
    Some(match array.data_type() {
        DataType::Boolean => Datum::bool(array.as_boolean().value(0)),
        DataType::Int32 => Datum::int(array.as_primitive::<Int32Type>().value(0)),
        DataType::Int64 => Datum::long(array.as_primitive::<Int64Type>().value(0)),
        DataType::Float32 => Datum::float(array.as_primitive::<Float32Type>().value(0)),
        DataType::Float64 => Datum::double(array.as_primitive::<Float64Type>().value(0)),
        DataType::Date32 => Datum::date(array.as_primitive::<Date32Type>().value(0)),
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            Datum::timestamp_micros(array.as_primitive::<TimestampMicrosecondType>().value(0))
        }
        DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => {
            Datum::timestamptz_micros(array.as_primitive::<TimestampMicrosecondType>().value(0))
        }
        &DataType::Decimal64(precision, scale) => decimal_datum(
            i128::from(array.as_primitive::<Decimal64Type>().value(0)),
            precision,
            scale,
        )?,
        &DataType::Decimal128(precision, scale) => decimal_datum(
            array.as_primitive::<Decimal128Type>().value(0),
            precision,
            scale,
        )?,
        DataType::Utf8View => Datum::string(array.as_string_view().value(0)),
        DataType::Utf8 => Datum::string(array.as_string::<i32>().value(0)),
        _ => return None,
    })
}

/// The decimal datum of `unscaled` at `precision` and `scale`, built the way
/// a manifest stores one: from the unscaled value's big-endian bytes.
fn decimal_datum(unscaled: i128, precision: u8, scale: i8) -> Option<Datum> {
    let data_type = PrimitiveType::Decimal {
        precision: u32::from(precision),
        scale: u32::try_from(scale).ok()?,
    };
    Datum::try_from_bytes(&unscaled.to_be_bytes(), data_type).ok()
}
