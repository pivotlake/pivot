//! Exact scalar conversion for partition projection. These operate on query
//! constants; object bounds remain in shared Arrow arrays.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::{Array, ArrayRef, Datum, PrimitiveArray, Scalar};
use arrow_schema::{DataType, TimeUnit};

use crate::Comparison;

pub(crate) enum BoundConstant {
    Value(Scalar<ArrayRef>),
    Always(bool),
    Unknown,
}

/// Bind against the type that wrote the partition, without rounding a wider
/// query constant. Out-of-domain integer/decimal constants can prove a filter
/// impossible; unsupported conversions simply leave the object unpruned.
pub(crate) fn bind(
    value: &Scalar<ArrayRef>,
    source: &DataType,
    compare: Comparison,
) -> BoundConstant {
    let (array, _) = value.get();
    let input = array.data_type();
    if input == &DataType::Null || array.is_null(0) {
        return BoundConstant::Always(false);
    }
    if input == source {
        return BoundConstant::Value(value.clone());
    }
    let compatible_numbers = (input.is_integer() && source.is_integer())
        || decimal_scale(input)
            .zip(decimal_scale(source))
            .is_some_and(|(a, b)| a == b);
    if compatible_numbers
        && let Some(value) = integer(array)
        && let Some((min, max)) = domain(source)
    {
        if value < min || value > max {
            let above = value > max;
            return BoundConstant::Always(match compare {
                Comparison::Equal => false,
                Comparison::NotEqual => true,
                Comparison::Less | Comparison::LessEqual => above,
                Comparison::Greater | Comparison::GreaterEqual => !above,
            });
        }
        return from_integer(value, source).map_or(BoundConstant::Unknown, BoundConstant::Value);
    }
    // String layouts differ without changing the value. Floating-point type
    // changes, decimal rescaling and temporal unit changes need their own
    // rounding analysis and are deliberately not implicit casts here.
    if input.is_string() && source.is_string() {
        return arrow_cast::cast(array, source)
            .ok()
            .map(Scalar::new)
            .map_or(BoundConstant::Unknown, BoundConstant::Value);
    }
    BoundConstant::Unknown
}

fn decimal_scale(data_type: &DataType) -> Option<i8> {
    match data_type {
        DataType::Decimal64(_, scale) | DataType::Decimal128(_, scale) => Some(*scale),
        _ => None,
    }
}

/// The discrete, unscaled value of an integer, decimal, date, or timestamp.
pub(crate) fn integer(array: &dyn Array) -> Option<i128> {
    macro_rules! read {
        ($ty:ty) => {
            Some(i128::from(array.as_primitive::<$ty>().value(0)))
        };
    }
    match array.data_type() {
        DataType::Int8 => read!(Int8Type),
        DataType::Int16 => read!(Int16Type),
        DataType::Int32 => read!(Int32Type),
        DataType::Int64 => read!(Int64Type),
        DataType::UInt8 => read!(UInt8Type),
        DataType::UInt16 => read!(UInt16Type),
        DataType::UInt32 => read!(UInt32Type),
        DataType::UInt64 => read!(UInt64Type),
        DataType::Decimal64(..) => read!(Decimal64Type),
        DataType::Decimal128(..) => read!(Decimal128Type),
        DataType::Date32 => read!(Date32Type),
        DataType::Timestamp(TimeUnit::Second, _) => read!(TimestampSecondType),
        DataType::Timestamp(TimeUnit::Millisecond, _) => read!(TimestampMillisecondType),
        DataType::Timestamp(TimeUnit::Microsecond, _) => read!(TimestampMicrosecondType),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => read!(TimestampNanosecondType),
        _ => None,
    }
}

pub(crate) fn from_integer(value: i128, data_type: &DataType) -> Option<Scalar<ArrayRef>> {
    match data_type {
        DataType::Int8 => primitive::<Int8Type>(value, data_type),
        DataType::Int16 => primitive::<Int16Type>(value, data_type),
        DataType::Int32 => primitive::<Int32Type>(value, data_type),
        DataType::Int64 => primitive::<Int64Type>(value, data_type),
        DataType::UInt8 => primitive::<UInt8Type>(value, data_type),
        DataType::UInt16 => primitive::<UInt16Type>(value, data_type),
        DataType::UInt32 => primitive::<UInt32Type>(value, data_type),
        DataType::UInt64 => primitive::<UInt64Type>(value, data_type),
        DataType::Decimal64(..) => primitive::<Decimal64Type>(value, data_type),
        DataType::Decimal128(..) => primitive::<Decimal128Type>(value, data_type),
        DataType::Date32 => primitive::<Date32Type>(value, data_type),
        DataType::Timestamp(TimeUnit::Second, _) => {
            primitive::<TimestampSecondType>(value, data_type)
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            primitive::<TimestampMillisecondType>(value, data_type)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            primitive::<TimestampMicrosecondType>(value, data_type)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            primitive::<TimestampNanosecondType>(value, data_type)
        }
        _ => None,
    }
}

fn primitive<T: ArrowPrimitiveType>(value: i128, data_type: &DataType) -> Option<Scalar<ArrayRef>>
where
    T::Native: TryFrom<i128>,
{
    let value = T::Native::try_from(value).ok()?;
    let array = PrimitiveArray::<T>::from_iter_values([value]).with_data_type(data_type.clone());
    Some(Scalar::new(Arc::new(array) as ArrayRef))
}

pub(crate) fn domain(data_type: &DataType) -> Option<(i128, i128)> {
    macro_rules! limits {
        ($ty:ty) => {
            Some((i128::from(<$ty>::MIN), i128::from(<$ty>::MAX)))
        };
    }
    match data_type {
        DataType::Int8 => limits!(i8),
        DataType::Int16 => limits!(i16),
        DataType::Int32 | DataType::Date32 => limits!(i32),
        DataType::Int64 | DataType::Timestamp(..) => limits!(i64),
        DataType::UInt8 => limits!(u8),
        DataType::UInt16 => limits!(u16),
        DataType::UInt32 => limits!(u32),
        DataType::UInt64 => limits!(u64),
        DataType::Decimal64(precision, _) | DataType::Decimal128(precision, _) => {
            let max = 10_i128.checked_pow(u32::from(*precision))?.checked_sub(1)?;
            Some((-max, max))
        }
        _ => None,
    }
}
