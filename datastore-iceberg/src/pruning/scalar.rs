//! One-value arrays of discrete types, read and written as integers.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::{Array, ArrayRef, PrimitiveArray, Scalar};
use arrow_schema::{DataType, TimeUnit};

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

/// `value` as a constant of `data_type`, or `None` when the type is not
/// discrete or cannot hold it.
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
