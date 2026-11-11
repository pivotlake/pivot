//! Arrow values as Iceberg values.
//!
//! A filter's constant and a file's column bound both arrive as one-value
//! Arrow arrays and both have to become an Iceberg `Datum`. A bound is read
//! as its column's type; a filter's constant as its own, since the planner
//! may widen a constant beyond the column it is compared with. A value the
//! type cannot hold is `None`, which a filter takes as "prune nothing" and a
//! bound as "unbounded".

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, TimestampMicrosecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_schema::{DataType, TimeUnit};
use iceberg::spec::{Datum, PrimitiveType};

/// The single value in `array` as an Iceberg value of `primitive`, or `None`
/// when the value has no such form: a null, an unsigned value above the
/// signed range, a NaN, a decimal of another scale than the column's, or an
/// Arrow type that is not a Pivot column's.
pub(crate) fn array_to_datum(array: &dyn Array, primitive: &PrimitiveType) -> Option<Datum> {
    if array.len() != 1 || array.is_null(0) {
        return None;
    }
    let datum = match (array.data_type(), primitive) {
        (DataType::Boolean, PrimitiveType::Boolean) => Datum::bool(array.as_boolean().value(0)),
        (DataType::Int8, PrimitiveType::Int) => {
            Datum::int(i32::from(array.as_primitive::<Int8Type>().value(0)))
        }
        (DataType::Int16, PrimitiveType::Int) => {
            Datum::int(i32::from(array.as_primitive::<Int16Type>().value(0)))
        }
        (DataType::Int32, PrimitiveType::Int) => {
            Datum::int(array.as_primitive::<Int32Type>().value(0))
        }
        (DataType::UInt8, PrimitiveType::Int) => {
            Datum::int(i32::from(array.as_primitive::<UInt8Type>().value(0)))
        }
        (DataType::UInt16, PrimitiveType::Int) => {
            Datum::int(i32::from(array.as_primitive::<UInt16Type>().value(0)))
        }
        (DataType::Int64, PrimitiveType::Long) => {
            Datum::long(array.as_primitive::<Int64Type>().value(0))
        }
        (DataType::UInt32, PrimitiveType::Long) => {
            Datum::long(i64::from(array.as_primitive::<UInt32Type>().value(0)))
        }
        (DataType::UInt64, PrimitiveType::Long) => {
            Datum::long(i64::try_from(array.as_primitive::<UInt64Type>().value(0)).ok()?)
        }
        (DataType::Float32, PrimitiveType::Float) => {
            let value = array.as_primitive::<Float32Type>().value(0);
            if value.is_nan() {
                return None;
            }
            Datum::float(value)
        }
        (DataType::Float64, PrimitiveType::Double) => {
            let value = array.as_primitive::<Float64Type>().value(0);
            if value.is_nan() {
                return None;
            }
            Datum::double(value)
        }
        (DataType::Utf8, PrimitiveType::String) => Datum::string(array.as_string::<i32>().value(0)),
        (DataType::LargeUtf8, PrimitiveType::String) => {
            Datum::string(array.as_string::<i64>().value(0))
        }
        (DataType::Utf8View, PrimitiveType::String) => {
            Datum::string(array.as_string_view().value(0))
        }
        (DataType::Date32, PrimitiveType::Date) => {
            Datum::date(array.as_primitive::<Date32Type>().value(0))
        }
        (DataType::Timestamp(TimeUnit::Microsecond, None), PrimitiveType::Timestamp) => {
            Datum::timestamp_micros(array.as_primitive::<TimestampMicrosecondType>().value(0))
        }
        (DataType::Timestamp(TimeUnit::Microsecond, Some(_)), PrimitiveType::Timestamptz) => {
            Datum::timestamptz_micros(array.as_primitive::<TimestampMicrosecondType>().value(0))
        }
        (DataType::Decimal64(_, arrow_scale), PrimitiveType::Decimal { scale, .. })
            if u32::try_from(*arrow_scale).ok() == Some(*scale) =>
        {
            let unscaled = i128::from(array.as_primitive::<Decimal64Type>().value(0));
            Datum::try_from_bytes(&unscaled.to_be_bytes(), primitive.clone()).ok()?
        }
        (DataType::Decimal128(_, arrow_scale), PrimitiveType::Decimal { scale, .. })
            if u32::try_from(*arrow_scale).ok() == Some(*scale) =>
        {
            let unscaled = array.as_primitive::<Decimal128Type>().value(0);
            Datum::try_from_bytes(&unscaled.to_be_bytes(), primitive.clone()).ok()?
        }
        _ => return None,
    };
    Some(datum)
}

/// The single value in `array` as an Iceberg value of the type the array's
/// own is written as, or `None` when the array's type is not one a Pivot
/// column has, or the value has no Iceberg form.
pub(crate) fn constant_to_datum(array: &dyn Array) -> Option<Datum> {
    let primitive = match array.data_type() {
        DataType::Boolean => PrimitiveType::Boolean,
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::UInt8 | DataType::UInt16 => {
            PrimitiveType::Int
        }
        DataType::Int64 | DataType::UInt32 | DataType::UInt64 => PrimitiveType::Long,
        DataType::Float32 => PrimitiveType::Float,
        DataType::Float64 => PrimitiveType::Double,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => PrimitiveType::String,
        DataType::Date32 => PrimitiveType::Date,
        DataType::Timestamp(TimeUnit::Microsecond, None) => PrimitiveType::Timestamp,
        DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => PrimitiveType::Timestamptz,
        DataType::Decimal64(precision, scale) | DataType::Decimal128(precision, scale) => {
            PrimitiveType::Decimal {
                precision: u32::from(*precision),
                scale: u32::try_from(*scale).ok()?,
            }
        }
        _ => return None,
    };
    array_to_datum(array, &primitive)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        ArrayRef, BooleanArray, Date32Array, Decimal64Array, Decimal128Array, Float32Array,
        Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeStringArray, StringArray,
        StringViewArray, TimestampMicrosecondArray, UInt8Array, UInt16Array, UInt32Array,
        UInt64Array,
    };

    use super::*;

    #[test]
    fn every_column_type_takes_the_arrow_values_a_column_of_it_is_written_as() {
        let decimal = PrimitiveType::Decimal {
            precision: 10,
            scale: 2,
        };
        let cases: Vec<(ArrayRef, PrimitiveType, Datum)> = vec![
            (
                Arc::new(BooleanArray::from(vec![true])),
                PrimitiveType::Boolean,
                Datum::bool(true),
            ),
            (
                Arc::new(Int8Array::from(vec![-3])),
                PrimitiveType::Int,
                Datum::int(-3),
            ),
            (
                Arc::new(Int16Array::from(vec![-300])),
                PrimitiveType::Int,
                Datum::int(-300),
            ),
            (
                Arc::new(Int32Array::from(vec![7])),
                PrimitiveType::Int,
                Datum::int(7),
            ),
            (
                Arc::new(UInt8Array::from(vec![200])),
                PrimitiveType::Int,
                Datum::int(200),
            ),
            (
                Arc::new(UInt16Array::from(vec![60_000])),
                PrimitiveType::Int,
                Datum::int(60_000),
            ),
            (
                Arc::new(Int64Array::from(vec![1_i64 << 40])),
                PrimitiveType::Long,
                Datum::long(1_i64 << 40),
            ),
            (
                Arc::new(UInt32Array::from(vec![u32::MAX])),
                PrimitiveType::Long,
                Datum::long(i64::from(u32::MAX)),
            ),
            (
                Arc::new(UInt64Array::from(vec![7])),
                PrimitiveType::Long,
                Datum::long(7),
            ),
            (
                Arc::new(Float32Array::from(vec![1.5])),
                PrimitiveType::Float,
                Datum::float(1.5_f32),
            ),
            (
                Arc::new(Float64Array::from(vec![2.5])),
                PrimitiveType::Double,
                Datum::double(2.5),
            ),
            (
                Arc::new(StringArray::from(vec!["a"])),
                PrimitiveType::String,
                Datum::string("a"),
            ),
            (
                Arc::new(LargeStringArray::from(vec!["b"])),
                PrimitiveType::String,
                Datum::string("b"),
            ),
            (
                Arc::new(StringViewArray::from(vec!["c"])),
                PrimitiveType::String,
                Datum::string("c"),
            ),
            (
                Arc::new(Date32Array::from(vec![15])),
                PrimitiveType::Date,
                Datum::date(15),
            ),
            (
                Arc::new(TimestampMicrosecondArray::from(vec![15])),
                PrimitiveType::Timestamp,
                Datum::timestamp_micros(15),
            ),
            (
                Arc::new(TimestampMicrosecondArray::from(vec![15]).with_timezone("UTC")),
                PrimitiveType::Timestamptz,
                Datum::timestamptz_micros(15),
            ),
            (
                Arc::new(
                    Decimal64Array::from(vec![-1234i64])
                        .with_precision_and_scale(10, 2)
                        .unwrap(),
                ),
                decimal.clone(),
                Datum::try_from_bytes(&(-1234i128).to_be_bytes(), decimal.clone()).unwrap(),
            ),
            (
                Arc::new(
                    Decimal128Array::from(vec![-1234i128])
                        .with_precision_and_scale(10, 2)
                        .unwrap(),
                ),
                decimal.clone(),
                Datum::try_from_bytes(&(-1234i128).to_be_bytes(), decimal).unwrap(),
            ),
        ];

        for (array, column_type, expected) in cases {
            let datum = array_to_datum(array.as_ref(), &column_type);

            assert_eq!(datum, Some(expected), "{array:?} as {column_type}");
        }
    }

    #[test]
    fn a_value_the_column_cannot_hold_has_no_datum() {
        let rejected: Vec<(ArrayRef, PrimitiveType)> = vec![
            (Arc::new(Int64Array::from(vec![None])), PrimitiveType::Long),
            (
                Arc::new(UInt64Array::from(vec![u64::MAX])),
                PrimitiveType::Long,
            ),
            (
                Arc::new(Float32Array::from(vec![f32::NAN])),
                PrimitiveType::Float,
            ),
            (
                Arc::new(Float64Array::from(vec![f64::NAN])),
                PrimitiveType::Double,
            ),
            (Arc::new(Int64Array::from(vec![7])), PrimitiveType::Int),
            (
                Arc::new(StringViewArray::from(vec!["seven"])),
                PrimitiveType::Long,
            ),
            (Arc::new(Int32Array::from(vec![1, 2])), PrimitiveType::Int),
        ];

        for (array, column_type) in rejected {
            assert_eq!(
                array_to_datum(array.as_ref(), &column_type),
                None,
                "{array:?} as {column_type}"
            );
        }
    }

    #[test]
    fn a_constant_is_read_as_the_type_its_array_is_written_as() {
        let wide = Int64Array::from(vec![i64::from(i32::MAX) + 1]);
        let priced = Decimal128Array::from(vec![5000i128])
            .with_precision_and_scale(10, 2)
            .unwrap();

        let long = constant_to_datum(&wide);
        let decimal = constant_to_datum(&priced);
        let nan = constant_to_datum(&Float64Array::from(vec![f64::NAN]));

        assert_eq!(long, Some(Datum::long(i64::from(i32::MAX) + 1)));
        assert_eq!(decimal.unwrap().to_string(), "50.00");
        assert_eq!(nan, None);
    }

    #[test]
    fn a_decimal_of_another_scale_than_the_columns_has_no_datum() {
        let column = PrimitiveType::Decimal {
            precision: 10,
            scale: 2,
        };
        let other_scale = Decimal128Array::from(vec![-1234i128])
            .with_precision_and_scale(10, 3)
            .unwrap();

        let datum = array_to_datum(&other_scale, &column);

        assert_eq!(datum, None);
    }
}
