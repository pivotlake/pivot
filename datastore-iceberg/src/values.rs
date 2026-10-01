//! Iceberg values as Arrow arrays, so a predicate's constant compares against
//! every file in one kernel, and a predicate's constant as an Iceberg datum,
//! which a partition transform can project.

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
use planner::types::{Type, physical_arrow_type};

/// `literals` as an array of `value_type`'s physical Arrow type, one slot
/// per literal. Null where there is no literal or it is not of that type. A
/// type Pivot does not compare gives all nulls.
pub(crate) fn build_array<'a>(
    value_type: &Type,
    literals: impl Iterator<Item = Option<&'a PrimitiveLiteral>>,
) -> ArrayRef {
    let data_type = physical_arrow_type(value_type);
    match value_type {
        Type::Boolean => Arc::new(BooleanArray::from_iter(literals.map(
            |literal| match literal {
                Some(PrimitiveLiteral::Boolean(value)) => Some(*value),
                _ => None,
            },
        ))),
        Type::Int32 => primitive_array::<Int32Type>(
            data_type,
            literals.map(|literal| match literal {
                Some(PrimitiveLiteral::Int(value)) => Some(*value),
                _ => None,
            }),
        ),
        Type::Date => primitive_array::<Date32Type>(
            data_type,
            literals.map(|literal| match literal {
                Some(PrimitiveLiteral::Int(days)) => Some(*days),
                _ => None,
            }),
        ),
        Type::Int64 => primitive_array::<Int64Type>(
            data_type,
            literals.map(|literal| match literal {
                Some(PrimitiveLiteral::Long(value)) => Some(*value),
                _ => None,
            }),
        ),
        Type::Timestamp | Type::TimestampTz => primitive_array::<TimestampMicrosecondType>(
            data_type,
            literals.map(|literal| match literal {
                Some(PrimitiveLiteral::Long(micros)) => Some(*micros),
                _ => None,
            }),
        ),
        Type::Float32 => primitive_array::<Float32Type>(
            data_type,
            literals.map(|literal| match literal {
                Some(PrimitiveLiteral::Float(value)) => Some(value.0),
                _ => None,
            }),
        ),
        Type::Float64 => primitive_array::<Float64Type>(
            data_type,
            literals.map(|literal| match literal {
                Some(PrimitiveLiteral::Double(value)) => Some(value.0),
                _ => None,
            }),
        ),
        Type::Decimal { .. } if matches!(data_type, DataType::Decimal64(..)) => {
            primitive_array::<Decimal64Type>(
                data_type,
                literals.map(|literal| match literal {
                    Some(PrimitiveLiteral::Int128(unscaled)) => i64::try_from(*unscaled).ok(),
                    _ => None,
                }),
            )
        }
        Type::Decimal { .. } => primitive_array::<Decimal128Type>(
            data_type,
            literals.map(|literal| match literal {
                Some(PrimitiveLiteral::Int128(unscaled)) => Some(*unscaled),
                _ => None,
            }),
        ),
        Type::Utf8 => {
            let mut builder = StringViewBuilder::new();
            for literal in literals {
                builder.append_option(match literal {
                    Some(PrimitiveLiteral::String(value)) => Some(value.as_str()),
                    _ => None,
                });
            }
            Arc::new(builder.finish())
        }
        _ => new_null_array(&data_type, literals.count()),
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

/// `literal` as a one-value scalar of `value_type`, or `None` when it is not
/// of that type.
pub(crate) fn build_scalar(
    value_type: &Type,
    literal: &PrimitiveLiteral,
) -> Option<Scalar<ArrayRef>> {
    let array = build_array(value_type, [Some(literal)].into_iter());
    array.is_valid(0).then(|| Scalar::new(array))
}

/// A predicate constant as an Iceberg datum of the same value, which a
/// transform can project. `None` when it is NULL or of a type Iceberg has no
/// primitive for.
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

/// A decimal datum, built the way a manifest stores one: from the unscaled
/// value's big-endian bytes.
fn decimal_datum(unscaled: i128, precision: u8, scale: i8) -> Option<Datum> {
    let data_type = PrimitiveType::Decimal {
        precision: u32::from(precision),
        scale: u32::try_from(scale).ok()?,
    };
    Datum::try_from_bytes(&unscaled.to_be_bytes(), data_type).ok()
}

#[cfg(test)]
mod tests {
    use arrow_array::{Decimal128Array, Int64Array, StringViewArray};

    use super::*;

    /// A decimal manifest bound, as the manifest stores it: the unscaled
    /// value's big-endian bytes.
    fn decimal_bound(unscaled: i128, precision: u32, scale: u32) -> Datum {
        Datum::try_from_bytes(
            &unscaled.to_be_bytes(),
            PrimitiveType::Decimal { precision, scale },
        )
        .unwrap()
    }

    #[test]
    fn manifest_bounds_become_arrays_of_the_columns_physical_type() {
        let bounds = [
            (Datum::bool(true), Type::Boolean),
            (Datum::int(7), Type::Int32),
            (Datum::date(19_000), Type::Date),
            (Datum::long(7), Type::Int64),
            (Datum::timestamp_micros(7), Type::Timestamp),
            (Datum::timestamptz_micros(7), Type::TimestampTz),
            (Datum::float(1.5_f32), Type::Float32),
            (Datum::double(1.5), Type::Float64),
            (
                decimal_bound(1234, 10, 2),
                Type::Decimal {
                    precision: 10,
                    scale: 2,
                },
            ),
            (
                decimal_bound(1234, 30, 2),
                Type::Decimal {
                    precision: 30,
                    scale: 2,
                },
            ),
            (Datum::string("ann"), Type::Utf8),
        ];

        for (datum, value_type) in bounds {
            let array = build_array(&value_type, [Some(datum.literal()), None].into_iter());

            assert_eq!(
                array.data_type(),
                &physical_arrow_type(&value_type),
                "{value_type:?}"
            );
            assert_eq!(array.len(), 2);
            assert!(array.is_valid(0) && array.is_null(1), "{value_type:?}");
        }
    }

    #[test]
    fn a_bound_of_another_type_is_a_missing_bound() {
        let array = build_array(
            &Type::Int64,
            [Some(Datum::string("seven").literal())].into_iter(),
        );

        assert!(array.is_null(0));
        assert!(build_scalar(&Type::Int64, Datum::string("seven").literal()).is_none());
    }

    #[test]
    fn a_constant_round_trips_through_its_datum() {
        let constants: Vec<(ArrayRef, Type)> = vec![
            (Arc::new(Int64Array::from(vec![7])), Type::Int64),
            (Arc::new(StringViewArray::from(vec!["ann"])), Type::Utf8),
            (
                Arc::new(
                    Decimal128Array::from(vec![1234])
                        .with_precision_and_scale(30, 2)
                        .unwrap(),
                ),
                Type::Decimal {
                    precision: 30,
                    scale: 2,
                },
            ),
        ];

        for (constant, value_type) in constants {
            let datum = to_datum(&Scalar::new(constant.clone())).expect("a datum of the type");

            let back = build_scalar(&value_type, datum.literal()).expect("the same value");
            assert_eq!(back.get().0, constant.as_ref());
        }
    }

    #[test]
    fn a_null_constant_has_no_datum() {
        let constant: ArrayRef = Arc::new(Int64Array::from(vec![None::<i64>]));

        assert!(to_datum(&Scalar::new(constant)).is_none());
    }
}
