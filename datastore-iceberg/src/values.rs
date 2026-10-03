//! Iceberg manifest literals as arrays of the table's physical Arrow types.

use std::sync::Arc;

use arrow_array::builder::{PrimitiveBuilder, StringViewBuilder};
use arrow_array::types::{
    ArrowPrimitiveType, Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type,
    Int32Type, Int64Type, TimestampMicrosecondType,
};
use arrow_array::{ArrayRef, BooleanArray, new_null_array};
use arrow_schema::{DataType, TimeUnit};
use iceberg::spec::PrimitiveLiteral;

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
