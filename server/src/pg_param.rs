//! Decoding bound prepared-statement parameters straight into Arrow arrays.
//!
//! The extended query protocol carries a statement's parameter values in the
//! Bind message, each as raw bytes plus a text/binary format code. For the
//! prepared `INSERT ... VALUES ($1, …)` path the server decodes each bound value
//! directly into a length-1 Arrow array of the target column's type, with no
//! intermediate scalar or expression evaluation, and assembles the row into
//! one `RecordBatch` handed to the table's insert sink.

use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Date32Array, Float32Array, Float64Array, Int8Array, Int16Array,
    Int32Array, Int64Array, Scalar, StringArray, StringViewArray, TimestampSecondArray, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::DataType;
use chrono::{NaiveDate, NaiveDateTime};
use pgwire::api::Type;
use pgwire::api::portal::Portal;
use pgwire::api::results::FieldInfo;
use pgwire::error::{ErrorInfo, PgWireError, PgWireResult};

use crate::arrow_to_pgwire::pg_type_for_arrow;

/// Decode the `param_index`th bound parameter of `portal` into a length-1 Arrow
/// [`Scalar`] of `arrow_type` (the parameter's resolved type). pgwire reads the
/// value in whichever format (text/binary) the client bound it in; a NULL bound
/// value yields a one-element null array. The scalar is the "single value
/// pointer" a bound `Parameter` becomes, and the cell a prepared `VALUES`
/// gathers.
pub(crate) fn bind_param<S: Clone>(
    portal: &Portal<S>,
    param_index: usize,
    arrow_type: &DataType,
) -> PgWireResult<Scalar<ArrayRef>> {
    Ok(Scalar::new(bind_value_to_array(
        portal,
        param_index,
        arrow_type,
    )?))
}

fn bind_value_to_array<S: Clone>(
    portal: &Portal<S>,
    param_index: usize,
    arrow_type: &DataType,
) -> PgWireResult<ArrayRef> {
    let pg_type = pg_type_for_arrow(arrow_type);
    // Read one bound parameter as `rust_ty` and wrap it in a length-1 array of
    // `array_ty` (arrow's `From<Vec<Option<T>>>` fills a null for `None`).
    macro_rules! primitive {
        ($rust_ty:ty, $array_ty:ty) => {{
            let value: Option<$rust_ty> = portal.parameter::<$rust_ty>(param_index, &pg_type)?;
            Arc::new(<$array_ty>::from(vec![value])) as ArrayRef
        }};
    }
    macro_rules! converted {
        ($wire_ty:ty, $value_ty:ty, $array_ty:ty) => {{
            let value: Option<$wire_ty> = portal.parameter::<$wire_ty>(param_index, &pg_type)?;
            let value: Option<$value_ty> = value
                .map(|value| {
                    <$value_ty>::try_from(value)
                        .map_err(|error| invalid_param_value(param_index, error.to_string()))
                })
                .transpose()?;
            Arc::new(<$array_ty>::from(vec![value])) as ArrayRef
        }};
    }
    let array = match arrow_type {
        DataType::Boolean => primitive!(bool, BooleanArray),
        DataType::Int8 => converted!(i16, i8, Int8Array),
        DataType::Int16 => primitive!(i16, Int16Array),
        DataType::Int32 => primitive!(i32, Int32Array),
        DataType::Int64 => primitive!(i64, Int64Array),
        DataType::UInt8 => converted!(i16, u8, UInt8Array),
        DataType::UInt16 => converted!(i32, u16, UInt16Array),
        DataType::UInt32 => converted!(i64, u32, UInt32Array),
        DataType::UInt64 => {
            let value: Option<String> = portal.parameter::<String>(param_index, &pg_type)?;
            let value = value
                .map(|value| {
                    value
                        .parse::<u64>()
                        .map_err(|error| invalid_param_value(param_index, error.to_string()))
                })
                .transpose()?;
            Arc::new(UInt64Array::from(vec![value])) as ArrayRef
        }
        DataType::Float32 => primitive!(f32, Float32Array),
        DataType::Float64 => primitive!(f64, Float64Array),
        // Pivot stores strings as either Utf8 or (the planner's default) Utf8View.
        DataType::Utf8 => {
            let value: Option<String> = portal.parameter::<String>(param_index, &pg_type)?;
            Arc::new(StringArray::from_iter(std::iter::once(value))) as ArrayRef
        }
        DataType::Utf8View => {
            let value: Option<String> = portal.parameter::<String>(param_index, &pg_type)?;
            Arc::new(StringViewArray::from_iter(std::iter::once(value))) as ArrayRef
        }
        DataType::Date32 => {
            let value: Option<NaiveDate> = portal.parameter(param_index, &pg_type)?;
            let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
            let value = value
                .map(|date| {
                    i32::try_from(date.signed_duration_since(epoch).num_days())
                        .map_err(|error| invalid_param_value(param_index, error.to_string()))
                })
                .transpose()?;
            Arc::new(Date32Array::from(vec![value])) as ArrayRef
        }
        DataType::Timestamp(arrow_schema::TimeUnit::Second, None) => {
            let value: Option<NaiveDateTime> = portal.parameter(param_index, &pg_type)?;
            Arc::new(TimestampSecondArray::from(vec![
                value.map(|datetime| datetime.and_utc().timestamp()),
            ])) as ArrayRef
        }
        other => {
            return Err(unsupported_param_type(other));
        }
    };
    Ok(array)
}

fn invalid_param_value(param_index: usize, message: String) -> PgWireError {
    PgWireError::UserError(Box::new(ErrorInfo::new(
        "ERROR".to_string(),
        "22003".to_string(),
        format!(
            "invalid value for parameter ${}: {message}",
            param_index + 1
        ),
    )))
}

fn unsupported_param_type(arrow_type: &DataType) -> PgWireError {
    PgWireError::UserError(Box::new(ErrorInfo::new(
        "ERROR".to_string(),
        // feature_not_supported
        "0A000".to_string(),
        format!("bound parameter of type {arrow_type} is not supported"),
    )))
}

/// The Postgres type OID a bound parameter of `arrow_type` is decoded against,
/// used to build the statement's `ParameterDescription`.
pub(crate) fn pg_type_for_param(arrow_type: &DataType) -> Type {
    pg_type_for_arrow(arrow_type)
}

/// Build the `RowDescription` fields for a result-producing plan from its
/// output column names and Arrow types. The statement-describe row description
/// is advertised in text; the actual result format is applied per Execute from
/// the portal's requested formats.
pub(crate) fn result_fields(names: &[String], arrow_types: &[DataType]) -> Arc<Vec<FieldInfo>> {
    use arrow_schema::{Field, Schema};
    use pgwire::api::portal::Format;
    let fields: Vec<Field> = arrow_types
        .iter()
        .enumerate()
        .map(|(i, arrow_type)| {
            let name = names.get(i).cloned().unwrap_or_else(|| format!("col{i}"));
            Field::new(name, arrow_type.clone(), true)
        })
        .collect();
    crate::arrow_to_pgwire::build_field_info_with_format(
        &Arc::new(Schema::new(fields)),
        &Format::UnifiedText,
    )
}
