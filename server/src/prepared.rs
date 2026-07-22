//! Prepared-statement metadata and pgwire Bind value decoding.

use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Date32Array, Float32Array, Float64Array, Int8Array, Int16Array,
    Int32Array, Int64Array, Scalar, StringArray, TimestampSecondArray, UInt32Array,
};
use chrono::{DateTime, FixedOffset, NaiveDate, NaiveDateTime};
use pgwire::api::Type as PgType;
use pgwire::api::portal::{Format, Portal};
use pgwire::api::results::{FieldFormat, FieldInfo};
use pgwire::error::{ErrorInfo, PgWireError, PgWireResult};

use crate::arrow_to_pgwire::pg_type_for_planner;

/// The reusable Parse product kept in pgwire's per-connection statement store.
#[derive(Debug, Clone)]
pub struct PreparedStatement {
    pub sql: String,
    pub plan: Arc<planner::Plan>,
    pub output_types: Vec<planner::types::Type>,
}

impl PreparedStatement {
    pub fn new(sql: String, plan: Arc<planner::Plan>) -> Result<Self, planner::compile::Error> {
        let is_insert = matches!(&plan.root.operator, planner::Operator::Insert(_));
        let output_types = if is_insert {
            Vec::new()
        } else {
            plan.root.output_types()?
        };
        Ok(Self {
            sql,
            output_types,
            plan,
        })
    }

    pub fn pg_parameter_types(&self) -> Vec<PgType> {
        self.plan
            .parameter_types
            .iter()
            .map(pg_type_for_planner)
            .collect()
    }

    pub fn result_fields(&self, format: Option<&Format>) -> Vec<FieldInfo> {
        self.output_types
            .iter()
            .enumerate()
            .map(|(index, ty)| {
                FieldInfo::new(
                    self.plan
                        .output_names
                        .get(index)
                        .cloned()
                        .unwrap_or_else(|| format!("col{index}")),
                    None,
                    None,
                    pg_type_for_planner(ty),
                    format.map_or(FieldFormat::Text, |format| format.format_for(index)),
                )
            })
            .collect()
    }
}

/// Decode a portal's raw text/binary fields and cast each length-one array to
/// the type DuckDB fixed in the cached plan.
pub fn decode_parameters(
    portal: &Portal<PreparedStatement>,
) -> PgWireResult<Vec<Scalar<ArrayRef>>> {
    let statement = &portal.statement.statement;
    if portal.parameter_len() != statement.plan.parameter_types.len() {
        return Err(protocol_error(format!(
            "prepared statement expects {} parameters, Bind supplied {}",
            statement.plan.parameter_types.len(),
            portal.parameter_len()
        )));
    }

    statement
        .plan
        .parameter_types
        .iter()
        .enumerate()
        .map(|(index, target)| {
            let target_type = planner::types::physical_arrow_type(target);
            if portal.parameters[index].is_none() {
                return Ok(Scalar::new(arrow_array::new_null_array(&target_type, 1)));
            }
            let inferred_type = pg_type_for_planner(target);
            let wire_type = portal
                .statement
                .parameter_types
                .get(index)
                .and_then(Option::as_ref)
                .unwrap_or(&inferred_type);
            let source = decode_one(portal, index, wire_type)?;
            let value = if source.data_type() == &target_type {
                source
            } else {
                arrow::compute::cast(&source, &target_type).map_err(|error| {
                    data_error(format!("cannot bind parameter ${}: {error}", index + 1))
                })?
            };
            Ok(Scalar::new(value))
        })
        .collect()
}

fn decode_one(
    portal: &Portal<PreparedStatement>,
    index: usize,
    ty: &PgType,
) -> PgWireResult<ArrayRef> {
    macro_rules! scalar {
        ($rust:ty, $array:ty) => {{
            let value = portal.parameter::<$rust>(index, ty)?;
            Arc::new(<$array>::from(vec![value])) as ArrayRef
        }};
    }

    Ok(match *ty {
        PgType::BOOL => scalar!(bool, BooleanArray),
        PgType::CHAR => scalar!(i8, Int8Array),
        PgType::INT2 => scalar!(i16, Int16Array),
        PgType::INT4 => scalar!(i32, Int32Array),
        PgType::INT8 => scalar!(i64, Int64Array),
        PgType::OID => scalar!(u32, UInt32Array),
        PgType::FLOAT4 => scalar!(f32, Float32Array),
        PgType::FLOAT8 => scalar!(f64, Float64Array),
        PgType::TEXT | PgType::VARCHAR | PgType::BPCHAR | PgType::NAME | PgType::UNKNOWN => {
            let value = portal.parameter::<String>(index, ty)?;
            Arc::new(StringArray::from(vec![value]))
        }
        PgType::DATE => {
            let value = portal.parameter::<NaiveDate>(index, ty)?;
            let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
            let days = value.map(|date| date.signed_duration_since(epoch).num_days() as i32);
            Arc::new(Date32Array::from(vec![days]))
        }
        PgType::TIMESTAMP => {
            let value = portal.parameter::<NaiveDateTime>(index, ty)?;
            Arc::new(TimestampSecondArray::from(vec![
                value.map(|value| value.and_utc().timestamp()),
            ]))
        }
        PgType::TIMESTAMPTZ => {
            let value = portal.parameter::<DateTime<FixedOffset>>(index, ty)?;
            Arc::new(TimestampSecondArray::from(vec![
                value.map(|value| value.timestamp()),
            ]))
        }
        PgType::NUMERIC => {
            let value = portal.parameter::<rust_decimal::Decimal>(index, ty)?;
            Arc::new(StringArray::from(vec![
                value.map(|value| value.to_string()),
            ]))
        }
        PgType::JSON | PgType::JSONB => {
            let value = portal.parameter::<serde_json::Value>(index, ty)?;
            Arc::new(StringArray::from(vec![
                value.map(|value| value.to_string()),
            ]))
        }
        _ => {
            return Err(data_error(format!(
                "Postgres parameter type {} is not supported",
                ty.name()
            )));
        }
    })
}

fn protocol_error(message: String) -> PgWireError {
    PgWireError::UserError(Box::new(ErrorInfo::new(
        "ERROR".to_string(),
        "08P01".to_string(),
        message,
    )))
}

fn data_error(message: String) -> PgWireError {
    PgWireError::UserError(Box::new(ErrorInfo::new(
        "ERROR".to_string(),
        "22P02".to_string(),
        message,
    )))
}
