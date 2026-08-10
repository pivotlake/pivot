//! Extended-query-protocol prepared statements: the statement the wire layer
//! stores per Parse, the parser that plans it, and the decoding of bound
//! parameter values (text and binary formats) into planner scalars.

use std::sync::{Arc, Mutex};

use arrow::compute::kernels::cast_utils::Parser;
use arrow::datatypes::{Date32Type, TimestampMicrosecondType};
use async_trait::async_trait;
use chrono::{NaiveDate, NaiveDateTime};
use pgwire::api::portal::{Format, Portal};
use pgwire::api::results::FieldInfo;
use pgwire::api::stmt::QueryParser;
use pgwire::api::{ClientInfo, Type};
use pgwire::error::{ErrorInfo, PgWireError, PgWireResult};
use planner::ScalarValue;
use planner::types::{Type as PivotType, logical_from_type, physical_arrow_type};
use postgres_types::FromSql;
use tracing::info;

use crate::arrow_to_pgwire::{numeric_binary_to_string, pg_type_for_arrow};
use crate::query_handler::{planner_error_to_pgwire, with_planner};

/// One prepared statement, planned at Parse time. Cheap to clone (pgwire
/// requires `Clone`); every field is shared.
#[derive(Debug, Clone)]
pub struct PreparedStatement {
    pub sql: Arc<str>,
    /// The inferred pivot types of `$1..$n`, driving parameter decoding.
    pub param_types: Arc<Vec<PivotType>>,
    /// The result columns (name and Postgres type) a Describe advertises.
    /// Empty for a statement that returns a command tag instead of rows.
    pub output_columns: Arc<Vec<(String, Type)>>,
    /// The reusable placeholder plan, when the statement qualifies for one
    /// (see [`planner::Planner::plan_prepare`]). Cleared when an execution
    /// discovers the target's schema drifted; executions replan with their
    /// values from then on, until the client prepares the statement again.
    pub cached_plan: Arc<Mutex<Option<Arc<planner::Plan>>>>,
}

impl PreparedStatement {
    /// An empty statement (`""` or `";"`) executes as `EmptyQueryResponse`.
    pub fn is_empty(&self) -> bool {
        matches!(self.sql.trim(), "" | ";")
    }

    /// Take the cached placeholder plan, if the statement still holds one.
    pub fn plan(&self) -> Option<Arc<planner::Plan>> {
        self.cached_plan.lock().unwrap().clone()
    }

    /// Drop the cached placeholder plan after an execution found it stale.
    pub fn clear_plan(&self) {
        self.cached_plan.lock().unwrap().take();
    }
}

/// The Postgres type a pivot-typed value is presented as on the wire.
fn pg_type_for(ty: &PivotType) -> Type {
    pg_type_for_arrow(&physical_arrow_type(ty))
}

/// Plans each Parse'd statement through the shared planner in prepare mode.
pub struct PivotQueryParser {
    catalog: Arc<catalog::PivotCatalog>,
}

impl PivotQueryParser {
    pub fn new(catalog: Arc<catalog::PivotCatalog>) -> Self {
        Self { catalog }
    }
}

#[async_trait]
impl QueryParser for PivotQueryParser {
    type Statement = PreparedStatement;

    async fn parse_sql<C>(
        &self,
        _client: &C,
        sql: &str,
        _types: &[Option<Type>],
    ) -> PgWireResult<PreparedStatement>
    where
        C: ClientInfo + Unpin + Send + Sync,
    {
        let statement_shell = |sql: &str| PreparedStatement {
            sql: Arc::from(sql),
            param_types: Arc::new(Vec::new()),
            output_columns: Arc::new(Vec::new()),
            cached_plan: Arc::new(Mutex::new(None)),
        };
        let trimmed = sql.trim();
        if trimmed.is_empty() || trimmed == ";" {
            return Ok(statement_shell(sql));
        }
        info!(%sql, "prepared statement parsed");

        // The transaction only provides the catalog snapshot planning binds
        // against; nothing is staged, so it is released by rollback either way.
        let transaction = self.catalog.begin_transaction();
        let catalog = self.catalog.clone();
        let owned_sql = sql.to_string();
        let planning_transaction = transaction.clone();
        let prepared = tokio::task::spawn_blocking(move || {
            with_planner(&catalog, |planner| {
                planner.plan_prepare(&owned_sql, planning_transaction)
            })?
        })
        .await
        .map_err(|e| PgWireError::ApiError(format!("planner thread panicked: {e}").into()))?;
        transaction.rollback();
        let prepared = prepared.map_err(planner_error_to_pgwire)?;

        let output_columns = if prepared.returns_rows {
            prepared
                .output_names
                .iter()
                .cloned()
                .zip(prepared.output_types.iter().map(pg_type_for))
                .collect()
        } else {
            // A command (INSERT, SET, ...) answers a Describe with no row
            // shape; its execution sends a command tag.
            Vec::new()
        };
        Ok(PreparedStatement {
            param_types: Arc::new(prepared.param_types),
            output_columns: Arc::new(output_columns),
            cached_plan: Arc::new(Mutex::new(prepared.plan.map(Arc::new))),
            ..statement_shell(sql)
        })
    }

    fn get_parameter_types(&self, stmt: &PreparedStatement) -> PgWireResult<Vec<Type>> {
        Ok(stmt.param_types.iter().map(pg_type_for).collect())
    }

    fn get_result_schema(
        &self,
        stmt: &PreparedStatement,
        column_format: Option<&Format>,
    ) -> PgWireResult<Vec<FieldInfo>> {
        if let Some(Format::Individual(codes)) = column_format
            && codes.len() < stmt.output_columns.len()
        {
            return Err(PgWireError::UserError(Box::new(ErrorInfo::new(
                "ERROR".to_string(),
                // Protocol violation: the Bind's format-code list is short.
                "08P01".to_string(),
                format!(
                    "the result has {} columns but only {} format codes were bound",
                    stmt.output_columns.len(),
                    codes.len()
                ),
            ))));
        }
        Ok(stmt
            .output_columns
            .iter()
            .enumerate()
            .map(|(i, (name, ty))| {
                let format = match column_format {
                    Some(format) => format.format_for(i),
                    None => pgwire::api::results::FieldFormat::Text,
                };
                FieldInfo::new(name.clone(), None, None, ty.clone(), format)
            })
            .collect())
    }
}

/// Decode a portal's bound parameter values into planner scalars, one per
/// `$n`, ordered. Every failure is an error: a value that cannot be decoded
/// as its parameter's type must not silently reach the plan.
pub(crate) fn decode_parameters(
    portal: &Portal<PreparedStatement>,
) -> Result<Vec<ScalarValue>, String> {
    let statement = &portal.statement.statement;
    let expected = statement.param_types.len();
    if portal.parameters.len() != expected {
        return Err(format!(
            "statement takes {expected} parameters but {} were bound",
            portal.parameters.len()
        ));
    }

    if let Format::Individual(codes) = &portal.parameter_format
        && codes.len() < expected
    {
        return Err(format!(
            "{expected} parameters were bound but only {} format codes were sent",
            codes.len()
        ));
    }

    let declared_oids = &portal.statement.parameter_types;
    let mut values = Vec::with_capacity(expected);
    for (i, bytes) in portal.parameters.iter().enumerate() {
        let target = &statement.param_types[i];
        let value = match bytes {
            None => ScalarValue::Null(logical_from_type(target)),
            Some(bytes) if portal.parameter_format.is_binary(i) => {
                // The client encoded by the OID it believes the parameter has:
                // what it declared at Parse, or what our Describe answered
                // (which matches the inferred type). The decoded value then
                // fits to the inferred type.
                let oid = declared_oids
                    .get(i)
                    .cloned()
                    .flatten()
                    .unwrap_or_else(|| pg_type_for(target));
                let decoded = decode_binary(bytes, &oid)?;
                coerce(decoded, target).map_err(|e| format!("parameter ${}: {e}", i + 1))?
            }
            // A text value parses directly as the inferred type; there is no
            // wire type to fit from.
            Some(bytes) => {
                let text = std::str::from_utf8(bytes)
                    .map_err(|_| format!("parameter ${} is not valid UTF-8", i + 1))?;
                decode_text(text, target)?
            }
        };
        values.push(value);
    }
    Ok(values)
}

/// Parse a text-format parameter directly as its inferred type, the way a
/// Postgres server reads text input through the target type's `in` function.
fn decode_text(text: &str, target: &PivotType) -> Result<ScalarValue, String> {
    let parse_int = |what: &str| {
        text.parse::<i128>()
            .map_err(|_| format!("`{text}` is not a valid {what}"))
    };
    Ok(match target {
        PivotType::Boolean => match text.to_ascii_lowercase().as_str() {
            "t" | "true" | "1" | "yes" | "on" => ScalarValue::Boolean(true),
            "f" | "false" | "0" | "no" | "off" => ScalarValue::Boolean(false),
            _ => return Err(format!("`{text}` is not a valid boolean")),
        },
        PivotType::Int8 | PivotType::Int16 | PivotType::Int32 | PivotType::Int64 => {
            let value = parse_int("integer")?;
            narrow_int(value, target)?
        }
        PivotType::UInt8 | PivotType::UInt16 | PivotType::UInt32 | PivotType::UInt64 => {
            let value = parse_int("integer")?;
            narrow_int(value, target)?
        }
        PivotType::Int128 => ScalarValue::Int128(parse_int("integer")?),
        PivotType::Float32 => ScalarValue::Float32(
            text.parse()
                .map_err(|_| format!("`{text}` is not a valid real"))?,
        ),
        PivotType::Float64 => ScalarValue::Float64(
            text.parse()
                .map_err(|_| format!("`{text}` is not a valid double"))?,
        ),
        PivotType::Decimal { precision, scale } => parse_decimal(text, *precision, *scale)?,
        PivotType::Utf8 => ScalarValue::Utf8(text.to_string()),
        // Arrow's temporal parsers accept the ISO forms clients send,
        // including a timezone offset (converted to UTC), which drivers
        // binding tz-aware datetimes in text format rely on.
        PivotType::Date => ScalarValue::Date(
            Date32Type::parse(text).ok_or_else(|| format!("`{text}` is not a valid date"))?,
        ),
        PivotType::Timestamp => ScalarValue::Timestamp(
            TimestampMicrosecondType::parse(text)
                .ok_or_else(|| format!("`{text}` is not a valid timestamp"))?,
        ),
        PivotType::Variant => {
            return Err("VARIANT parameters are not supported".to_string());
        }
    })
}

/// Decode a binary-format parameter by the Postgres wire encoding of `oid`,
/// through postgres-types' own decoders (which also reject out-of-range
/// values such as the 'infinity' date/timestamp sentinels).
fn decode_binary(bytes: &[u8], oid: &Type) -> Result<ScalarValue, String> {
    fn from_sql<'a, T: FromSql<'a>>(oid: &Type, bytes: &'a [u8]) -> Result<T, String> {
        T::from_sql(oid, bytes).map_err(|e| format!("decoding a binary {oid}: {e}"))
    }
    Ok(match *oid {
        Type::BOOL => ScalarValue::Boolean(from_sql(oid, bytes)?),
        Type::INT2 => ScalarValue::Int16(from_sql(oid, bytes)?),
        Type::INT4 => ScalarValue::Int32(from_sql(oid, bytes)?),
        Type::INT8 => ScalarValue::Int64(from_sql(oid, bytes)?),
        Type::FLOAT4 => ScalarValue::Float32(from_sql(oid, bytes)?),
        Type::FLOAT8 => ScalarValue::Float64(from_sql(oid, bytes)?),
        Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME | Type::UNKNOWN => {
            ScalarValue::Utf8(from_sql(oid, bytes)?)
        }
        Type::DATE => {
            let date: NaiveDate = from_sql(oid, bytes)?;
            ScalarValue::Date(
                (date - NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()).num_days() as i32,
            )
        }
        Type::TIMESTAMP => {
            let timestamp: NaiveDateTime = from_sql(oid, bytes)?;
            ScalarValue::Timestamp(timestamp.and_utc().timestamp_micros())
        }
        Type::NUMERIC => ScalarValue::Utf8(numeric_binary_to_string(bytes)?),
        ref other => {
            return Err(format!(
                "binary parameters of type {other} are not supported"
            ));
        }
    })
}

/// Fit a binary-decoded value (typed by its wire OID) to the parameter's
/// inferred type. Integers widen (and narrow with a range check), a float
/// widens, and a textual value re-parses as the target; any other pairing is
/// an error, not a cast. Text-format parameters never come through here: they
/// parse as the inferred type directly.
fn coerce(value: ScalarValue, target: &PivotType) -> Result<ScalarValue, String> {
    let as_i128 = |value: &ScalarValue| -> Option<i128> {
        Some(match value {
            ScalarValue::Int8(v) => *v as i128,
            ScalarValue::Int16(v) => *v as i128,
            ScalarValue::Int32(v) => *v as i128,
            ScalarValue::Int64(v) => *v as i128,
            ScalarValue::Int128(v) => *v,
            _ => return None,
        })
    };
    match (&value, target) {
        (ScalarValue::Boolean(_), PivotType::Boolean)
        | (ScalarValue::Int8(_), PivotType::Int8)
        | (ScalarValue::Int16(_), PivotType::Int16)
        | (ScalarValue::Int32(_), PivotType::Int32)
        | (ScalarValue::Int64(_), PivotType::Int64)
        | (ScalarValue::Int128(_), PivotType::Int128)
        | (ScalarValue::Float32(_), PivotType::Float32)
        | (ScalarValue::Float64(_), PivotType::Float64)
        | (ScalarValue::Utf8(_), PivotType::Utf8)
        | (ScalarValue::Date(_), PivotType::Date)
        | (ScalarValue::Timestamp(_), PivotType::Timestamp)
        | (ScalarValue::Decimal { .. }, PivotType::Decimal { .. }) => Ok(value),
        // A textual value (a NUMERIC rendered to digits, or a parameter the
        // client declared as text) re-parses as the target type.
        (ScalarValue::Utf8(text), _) => decode_text(text, target),
        (ScalarValue::Float32(v), PivotType::Float64) => Ok(ScalarValue::Float64(*v as f64)),
        _ => match as_i128(&value) {
            Some(integer) => match target {
                PivotType::Float64 => Ok(ScalarValue::Float64(integer as f64)),
                PivotType::Float32 => Ok(ScalarValue::Float32(integer as f32)),
                PivotType::Decimal { precision, scale } => {
                    scale_decimal(integer, 0, *precision, *scale)
                }
                _ => narrow_int(integer, target),
            },
            None => Err(format!("a {value} value does not fit the parameter's type")),
        },
    }
}

/// Check that an execution plan still produces the row shape the statement's
/// Describe advertised, mirroring Postgres's "cached plan must not change
/// result type" guard: a replanned execution encodes rows by the live
/// catalog's schema while the client decodes by the Parse-time one.
///
/// Only the leading described columns are compared: a statement whose plan
/// legitimately emits fewer columns than the binder named (EXPLAIN) fails on
/// the count downstream in a visible way, not as misdecoded values.
pub(crate) fn validate_result_shape(
    statement: &PreparedStatement,
    plan: &planner::Plan,
) -> Result<(), String> {
    if statement.output_columns.is_empty() {
        return Ok(());
    }
    let types = plan.root.output_types().map_err(|e| e.to_string())?;
    for (produced, (name, described)) in types.iter().zip(statement.output_columns.iter()) {
        if pg_type_for(produced) != *described {
            return Err(format!(
                "cached plan must not change result type: column {name} is no longer {described}; \
                 prepare the statement again"
            ));
        }
    }
    Ok(())
}

/// Fit an integer into the integer type `target`, erroring when out of range.
fn narrow_int(value: i128, target: &PivotType) -> Result<ScalarValue, String> {
    let out_of_range = |what: &str| format!("{value} is out of range for {what}");
    Ok(match target {
        PivotType::Int8 => {
            ScalarValue::Int8(i8::try_from(value).map_err(|_| out_of_range("a tinyint"))?)
        }
        PivotType::Int16 => {
            ScalarValue::Int16(i16::try_from(value).map_err(|_| out_of_range("a smallint"))?)
        }
        PivotType::Int32 => {
            ScalarValue::Int32(i32::try_from(value).map_err(|_| out_of_range("an integer"))?)
        }
        PivotType::Int64 => {
            ScalarValue::Int64(i64::try_from(value).map_err(|_| out_of_range("a bigint"))?)
        }
        PivotType::Int128 => ScalarValue::Int128(value),
        PivotType::UInt8 => {
            ScalarValue::UInt8(u8::try_from(value).map_err(|_| out_of_range("a utinyint"))?)
        }
        PivotType::UInt16 => {
            ScalarValue::UInt16(u16::try_from(value).map_err(|_| out_of_range("a usmallint"))?)
        }
        PivotType::UInt32 => {
            ScalarValue::UInt32(u32::try_from(value).map_err(|_| out_of_range("a uinteger"))?)
        }
        PivotType::UInt64 => {
            ScalarValue::UInt64(u64::try_from(value).map_err(|_| out_of_range("a ubigint"))?)
        }
        other => return Err(format!("an integer value does not fit a {other} parameter")),
    })
}

/// Parse a decimal literal (`-12.345`) into an unscaled integer at the
/// parameter's scale. Excess fractional digits round half away from zero, as
/// Postgres rounds; an integer part wider than the precision allows errors.
fn parse_decimal(text: &str, precision: u8, scale: i8) -> Result<ScalarValue, String> {
    if scale < 0 {
        return Err(format!("unsupported negative decimal scale {scale}"));
    }
    let invalid = || format!("`{text}` is not a valid decimal");
    let (negative, digits) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    let (int_part, frac_part) = digits.split_once('.').unwrap_or((digits, ""));
    if int_part.is_empty() && frac_part.is_empty() {
        return Err(invalid());
    }
    if !int_part.bytes().all(|b| b.is_ascii_digit())
        || !frac_part.bytes().all(|b| b.is_ascii_digit())
    {
        return Err(invalid());
    }

    let scale_usize = scale as usize;
    let mut unscaled: i128 = 0;
    for b in int_part.bytes().chain(frac_part.bytes().take(scale_usize)) {
        unscaled = unscaled
            .checked_mul(10)
            .and_then(|v| v.checked_add((b - b'0') as i128))
            .ok_or_else(invalid)?;
    }
    // Pad when the literal has fewer fractional digits than the scale.
    for _ in frac_part.len()..scale_usize {
        unscaled = unscaled.checked_mul(10).ok_or_else(invalid)?;
    }
    // Round half away from zero on the first excess fractional digit.
    if frac_part.len() > scale_usize && frac_part.as_bytes()[scale_usize] >= b'5' {
        unscaled += 1;
    }
    if negative {
        unscaled = -unscaled;
    }
    scale_decimal(unscaled, scale, precision, scale)
}

/// Wrap an unscaled integer (already at scale `from_scale`) into a decimal
/// scalar at the target width, rescaling when the scales differ and checking
/// the precision bound.
fn scale_decimal(
    mut unscaled: i128,
    from_scale: i8,
    precision: u8,
    scale: i8,
) -> Result<ScalarValue, String> {
    for _ in from_scale..scale {
        unscaled = unscaled
            .checked_mul(10)
            .ok_or_else(|| "decimal value overflows".to_string())?;
    }
    let bound = 10_i128
        .checked_pow(precision as u32)
        .ok_or_else(|| "decimal value overflows".to_string())?;
    if unscaled.abs() >= bound {
        return Err(format!(
            "decimal value does not fit DECIMAL({precision},{scale})"
        ));
    }
    Ok(ScalarValue::Decimal {
        value: unscaled,
        width: precision,
        scale: scale as u8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // The decimal rounding corner has no blackbox equivalent: no test client
    // in the tree binds NUMERIC parameters without an extra decimal
    // dependency, so the rounding is unobservable through the wire tests.
    #[test]
    fn decimal_text_pads_and_rounds_to_scale() {
        let ScalarValue::Decimal { value, scale, .. } = parse_decimal("12.3", 10, 2).unwrap()
        else {
            panic!("expected a decimal");
        };
        assert_eq!((value, scale), (1230, 2));

        let ScalarValue::Decimal { value, .. } = parse_decimal("-1.005", 10, 2).unwrap() else {
            panic!("expected a decimal");
        };
        assert_eq!(value, -101);

        assert!(parse_decimal("123456789.0", 6, 2).is_err());
    }
}
