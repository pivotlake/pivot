use std::fmt;

use crate::duckdb_bridge::duckdb_types::LogicalTypeId;
use crate::duckdb_bridge::ffi;

/// The payload that completes a parameterized DuckDB type, mirroring
/// DuckDB's own `ExtraTypeInfo`. Most ids fully describe their type and
/// carry [`ExtraTypeInfo::None`]; a parameterized id (today only `DECIMAL`)
/// carries its parameters here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtraTypeInfo {
    None,
    /// `DECIMAL(width, scale)`: total digits and fractional digits.
    Decimal {
        width: u8,
        scale: u8,
    },
}

/// A DuckDB logical type as read off a bound plan: the type id plus the
/// extra info that completes a parameterized id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundLogicalType {
    pub id: LogicalTypeId,
    pub extra: ExtraTypeInfo,
}

impl BoundLogicalType {
    /// A type made of the id alone, for the ids that need nothing else.
    pub fn plain(id: LogicalTypeId) -> Self {
        BoundLogicalType {
            id,
            extra: ExtraTypeInfo::None,
        }
    }

    /// Decode the flat FFI type struct. CXX cannot pass an enum with a
    /// payload across the bridge, so the struct carries every parameter
    /// field flat and the id decides which of them are meaningful; this is
    /// the one place that mapping is interpreted.
    pub(crate) fn from_bridge(raw: ffi::BridgeLogicalType) -> Self {
        let id = LogicalTypeId::from_u8(raw.id);
        let extra = match id {
            LogicalTypeId::DECIMAL => ExtraTypeInfo::Decimal {
                width: raw.decimal_width,
                scale: raw.decimal_scale,
            },
            _ => ExtraTypeInfo::None,
        };
        BoundLogicalType { id, extra }
    }

    /// Encode into the flat FFI column descriptor the DuckDB catalog glue
    /// consumes, the inverse of `from_bridge`: every parameter field is
    /// written flat, zeroed when the id has none.
    pub fn to_duckdb_column(&self, name: String) -> ffi::DuckDBColumn {
        let (decimal_width, decimal_scale) = match self.extra {
            ExtraTypeInfo::Decimal { width, scale } => (width, scale),
            ExtraTypeInfo::None => (0, 0),
        };
        ffi::DuckDBColumn {
            name,
            duckdb_logical_type_id: self.id.clone() as u8,
            decimal_width,
            decimal_scale,
        }
    }
}

/// A prepared-statement parameter value handed to the planner, which
/// substitutes it into the plan as a constant of the parameter's
/// binder-inferred type (a mismatched value fails that cast).
///
/// The variants are deliberately few: every integer width rides `Int` and a
/// type with no natural carrier here (e.g. DECIMAL) rides `Text` in its SQL
/// text form, both relying on that cast.
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterValue {
    Null,
    Boolean(bool),
    Int(i64),
    Float(f64),
    Text(String),
    /// `DATE`: days since the Unix epoch (1970-01-01).
    Date(i32),
    /// `TIMESTAMP`: microseconds since the Unix epoch.
    Timestamp(i64),
}

impl ParameterValue {
    /// Encode into the flat FFI struct `extract_plan` takes: the carrier
    /// fields are written flat and `kind` names the meaningful one.
    pub(crate) fn to_bridge(&self) -> ffi::BridgeParameterValue {
        let mut bridge = ffi::BridgeParameterValue {
            kind: LogicalTypeId::SQLNULL as u8,
            bool_value: false,
            int_value: 0,
            float_value: 0.0,
            string_value: String::new(),
        };
        match self {
            ParameterValue::Null => {}
            ParameterValue::Boolean(v) => {
                bridge.kind = LogicalTypeId::BOOLEAN as u8;
                bridge.bool_value = *v;
            }
            ParameterValue::Int(v) => {
                bridge.kind = LogicalTypeId::BIGINT as u8;
                bridge.int_value = *v;
            }
            ParameterValue::Float(v) => {
                bridge.kind = LogicalTypeId::DOUBLE as u8;
                bridge.float_value = *v;
            }
            ParameterValue::Text(v) => {
                bridge.kind = LogicalTypeId::VARCHAR as u8;
                bridge.string_value = v.clone();
            }
            ParameterValue::Date(v) => {
                bridge.kind = LogicalTypeId::DATE as u8;
                bridge.int_value = *v as i64;
            }
            ParameterValue::Timestamp(v) => {
                bridge.kind = LogicalTypeId::TIMESTAMP as u8;
                bridge.int_value = *v;
            }
        }
        bridge
    }
}

/// A constant value from the query (or a table-function argument), extracted from
/// DuckDB as a typed value rather than a string.
///
/// Each variant holds the value already decoded into the corresponding Rust type;
/// [`Other`](ScalarValue::Other) covers DuckDB types the bridge doesn't decode
/// into a typed variant and carries the logical type so the consumer can
/// report it.
#[derive(Debug, Clone)]
pub enum ScalarValue {
    /// A SQL NULL constant, carrying its full logical type (a NULL is always
    /// typed in a bound plan, e.g. `NULL::VARCHAR` in a VALUES row).
    Null(BoundLogicalType),
    Boolean(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    /// `HUGEINT`: a 128-bit integer.
    Int128(i128),
    Float32(f32),
    Float64(f64),
    /// `DECIMAL(width, scale)`: the unscaled 128-bit integer, so the numeric
    /// value is `value / 10^scale`.
    Decimal {
        value: i128,
        width: u8,
        scale: u8,
    },
    Utf8(String),
    /// `DATE`: days since the Unix epoch (1970-01-01).
    Date(i32),
    /// `TIMESTAMP`: microseconds since the Unix epoch.
    Timestamp(i64),
    /// `INTERVAL` kept as its three independent components.
    Interval {
        months: i32,
        days: i32,
        micros: i64,
    },
    /// A `VARIANT`: the document text it holds. DuckDB builds a variant from a
    /// string by wrapping the raw text, and hands it back as that same text
    /// under a variant-to-VARCHAR cast, so the consumer parses it as JSON.
    Variant(String),
    /// A DuckDB type the bridge does not decode into a typed variant.
    Other(LogicalTypeId),
}

impl fmt::Display for ScalarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarValue::Null(_) => write!(f, "NULL"),
            ScalarValue::Boolean(v) => write!(f, "{v}"),
            ScalarValue::Int8(v) => write!(f, "{v}"),
            ScalarValue::Int16(v) => write!(f, "{v}"),
            ScalarValue::Int32(v) => write!(f, "{v}"),
            ScalarValue::Int64(v) => write!(f, "{v}"),
            ScalarValue::UInt8(v) => write!(f, "{v}"),
            ScalarValue::UInt16(v) => write!(f, "{v}"),
            ScalarValue::UInt32(v) => write!(f, "{v}"),
            ScalarValue::UInt64(v) => write!(f, "{v}"),
            ScalarValue::Int128(v) => write!(f, "{v}"),
            ScalarValue::Float32(v) => write!(f, "{v}"),
            ScalarValue::Float64(v) => write!(f, "{v}"),
            ScalarValue::Decimal {
                value,
                width,
                scale,
            } => write!(f, "{value} as DECIMAL({width},{scale})"),
            ScalarValue::Utf8(v) => write!(f, "{v}"),
            ScalarValue::Date(v) => write!(f, "{v}"),
            ScalarValue::Timestamp(v) => write!(f, "{v}"),
            ScalarValue::Interval {
                months,
                days,
                micros,
            } => write!(f, "{months} {days} {micros}"),
            ScalarValue::Variant(v) => write!(f, "{v}"),
            ScalarValue::Other(ty) => write!(f, "{ty:?}"),
        }
    }
}
