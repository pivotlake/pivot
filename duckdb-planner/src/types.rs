use std::fmt;

use crate::duckdb_bridge::duckdb_types::LogicalTypeId;

/// A constant value from the query (or a table-function argument), extracted from
/// DuckDB as a typed value rather than a string.
///
/// Each variant holds the value already decoded into the corresponding Rust type;
/// [`Other`](ScalarValue::Other) covers DuckDB types the bridge doesn't decode
/// into a typed variant (e.g. `HUGEINT`, `DECIMAL`) and carries the logical type
/// so the consumer can report it.
#[derive(Debug, Clone)]
pub enum ScalarValue {
    Boolean(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
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
    /// A `NULL` constant of the given logical type.
    Null(LogicalTypeId),
    /// A DuckDB type the bridge does not decode into a typed variant.
    Other(LogicalTypeId),
}

impl fmt::Display for ScalarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarValue::Boolean(v) => write!(f, "{v}"),
            ScalarValue::Int8(v) => write!(f, "{v}"),
            ScalarValue::Int16(v) => write!(f, "{v}"),
            ScalarValue::Int32(v) => write!(f, "{v}"),
            ScalarValue::Int64(v) => write!(f, "{v}"),
            ScalarValue::UInt8(v) => write!(f, "{v}"),
            ScalarValue::UInt16(v) => write!(f, "{v}"),
            ScalarValue::UInt32(v) => write!(f, "{v}"),
            ScalarValue::UInt64(v) => write!(f, "{v}"),
            ScalarValue::Float32(v) => write!(f, "{v}"),
            ScalarValue::Float64(v) => write!(f, "{v}"),
            ScalarValue::Utf8(v) => write!(f, "{v}"),
            ScalarValue::Date(v) => write!(f, "{v}"),
            ScalarValue::Timestamp(v) => write!(f, "{v}"),
            ScalarValue::Interval {
                months,
                days,
                micros,
            } => write!(f, "{months} {days} {micros}"),
            ScalarValue::Null(ty) => write!(f, "NULL:{ty:?}"),
            ScalarValue::Other(ty) => write!(f, "{ty:?}"),
        }
    }
}
