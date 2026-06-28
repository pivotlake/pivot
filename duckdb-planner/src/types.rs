use crate::duckdb_bridge::duckdb_types::LogicalTypeId;
use std::fmt::Debug;

impl LogicalTypeId {
    /// Reconstruct a [`LogicalTypeId`] from the `u8` discriminant the bridge
    /// reports. DuckDB defines `LogicalTypeId` as `enum class : uint8_t`, so the
    /// discriminant round-trips exactly.
    pub(crate) fn from_u8(value: u8) -> Self {
        unsafe { std::mem::transmute::<u8, LogicalTypeId>(value) }
    }
}

impl Debug for LogicalTypeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.clone() as u8)
    }
}

/// A constant value from the query, stored as the DuckDB type tag plus its
/// string representation.
///
/// This intentionally avoids a typed enum so that every DuckDB type is
/// supported without needing a variant for each one — the consumer can
/// parse `raw_value` according to `logical_type` as needed.
#[derive(Debug, Clone)]
pub struct ScalarValue {
    /// The DuckDB logical type (e.g. `INTEGER`, `VARCHAR`).
    pub logical_type: LogicalTypeId,
    /// The value as a string exactly as DuckDB serialized it (e.g. `"42"`, `"alice"`).
    pub raw_value: String,
}
