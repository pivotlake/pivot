use crate::duckdb_bridge::duckdb_types::LogicalTypeId;
use custom_deserializer::CustomDeserializer;
use std::fmt::Debug;

impl<'de> serde::Deserialize<'de> for LogicalTypeId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = <u8 as serde::Deserialize>::deserialize(deserializer)?;
        Ok(unsafe { std::mem::transmute::<u8, LogicalTypeId>(value) })
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
#[derive(CustomDeserializer, Debug, Clone)]
pub struct ScalarValue {
    /// The DuckDB logical type (e.g. `INTEGER`, `VARCHAR`).
    pub logical_type: LogicalTypeId,
    /// The value as a string exactly as DuckDB serialized it (e.g. `"42"`, `"alice"`).
    pub raw_value: String,
}
