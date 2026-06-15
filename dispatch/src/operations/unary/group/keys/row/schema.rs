//! The row extractor's per-query key schema.

use arrow_schema::DataType;
use std::sync::Arc;

/// The key columns' arrow types, in GROUP BY order — the row extractor's
/// `Config`. Supports the eight fixed-width integer types and `Utf8View`; the
/// planner must only route those shapes here.
#[derive(Clone)]
pub struct RowKeySchema(Arc<[DataType]>);

impl RowKeySchema {
    /// Build a schema from the key columns' arrow types, panicking on an
    /// unsupported type (the planner is responsible for only routing supported
    /// shapes to the row extractor).
    pub fn new(types: impl Into<Arc<[DataType]>>) -> Self {
        let types = types.into();
        for t in types.iter() {
            assert!(
                encoded_width(t).is_some() || *t == DataType::Utf8View,
                "row key column type not supported: {t}"
            );
        }
        Self(types)
    }

    pub(super) fn types(&self) -> &[DataType] {
        &self.0
    }
}

/// Encoded byte width of a fixed-width integer type; `None` for variable-width
/// (string) columns.
fn encoded_width(dt: &DataType) -> Option<usize> {
    Some(match dt {
        DataType::Int8 | DataType::UInt8 => 1,
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int32 | DataType::UInt32 => 4,
        DataType::Int64 | DataType::UInt64 => 8,
        _ => return None,
    })
}
