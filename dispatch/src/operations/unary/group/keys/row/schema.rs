//! The row extractor's per-query key schema.

use arrow_schema::DataType;
use std::sync::Arc;

/// The key columns' arrow types, in GROUP BY order — the row extractor's
/// `Config`. Supports the eight fixed-width integer types, the 8-byte
/// `Decimal64`, the 16-byte `Decimal128`, and `Utf8View`; the planner must only
/// route those shapes here.
#[derive(Clone)]
pub struct RowKeySchema(Arc<[DataType]>);

impl RowKeySchema {
    /// Build a schema from the key columns' arrow types, panicking on an
    /// unsupported type (the planner is responsible for only routing supported
    /// shapes to the row extractor).
    pub fn new(types: impl Into<Arc<[DataType]>>) -> Self {
        let types = types.into();
        for t in types.iter() {
            // The extractor encodes the eight fixed-width integer types
            // (`is_integer`), the two decimal widths (whose raw unscaled
            // integers pack like integers of the same width), and `Utf8View`;
            // the planner must only route those.
            assert!(
                t.is_integer()
                    || matches!(t, DataType::Decimal64(_, _) | DataType::Decimal128(_, _))
                    || *t == DataType::Utf8View,
                "row key column type not supported: {t}"
            );
        }
        Self(types)
    }

    pub(super) fn types(&self) -> &[DataType] {
        &self.0
    }
}
