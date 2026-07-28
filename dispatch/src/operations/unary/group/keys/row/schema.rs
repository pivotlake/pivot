//! The row extractor's per-query key schema.

use arrow_schema::DataType;
use std::sync::Arc;

/// The key columns' arrow types and nullability, in GROUP BY order: the row
/// extractor's `Config`. Supports the eight fixed-width integer types, the
/// 8-byte `Decimal64`, the 16-byte `Decimal128`, and `Utf8View`; the planner
/// must only route those shapes here.
///
/// A nullable field is encoded with a leading validity byte (`1` = value bytes
/// follow, `0` = NULL, nothing follows), so SQL's "NULLs group together" falls
/// out of byte equality. A non-nullable field encodes without a validity byte,
/// so a NULL-free schema's blobs pay nothing for the nullable machinery.
#[derive(Clone)]
pub struct RowKeySchema {
    types: Arc<[DataType]>,
    nullable: Arc<[bool]>,
}

impl RowKeySchema {
    /// Build a schema from the key columns' arrow types and nullability,
    /// panicking on an unsupported type (the planner is responsible for only
    /// routing supported shapes to the row extractor).
    pub fn new(types: impl Into<Arc<[DataType]>>, nullable: impl Into<Arc<[bool]>>) -> Self {
        let types = types.into();
        let nullable = nullable.into();
        assert_eq!(
            types.len(),
            nullable.len(),
            "one nullability flag per key column"
        );
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
        Self { types, nullable }
    }

    pub(super) fn types(&self) -> &[DataType] {
        &self.types
    }

    pub(super) fn nullable(&self) -> &[bool] {
        &self.nullable
    }
}
