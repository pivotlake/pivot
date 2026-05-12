//! Shared runtime cells for dynamic filters injected by DuckDB optimizer
//! passes (TopN pushdown, hash-join probe filters).
//!
//! At plan-extract time we know the wiring (which producer writes a slot,
//! which consumer scans read from it) and the comparison shape; the
//! boundary value itself is filled in at runtime by the producer.

use std::fmt;
use std::sync::Arc;

use custom_deserializer::CustomDeserializer;

use crate::duckdb_bridge::duckdb_types::ExpressionType;

/// Shared cell between one producer and N consumers. Defined in `dispatch`
/// so the executor's TopN operator can write into it directly without
/// going back through the planner crate.
pub use dispatch::DynamicFilterSlot;

/// A reference to a shared slot. Shape mirrors [`ConstantComparison`] but
/// with the constant deferred into the slot. Same struct on both producer
/// and consumer sides:
/// * Consumer (Input): `column_idx` is the position in `Input.columns`
///   the filter applies to.
/// * Producer (TopN): `column_idx` is the position in `order_bys` whose
///   value is published.
///
/// [`ConstantComparison`]: crate::expression::ConstantComparison
#[derive(CustomDeserializer)]
pub struct DynamicFilter {
    pub slot_id: usize,
    pub column_idx: usize,
    pub compare_type: ExpressionType,
    #[skip_deserialize]
    pub slot: Option<Arc<DynamicFilterSlot>>,
}

impl fmt::Debug for DynamicFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynamicFilter")
            .field("slot_id", &self.slot_id)
            .field("column_idx", &self.column_idx)
            .field("compare_type", &self.compare_type)
            .field("resolved", &self.slot.is_some())
            .finish()
    }
}