//! Dynamic filters extracted from DuckDB's optimizer (currently the Top-N
//! pushdown pass; hash-join probe filters would slot in the same way).
//!
//! At plan-extract time the bridge tells us the wiring — which producer writes
//! a slot, which consumer scans read it — and the comparison shape. The
//! boundary value itself is filled in at runtime by the producer. Producer and
//! consumer entries that share a `slot_id` are bound to the same
//! [`DynamicFilterSlot`] during [`resolve_inputs`](crate::PlanNode::resolve_inputs).

use std::fmt;
use std::sync::Arc;

use custom_deserializer::CustomDeserializer;

use crate::duckdb_bridge::duckdb_types::ExpressionType;

/// The shared cell between one producer and N consumers. Defined in `dispatch`
/// so the executor's Top-N operator can write to it directly without going back
/// through the planner crates.
pub use dispatch::DynamicFilterSlot;

/// A reference to a shared dynamic-filter slot. Its shape mirrors a constant
/// comparison, but the constant is deferred into the slot:
/// * Consumer (`Input`): `column_idx` is the storage column the filter tests.
/// * Producer (`TopN`): `column_idx` is the position in the child output whose
///   value is published.
///
/// `slot` is `None` until [`resolve_inputs`](crate::PlanNode::resolve_inputs)
/// binds it by `slot_id`.
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
