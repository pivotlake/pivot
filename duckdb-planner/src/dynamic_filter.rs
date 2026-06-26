//! Dynamic filters extracted from DuckDB's optimizer (currently the Top-N
//! pushdown pass; hash-join probe filters would slot in the same way).
//!
//! At plan-extract time the bridge tells us the wiring — which producer writes
//! a slot, which consumer scans read it, correlated by a `slot_id` — and the
//! comparison shape. This is pure plan data; the shared cell the boundary is
//! published into is allocated later, at compile time, and keyed back to these
//! references by `slot_id`.


use crate::duckdb_bridge::duckdb_types::ExpressionType;

/// A reference to a shared dynamic-filter slot. Its shape mirrors a constant
/// comparison, but the constant is deferred into the slot:
/// * Consumer (`Input`): `column_idx` is the storage column the filter tests.
/// * Producer (`TopN`): `column_idx` is the position in the child output whose
///   value is published.
///
/// `slot_id` correlates a producer with the consumer scans that read its slot.
#[derive(Debug)]
pub struct DynamicFilter {
    pub slot_id: usize,
    pub column_idx: usize,
    pub compare_type: ExpressionType,
}
