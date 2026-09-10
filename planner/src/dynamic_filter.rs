//! Logical runtime-filter wiring stored in a query plan.
//!
//! These structures describe connections, not live filter state. A producer
//! and its consumer scans carry the same `slot_id`; [`Plan::compile`](crate::Plan::compile) resolves
//! that ID through its runtime-slot registry so both ends receive the same
//! shared [`BoundarySlot`](dispatch::BoundarySlot) or key-bitset slot.
//!
//! Keeping only IDs in the plan matters for plan caching: every execution gets
//! newly allocated, unarmed slots, so no boundary or build-key set can leak
//! from an earlier execution.

use std::fmt;

use crate::expression::CompareType;

/// One boundary-filter endpoint in the logical plan.
///
/// On a consumer scan, `column_idx` and `compare_type` say how to compare scan
/// data with the current boundary. On a producer, `slot_id` identifies the same
/// runtime cell. The cell itself is allocated only when the plan is compiled.
#[derive(Debug)]
pub struct DynamicFilter {
    pub slot_id: usize,
    pub column_idx: usize,
    pub compare_type: CompareType,
}

/// A pair of dynamic filters a hash join's build side produces: once the
/// build seals, the minimum and maximum of one build key column are published
/// into the two slots, and consumer scans of the probe side prune row groups
/// whose key range falls entirely outside `[min, max]`.
#[derive(Debug)]
pub struct JoinProducedFilter {
    /// Position of the key among the join's equality conditions (indexes
    /// `build_keys`/`key_types`).
    pub key_position: usize,
    /// Slot the build key minimum is published into; its consumers compare
    /// with `>=`.
    pub min_slot_id: usize,
    /// Slot the build key maximum is published into; its consumers compare
    /// with `<=`.
    pub max_slot_id: usize,
    /// Slot the build key set is published into as an exact key bitset, when
    /// its shape allows one; consumer scans drop rows whose key
    /// it does not hold. Its own id space, separate from the boundary slots.
    pub key_bitset_slot_id: usize,
}

/// A key-bitset consumer on a scan: rows whose `column_idx` value the
/// slot's sealed key set does not hold are dropped directly above the scan,
/// before any other operator sees them.
#[derive(Debug)]
pub struct KeyBitsetFilter {
    pub slot_id: usize,
    pub column_idx: usize,
}

/// Join-filter-pushdown wiring read off a scan's DuckDB get at plan-build
/// time, consumed by the ancestor join's builder: it appends the consumer
/// [`DynamicFilter`]s to the scan and records the producing side on itself.
#[derive(Debug)]
pub struct JoinFilterScanInfo {
    /// Pointer identity of the shared filter set pairing this scan with the
    /// joins that push into it.
    pub filter_set_id: usize,
    /// The get's `column_ids` index -> storage column mapping, the space the
    /// join's probe column indexes arrive in.
    pub proj_to_storage: Vec<usize>,
}

impl fmt::Display for DynamicFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DynamicFilter(#{} {} ?)",
            self.column_idx, self.compare_type
        )
    }
}
