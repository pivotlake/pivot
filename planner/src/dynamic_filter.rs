//! Planner-side view of a dynamic-filter reference.
//!
//! Carries the comparison translated into the planner's [`CompareType`]. This is
//! pure plan data: it carries the `slot_id` that correlates a producer (`TopN`)
//! with its consumer scans, but *not* the shared slot itself. The
//! [`DynamicFilterSlot`](dispatch::DynamicFilterSlot) is allocated fresh on each
//! [`Plan::compile`](crate::Plan::compile) and handed to the operators by
//! `slot_id`, so a cached `Plan` holds no runtime state.

use std::fmt;

use crate::expression::CompareType;

/// A dynamic-filter reference on an operator: which column it tests, the
/// comparison, and the `slot_id` correlating it with the other end of the
/// producer↔consumer pair. The shared slot is resolved at compile time (see
/// [`Plan::compile`](crate::Plan::compile)).
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
    /// Slot the build key set is published into as an exact membership
    /// filter, when its shape allows one; consumer scans drop rows whose key
    /// it does not hold. Its own id space, separate from the boundary slots.
    pub membership_slot_id: usize,
}

/// A membership-filter consumer on a scan: rows whose `column_idx` value the
/// slot's sealed key set does not hold are dropped directly above the scan,
/// before any other operator sees them.
#[derive(Debug)]
pub struct MembershipFilter {
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
