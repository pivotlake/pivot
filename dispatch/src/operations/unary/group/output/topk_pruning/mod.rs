//! Merge-phase pruning for a pushed-down grouped top-k.
//!
//! When a grouped `ORDER BY <count slot> DESC LIMIT k` is pushed into the
//! group operator, each worker also sums the ordering aggregate's partial
//! values into a fixed array of hash bins indexed by the top bits of the
//! group hash. The ordering aggregate is additive and nonnegative, so after
//! summing every worker's bin totals, a bin's total is an upper bound on the
//! final value of any single group hashing into that bin.
//!
//! Merge partitions cover contiguous bin ranges of the same top hash bits, so
//! each partition gets a bound: the maximum bin total in its range. The merge
//! phase runs partition jobs in descending bound order and, once any worker's
//! top-k heap holds `k` exact (fully merged) groups, skips every partition
//! whose bound is below that k-th value: no group there can displace one of
//! the `k` already found.
//!
//! [`prunable_topk_slot`] decides whether a pushed limit qualifies. The
//! per-worker [`HashBinTotals`] are summed through [`SharedBinTotals`] into
//! [`TopKBounds`], which merge jobs check against the [`TopKThreshold`] to
//! form the [`TopKCutoff`] their scans apply.

mod bin_totals;
mod bounds;
mod shared_bin_totals;
mod threshold;

pub use bin_totals::HashBinTotals;
pub(crate) use bounds::{TopKBounds, TopKCutoff};
pub(crate) use shared_bin_totals::SharedBinTotals;
pub use threshold::TopKThreshold;

use crate::operations::unary::group::GroupLimit;
use crate::operations::unary::group::values::{AggregationKind, AggregationSlot};

/// Bin resolution in top hash bits. Bounds tighten as this grows (the noise
/// under any single group's bound is roughly `rows >> HASH_BIN_BITS`); the
/// per-worker array is `8 << HASH_BIN_BITS` bytes.
pub(super) const HASH_BIN_BITS: u32 = 16;

/// Number of hash bins.
pub(crate) const HASH_BINS: usize = 1 << HASH_BIN_BITS;

/// Table entries below which a worker that never drained skips building its
/// flush-time bin totals. The array's allocation and page churn are
/// measurable against a small query's total runtime, and so few entries mean
/// the whole merge is too small for pruning to pay anyway. The merge only
/// prunes when every contributing worker reported bin totals, so a skipped
/// worker disables pruning rather than invalidating bounds. Tests all but
/// drop the floor so the pruning path stays exercised at test-sized group
/// counts.
pub(crate) const MIN_BINNED_ENTRIES: usize = if cfg!(test) { 1 } else { 8192 };

/// The hash bin a hash falls into.
#[inline(always)]
pub(crate) fn hash_bin(hash: u64) -> usize {
    (hash >> (u64::BITS - HASH_BIN_BITS)) as usize
}

/// The pushed top-k ordering slot, when its aggregate makes bin sums valid
/// upper bounds: the partials must be nonnegative and must not merge to more
/// than their sum. `COUNT` forms qualify; `SUM` may go negative and extremes
/// are not additive, so they do not.
pub(crate) fn prunable_topk_slot(
    output_limit: Option<GroupLimit>,
    slots: &[AggregationSlot],
) -> Option<usize> {
    // Experiment kill switch: disables the whole pruning apparatus so an A/B
    // can attribute a regression to it or to the surrounding changes. Read
    // once: every worker builds its group operator at the same moment, and
    // reading the environment takes a process-wide lock.
    static PRUNING_DISABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("PIVOT_DISABLE_TOPK_PRUNING").is_some());
    if *PRUNING_DISABLED {
        return None;
    }
    let (slot, limit) = match output_limit? {
        GroupLimit::TopK { slot, limit, .. } => (slot, limit),
        GroupLimit::First { .. } => return None,
    };
    // A zero limit emits nothing; the pruning threshold needs a full heap.
    if limit == 0 {
        return None;
    }
    matches!(
        slots.get(slot)?.kind,
        AggregationKind::CountStar | AggregationKind::Count
    )
    .then_some(slot)
}

/// Converts a pushed ORDER BY sort key into a pruning weight, clamping into
/// `u64`. Order-preserving for the nonnegative values pruning is gated to.
pub trait TopKWeight {
    fn saturating_weight(self) -> u64;
}

impl TopKWeight for i64 {
    #[inline(always)]
    fn saturating_weight(self) -> u64 {
        self.max(0) as u64
    }
}

impl TopKWeight for i128 {
    #[inline(always)]
    fn saturating_weight(self) -> u64 {
        self.clamp(0, u64::MAX as i128) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_slots_are_prunable_and_others_are_not() {
        use arrow_schema::DataType;
        let slots = vec![
            AggregationSlot::new(AggregationKind::Sum, 0, DataType::Int64),
            AggregationSlot::new(AggregationKind::CountStar, 0, DataType::Int64),
        ];
        let topk = |slot| {
            Some(GroupLimit::TopK {
                slot,
                limit: 10,
                with_ties: false,
            })
        };

        assert_eq!(prunable_topk_slot(topk(1), &slots), Some(1));
        assert_eq!(prunable_topk_slot(topk(0), &slots), None);
        assert_eq!(
            prunable_topk_slot(Some(GroupLimit::First { limit: 10 }), &slots),
            None
        );
        assert_eq!(prunable_topk_slot(None, &slots), None);
    }
}
