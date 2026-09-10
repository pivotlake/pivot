//! Per-slot count totals that bound how heavy a group can be, used to prune
//! the merge of a pushed-down grouped top-k.
//!
//! A grouped `ORDER BY COUNT(...) DESC LIMIT k` only needs the `k` heaviest
//! groups, but a plain merge folds every group across every worker before the
//! heap above discards all but `k`. The totals here bound how heavy any group
//! in a hash range can be, so the merge can skip ranges that cannot reach the
//! top.
//!
//! Every worker folds each row's partial count into a fixed array indexed by
//! the top bits of the group hash. Counts are additive and nonnegative, so once
//! all workers' arrays are summed, a slot's total bounds the final count of
//! every group hashing into it. Merge partitions are contiguous ranges of the
//! same top bits, so a partition's bound is the largest slot total in its
//! range, and any single entry is bounded by its own slot.
//!
//! The threshold the bounds are checked against is the k-th largest exact
//! count seen so far. A worker's top-k heap holds exact, fully merged groups,
//! so its k-th value is a lower bound on the final k-th best, and the shared
//! threshold only ever rises.

use crate::memory::{SlabAllocator, SlabBuffer};
use std::sync::atomic::{AtomicU64, Ordering};

/// Slot resolution in top hash bits. The per-worker array is `8 << BITS`
/// bytes, and any single group's bound carries the counts of the other groups
/// sharing its slot, roughly `groups >> BITS` of them.
const SLOT_BITS: u32 = 16;

/// Number of slots in a totals array.
pub(crate) const TOTAL_SLOTS: usize = 1 << SLOT_BITS;

/// Converts a group's ORDER BY value into the weight added to its slot total.
///
/// Only COUNT slots are totalled, so the value is never negative; the clamp
/// keeps a corrupt value from wrapping into a huge bound.
#[inline(always)]
pub(crate) fn count_weight(sort_key: impl Into<i128>) -> u64 {
    u64::try_from(sort_key.into().max(0)).unwrap_or(u64::MAX)
}

/// Per-slot sums of the ordering aggregate's partial values, by top hash bits.
///
/// Slab-backed so a query costs one memset of pooled, already mapped pages
/// per worker rather than a fresh mapping each time.
pub struct SlotCountTotals {
    sums: SlabBuffer<u64>,
}

// SAFETY: the slab is address-stable pool memory owned by the buffer, and the
// array has one owner at a time (a worker, then the gather).
unsafe impl Send for SlotCountTotals {}

impl SlotCountTotals {
    pub(crate) fn new(allocator: &mut SlabAllocator) -> Self {
        Self {
            sums: allocator.create_slab_buffer(TOTAL_SLOTS, true),
        }
    }

    #[inline(always)]
    fn slot_of(hash: u64) -> usize {
        (hash >> (u64::BITS - SLOT_BITS)) as usize
    }

    fn as_slice(&self) -> &[u64] {
        // SAFETY: the buffer was created with TOTAL_SLOTS zeroed slots.
        unsafe { std::slice::from_raw_parts(self.sums.ptr_at_index(0), TOTAL_SLOTS) }
    }

    fn as_mut_slice(&mut self) -> &mut [u64] {
        // SAFETY: as `as_slice`, and `&mut self` makes the access exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.sums.ptr_at_index(0), TOTAL_SLOTS) }
    }

    /// Folds one partial value of the ordering aggregate into its slot.
    ///
    /// Saturating: a saturated slot only loosens its bound.
    #[inline(always)]
    pub(crate) fn fold(&mut self, hash: u64, weight: u64) {
        let slot = &mut self.as_mut_slice()[Self::slot_of(hash)];
        *slot = slot.saturating_add(weight);
    }

    /// Adds another worker's totals into this array.
    pub(crate) fn merge_from(&mut self, other: &SlotCountTotals) {
        for (sum, other_sum) in self.as_mut_slice().iter_mut().zip(other.as_slice()) {
            *sum = sum.saturating_add(*other_sum);
        }
    }

    /// Upper bound on the final value of any group with this hash.
    #[inline(always)]
    pub(crate) fn slot_bound(&self, hash: u64) -> u64 {
        self.as_slice()[Self::slot_of(hash)]
    }

    /// Upper bound per merge partition, for a merge routing rows by the top
    /// `log2(num_partitions)` hash bits.
    ///
    /// A partition coarser than a slot spans a contiguous slot range and takes
    /// the range's maximum; a finer one lies inside a single slot and takes
    /// that slot's total.
    pub(crate) fn partition_bounds(&self, num_partitions: usize) -> Vec<u64> {
        debug_assert!(num_partitions.is_power_of_two());
        let sums = self.as_slice();
        if num_partitions <= TOTAL_SLOTS {
            let slots_per_partition = TOTAL_SLOTS / num_partitions;
            sums.chunks(slots_per_partition)
                .map(|range| range.iter().copied().max().unwrap_or(0))
                .collect()
        } else {
            let partitions_per_slot = num_partitions / TOTAL_SLOTS;
            (0..num_partitions)
                .map(|partition| sums[partition / partitions_per_slot])
                .collect()
        }
    }
}

/// The pool-wide lower bound on the k-th best exact group value.
///
/// Raised by workers as their top-k heaps fill with fully merged groups and
/// read by merge jobs to skip work that cannot reach it.
pub struct PruneThreshold {
    kth_best: AtomicU64,
}

impl PruneThreshold {
    pub(crate) fn new() -> Self {
        Self {
            kth_best: AtomicU64::new(0),
        }
    }

    /// Records that `k` exact groups of at least `kth_best` exist.
    #[inline]
    pub(crate) fn raise(&self, kth_best: u64) {
        self.kth_best.fetch_max(kth_best, Ordering::Relaxed);
    }

    /// The value a bound must reach for its groups to still matter.
    #[inline]
    pub(crate) fn current(&self) -> u64 {
        self.kth_best.load(Ordering::Relaxed)
    }
}

/// Everything a merge job needs to prune: the pool-wide totals, their bound per
/// partition, and the shared threshold.
pub struct TopKBounds {
    totals: SlotCountTotals,
    partition_bounds: Vec<u64>,
    threshold: PruneThreshold,
}

impl TopKBounds {
    pub(crate) fn new(totals: SlotCountTotals, num_partitions: usize) -> Self {
        Self {
            partition_bounds: totals.partition_bounds(num_partitions),
            totals,
            threshold: PruneThreshold::new(),
        }
    }

    pub(crate) fn threshold(&self) -> &PruneThreshold {
        &self.threshold
    }

    /// Upper bound on any group's final value in `partition`.
    pub(crate) fn partition_bound(&self, partition: usize) -> u64 {
        self.partition_bounds[partition]
    }

    /// Whether the whole partition can be skipped at the current threshold.
    pub(crate) fn skips_partition(&self, partition: usize) -> bool {
        self.partition_bound(partition) < self.threshold.current()
    }

    /// A filter over single entries at the current threshold, or `None` when
    /// the threshold is still zero and nothing could be rejected.
    pub(crate) fn entry_filter(&self) -> Option<SlotBoundFilter<'_>> {
        let kth_best = self.threshold.current();
        (kth_best > 0).then_some(SlotBoundFilter {
            totals: &self.totals,
            kth_best,
        })
    }
}

/// Rejects entries whose slot bound cannot reach the k-th best.
#[derive(Clone, Copy)]
pub struct SlotBoundFilter<'a> {
    totals: &'a SlotCountTotals,
    kth_best: u64,
}

impl SlotBoundFilter<'_> {
    /// Whether the group with `hash` could still enter the top-k.
    #[inline(always)]
    pub(crate) fn could_reach(&self, hash: u64) -> bool {
        self.totals.slot_bound(hash) >= self.kth_best
    }
}

/// Applies an optional filter; no filter passes everything.
#[inline(always)]
pub(crate) fn passes_filter(filter: Option<SlotBoundFilter<'_>>, hash: u64) -> bool {
    match filter {
        Some(filter) => filter.could_reach(hash),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;

    fn hash_in_slot(slot: u64) -> u64 {
        slot << (u64::BITS - SLOT_BITS)
    }

    #[test]
    fn slot_total_bounds_every_group_in_the_slot() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut totals = SlotCountTotals::new(&mut allocator);

        totals.fold(hash_in_slot(7), 3);
        totals.fold(hash_in_slot(7) | 1, 5);
        totals.fold(hash_in_slot(8), 1);

        assert_eq!(totals.slot_bound(hash_in_slot(7) | 99), 8);
        assert_eq!(totals.slot_bound(hash_in_slot(8)), 1);
        assert_eq!(totals.slot_bound(hash_in_slot(9)), 0);
    }

    #[test]
    fn merged_totals_sum_workers() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut first = SlotCountTotals::new(&mut allocator);
        let mut second = SlotCountTotals::new(&mut allocator);
        first.fold(hash_in_slot(3), 2);
        second.fold(hash_in_slot(3), 4);

        first.merge_from(&second);

        assert_eq!(first.slot_bound(hash_in_slot(3)), 6);
    }

    #[test]
    fn partition_bounds_take_the_range_maximum() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut totals = SlotCountTotals::new(&mut allocator);
        // Four partitions cover 16384 slots each; slot 16384 is the first of
        // partition 1.
        totals.fold(hash_in_slot(0), 5);
        totals.fold(hash_in_slot(16384), 9);
        totals.fold(hash_in_slot(16385), 2);

        let bounds = totals.partition_bounds(4);

        assert_eq!(bounds, vec![5, 9, 0, 0]);
    }

    #[test]
    fn partitions_finer_than_slots_share_the_slot_total() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut totals = SlotCountTotals::new(&mut allocator);
        totals.fold(hash_in_slot(1), 7);

        let bounds = totals.partition_bounds(TOTAL_SLOTS * 2);

        assert_eq!(&bounds[..4], &[0, 0, 7, 7]);
    }

    #[test]
    fn partition_skips_once_the_threshold_passes_its_bound() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut totals = SlotCountTotals::new(&mut allocator);
        totals.fold(hash_in_slot(0), 5);
        totals.fold(hash_in_slot(16384), 9);
        let prune = TopKBounds::new(totals, 4);

        assert!(!prune.skips_partition(0));
        prune.threshold().raise(6);

        assert!(prune.skips_partition(0));
        assert!(!prune.skips_partition(1));
        assert!(prune.skips_partition(2));
    }

    #[test]
    fn entry_filter_keeps_slots_that_can_tie_the_threshold() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut totals = SlotCountTotals::new(&mut allocator);
        totals.fold(hash_in_slot(0), 5);
        totals.fold(hash_in_slot(1), 4);
        let prune = TopKBounds::new(totals, 2);
        assert!(prune.entry_filter().is_none());
        prune.threshold().raise(5);

        let filter = prune.entry_filter().unwrap();

        assert!(filter.could_reach(hash_in_slot(0)));
        assert!(!filter.could_reach(hash_in_slot(1)));
    }

    #[test]
    fn threshold_only_rises() {
        let threshold = PruneThreshold::new();

        threshold.raise(10);
        threshold.raise(4);

        assert_eq!(threshold.current(), 10);
    }
}
