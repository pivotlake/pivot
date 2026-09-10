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
//! the top bits of the group hash. Counts are additive and nonnegative, so the
//! sum of every worker's slot bounds the final count of every group hashing
//! into it. Merge partitions are contiguous ranges of the same top bits, so a
//! merge job sums just its own range across the workers' arrays: the range's
//! largest total bounds the whole partition, and each slot bounds its own
//! entries. Nothing is summed up front, and a job only sums once there is a
//! threshold to compare against.
//!
//! That threshold is the k-th largest exact count seen so far. A worker's
//! top-k heap holds exact, fully merged groups, so its k-th value is a lower
//! bound on the final k-th best, and the shared threshold only ever rises.

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

/// The slots a merge partition covers when the merge routes rows by the top
/// `log2(num_partitions)` hash bits. A partition finer than a slot lies inside
/// one slot and shares it with its neighbours.
fn slot_range(partition: usize, num_partitions: usize) -> std::ops::Range<usize> {
    debug_assert!(num_partitions.is_power_of_two());
    if num_partitions <= TOTAL_SLOTS {
        let slots_per_partition = TOTAL_SLOTS / num_partitions;
        partition * slots_per_partition..(partition + 1) * slots_per_partition
    } else {
        let partitions_per_slot = num_partitions / TOTAL_SLOTS;
        let slot = partition / partitions_per_slot;
        slot..slot + 1
    }
}

/// One worker's sums of the ordering aggregate's partial values, by top hash
/// bits.
///
/// Slab-backed so a query costs one memset of pooled, already mapped pages
/// per worker rather than a fresh mapping each time.
pub struct SlotCountTotals {
    sums: SlabBuffer<u64>,
}

// SAFETY: the slab is address-stable pool memory owned by the buffer, and the
// array has one owner at a time (a worker, then the gather).
unsafe impl Send for SlotCountTotals {}
unsafe impl Sync for SlotCountTotals {}

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

    /// Folds one partial value of the ordering aggregate into its slot.
    ///
    /// Saturating: a saturated slot only loosens its bound.
    #[inline(always)]
    pub(crate) fn fold(&mut self, hash: u64, weight: u64) {
        // SAFETY: `slot_of` is below TOTAL_SLOTS and `&mut self` makes the
        // access exclusive.
        let slot = unsafe { &mut *self.sums.ptr_at_index(Self::slot_of(hash)) };
        *slot = slot.saturating_add(weight);
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

/// Every worker's totals plus the shared threshold, from which each merge job
/// derives the bounds for its own partition.
pub struct TopKBounds {
    worker_totals: Vec<SlotCountTotals>,
    num_partitions: usize,
    threshold: PruneThreshold,
}

impl TopKBounds {
    pub(crate) fn new(worker_totals: Vec<SlotCountTotals>, num_partitions: usize) -> Self {
        Self {
            worker_totals,
            num_partitions,
            threshold: PruneThreshold::new(),
        }
    }

    pub(crate) fn threshold(&self) -> &PruneThreshold {
        &self.threshold
    }

    /// The pool-wide totals of `partition`'s slots at the current threshold,
    /// or `None` while the threshold is zero and nothing could be pruned.
    pub(crate) fn partition_totals(&self, partition: usize) -> Option<PartitionTotals> {
        let kth_best = self.threshold.current();
        if kth_best == 0 {
            return None;
        }
        let range = slot_range(partition, self.num_partitions);
        let mut totals = vec![0u64; range.len()];
        for worker in &self.worker_totals {
            for (total, sum) in totals.iter_mut().zip(&worker.as_slice()[range.clone()]) {
                *total = total.saturating_add(*sum);
            }
        }
        Some(PartitionTotals {
            first_slot: range.start,
            totals,
            kth_best,
        })
    }
}

/// One partition's summed slot totals, checked against the threshold they
/// were summed at.
pub struct PartitionTotals {
    first_slot: usize,
    totals: Vec<u64>,
    kth_best: u64,
}

impl PartitionTotals {
    /// Whether no group in the partition can reach the k-th best.
    pub(crate) fn skips_partition(&self) -> bool {
        self.totals.iter().all(|&total| total < self.kth_best)
    }

    /// A filter over the partition's single entries.
    pub(crate) fn filter(&self) -> SlotBoundFilter<'_> {
        SlotBoundFilter { totals: self }
    }
}

/// Rejects entries whose slot total cannot reach the k-th best.
#[derive(Clone, Copy)]
pub struct SlotBoundFilter<'a> {
    totals: &'a PartitionTotals,
}

impl SlotBoundFilter<'_> {
    /// Whether the group with `hash` could still enter the top-k.
    #[inline(always)]
    pub(crate) fn could_reach(&self, hash: u64) -> bool {
        let totals = self.totals;
        let slot = SlotCountTotals::slot_of(hash) - totals.first_slot;
        totals.totals[slot] >= totals.kth_best
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

    /// Two workers' totals: slot 0 holds 3 and 5, slot 16384 (the first of
    /// partition 1 of 4) holds 9, slot 16385 holds 2.
    fn two_worker_bounds(num_partitions: usize) -> TopKBounds {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut first = SlotCountTotals::new(&mut allocator);
        let mut second = SlotCountTotals::new(&mut allocator);
        first.fold(hash_in_slot(0), 3);
        second.fold(hash_in_slot(0) | 1, 5);
        first.fold(hash_in_slot(16384), 9);
        second.fold(hash_in_slot(16385), 2);
        TopKBounds::new(vec![first, second], num_partitions)
    }

    #[test]
    fn nothing_is_summed_before_a_threshold_exists() {
        let bounds = two_worker_bounds(4);

        assert!(bounds.partition_totals(0).is_none());
    }

    #[test]
    fn partition_totals_sum_the_workers() {
        let bounds = two_worker_bounds(4);
        bounds.threshold().raise(1);

        let totals = bounds.partition_totals(0).unwrap();

        assert_eq!(totals.totals[0], 8);
        assert_eq!(totals.totals.len(), 16384);
    }

    #[test]
    fn partition_skips_once_the_threshold_passes_its_largest_slot() {
        let bounds = two_worker_bounds(4);
        bounds.threshold().raise(9);

        assert!(bounds.partition_totals(0).unwrap().skips_partition());
        assert!(!bounds.partition_totals(1).unwrap().skips_partition());
        assert!(bounds.partition_totals(2).unwrap().skips_partition());
    }

    #[test]
    fn entry_filter_keeps_slots_that_can_tie_the_threshold() {
        let bounds = two_worker_bounds(4);
        bounds.threshold().raise(8);
        let totals = bounds.partition_totals(0).unwrap();

        let filter = totals.filter();

        assert!(filter.could_reach(hash_in_slot(0) | 7));
        assert!(!filter.could_reach(hash_in_slot(1)));
    }

    #[test]
    fn partitions_finer_than_slots_share_the_slot() {
        let bounds = two_worker_bounds(TOTAL_SLOTS * 2);
        bounds.threshold().raise(1);

        let totals = bounds.partition_totals(1).unwrap();

        assert_eq!(totals.totals, vec![8]);
        assert_eq!(totals.first_slot, 0);
    }

    #[test]
    fn threshold_only_rises() {
        let threshold = PruneThreshold::new();

        threshold.raise(10);
        threshold.raise(4);

        assert_eq!(threshold.current(), 10);
    }
}
