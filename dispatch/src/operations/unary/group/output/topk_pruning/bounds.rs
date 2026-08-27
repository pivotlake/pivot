//! Per-bin and per-partition upper bounds built from the pool-wide totals.

use crate::operations::unary::group::output::topk_pruning::{HASH_BINS, HashBinTotals, hash_bin};

/// Per-bin upper bounds on any single group's final value, shared by every
/// merge job of one query.
///
/// Built from the pool-wide bin totals. Bounds prune at two
/// granularities: a job whose partition's maximum bound is below the k-th
/// best skips entirely, and a job that does run skips the individual source
/// entries of bins that cannot reach it (a group's partials all share a hash
/// and therefore a bin, so bin-level skipping drops a group in full or not at
/// all).
pub(crate) struct TopKBounds {
    bin_bounds: Vec<u64>,
}

impl TopKBounds {
    /// Takes the pool-wide bin totals as the per-bin bounds.
    pub(crate) fn build(totals: HashBinTotals) -> Self {
        Self {
            bin_bounds: totals.into_bin_totals(),
        }
    }

    /// Builds the cutoff a merge job should apply while scanning one
    /// partition, or `None` when applying one would be wasted work.
    ///
    /// `kth_best` is the k-th best exact count found so far. A group can only
    /// be rejected when its bin bound is below that count, so there are two
    /// situations where the cutoff could never reject anything:
    ///
    /// - `kth_best` is still 0, meaning fewer than k groups have been merged
    ///   pool-wide. No bound is below 0.
    /// - Every bin in this partition has a bound of at least `kth_best`. The
    ///   check is made on the partition's smallest bin bound: if even that
    ///   one reaches `kth_best`, all of them do.
    ///
    /// Returning `None` in those cases lets the job merge the partition with
    /// no per-entry check at all, which matters because the check costs a
    /// bounds lookup for every entry the scan touches.
    pub(crate) fn cutoff_for_partition(
        &self,
        partition: usize,
        num_partitions: usize,
        kth_best: u64,
    ) -> Option<TopKCutoff<'_>> {
        if kth_best == 0 {
            return None;
        }
        let start = partition * HASH_BINS / num_partitions;
        let end = ((partition + 1) * HASH_BINS / num_partitions).max(start + 1);
        let min_bound = self.bin_bounds[start..end]
            .iter()
            .copied()
            .min()
            .expect("a partition covers at least one bin");
        (min_bound < kth_best).then_some(TopKCutoff {
            bounds: self,
            kth_best,
        })
    }

    /// The per-partition upper bound on any single group's final value, for a
    /// merge over `num_partitions` (a power of two) hash-prefix partitions.
    ///
    /// A partition's bound is the maximum bin bound over the bins its hash
    /// range covers (a partition finer than a bin shares that bin's bound).
    pub(crate) fn partition_bounds(&self, num_partitions: usize) -> Vec<u64> {
        (0..num_partitions)
            .map(|partition| {
                let start = partition * HASH_BINS / num_partitions;
                let end = ((partition + 1) * HASH_BINS / num_partitions).max(start + 1);
                self.bin_bounds[start..end].iter().copied().max().unwrap()
            })
            .collect()
    }
}

/// The count a group must be able to reach to be worth merging: the k-th
/// best exact count some worker already holds, snapshotted once per merge
/// job, with the bin bounds that say which groups can reach it.
///
/// A group's partials all share a hash and therefore a bin, so the cutoff
/// admits or rejects a group in full, never a single partial of it.
#[derive(Clone, Copy)]
pub(crate) struct TopKCutoff<'a> {
    bounds: &'a TopKBounds,
    kth_best: u64,
}

impl TopKCutoff<'_> {
    /// Whether the group hashing to `hash` could still reach the cutoff.
    #[inline(always)]
    pub(crate) fn admits(&self, hash: u64) -> bool {
        self.bounds.bin_bounds[hash_bin(hash)] >= self.kth_best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{SlabAllocator, init_test_free_pool};
    use crate::operations::unary::group::output::topk_pruning::HASH_BIN_BITS;

    fn test_bin_totals() -> (SlabAllocator, HashBinTotals) {
        init_test_free_pool(8);
        let mut allocator = SlabAllocator::new(false);
        let totals = HashBinTotals::new(&mut allocator);
        (allocator, totals)
    }

    #[test]
    fn partition_bounds_take_range_maxima() {
        let (_allocator, mut totals) = test_bin_totals();
        // Two bins inside partition 1 of 2: the bound takes the larger.
        let upper_half = 1u64 << (u64::BITS - 1);
        totals.add(upper_half, 100);
        totals.add(upper_half | (1 << (u64::BITS - HASH_BIN_BITS)), 250);

        let bounds = TopKBounds::build(totals).partition_bounds(2);

        assert_eq!(bounds, vec![0, 250]);
    }

    #[test]
    fn partitions_finer_than_bins_share_the_bin_total() {
        let (_allocator, mut totals) = test_bin_totals();
        totals.add(0, 8);

        let bounds = TopKBounds::build(totals).partition_bounds(2 * HASH_BINS);

        assert_eq!(bounds[0], 8);
        assert_eq!(bounds[1], 8);
        assert_eq!(bounds[2], 0);
    }
}
