//! One worker's sums of the ordering aggregate's partials by hash bin.

use crate::memory::{SlabAllocator, SlabBuffer};
use crate::operations::unary::group::output::topk_pruning::{HASH_BINS, hash_bin};

/// Per-worker (and, after merging, global) sums of the ordering aggregate's
/// partials by top hash bits.
///
/// Slab-backed on purpose: a fresh heap allocation this size per worker per
/// query means an mmap and munmap each time, and on a many-core box the
/// munmaps alone are a TLB-shootdown storm across every CPU. Slab memory is
/// pooled and stays mapped, so the array costs one memset of already-resident
/// pages.
pub struct HashBinTotals {
    sums: SlabBuffer<u64>,
}

// The slab's pages are address-stable pool memory and the totals are accessed
// by one owner at a time (a worker, then whichever worker steals it from
// `SharedBinTotals`, then the gather's final arrival).
unsafe impl Send for HashBinTotals {}

impl HashBinTotals {
    pub(crate) fn new(allocator: &mut SlabAllocator) -> Self {
        Self {
            sums: allocator.create_slab_buffer(HASH_BINS, true),
        }
    }

    fn as_slice(&self) -> &[u64] {
        // In-bounds: the buffer was created with HASH_BINS bins.
        unsafe { std::slice::from_raw_parts(self.sums.ptr_at_index(0), HASH_BINS) }
    }

    /// Folds one partial value of the ordering aggregate into its bin.
    ///
    /// Saturating: a saturated bin only loosens the bound, which stays valid.
    #[inline(always)]
    pub(crate) fn add(&mut self, hash: u64, weight: u64) {
        let bin = hash_bin(hash);
        self.sums[bin] = self.sums[bin].saturating_add(weight);
    }

    /// Sums another worker's bin totals into this one.
    pub(crate) fn merge(&mut self, other: &HashBinTotals) {
        let base = self.sums.ptr_at_index(0);
        for (bin, other_sum) in other.as_slice().iter().enumerate() {
            // One base pointer for the whole pass; `SlabBuffer` indexing would
            // reload it per bin.
            unsafe {
                let sum = base.add(bin);
                *sum = (*sum).saturating_add(*other_sum);
            }
        }
    }

    /// Copies the bin totals out of the slab, freeing it for reuse.
    pub(crate) fn into_bin_totals(self) -> Vec<u64> {
        self.as_slice().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use crate::operations::unary::group::output::topk_pruning::HASH_BIN_BITS;

    fn test_bin_totals() -> (SlabAllocator, HashBinTotals) {
        init_test_free_pool(8);
        let mut allocator = SlabAllocator::new(false);
        let totals = HashBinTotals::new(&mut allocator);
        (allocator, totals)
    }

    #[test]
    fn bin_sums_accumulate_by_top_bits() {
        let (_allocator, mut totals) = test_bin_totals();
        let hash_a = 3u64 << (u64::BITS - HASH_BIN_BITS);
        let hash_b = hash_a | 0x1fff; // Same top bits, different low bits.

        totals.add(hash_a, 5);
        totals.add(hash_b, 7);

        assert_eq!(totals.sums[3], 12);
        assert_eq!(totals.sums[2], 0);
    }

    #[test]
    fn merge_sums_workers_elementwise() {
        let (mut allocator, mut a) = test_bin_totals();
        let mut b = HashBinTotals::new(&mut allocator);
        let hash = 9u64 << (u64::BITS - HASH_BIN_BITS);
        a.add(hash, 10);
        b.add(hash, 32);

        a.merge(&b);

        assert_eq!(a.sums[9], 42);
    }
}
