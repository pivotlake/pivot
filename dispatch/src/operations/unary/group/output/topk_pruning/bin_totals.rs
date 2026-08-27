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

    /// Adds each scatter bucket's raw-scattered row count to every bin the
    /// bucket covers. Buckets and bins are both prefixes of the same top
    /// hash bits, so a bucket covers a contiguous run of bins (or a bin a
    /// run of buckets).
    ///
    /// This is most definitely a hack and a patch. Raw-scattered rows never
    /// reach the totals, and at worst one group absorbed a whole bucket of
    /// them, so the bucket's row count is added to each of its bins to keep
    /// the bounds valid. It will not be relevant once raw rows are summed
    /// into the totals properly.
    pub(crate) fn add_raw_bucket_rows(&mut self, bucket_rows: &[u64]) {
        let buckets = bucket_rows.len();
        if buckets <= HASH_BINS {
            let bins_per_bucket = HASH_BINS / buckets;
            for bin in 0..HASH_BINS {
                let rows = bucket_rows[bin / bins_per_bucket];
                self.sums[bin] = self.sums[bin].saturating_add(rows);
            }
        } else {
            let buckets_per_bin = buckets / HASH_BINS;
            for (bucket, rows) in bucket_rows.iter().enumerate() {
                let bin = bucket / buckets_per_bin;
                self.sums[bin] = self.sums[bin].saturating_add(*rows);
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

    #[test]
    fn raw_bucket_rows_widen_every_bin_of_the_bucket() {
        let (_allocator, mut totals) = test_bin_totals();
        totals.add(0, 5);
        let mut bucket_rows = vec![0u64; HASH_BINS / 4];
        bucket_rows[0] = 100;

        totals.add_raw_bucket_rows(&bucket_rows);

        assert_eq!(totals.sums[0], 105);
        assert_eq!(totals.sums[3], 100);
        assert_eq!(totals.sums[4], 0);
    }
}
