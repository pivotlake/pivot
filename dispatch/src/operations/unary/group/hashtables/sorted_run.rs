//! A bucket-grouped view over a retired table's occupied slots.
//!
//! The merge folds each partition's entries into a hash table, so it needs
//! every entry of a hash range delivered densely, but in no particular order
//! within the range. Sealing therefore counting-sorts the occupied slots by
//! the top [`BUCKET_BITS`] of their hash: two branchless passes over the
//! table, no comparisons. A merge partition's slice of the run is then two
//! array reads for any power-of-two partition count up to [`BUCKET_COUNT`],
//! and the merge walks only its own entries with no occupancy or partition
//! test per slot.
//!
//! Within a bucket, slots keep their table order, which top-bit placement
//! makes nearly hash-ascending: the fold's inserts still sweep its target
//! left to right.

use crate::operations::unary::group::hll::Hll;

/// Bits of hash prefix resolved by the per-run bucket index.
pub const BUCKET_BITS: u32 = 12;

/// Number of equal hash ranges each run's bucket index resolves.
pub const BUCKET_COUNT: usize = 1 << BUCKET_BITS;

/// Occupied slots of one retired table, grouped by hash bucket.
pub struct SortedRun {
    /// Slot indices into the owning table, ascending by the slot's bucket.
    positions: Vec<u32>,
    /// `bucket_starts[b]` is the first position of bucket `b`; the extra
    /// final element is the total entry count.
    bucket_starts: Vec<u32>,
}

impl SortedRun {
    /// Counting-sorts `(hash, slot)` pairs by bucket, folding each hash into
    /// the worker's distinct-count sketch on the way.
    ///
    /// The pairs arrive in slot-scan order; each bucket keeps that order.
    pub(super) fn from_slot_scan(entries: &[(u64, u32)], hll: &mut Hll) -> Self {
        let mut bucket_starts = vec![0u32; BUCKET_COUNT + 1];
        for &(hash, _) in entries {
            hll.add(hash);
            let bucket = (hash >> (64 - BUCKET_BITS)) as usize;
            bucket_starts[bucket + 1] += 1;
        }
        for b in 0..BUCKET_COUNT {
            bucket_starts[b + 1] += bucket_starts[b];
        }
        let mut cursors = bucket_starts.clone();
        let mut positions = vec![0u32; entries.len()];
        for &(hash, slot) in entries {
            let bucket = (hash >> (64 - BUCKET_BITS)) as usize;
            positions[cursors[bucket] as usize] = slot;
            cursors[bucket] += 1;
        }
        Self {
            positions,
            bucket_starts,
        }
    }

    /// This run's positions within the half-open bucket range `[lo, hi)`.
    #[inline(always)]
    pub(crate) fn bucket_range(&self, lo: usize, hi: usize) -> (usize, usize) {
        (
            self.bucket_starts[lo] as usize,
            self.bucket_starts[hi] as usize,
        )
    }

    /// Raw pointer to the grouped slot indices, for the merge's slice walks.
    #[inline(always)]
    pub(crate) fn positions_ptr(&self) -> *const u32 {
        self.positions.as_ptr()
    }

    /// Number of entries in the run.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.positions.len()
    }
}
