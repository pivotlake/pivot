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

/// Most hash-prefix bits a run's bucket index resolves.
///
/// Fine enough that a large merge splits into cache-resident targets and
/// folds them in cache-resident steps, while consolidation's per-bucket
/// write cursors still fit in L2: at this resolution the placement pass
/// keeps at most 8192 open write streams.
pub const BUCKET_BITS: u32 = 13;

/// Fewest hash-prefix bits a run's bucket index resolves. Bounds how far the
/// merge can split the key space, so it stays above the partition floor a
/// full worker pool asks for.
pub const MIN_BUCKET_BITS: u32 = 10;

/// Bucket resolution for a lone table of `len` entries: roughly 64 entries
/// per bucket, clamped to `[MIN_BUCKET_BITS, BUCKET_BITS]`. Small runs get
/// small index arrays, so the fixed cost of building and walking them never
/// outweighs the entries themselves. Tables in a stack always seal at full
/// resolution instead: a stack implies high cardinality, and the coarsest
/// source bounds how finely the merge can split the key space.
pub(super) fn bucket_bits_for(len: usize) -> u32 {
    let magnitude = usize::BITS - len.max(1).leading_zeros();
    magnitude
        .saturating_sub(6)
        .clamp(MIN_BUCKET_BITS, BUCKET_BITS)
}

/// Occupied slots of one retired table, grouped by hash bucket.
pub struct SortedRun {
    /// Slot indices into the owning table, ascending by the slot's bucket.
    positions: Vec<u32>,
    /// `bucket_starts[b]` is the first position of bucket `b`; the extra
    /// final element is the total entry count.
    bucket_starts: Vec<u32>,
    /// Hash-prefix bits this run's index resolves.
    bucket_bits: u32,
}

impl SortedRun {
    /// Counting-sorts `(hash, slot)` pairs by bucket, folding each hash into
    /// the worker's distinct-count sketch on the way.
    ///
    /// The pairs arrive in slot-scan order; each bucket keeps that order.
    pub(super) fn from_slot_scan(entries: &[(u64, u32)], hll: &mut Hll, bucket_bits: u32) -> Self {
        let bucket_count = 1usize << bucket_bits;
        let mut bucket_starts = vec![0u32; bucket_count + 1];
        for &(hash, _) in entries {
            hll.add(hash);
            let bucket = (hash >> (64 - bucket_bits)) as usize;
            bucket_starts[bucket + 1] += 1;
        }
        for b in 0..bucket_count {
            bucket_starts[b + 1] += bucket_starts[b];
        }
        let mut cursors = bucket_starts.clone();
        let mut positions = vec![0u32; entries.len()];
        for &(hash, slot) in entries {
            let bucket = (hash >> (64 - bucket_bits)) as usize;
            positions[cursors[bucket] as usize] = slot;
            cursors[bucket] += 1;
        }
        Self {
            positions,
            bucket_starts,
            bucket_bits,
        }
    }

    /// Hash-prefix bits this run's index resolves.
    #[inline(always)]
    pub(crate) fn bucket_bits(&self) -> u32 {
        self.bucket_bits
    }

    /// This run's positions within the global bucket range `[lo, hi)`.
    ///
    /// `lo` and `hi` are indices at [`BUCKET_BITS`] resolution and must be
    /// aligned to this run's coarser resolution, which the merge guarantees
    /// by never splitting finer than any source resolves.
    #[inline(always)]
    pub(crate) fn bucket_range(&self, lo: usize, hi: usize) -> (usize, usize) {
        let shift = BUCKET_BITS - self.bucket_bits;
        (
            self.bucket_starts[lo >> shift] as usize,
            self.bucket_starts[hi >> shift] as usize,
        )
    }

    /// Raw pointer to the grouped slot indices, for the merge's slice walks.
    #[inline(always)]
    pub(crate) fn positions_ptr(&self) -> *const u32 {
        self.positions.as_ptr()
    }

    /// Number of entries in the run.
    pub fn len(&self) -> usize {
        self.positions.len()
    }
}
