//! A hash-ordered view over a retired table's occupied slots.
//!
//! Tables place entries by the high bits of their hash, so a slot scan is
//! already nearly hash-ordered: linear probing only displaces an entry past
//! its ideal slot by the length of its probe chain, and a probe never crosses
//! an empty slot. Rotating the scan to begin at an empty slot removes the
//! wrap-around cluster, and an insertion pass fixes the remaining bounded,
//! local inversions in near-linear time.

/// Bits of hash prefix resolved by a dense run's bucket index.
pub const BUCKET_BITS: u32 = 12;

/// Number of equal hash ranges a dense run's bucket index resolves.
pub const BUCKET_COUNT: usize = 1 << BUCKET_BITS;

/// Occupied slots of one retired table, ordered by their stored hash.
pub struct SortedRun {
    /// Slot indices into the owning table, ascending by the slot's hash.
    positions: Vec<u32>,
}

impl SortedRun {
    /// Builds a run from `(hash, slot)` pairs collected in rotated slot order.
    ///
    /// The pairs must be nearly sorted (bounded, local inversions only), which
    /// is what a slot scan starting at an empty slot yields.
    pub(super) fn from_nearly_sorted(mut entries: Vec<(u64, u32)>) -> Self {
        let len = entries.len();
        for i in 1..len {
            let current = entries[i];
            if entries[i - 1].0 <= current.0 {
                continue;
            }
            let mut j = i;
            while j > 0 && entries[j - 1].0 > current.0 {
                entries[j] = entries[j - 1];
                j -= 1;
            }
            entries[j] = current;
        }
        Self {
            positions: entries.iter().map(|&(_, slot)| slot).collect(),
        }
    }

    /// Raw pointer to the ordered slot indices, for merge cursors.
    #[inline(always)]
    pub(crate) fn positions_ptr(&self) -> *const u32 {
        self.positions.as_ptr()
    }

    /// Number of entries in the run.
    pub fn len(&self) -> usize {
        self.positions.len()
    }
}
