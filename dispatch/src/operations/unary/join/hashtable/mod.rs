const PTR_SHIFT: u32 = 16;
const BLOOM_MASK: u64 = 0xFFFF;

pub struct Directory {
    pub(crate) entries: Vec<u64>,
    pub(crate) shift: u32,
}

impl Directory {
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
            shift: 64,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        debug_assert!(capacity.is_power_of_two());
        Self {
            entries: vec![0u64; capacity],
            shift: 64 - capacity.trailing_zeros(),
        }
    }

    #[inline(always)]
    pub fn bloom(hash: u64) -> u64 {
        hash & BLOOM_MASK
    }

    #[inline(always)]
    pub fn slot_for(&self, hash: u64) -> usize {
        (hash >> self.shift) as usize
    }

    #[inline(always)]
    pub fn matches_bloom(&self, slot: usize, hash: u64) -> bool {
        let stored = self.entries[slot] & BLOOM_MASK;
        let probe = Self::bloom(hash);
        stored & probe == probe
    }

    /// End-pointer (exclusive) stored in the upper 48 bits.
    #[inline(always)]
    pub fn end_ptr(&self, slot: usize) -> usize {
        (self.entries[slot] >> PTR_SHIFT) as usize
    }

    pub fn capacity(&self) -> usize {
        self.entries.len()
    }
}

mod builder;
