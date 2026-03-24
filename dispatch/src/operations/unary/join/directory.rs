const PTR_SHIFT: u32 = 16;

const fn build_tag_table() -> [u16; 2048] {
    let mut table = [0u16; 2048];
    let mut i = 0u32;
    while i < 2048 {
        let mut tag: u16 = 0;
        tag |= 1 << (i & 0xF);           // bits 0-3
        tag |= 1 << ((i >> 3) & 0xF);    // bits 3-6 (overlapping)
        tag |= 1 << ((i >> 6) & 0xF);    // bits 6-9 (overlapping)
        tag |= 1 << ((i >> 8) & 0xF);    // bits 8-10 (only 3 useful bits, max 7)
        table[i as usize] = tag;
        i += 1;
    }
    table
}
static TAG_TABLE: [u16; 2048] = build_tag_table();

pub struct Directory {
    /// Sentinel at index 0 (always zero), then `capacity` real slots.
    entries: Vec<u64>,
    /// Base pointer into `entries`, past the sentinel (i.e. &entries[1]).
    /// All slot indexing goes through this so hash-derived indices work directly.
    base: *mut u64,
    pub(crate) shift: u32,
}

unsafe impl Send for Directory {}
unsafe impl Sync for Directory {}

impl Directory {
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
            base: std::ptr::null_mut(),
            shift: 64,
        }
    }

    /// Allocate a directory with a zeroed sentinel at index 0 followed by
    /// `capacity` zeroed real slots. `base` points past the sentinel so
    /// `slot_for(hash)` indexes directly without any +1 adjustment.
    pub fn with_capacity(capacity: usize) -> Self {
        debug_assert!(capacity.is_power_of_two());
        let mut entries = vec![0u64; capacity + 1];
        let base = unsafe { entries.as_mut_ptr().add(1) };
        Self {
            entries,
            base,
            shift: 64 - capacity.trailing_zeros(),
        }
    }

    #[inline(always)]
    pub fn compute_tag(hash: u64) -> u16 {
        let slot = ((hash as u32) >> (32 - 11)) as usize;
        TAG_TABLE[slot]
    }

    #[inline(always)]
    pub fn slot_for(&self, hash: u64) -> usize {
        (hash >> self.shift) as usize
    }

    /// Raw base pointer past the sentinel. Slot indices from `slot_for`
    /// can be used directly with this pointer.
    #[inline(always)]
    pub fn base(&self) -> *mut u64 {
        self.base
    }

    #[inline(always)]
    pub fn matches_bloom(&self, hash: u64) -> bool {
        let slot = self.slot_for(hash);
        let stored = unsafe { *self.base.add(slot) };
        let probe = Self::compute_tag(hash) as u64;
        (stored & probe) == probe
    }

    #[inline(always)]
    pub fn bloom(&self, slot: usize) -> u16 {
        let stored = unsafe { *self.base.add(slot) };
        (stored & 0xFFFF) as u16
    }

    /// End-pointer (exclusive) stored in the upper 48 bits.
    #[inline(always)]
    pub fn end_ptr(&self, slot: isize) -> usize {
        (unsafe { *self.base.offset(slot) } >> PTR_SHIFT) as usize
    }

    pub fn capacity(&self) -> usize {
        self.entries.len() - 1
    }
}
