use std::cell::UnsafeCell;
use crate::memory::{MultiSlabBuffer, SlabAllocator};

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
    entries: UnsafeCell<MultiSlabBuffer<u64>>,
    capacity: usize,
    pub(crate) shift: u32,
}

unsafe impl Send for Directory {}
unsafe impl Sync for Directory {}

impl Directory {
    pub fn empty() -> Self {
        Self {
            entries: UnsafeCell::new(MultiSlabBuffer::new(vec![])),
            capacity: 0,
            shift: 64,
        }
    }

    /// Allocate a directory with a zeroed sentinel at index 0 followed by
    /// `capacity` zeroed real slots.
    pub fn with_capacity(capacity: usize, allocator: &mut SlabAllocator) -> Self {
        debug_assert!(capacity.is_power_of_two());
        let entries = allocator.create_multi_slab_buffer::<u64>(capacity + 1, true);
        Self {
            entries: UnsafeCell::new(entries),
            capacity,
            shift: 64 - capacity.trailing_zeros(),
        }
    }

    #[inline(always)]
    fn entries(&self) -> &MultiSlabBuffer<u64> {
        unsafe { &*self.entries.get() }
    }

    #[inline(always)]
    fn entries_mut(&self) -> &mut MultiSlabBuffer<u64> {
        unsafe { &mut *self.entries.get() }
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

    /// Read entry at slot (0-indexed into real slots, sentinel is slot -1).
    #[inline(always)]
    pub fn entry(&self, slot: usize) -> u64 {
        self.entries()[slot + 1]
    }

    /// Write entry at slot.
    ///
    /// # Safety
    /// Caller must ensure exclusive access to this slot.
    #[inline(always)]
    pub unsafe fn set_entry(&self, slot: usize, value: u64) {
        self.entries_mut()[slot + 1] = value;
    }

    /// Add `value` to the entry at slot.
    ///
    /// # Safety
    /// Caller must ensure exclusive access to this slot.
    #[inline(always)]
    pub unsafe fn add_to_entry(&self, slot: usize, value: u64) {
        self.entries_mut()[slot + 1] += value;
    }

    #[inline(always)]
    pub fn matches_bloom(&self, hash: u64) -> bool {
        let slot = self.slot_for(hash);
        let stored = self.entries()[slot + 1];
        let probe = Self::compute_tag(hash) as u64;
        (stored & probe) == probe
    }

    #[inline(always)]
    pub fn bloom(&self, slot: usize) -> u16 {
        (self.entries()[slot + 1] & 0xFFFF) as u16
    }

    /// End-pointer (exclusive) stored in the upper 48 bits.
    /// Slot -1 reads the sentinel (always 0).
    #[inline(always)]
    pub fn end_ptr(&self, slot: isize) -> usize {
        (self.entries()[(slot + 1) as usize] >> PTR_SHIFT) as usize
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}
