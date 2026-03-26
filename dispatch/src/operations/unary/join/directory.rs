use std::cell::UnsafeCell;
use std::ops::{Index, IndexMut};
use crate::memory::{ContiguousMultiBuffer, MultiSlabBuffer};

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

pub struct Directory<B> {
    entries: UnsafeCell<B>,
    capacity: usize,
    pub(crate) shift: u32,
}

unsafe impl<B> Send for Directory<B> {}
unsafe impl<B> Sync for Directory<B> {}

impl<B> Directory<B> {
    pub fn new(entries: B, capacity: usize) -> Self {
        Self {
            entries: UnsafeCell::new(entries),
            capacity,
            shift: if capacity > 0 {
                debug_assert!(capacity.is_power_of_two());
                64 - capacity.trailing_zeros()
            } else {
                64
            },
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

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[inline(always)]
pub fn prefetch_ptr(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = ptr;
}

impl<B: Index<usize, Output = u64> + IndexMut<usize>> Directory<B> {
    #[inline(always)]
    fn entries(&self) -> &B {
        unsafe { &*self.entries.get() }
    }

    #[inline(always)]
    fn entries_mut(&self) -> &mut B {
        unsafe { &mut *self.entries.get() }
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

    /// Prefetch the directory entry for the slot where `hash` would land.
    #[inline(always)]
    pub fn prefetch(&self, hash: u64) {
        let slot = self.slot_for(hash);
        let ptr = &self.entries()[slot + 1] as *const u64 as *const u8;
        prefetch_ptr(ptr);
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
}

/// Wraps two monomorphized `Directory` variants. The match happens once per
/// partition job / probe batch, then the inner loop runs on the concrete
/// `Directory<B>` — no branch per element.
pub enum JoinDirectory {
    Contiguous(Directory<ContiguousMultiBuffer<u64>>),
    NonContiguous(Directory<MultiSlabBuffer<u64>>),
}

impl JoinDirectory {
    pub fn initial() -> Self {
        JoinDirectory::NonContiguous(Directory::new(MultiSlabBuffer::new(vec![]), 0))
    }
}
