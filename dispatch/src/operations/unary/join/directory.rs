use crate::memory::MultiSlabBuffer;
use std::cell::UnsafeCell;

pub const PTR_SHIFT: u32 = 16;

const fn build_tag_table() -> [u16; 2048] {
    let mut table = [0u16; 2048];
    let mut i = 0u32;
    while i < 2048 {
        let mut tag: u16 = 0;
        tag |= 1 << (i & 0xF); // bits 0-3
        tag |= 1 << ((i >> 3) & 0xF); // bits 3-6 (overlapping)
        tag |= 1 << ((i >> 6) & 0xF); // bits 6-9 (overlapping)
        tag |= 1 << ((i >> 8) & 0xF); // bits 8-10 (only 3 useful bits, max 7)
        table[i as usize] = tag;
        i += 1;
    }
    table
}
static TAG_TABLE: [u16; 2048] = build_tag_table();

/// The join's hash directory: one `u64` entry per slot, held in ring-slab
/// chunks. Each entry packs an arena end pointer in the upper 48 bits over a
/// 16-bit bloom tag, so a probe rejects most non-matching rows on the entry
/// word alone and otherwise reads its arena range straight out of it.
pub struct JoinDirectory {
    entries: UnsafeCell<MultiSlabBuffer<u64>>,
    capacity: usize,
    pub(crate) shift: u32,
}

unsafe impl Send for JoinDirectory {}
unsafe impl Sync for JoinDirectory {}

impl JoinDirectory {
    pub fn new(entries: MultiSlabBuffer<u64>, capacity: usize) -> Self {
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

    /// The empty directory a [`super::JoinTable`] starts with; the build
    /// replaces it once the table's size is known.
    pub fn initial() -> Self {
        Self::new(MultiSlabBuffer::new(vec![]), 0)
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

    /// Raw pointer to the entry at `slot`, so probe code can compute the exact
    /// address for a hash slot ahead of time and feed many independent loads
    /// to the CPU's reorder buffer.
    #[inline(always)]
    pub fn ptr_for_slot(&self, slot: usize) -> *const u64 {
        unsafe { (*self.entries.get()).ptr_at_index(slot) as *const u64 }
    }

    #[inline(always)]
    pub(crate) fn entries(&self) -> &MultiSlabBuffer<u64> {
        unsafe { &*self.entries.get() }
    }

    /// Read entry at slot (0-indexed into real slots, sentinel is slot -1).
    #[inline(always)]
    pub fn entry(&self, slot: usize) -> u64 {
        self.entries()[slot]
    }

    /// Write entry at slot.
    ///
    /// # Safety
    /// Caller must ensure exclusive access to this slot.
    #[inline(always)]
    pub fn set_entry(&self, slot: usize, value: u64) {
        unsafe { (&mut *self.entries.get())[slot] = value };
    }

    /// Add `value` to the entry at slot.
    ///
    /// # Safety
    /// Caller must ensure exclusive access to this slot.
    #[inline(always)]
    pub unsafe fn add_to_entry(&self, slot: usize, value: u64) {
        unsafe { (&mut *self.entries.get())[slot] += value };
    }

    /// OR `value` into the entry at slot.
    ///
    /// # Safety
    /// Caller must ensure exclusive access to this slot.
    #[inline(always)]
    pub unsafe fn or_to_entry(&self, slot: usize, value: u64) {
        unsafe { (&mut *self.entries.get())[slot] |= value };
    }

    #[inline(always)]
    pub fn prefetch_l2(&self, hash: u64) {
        let slot = self.slot_for(hash);
        let ptr = &self.entries()[slot] as *const u64 as *const u8;
        prefetch_ptr_l2(ptr);
    }

    /// End-pointer (exclusive) stored in the upper 48 bits.
    /// Slot -1 reads the sentinel (always 0).
    #[inline(always)]
    pub fn end_ptr(&self, slot: isize) -> usize {
        (self.entries()[(slot) as usize] >> PTR_SHIFT) as usize
    }
}

#[inline(always)]
pub fn prefetch_ptr_l2(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T1 }>(ptr as *const i8);
    }
    #[cfg(target_arch = "aarch64")]
    #[allow(clippy::pointers_in_nomem_asm_block)]
    unsafe {
        std::arch::asm!("prfm pldl2keep, [{0}]", in(reg) ptr, options(nomem, nostack, preserves_flags));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = ptr;
}
