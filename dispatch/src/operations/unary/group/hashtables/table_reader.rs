use super::hash_table::{EntryLayout, EntryView, fast_div, reciprocal};
use super::{AggregationValue, PersistedKey};
use std::marker::PhantomData;

/// A copyable reader over a table's slots: how hashes and slot indices map
/// to table memory, plus entry access through that mapping.
///
/// A table is a logical array of slots split across fixed-size slabs. Entries
/// never cross a slab boundary, so a global slot index first selects a slab and
/// then selects an entry within it:
///
/// ```text
/// logical slots
///  0          entries_per_slab - 1  entries_per_slab             capacity - 1
///  |---------------------------|    |--------------------------------------|
///              slab 0                              slab 1
///
/// one entry
///  +----------------------- entry_stride ------------------------+
///  | padding | hash | padding | key | padding | stored value ... |
///  +-------------------------------------------------------------+
///            ^                ^               ^
///            hash_offset      key_offset      value_offset
/// ```
///
/// Each slab base is adjusted by the global index of its first entry:
///
/// ```text
/// adjusted_base[slab] =
///     allocated_base[slab] - first_global_index[slab] * entry_stride
///
/// entry_address(index) =
///     adjusted_base[index / entries_per_slab] + index * entry_stride
/// ```
///
/// This adjustment lets the final address calculation use the global index
/// directly. Division by `entries_per_slab` is implemented as a widening
/// multiply by `entries_per_slab_reciprocal`.
///
/// The initial slot comes from the high bits of the hash:
///
/// ```text
/// initial_slot = (hash << hash_left_shift) >> slot_shift
/// ```
///
/// Linear probing advances that slot and wraps with `slot_mask`.
///
/// This exists as a copyable value, separate from the table, so that hot
/// loops keep the layout in registers. Every field here is also stored on the
/// table itself, but as ordinary struct fields behind a reference, and the
/// loops that need them write entries through raw pointers. The optimizer
/// must assume a raw-pointer store can overwrite any memory it cannot prove
/// disjoint, including the table's own fields, so a loop reading the stride
/// or an offset through the table reference has to reload it from memory
/// after every entry write, once per row and per probe step. A copy of this
/// struct is a bundle of local variables whose address is never exposed: no
/// store can alias them, and they stay in registers for the whole scan.
///
/// Constructing the reader is also where a specialized slot count enters:
/// built from constant metadata, the layout arithmetic constant-folds inside
/// that instantiation, which table fields, being runtime values shared by
/// every instantiation, never could.
///
/// The lifetime records that the raw slab addresses belong to a borrowed
/// table. A reader must be rebuilt after resizing because resizing replaces
/// the slab list and changes the slot mask and shift.
pub(crate) struct TableReader<'table, K, V: AggregationValue + ?Sized> {
    /// Reciprocal of the number of whole entries in one slab.
    ///
    /// `fast_div(index, entries_per_slab_reciprocal)` returns the slab that
    /// owns the global slot index without issuing an integer division.
    pub(super) entries_per_slab_reciprocal: u64,

    /// Distance in bytes from one entry to the next entry in the same slab.
    pub(super) entry_stride: usize,

    /// Byte offset of the stored hash from the start of an entry.
    pub(super) hash_offset: usize,

    /// Byte offset of the persisted group key from the start of an entry.
    pub(super) key_offset: usize,

    /// Byte offset of the stored aggregation value from the start of an entry.
    pub(super) value_offset: usize,

    /// Bit mask used to wrap a linear probe at the table capacity.
    ///
    /// The capacity is a power of two, so this is always `capacity - 1`.
    pub(super) slot_mask: usize,

    /// Number of bits to shift right after applying `hash_left_shift`.
    ///
    /// This selects exactly the high hash bits needed for the current table
    /// capacity.
    pub(super) slot_shift: u32,

    /// Number of partition bits to discard from the high end of the hash.
    ///
    /// Ordinary tables use zero. Partition-local merge tables shift away the
    /// partition prefix so the following bits choose the initial slot.
    pub(super) hash_left_shift: u32,

    /// Pointer to the table's array of adjusted slab base addresses.
    ///
    /// The pointer has `slab_count` readable elements and remains valid for
    /// `'table`. See the type-level diagram for the base adjustment.
    adjusted_slab_bases: *const usize,

    /// Number of adjusted slab base addresses available through
    /// `adjusted_slab_bases`.
    slab_count: usize,

    /// First adjusted slab base, cached for the common one-slab case.
    ///
    /// This avoids loading `adjusted_slab_bases` when every slot is in slab
    /// zero.
    first_adjusted_slab_base: usize,

    /// Runtime metadata required to interpret the stored aggregation value.
    ///
    /// For a dynamic aggregation tuple this includes its arity. Compiled value
    /// types generally use zero-sized metadata.
    pub(super) storage_metadata: V::StorageMetadata,

    /// Associates the raw slab addresses with the table and key type.
    ///
    /// No `K` value is stored in this snapshot, and raw pointers do not carry
    /// the lifetime of the allocation they address, so this marker records both.
    table_borrow: PhantomData<&'table K>,
}

impl<K, V: AggregationValue + ?Sized> Clone for TableReader<'_, K, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K, V: AggregationValue + ?Sized> Copy for TableReader<'_, K, V> {}

impl<'table, K: PersistedKey, V: AggregationValue + ?Sized> TableReader<'table, K, V> {
    /// Creates a layout whose raw addresses are valid for `'table`.
    ///
    /// # Safety
    ///
    /// `adjusted_slab_bases` must remain allocated and unchanged for `'table`.
    /// Every base must address a slab laid out according to `entry_layout`.
    /// `slot_mask` must not name slots beyond those slabs.
    #[inline(always)]
    pub(super) unsafe fn new(
        entry_layout: EntryLayout,
        entries_per_slab: usize,
        slot_mask: usize,
        slot_shift: u32,
        hash_left_shift: u32,
        adjusted_slab_bases: &[usize],
        storage_metadata: V::StorageMetadata,
    ) -> Self {
        debug_assert!(!adjusted_slab_bases.is_empty());
        Self {
            entries_per_slab_reciprocal: reciprocal(entries_per_slab as u64),
            entry_stride: entry_layout.stride,
            hash_offset: entry_layout.hash_offset,
            key_offset: entry_layout.key_offset,
            value_offset: entry_layout.value_offset,
            slot_mask,
            slot_shift,
            hash_left_shift,
            adjusted_slab_bases: adjusted_slab_bases.as_ptr(),
            slab_count: adjusted_slab_bases.len(),
            first_adjusted_slab_base: adjusted_slab_bases[0],
            storage_metadata,
            table_borrow: PhantomData,
        }
    }

    /// Maps a hash to its initial slot using the table's selected high bits.
    #[inline(always)]
    pub(super) fn slot_for(&self, hash: u64) -> usize {
        ((hash << self.hash_left_shift) >> self.slot_shift) as usize
    }

    /// Returns the address of the entry at `index`.
    #[inline(always)]
    pub(crate) fn entry_ptr(&self, index: usize) -> *mut u8 {
        if self.slab_count == 1 {
            return self
                .first_adjusted_slab_base
                .wrapping_add(index * self.entry_stride) as *mut u8;
        }

        let slab_index = fast_div(index, self.entries_per_slab_reciprocal);
        assert!(slab_index < self.slab_count);
        let adjusted_slab_base = unsafe { *self.adjusted_slab_bases.add(slab_index) };
        adjusted_slab_base.wrapping_add(index * self.entry_stride) as *mut u8
    }

    /// Reads the hash of an entry the collector already located.
    #[inline(always)]
    pub(crate) fn hash_of(&self, entry: *const u8) -> u64 {
        unsafe { *(entry.add(self.hash_offset) as *const u64) }
    }

    /// Returns the hash stored at `index`, or zero when the slot is empty.
    #[cfg(test)]
    #[inline(always)]
    pub(crate) fn hash_at(&self, index: usize) -> u64 {
        self.hash_of(self.entry_ptr(index))
    }

    /// Builds a view of an entry the collector already located.
    ///
    /// # Safety
    ///
    /// `entry` must be an occupied entry address from this reader's table.
    #[inline(always)]
    pub(crate) unsafe fn view_of(&self, entry: *const u8, hash: u64) -> EntryView<'table, K, V> {
        unsafe { self.view(entry, hash) }
    }

    /// Returns a borrowed view of the entry at `index`.
    ///
    /// Empty slots are represented by a view whose `hash` is zero. Callers must
    /// inspect the hash before reading the key or stored value of an empty slot.
    #[cfg(test)]
    #[inline(always)]
    pub(crate) fn view_at(&self, index: usize) -> EntryView<'table, K, V> {
        let entry = self.entry_ptr(index);
        let hash = unsafe { *(entry.add(self.hash_offset) as *const u64) };
        unsafe { self.view(entry, hash) }
    }

    /// Builds a view of an occupied entry after its hash has already been read.
    ///
    /// # Safety
    ///
    /// `entry` must point to an initialized entry in the borrowed table.
    #[inline(always)]
    pub(super) unsafe fn view(&self, entry: *const u8, hash: u64) -> EntryView<'table, K, V> {
        unsafe {
            EntryView {
                hash,
                key: &*(entry.add(self.key_offset) as *const K),
                stored: V::from_entry(entry.add(self.value_offset), self.storage_metadata),
            }
        }
    }
}
