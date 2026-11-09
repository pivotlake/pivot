//! [`SlabList`] - an indexable, append-only list in slab memory.

use super::{Slab, SlabAllocator};
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::{Index, IndexMut};

/// Bytes a chunk aims for; a chunk holds the largest power-of-two count of
/// elements that fits in it.
const TARGET_CHUNK_BYTES: usize = 128 * 1024;

/// An append-only list that owns its elements like a `Vec` but keeps them in
/// slab memory: fixed-size chunks carved from an allocator's ring buffers, so
/// appends never move an element and dropping the list hands back references
/// to those buffers instead of freeing heap blocks. That matters for a list
/// built on one thread and dropped on another, where a large heap block goes
/// back to an allocator arena the dropping thread does not own.
///
/// Every chunk holds the same power-of-two count of elements, so indexing is
/// a shift and a mask.
pub struct SlabList<T> {
    chunks: Vec<Slab>,
    len: usize,
    _phantom: PhantomData<T>,
}

impl<T> Default for SlabList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SlabList<T> {
    const CHUNK_LEN: usize = {
        assert!(size_of::<T>() > 0 && size_of::<T>() <= TARGET_CHUNK_BYTES);
        1 << (TARGET_CHUNK_BYTES / size_of::<T>()).ilog2()
    };
    const CHUNK_SHIFT: u32 = Self::CHUNK_LEN.trailing_zeros();

    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            len: 0,
            _phantom: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Appends `value`, taking a new chunk from `allocator` when the last one
    /// is full.
    pub fn push(&mut self, allocator: &mut SlabAllocator, value: T) {
        if self.len == self.chunks.len() << Self::CHUNK_SHIFT {
            self.chunks.push(allocator.get_aligned_slab(
                Self::CHUNK_LEN * size_of::<T>(),
                align_of::<T>(),
                false,
            ));
        }
        // SAFETY: the chunk for index `len` exists, just above at the latest,
        // and the slot has never held an element.
        unsafe { self.ptr_at_index(self.len).write(value) };
        self.len += 1;
    }

    /// A raw pointer to the slot of `index`, which must lie in a chunk the
    /// list has taken. The slot holds an element only below [`len`](Self::len).
    #[inline(always)]
    pub fn ptr_at_index(&self, index: usize) -> *mut T {
        let chunk = &self.chunks[index >> Self::CHUNK_SHIFT];
        // SAFETY: a chunk holds `CHUNK_LEN` elements and the masked index is
        // below that.
        unsafe { (chunk.ptr as *mut T).add(index & (Self::CHUNK_LEN - 1)) }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.len).map(|index| &self[index])
    }

    /// The same list with every element in an `UnsafeCell`, for a list whose
    /// elements are from now on reached through a shared reference. No
    /// element moves.
    pub fn into_cells(self) -> SlabList<UnsafeCell<T>> {
        let mut list = ManuallyDrop::new(self);
        SlabList {
            chunks: std::mem::take(&mut list.chunks),
            len: list.len,
            _phantom: PhantomData,
        }
    }

    /// Empties the list without dropping its elements, leaking whatever they
    /// own. For a list whose elements are known to own nothing any more, so
    /// that dropping the list does not walk them.
    pub fn forget_elements(&mut self) {
        self.len = 0;
    }
}

impl<T> Index<usize> for SlabList<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &T {
        assert!(index < self.len, "index {index} out of {}", self.len);
        // SAFETY: slots below `len` hold elements.
        unsafe { &*self.ptr_at_index(index) }
    }
}

impl<T> IndexMut<usize> for SlabList<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut T {
        assert!(index < self.len, "index {index} out of {}", self.len);
        // SAFETY: slots below `len` hold elements.
        unsafe { &mut *self.ptr_at_index(index) }
    }
}

impl<T> Drop for SlabList<T> {
    fn drop(&mut self) {
        if !std::mem::needs_drop::<T>() {
            return;
        }
        for index in 0..self.len {
            // SAFETY: slots below `len` hold elements, each dropped once here.
            unsafe { self.ptr_at_index(index).drop_in_place() };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use std::rc::Rc;

    #[test]
    fn elements_read_back_by_index_across_chunks() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(false);
        let mut list = SlabList::new();
        let count = 3 * TARGET_CHUNK_BYTES / size_of::<u64>();

        for value in 0..count as u64 {
            list.push(&mut allocator, value);
        }

        assert_eq!(list.len(), count);
        assert!(list.iter().copied().eq(0..count as u64));
        assert_eq!(list[count - 1], count as u64 - 1);
    }

    #[test]
    fn dropping_the_list_drops_its_elements() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(false);
        let tracked = Rc::new(());
        let mut list = SlabList::new();
        list.push(&mut allocator, tracked.clone());
        list.push(&mut allocator, tracked.clone());

        drop(list);

        assert_eq!(Rc::strong_count(&tracked), 1);
    }

    #[test]
    fn forgotten_elements_are_not_dropped_with_the_list() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(false);
        let tracked = Rc::new(());
        let mut list = SlabList::new();
        list.push(&mut allocator, tracked.clone());

        list.forget_elements();
        drop(list);

        assert_eq!(Rc::strong_count(&tracked), 2);
    }

    #[test]
    fn cells_keep_the_elements_in_place() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(false);
        let mut list = SlabList::new();
        list.push(&mut allocator, 7u32);
        let slot = list.ptr_at_index(0) as usize;

        let cells = list.into_cells();

        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].get() as usize, slot);
        assert_eq!(unsafe { *cells[0].get() }, 7);
    }
}
