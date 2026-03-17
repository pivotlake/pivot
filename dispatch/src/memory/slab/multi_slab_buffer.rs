use crate::memory::BUFFER_SIZE;
use crate::memory::slab::Slab;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

const BUFFER_SHIFT: u32 = BUFFER_SIZE.trailing_zeros();
const BUFFER_MASK: usize = BUFFER_SIZE - 1;

/// Typed buffer backed by one or more [`Slab`]s, supporting allocations that span
/// multiple 2MB `WriteBuffer`s.
///
/// Created via [`SlabAllocator::create_multi_slab_buffer`]. Each slab is aligned to the
/// start of a 2MB buffer, so element indexing works by computing a byte offset, then using
/// bit-shift (`>> 21`) to find the slab and bit-mask (`& 0x1FFFFF`) for the offset within it.
///
/// Supports `Index<usize>` and `IndexMut<usize>` for typed element access. The buffer does
/// not track its logical length — callers must track how many elements have been written. This
/// means accessing indexes that have not yet been set is UB.
pub struct MultiSlabBuffer<T> {
    slabs: Vec<Slab>,
    _phantom: PhantomData<T>,
}

impl<T> MultiSlabBuffer<T> {
    /// Wraps existing slabs into a `MultiSlabBuffer`. Each slab must start at offset 0
    /// of its `WriteBuffer` for indexing to work correctly.
    pub fn new(slabs: Vec<Slab>) -> Self {
        Self {
            slabs,
            _phantom: Default::default(),
        }
    }

    /// Returns a raw pointer to the element at `index`.
    ///
    /// Computes the byte offset, determines which slab it falls in via bit-shift, and
    /// the offset within that slab via bit-mask.
    pub(crate) fn ptr_at_index(&self, index: usize) -> *mut T {
        let byte_offset = index * size_of::<T>();
        let buffer_idx = byte_offset >> BUFFER_SHIFT;
        let offset_in_buffer = byte_offset & BUFFER_MASK;
        unsafe { self.slabs[buffer_idx].ptr.add(offset_in_buffer) as *mut T }
    }

    /// Consumes the buffer and returns the single backing `Slab`.
    ///
    /// Useful for zero-copy handoff to Arrow: wrap the returned `Slab` in `Arc` and pass
    /// to `Buffer::from_custom_allocation`.
    ///
    /// Panics (debug) if the buffer spans more than one slab.
    pub fn into_single_slab(self) -> Slab {
        debug_assert_eq!(
            self.slabs.len(),
            1,
            "MultiSlabBuffer spans {} slabs, expected 1",
            self.slabs.len()
        );
        self.slabs.into_iter().next().unwrap()
    }
}

impl<T> Index<usize> for MultiSlabBuffer<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.ptr_at_index(index) }
    }
}

impl<T> IndexMut<usize> for MultiSlabBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *self.ptr_at_index(index) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{BUFFER_SIZE, SlabAllocator, init_test_free_pool};

    #[test]
    fn write_and_read() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut buf: MultiSlabBuffer<u32> = alloc.create_multi_slab_buffer(64, true);

        buf[0] = 10;
        buf[63] = 20;

        assert_eq!(buf[0], 10);
        assert_eq!(buf[63], 20);
    }

    #[test]
    fn zeroed_on_creation() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);

        let buf: MultiSlabBuffer<u64> = alloc.create_multi_slab_buffer(256, true);

        for i in 0..256 {
            assert_eq!(buf[i], 0);
        }
    }

    #[test]
    fn spans_multiple_buffers() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let elements = BUFFER_SIZE / size_of::<u64>() + 100;

        let mut buf: MultiSlabBuffer<u64> = alloc.create_multi_slab_buffer(elements, true);
        buf[0] = 1;
        buf[elements - 1] = 2;

        assert_eq!(buf[0], 1);
        assert_eq!(buf[elements - 1], 2);
    }

    #[test]
    fn into_single_slab() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut buf: MultiSlabBuffer<u32> = alloc.create_multi_slab_buffer(8, true);
        buf[0] = 0xDEAD;

        let slab = buf.into_single_slab();

        let val = unsafe { *(slab.ptr as *const u32) };
        assert_eq!(val, 0xDEAD);
    }
}
