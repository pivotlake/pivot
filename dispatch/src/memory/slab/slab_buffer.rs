use crate::memory::slab::Slab;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// Typed buffer backed by a single [`Slab`] (must fit within one 2MB `WriteBuffer`).
///
/// Created via [`SlabAllocator::create_slab_buffer`]. Cheaper to index than [`MultiSlabBuffer`]
/// since there's no slab lookup — just a single pointer offset.
///
/// Supports `Index<usize>` and `IndexMut<usize>` for typed element access. Like `MultiSlabBuffer`,
/// does not track its logical length — callers must track how many elements have been written, and
/// it is UB to access an index not yet set.
pub struct SlabBuffer<T> {
    slab: Slab,
    _phantom: PhantomData<T>,
}

impl<T> SlabBuffer<T> {
    /// Wraps a slab into a typed `SlabBuffer`.
    pub fn new(slab: Slab) -> Self {
        Self {
            slab,
            _phantom: Default::default(),
        }
    }

    /// Returns a raw pointer to the element at `index`.
    /// Simple pointer arithmetic — no slab lookup needed.
    #[inline(always)]
    pub(crate) fn ptr_at_index(&self, index: usize) -> *mut T {
        let byte_offset = index * size_of::<T>();
        unsafe { self.slab.ptr.add(byte_offset) as *mut T }
    }

    /// Consumes the buffer and returns the backing [`Slab`].
    ///
    /// Useful for zero-copy handoff to Arrow: wrap the returned `Slab` in `Arc` and pass
    /// to `Buffer::from_custom_allocation`.
    pub fn into_slab(self) -> Slab {
        self.slab
    }
}

impl<T> Index<usize> for SlabBuffer<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &*self.ptr_at_index(index) }
    }
}

impl<T> IndexMut<usize> for SlabBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut *self.ptr_at_index(index) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{SlabAllocator, init_test_free_pool};

    #[test]
    fn write_and_read() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut buf: SlabBuffer<u64> = alloc.create_slab_buffer(16, true);

        buf[0] = 42;
        buf[15] = 99;

        assert_eq!(buf[0], 42);
        assert_eq!(buf[15], 99);
    }

    #[test]
    fn zeroed_on_creation() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);

        let buf: SlabBuffer<u64> = alloc.create_slab_buffer(128, true);

        for i in 0..128 {
            assert_eq!(buf[i], 0);
        }
    }
}
