use crate::memory::slab::slab_buffer::SlabBuffer;
use crate::memory::slab::{MultiSlabBuffer, Slab};
use crate::memory::{BUFFER_SIZE, WriteBuffer, memory_ctx};
use std::sync::Arc;

/// Bump allocator that carves [`Slab`]s out of 2MB [`WriteBuffer`]s.
///
/// Maintains a "working buffer" and an offset into it. Each allocation bumps the offset
/// forward; when the current buffer doesn't have enough room, a fresh buffer is acquired
/// from the free pool.
///
/// Because all slabs carved from the same buffer share an `Arc<WriteBuffer>`, the buffer
/// is pinned until the last slab is dropped. To avoid pinning buffers longer than necessary,
/// use separate allocators for data with different lifetimes (e.g. one for short-lived scratch
/// buffers, another for output that lives until Arrow arrays are consumed).
pub struct SlabAllocator {
    /// The current buffer we're bumping through.
    working_buffer: Arc<WriteBuffer>,
    /// Byte offset of the next free region in `working_buffer`.
    offset: usize,
}

impl SlabAllocator {
    /// Creates a new allocator with a fresh buffer from the free pool.
    /// If `zeroed` is true, prefers a pre-zeroed buffer.
    pub fn new(zeroed: bool) -> Self {
        Self {
            working_buffer: Arc::new(memory_ctx().get_write_buffer(zeroed)),
            offset: 0,
        }
    }

    /// Bytes remaining in the current working buffer.
    #[inline(always)]
    fn remaining_in_buffer(&self) -> usize {
        BUFFER_SIZE - self.offset
    }

    /// Carves a slab of `size` bytes from the current buffer without checking capacity.
    /// Zeroes the slab if `zeroed` is requested but the buffer wasn't pre-zeroed.
    fn get_slab_from_current_buffer(&mut self, size: usize, zeroed: bool) -> Slab {
        debug_assert!(size <= self.remaining_in_buffer());
        let mut slab = Slab {
            ptr: unsafe { self.working_buffer.ptr.add(self.offset) },
            size,
            _buffer: self.working_buffer.clone(),
        };
        if zeroed && !self.working_buffer.zeroed {
            slab.zero_out();
        }

        self.offset += size;
        slab
    }

    /// Discards the remaining space in the current buffer and acquires a fresh one.
    fn advance_to_new_buffer(&mut self, zeroed: bool) {
        self.offset = 0;
        self.working_buffer = Arc::new(memory_ctx().get_write_buffer(zeroed));
    }

    /// Allocates a single slab of exactly `size` bytes.
    ///
    /// Panics if `size >= BUFFER_SIZE` (2MB) — use [`get_slabs_of_size`](Self::get_slabs_of_size)
    /// for larger allocations. Advances to a new buffer if the current one doesn't have enough room.
    pub fn get_slab_of_size(&mut self, size: usize, zeroed: bool) -> Slab {
        assert!(size <= BUFFER_SIZE, "Size was {:?}", size);

        if self.remaining_in_buffer() < size {
            self.advance_to_new_buffer(zeroed);
        }

        self.get_slab_from_current_buffer(size, zeroed)
    }

    /// Allocates `size` bytes across one or more slabs, each at most 2MB.
    ///
    /// Returns the slabs in order — the caller is responsible for treating them as a
    /// contiguous logical buffer (see [`MultiSlabBuffer`]).
    pub fn get_slabs_of_size(&mut self, size: usize, zeroed: bool) -> Vec<Slab> {
        let mut slabs = vec![];
        let mut remaining = size;
        while remaining > 0 {
            let slab_size = std::cmp::min(remaining, self.remaining_in_buffer());
            slabs.push(self.get_slab_from_current_buffer(slab_size, zeroed));
            remaining -= slab_size;
            if self.remaining_in_buffer() == 0 {
                self.advance_to_new_buffer(zeroed);
            }
        }

        slabs
    }

    /// Rounds `self.offset` up so that the next allocation starts at an address satisfying
    /// `align_of::<T>()`. Without this, a prior odd-sized allocation (e.g. `SlabBuffer<u8>`)
    /// would leave the bump pointer misaligned for types with stricter requirements (e.g.
    /// `SlabBuffer<u128>` needs 16-byte alignment), causing UB in `slice::from_raw_parts_mut`.
    fn align_offset<T>(&mut self) {
        let align = std::mem::align_of::<T>();
        self.offset = (self.offset + align - 1) & !(align - 1);
    }

    /// Allocates a [`MultiSlabBuffer<T>`] that can hold `size` elements of type `T`.
    ///
    /// Always starts from a fresh buffer boundary so that `MultiSlabBuffer`'s bit-shift
    /// indexing works correctly (each slab is aligned to the start of a 2MB buffer).
    /// The resulting buffer is zeroed.
    ///
    /// Note: We did try playing with the idea of having a `start_offset` within the
    /// `MultiSlabBuffer`, but the extra addition was felt (~5% in some queries) since it appears
    /// in every single indexing operation.
    pub fn create_multi_slab_buffer<T>(&mut self, size: usize, zeroed: bool) -> MultiSlabBuffer<T> {
        let bytes = size * size_of::<T>();
        // For MultiSlabBuffer to work properly (with indexing), if the allocation doesn't fit
        // in the remaining space we must start a new buffer so each slab begins at offset 0.
        if self.remaining_in_buffer() < bytes {
            self.advance_to_new_buffer(zeroed);
        }
        self.align_offset::<T>();
        MultiSlabBuffer::new(self.get_slabs_of_size(bytes, zeroed))
    }

    /// Allocates a [`SlabBuffer<T>`] that can hold `size` elements of type `T`.
    ///
    /// The total byte size (`size * size_of::<T>()`) must be < 2MB since `SlabBuffer` is
    /// backed by a single slab. The resulting buffer is zeroed.
    pub fn create_slab_buffer<T>(&mut self, size: usize, zeroed: bool) -> SlabBuffer<T> {
        self.align_offset::<T>();
        let bytes = size * size_of::<T>();
        let slab = self.get_slab_of_size(bytes, zeroed);
        SlabBuffer::new(slab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{BUFFER_SIZE, init_test_free_pool};

    #[test]
    fn returns_zeroed_slab() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);

        let slab = alloc.get_slab_of_size(1024, true);

        let slice = unsafe { std::slice::from_raw_parts(slab.ptr, 1024) };
        assert!(slice.iter().all(|&b| b == 0));
    }

    #[test]
    fn packs_small_slabs_in_same_buffer() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);

        let a = alloc.get_slab_of_size(64, false);
        let b = alloc.get_slab_of_size(64, false);

        assert_eq!(unsafe { b.ptr.offset_from(a.ptr) }, 64);
    }

    #[test]
    fn advances_buffer_when_full() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);

        let a = alloc.get_slab_of_size(BUFFER_SIZE - 1, false);
        let b = alloc.get_slab_of_size(64, false);

        let distance = (b.ptr as usize).abs_diff(a.ptr as usize);
        assert!(distance >= BUFFER_SIZE);
    }

    #[test]
    fn get_slabs_spans_multiple_buffers() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);

        let slabs = alloc.get_slabs_of_size(BUFFER_SIZE + 1024, false);

        assert!(slabs.len() >= 2);
        let total: usize = slabs.iter().map(|s| s.size).sum();
        assert_eq!(total, BUFFER_SIZE + 1024);
    }
}
