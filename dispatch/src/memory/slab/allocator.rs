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
///
/// Once every slab carved from the working buffer has been dropped, the next allocation starts
/// over at the buffer's beginning (see [`rewind_if_unshared`](Self::rewind_if_unshared)), so an
/// allocator whose slabs die young keeps reusing the same cache-resident bytes.
pub struct SlabAllocator {
    /// The current buffer we're bumping through.
    working_buffer: Arc<WriteBuffer>,
    /// Byte offset of the next free region in `working_buffer`.
    offset: usize,
    /// Whether the bytes from `offset` onwards are still zero. Starts as the buffer's own
    /// `zeroed` flag and is cleared by a rewind, which hands out bytes already written.
    working_buffer_zeroed: bool,
}

impl SlabAllocator {
    /// Creates a new allocator with a fresh buffer from the free pool.
    /// If `zeroed` is true, prefers a pre-zeroed buffer.
    pub fn new(zeroed: bool) -> Self {
        let working_buffer = memory_ctx().get_write_buffer(zeroed);
        Self {
            working_buffer_zeroed: working_buffer.zeroed,
            working_buffer: Arc::new(working_buffer),
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
        if zeroed && !self.working_buffer_zeroed {
            slab.zero_out();
        }

        self.offset += size;
        slab
    }

    /// Discards the remaining space in the current buffer and acquires a fresh one.
    fn advance_to_new_buffer(&mut self, zeroed: bool) {
        self.offset = 0;
        let working_buffer = memory_ctx().get_write_buffer(zeroed);
        self.working_buffer_zeroed = working_buffer.zeroed;
        self.working_buffer = Arc::new(working_buffer);
    }

    /// Restarts the bump pointer at the beginning of the working buffer when no slab carved
    /// from it is still alive.
    ///
    /// Without this, a steady stream of short-lived allocations (a batch built, consumed and
    /// dropped before the next one is built) walks through fresh ring memory: every line it
    /// touches is first read in for ownership and later written back to DRAM on eviction, though
    /// its contents died long before. Rewinding keeps such a stream on the same few
    /// cache-resident lines.
    ///
    /// A zeroed request against a buffer whose untouched tail is still zero keeps consuming that
    /// tail instead: rewinding there would trade zeroing already done for an inline memset.
    #[inline(always)]
    fn rewind_if_unshared(&mut self, zeroed: bool) {
        if self.offset == 0 || (zeroed && self.working_buffer_zeroed) {
            return;
        }
        // `get_mut` succeeds only while this allocator holds the sole reference, and its
        // Acquire orders every access made through a since-dropped slab (possibly on another
        // thread) before the reuse.
        if Arc::get_mut(&mut self.working_buffer).is_some() {
            self.offset = 0;
            self.working_buffer_zeroed = false;
        }
    }

    /// Allocates a single slab of exactly `size` bytes.
    ///
    /// Panics if `size >= BUFFER_SIZE` (2MB) — use [`get_slabs_of_size`](Self::get_slabs_of_size)
    /// for larger allocations. Advances to a new buffer if the current one doesn't have enough room.
    pub fn get_slab_of_size(&mut self, size: usize, zeroed: bool) -> Slab {
        assert!(size <= BUFFER_SIZE, "Size was {:?}", size);
        self.rewind_if_unshared(zeroed);

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
        self.rewind_if_unshared(zeroed);
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
        // `MultiSlabBuffer` packs `elems_per_slab` elements into each slab (it does NOT treat
        // the slabs as one contiguous byte run — see its docs), so a single slab fits tightly
        // but anything larger takes one full 2MB slab per `elems_per_slab` elements. Sizing by
        // bytes alone would under-allocate by a slab once an element straddles the 2MB point.
        let elems_per_slab = BUFFER_SIZE / size_of::<T>();
        let bytes = if size <= elems_per_slab {
            size * size_of::<T>()
        } else {
            size.div_ceil(elems_per_slab) * BUFFER_SIZE
        };
        self.rewind_if_unshared(zeroed);
        // For MultiSlabBuffer to work properly (with indexing), if the allocation doesn't fit
        // in the remaining space we must start a new buffer so each slab begins at offset 0.
        if self.remaining_in_buffer() < bytes {
            self.advance_to_new_buffer(zeroed);
        }
        self.align_offset::<T>();
        MultiSlabBuffer::new(self.get_slabs_of_size(bytes, zeroed))
    }

    /// Allocates zeroed slabs for `count` runtime-sized entries.
    ///
    /// Entries never cross slab boundaries. Multi-slab allocations begin at a
    /// buffer boundary so every entry can be addressed from its slab base.
    pub fn create_strided_slabs(&mut self, count: usize, stride: usize, align: usize) -> Vec<Slab> {
        let entries_per_slab = BUFFER_SIZE / stride;
        let bytes = if count <= entries_per_slab {
            count * stride
        } else {
            count.div_ceil(entries_per_slab) * BUFFER_SIZE
        };
        self.rewind_if_unshared(true);
        if self.remaining_in_buffer() < bytes {
            self.advance_to_new_buffer(true);
        }
        self.offset = (self.offset + align - 1) & !(align - 1);
        self.get_slabs_of_size(bytes, true)
    }

    /// Allocates one slab at an explicitly aligned address.
    ///
    /// `size + align` must fit in one backing buffer.
    pub fn get_aligned_slab(&mut self, size: usize, align: usize, zeroed: bool) -> Slab {
        self.rewind_if_unshared(zeroed);
        if self.remaining_in_buffer() < size + align {
            self.advance_to_new_buffer(zeroed);
        }
        self.offset = (self.offset + align - 1) & !(align - 1);
        self.get_slab_from_current_buffer(size, zeroed)
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

    #[test]
    fn reuses_space_once_every_slab_is_dropped() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(false);
        let first = alloc.get_slab_of_size(1024, false);
        let first_ptr = first.ptr;
        drop(first);

        let second = alloc.get_slab_of_size(1024, false);

        assert_eq!(second.ptr, first_ptr);
    }

    #[test]
    fn keeps_space_of_a_live_slab() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(false);
        let first = alloc.get_slab_of_size(1024, false);

        let second = alloc.get_slab_of_size(1024, false);

        assert_eq!(unsafe { second.ptr.offset_from(first.ptr) }, 1024);
    }

    #[test]
    fn zeroes_reused_space_for_a_zeroed_request() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(false);
        let mut dirty = alloc.get_slab_of_size(1024, false);
        dirty.as_mut_slice().fill(0xFF);
        drop(dirty);
        alloc.get_slab_of_size(16, false);

        let zeroed = alloc.get_slab_of_size(1024, true);

        assert!(zeroed.as_slice().iter().all(|&b| b == 0));
    }
}
