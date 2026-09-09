use arrow_buffer::Buffer;

use crate::memory::{SlabAllocator, SlabBuffer};

/// A fixed-capacity column of `T` backed by a single [`SlabBuffer`], materialised
/// zero-copy into an Arrow [`Buffer`].
///
/// A column is built one slab-sized chunk at a time — the parquet decoder caps a
/// batch at `RECORD_BATCH_SIZE`, and GROUP BY output chunks to one slab — so a
/// single slab always suffices. Indexing is then a lone pointer write at
/// `ptr + len`, with none of
/// [`MultiSlabBuffer`](crate::memory::MultiSlabBuffer)'s per-element `slabs[idx]`
/// lookup, and `len` stays in a register across the build loop. Handing the slab
/// to Arrow keeps the column on our pre-faulted, huge-paged, accounted memory.
///
/// This is the shared core: [`PrimitiveBuilder`](crate::arrays::PrimitiveBuilder) wraps it for Arrow primitive
/// columns, and the GROUP BY string-view headers use it directly (their `u128`
/// view type is not an Arrow primitive type).
pub struct SlabColumn<T: Copy> {
    // `pub` so out-of-crate bulk decoders (the Parquet primitive decoder, now in
    // `catalog`) can copy straight into the backing buffer via
    // `SlabBuffer::ptr_at_index` and advance the length, rather than going
    // value-by-value through `push`.
    pub values: SlabBuffer<T>,
    pub len: usize,
}

impl<T: Copy> SlabColumn<T> {
    pub fn with_capacity(allocator: &mut SlabAllocator, capacity: usize) -> Self {
        Self {
            values: allocator.create_slab_buffer(capacity, false),
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, value: T) {
        unsafe { self.values.ptr_at_index(self.len).write(value) };
        self.len += 1;
    }

    /// Returns a mutable slice of `count` uninitialised slots at the end of the
    /// column, advancing the length. Callers must fill every slot. The slice is
    /// contiguous, since the backing is a single slab.
    #[inline(always)]
    pub fn spare_mut(&mut self, count: usize) -> &mut [T] {
        let start = self.len;
        self.len += count;
        unsafe { std::slice::from_raw_parts_mut(self.values.ptr_at_index(start), count) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The pushed elements as a slice: the backing is a single contiguous slab
    /// and every slot below `len` has been written.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.values.ptr_at_index(0), self.len) }
    }

    /// Consumes the column, handing its slab to Arrow as a zero-copy [`Buffer`].
    /// The caller wraps it in the appropriate typed buffer / array.
    pub fn into_buffer(self) -> Buffer {
        self.values.into_buffer(self.len * size_of::<T>())
    }
}
