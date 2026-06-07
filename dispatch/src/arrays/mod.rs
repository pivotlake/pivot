//! Building Arrow arrays directly over engine-owned ([`WriteBuffer`](crate::memory::WriteBuffer))
//! memory.
//!
//! [`ArrayBuilder`] accumulates values into a [`SlabAllocator`]-backed buffer and
//! produces the finished Arrow array with **zero copies** — the slab is handed to
//! Arrow via `Buffer::from_custom_allocation`, and its `Arc<WriteBuffer>` keeps the
//! memory pinned until the array is dropped (at which point the buffer returns to
//! the free pool). This keeps output allocations on our pre-faulted, huge-paged,
//! accounted memory instead of the global allocator.
//!
//! Used both by the parquet decoders (decoding pages into arrays) and the GROUP BY
//! output (building result columns).

use std::ptr::NonNull;
use std::sync::Arc;

use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{ArrayRef, PrimitiveArray};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};

use crate::memory::{SlabAllocator, SlabBuffer};

/// Accumulates values into engine memory and produces a finished Arrow array.
pub trait ArrayBuilder {
    type Element: Copy;

    /// Creates a builder pre-allocated for `capacity` elements.
    fn with_capacity(allocator: &mut SlabAllocator, capacity: usize) -> Self;

    /// Number of elements pushed so far.
    fn len(&self) -> usize;

    /// Whether no elements have been pushed yet.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Appends `element` repeated `amount` times (used by RLE runs).
    fn push(&mut self, element: &Self::Element, amount: usize);

    /// Returns a mutable slice of `count` uninitialised slots at the end of
    /// the buffer, advancing the length. Callers must fill every slot.
    fn spare_mut(&mut self, count: usize) -> &mut [Self::Element];

    /// Consumes the builder and returns the finished Arrow array.
    fn into_array(self, null_buffer: Option<Buffer>) -> ArrayRef;
}

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
/// This is the shared core: [`PrimitiveBuilder`] wraps it for Arrow primitive
/// columns, and the GROUP BY string-view headers use it directly (their `u128`
/// view type is not an [`ArrowPrimitiveType`]).
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

    /// Consumes the column, handing its slab to Arrow as a zero-copy [`Buffer`].
    /// The caller wraps it in the appropriate typed buffer / array.
    pub fn into_buffer(self) -> Buffer {
        let byte_len = self.len * size_of::<T>();
        let slab = self.values.into_slab();
        let ptr = NonNull::new(slab.ptr).unwrap();
        unsafe { Buffer::from_custom_allocation(ptr, byte_len, Arc::new(slab)) }
    }
}

/// [`ArrayBuilder`] for fixed-width primitive Arrow types, backed by a single-slab
/// [`SlabColumn`]. Used by both the parquet primitive decoders and the GROUP BY
/// output columns.
pub struct PrimitiveBuilder<T: ArrowPrimitiveType> {
    pub col: SlabColumn<T::Native>,
}

impl<T: ArrowPrimitiveType> ArrayBuilder for PrimitiveBuilder<T> {
    type Element = T::Native;

    fn with_capacity(allocator: &mut SlabAllocator, capacity: usize) -> Self {
        Self {
            col: SlabColumn::with_capacity(allocator, capacity),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.col.len()
    }

    #[inline(always)]
    fn push(&mut self, element: &T::Native, amount: usize) {
        self.col.spare_mut(amount).fill(*element);
    }

    #[inline]
    fn spare_mut(&mut self, count: usize) -> &mut [T::Native] {
        self.col.spare_mut(count)
    }

    fn into_array(self, null_buffer: Option<Buffer>) -> ArrayRef {
        let len = self.col.len();
        let values = ScalarBuffer::new(self.col.into_buffer(), 0, len);
        let nulls = null_buffer
            .map(|b| NullBuffer::new(BooleanBuffer::new(b, 0, len)))
            .filter(|n| n.null_count() != 0);
        Arc::new(PrimitiveArray::<T>::new(values, nulls))
    }
}
