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

use crate::memory::{MultiSlabBuffer, SlabAllocator};

/// Accumulates values into engine memory and produces a finished Arrow array.
pub trait ArrayBuilder {
    type Element: Copy;

    /// Creates a builder pre-allocated for `capacity` elements.
    fn with_capacity(allocator: &mut SlabAllocator, capacity: usize) -> Self;

    /// Number of elements pushed so far.
    fn len(&self) -> usize;

    /// Appends `element` repeated `amount` times (used by RLE runs).
    fn push(&mut self, element: &Self::Element, amount: usize);

    /// Returns a mutable slice of `count` uninitialised slots at the end of
    /// the buffer, advancing the length. Callers must fill every slot.
    fn spare_mut(&mut self, count: usize) -> &mut [Self::Element];

    /// Consumes the builder and returns the finished Arrow array.
    fn into_array(self, null_buffer: Option<Buffer>) -> ArrayRef;
}

/// [`ArrayBuilder`] for fixed-width primitive Arrow types.
///
/// Backed by a [`MultiSlabBuffer`] so that the final array can be produced
/// with zero copies via [`into_array`](ArrayBuilder::into_array). The capacity
/// must fit in a single 2MB slab (see [`into_array`](ArrayBuilder::into_array)).
pub struct PrimitiveBuilder<T: ArrowPrimitiveType> {
    // `pub(crate)` so bulk decoders (e.g. the parquet primitive decoder) can copy
    // straight into the backing buffer via `MultiSlabBuffer::ptr_at_index` and
    // advance the length, rather than going value-by-value through `push`.
    pub(crate) values: MultiSlabBuffer<T::Native>,
    pub(crate) len: usize,
}

impl<T: ArrowPrimitiveType> ArrayBuilder for PrimitiveBuilder<T> {
    type Element = T::Native;

    fn with_capacity(allocator: &mut SlabAllocator, capacity: usize) -> Self {
        Self {
            values: allocator.create_multi_slab_buffer(capacity, false),
            len: 0,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn push(&mut self, element: &T::Native, amount: usize) {
        let start = self.len;
        self.len += amount;
        let dest =
            unsafe { std::slice::from_raw_parts_mut(self.values.ptr_at_index(start), amount) };
        dest.fill(*element);
    }

    #[inline]
    fn spare_mut(&mut self, count: usize) -> &mut [T::Native] {
        let start = self.len;
        self.len += count;
        unsafe { std::slice::from_raw_parts_mut(self.values.ptr_at_index(start), count) }
    }

    fn into_array(self, null_buffer: Option<Buffer>) -> ArrayRef {
        let len = self.len;
        let byte_len = len * size_of::<T::Native>();
        let slab = self.values.into_single_slab();
        let ptr = NonNull::new(slab.ptr).unwrap();
        let buffer = unsafe { Buffer::from_custom_allocation(ptr, byte_len, Arc::new(slab)) };
        let values = ScalarBuffer::new(buffer, 0, len);
        let nulls = null_buffer
            .map(|b| NullBuffer::new(BooleanBuffer::new(b, 0, len)))
            .filter(|n| n.null_count() != 0);
        Arc::new(PrimitiveArray::<T>::new(values, nulls))
    }
}
