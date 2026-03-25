//! A slab-backed primitive array builder for join output.
//!
//! Unlike Arrow's `PrimitiveBuilder` or the parquet decoder's `PrimitiveBuilder`,
//! this builder has no internal length — the caller writes directly by index and
//! provides the final length at conversion time. This avoids a length field update
//! on every write.
//!
//! Backed by a single [`SlabBuffer`], so indexing is a single pointer offset
//! with no slab lookup. The slab is handed off to Arrow zero-copy via
//! `Buffer::from_custom_allocation`.

use std::ptr::NonNull;
use std::sync::Arc;

use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{ArrayRef, PrimitiveArray};
use arrow_buffer::{Buffer, ScalarBuffer};

use crate::memory::{SlabAllocator, SlabBuffer};

/// Fixed-capacity primitive builder backed by a single slab.
///
/// Write elements by index (`builder[i] = value`), then call
/// [`into_array`](Self::into_array) with the final length to produce
/// a zero-copy Arrow array.
pub struct JoinPrimitiveBuilder<T: ArrowPrimitiveType> {
    values: SlabBuffer<T::Native>,
}

impl<T: ArrowPrimitiveType> JoinPrimitiveBuilder<T> {
    pub fn new(allocator: &mut SlabAllocator, capacity: usize) -> Self {
        Self {
            values: allocator.create_slab_buffer(capacity, false),
        }
    }

    /// Write a value at the given index. No bounds checking, no length tracking.
    #[inline(always)]
    pub fn write(&mut self, index: usize, value: T::Native) {
        unsafe {
            self.values.ptr_at_index(index).write(value);
        }
    }

    /// Convert into an Arrow array with the given length.
    ///
    /// The caller must ensure all indices `0..len` have been written.
    pub fn into_array(self, len: usize) -> ArrayRef {
        let byte_len = len * size_of::<T::Native>();
        let slab = self.values.into_slab();
        let ptr = NonNull::new(slab.ptr).unwrap();
        let buffer = unsafe { Buffer::from_custom_allocation(ptr, byte_len, Arc::new(slab)) };
        let values = ScalarBuffer::new(buffer, 0, len);
        Arc::new(PrimitiveArray::<T>::new(values, None))
    }
}
