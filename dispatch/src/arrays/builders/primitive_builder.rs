use std::sync::Arc;

use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{ArrayRef, PrimitiveArray};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};

use super::ArrayBuilder;
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;

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
