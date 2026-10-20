use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};

use super::ArrayBuilder;
use crate::arrays::SlabColumn;
use crate::memory::{SlabAllocator, SlabBuffer};

/// [`ArrayBuilder`] for boolean arrays.
///
/// Values are appended one byte each into a [`SlabColumn`], which is what lets
/// a decoder fill a run of them through [`spare_mut`](ArrayBuilder::spare_mut)
/// like any other element, and packed into the bitmap arrow wants only once the
/// array is finished. The bitmap is allocated up front on slab memory alongside
/// the values, since the finishing step has no allocator to ask.
pub struct BooleanBuilder {
    values: SlabColumn<bool>,
    bits: SlabBuffer<u8>,
}

impl ArrayBuilder for BooleanBuilder {
    type Element = bool;

    fn with_capacity(allocator: &mut SlabAllocator, capacity: usize) -> Self {
        Self {
            values: SlabColumn::with_capacity(allocator, capacity),
            // Zeroed, so packing only ever sets the bits of true values.
            bits: allocator.create_slab_buffer(capacity.div_ceil(8), true),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.values.len()
    }

    #[inline(always)]
    fn push(&mut self, element: &bool, amount: usize) {
        self.values.spare_mut(amount).fill(*element);
    }

    #[inline]
    fn spare_mut(&mut self, count: usize) -> &mut [bool] {
        self.values.spare_mut(count)
    }

    fn into_array(self, null_buffer: Option<Buffer>) -> ArrayRef {
        let len = self.values.len();
        for (byte_index, byte_values) in self.values.as_slice().chunks(8).enumerate() {
            let mut byte = 0u8;
            for (bit, &value) in byte_values.iter().enumerate() {
                byte |= (value as u8) << bit;
            }
            unsafe { *self.bits.ptr_at_index(byte_index) = byte };
        }
        let values = BooleanBuffer::new(self.bits.into_buffer(len.div_ceil(8)), 0, len);
        let nulls = null_buffer
            .map(|b| NullBuffer::new(BooleanBuffer::new(b, 0, len)))
            .filter(|n| n.null_count() != 0);
        Arc::new(BooleanArray::new(values, nulls))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use arrow_array::Array;

    #[test]
    fn packs_pushed_values_in_order() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut builder = BooleanBuilder::with_capacity(&mut allocator, 20);

        builder.push(&true, 3);
        builder.push(&false, 7);
        builder.spare_mut(10).copy_from_slice(&[
            true, false, true, false, true, false, true, false, true, true,
        ]);
        let array = builder.into_array(None);

        let got: Vec<bool> = array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        let mut want = vec![
            true, true, true, false, false, false, false, false, false, false,
        ];
        want.extend([
            true, false, true, false, true, false, true, false, true, true,
        ]);
        assert_eq!(got, want);
        assert_eq!(array.null_count(), 0);
    }

    #[test]
    fn applies_the_null_buffer() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut builder = BooleanBuilder::with_capacity(&mut allocator, 3);
        builder.push(&true, 3);

        let array = builder.into_array(Some(Buffer::from(vec![0b101u8])));

        let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            array.iter().collect::<Vec<_>>(),
            vec![Some(true), None, Some(true)]
        );
    }
}
