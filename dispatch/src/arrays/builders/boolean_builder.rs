use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};

use super::ArrayBuilder;
use super::bitmap_builder::{BitSink, BitmapBuilder};
use crate::memory::SlabAllocator;

/// [`ArrayBuilder`] for boolean arrays, holding its values as the bitmap
/// arrow wants from the start.
///
/// A Parquet PLAIN boolean page is that same bitmap, so a decoder appends a
/// page's bits with [`append_bits`](Self::append_bits), a copy of whole bytes
/// when both sides are byte-aligned, and a run of one value is a bit fill.
/// Only [`spare_mut`](ArrayBuilder::spare_mut), which hands out element
/// slots to be filled after the call, cannot write bits directly: it stages
/// the slots in a small vector that the next call packs. For booleans that
/// path serves the dictionary-index gather alone, which no writer produces
/// for the type.
pub struct BooleanBuilder {
    bits: BitmapBuilder,
    /// Values handed out by `spare_mut` and not yet packed. They follow the
    /// packed ones, so the builder's length counts them too.
    staged: Vec<bool>,
}

impl BooleanBuilder {
    /// Appends `count` bits of `src`, starting at bit `src_bit` of it, least
    /// significant bit first.
    #[inline]
    pub fn append_bits(&mut self, src: &[u8], src_bit: usize, count: usize) {
        self.pack_staged();
        self.bits.append_bits(src, src_bit, count);
    }

    /// Packs the values `spare_mut` handed out, if any, onto the bitmap.
    #[inline]
    fn pack_staged(&mut self) {
        if self.staged.is_empty() {
            return;
        }
        let staged = std::mem::take(&mut self.staged);
        for &value in &staged {
            self.bits.append_run(value, 1);
        }
        self.staged = staged;
        self.staged.clear();
    }
}

impl BitSink for BooleanBuilder {
    #[inline]
    fn append_bits(&mut self, src: &[u8], src_bit: usize, count: usize) {
        BooleanBuilder::append_bits(self, src, src_bit, count);
    }

    #[inline]
    fn append_run(&mut self, value: bool, count: usize) {
        self.pack_staged();
        self.bits.append_run(value, count);
    }
}

impl ArrayBuilder for BooleanBuilder {
    type Element = bool;

    fn with_capacity(allocator: &mut SlabAllocator, capacity: usize) -> Self {
        Self {
            bits: BitmapBuilder::with_capacity(allocator, capacity),
            staged: Vec::new(),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.bits.len() + self.staged.len()
    }

    #[inline]
    fn push(&mut self, element: &bool, amount: usize) {
        self.pack_staged();
        self.bits.append_run(*element, amount);
    }

    fn spare_mut(&mut self, count: usize) -> &mut [bool] {
        self.pack_staged();
        self.staged.resize(count, false);
        &mut self.staged
    }

    fn into_array(mut self, null_buffer: Option<Buffer>) -> ArrayRef {
        self.pack_staged();
        let len = self.bits.len();
        let values = BooleanBuffer::new(self.bits.into_buffer(), 0, len);
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

    fn values_of(array: &ArrayRef) -> Vec<bool> {
        array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .iter()
            .flatten()
            .collect()
    }

    #[test]
    fn packs_pushed_and_staged_values_in_order() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut builder = BooleanBuilder::with_capacity(&mut allocator, 20);

        builder.push(&true, 3);
        builder.push(&false, 7);
        builder.spare_mut(10).copy_from_slice(&[
            true, false, true, false, true, false, true, false, true, true,
        ]);
        let array = builder.into_array(None);

        let mut want = vec![
            true, true, true, false, false, false, false, false, false, false,
        ];
        want.extend([
            true, false, true, false, true, false, true, false, true, true,
        ]);
        assert_eq!(values_of(&array), want);
        assert_eq!(array.null_count(), 0);
    }

    #[test]
    fn a_run_of_true_fills_whole_bytes() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut builder = BooleanBuilder::with_capacity(&mut allocator, 40);

        builder.push(&false, 5);
        builder.push(&true, 30);
        builder.push(&false, 5);
        let array = builder.into_array(None);

        let mut want = vec![false; 5];
        want.extend([true; 30]);
        want.extend([false; 5]);
        assert_eq!(values_of(&array), want);
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
