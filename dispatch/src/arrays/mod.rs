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
    /// `Default` is the element written under null slots: the slab is not
    /// zeroed on allocation, and downstream kernels (and unsafe array
    /// constructors) may touch masked slots, so they must hold a valid value.
    type Element: Copy + Default;

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

pub mod accumulator;
pub mod take;

/// Hand a slab's first `byte_len` bytes to Arrow as a zero-copy [`Buffer`].
/// The slab rides in the buffer's allocation `Arc`, so the memory returns to
/// the pool when the last downstream reference drops.
pub(crate) fn slab_into_buffer<T>(slab: SlabBuffer<T>, byte_len: usize) -> Buffer {
    let slab = slab.into_slab();
    let ptr = NonNull::new(slab.ptr).unwrap();
    // SAFETY: the slab owns at least `byte_len` bytes at `ptr` and lives as
    // long as the returned buffer via the custom-allocation `Arc`.
    unsafe { Buffer::from_custom_allocation(ptr, byte_len, Arc::new(slab)) }
}

/// A validity bitmap (`1` = present/valid) built run-wise on slab memory, the
/// slab-array analog of arrow's `BooleanBufferBuilder`. A nullable decoded
/// column's null buffer lives on the same pre-faulted, accounted slab memory as
/// its values, rather than on the global allocator. Append the rows in order,
/// then hand the bitmap to Arrow with [`into_buffer`](Self::into_buffer).
pub struct ValidityBuilder {
    bits: SlabBuffer<u8>,
    len: usize,
}

impl ValidityBuilder {
    /// A bitmap sized for `capacity` rows, all initially null.
    pub fn with_capacity(allocator: &mut SlabAllocator, capacity: usize) -> Self {
        // Zeroed slab: absent rows stay null without being written.
        Self {
            bits: allocator.create_slab_buffer(capacity.div_ceil(8), true),
            len: 0,
        }
    }

    /// Append `count` rows, all present (valid) or all null. Null runs are
    /// already zero in the slab, so only present runs are written: whole bytes
    /// where the run spans them, edge bits otherwise.
    pub fn append_n(&mut self, count: usize, present: bool) {
        if present {
            let (mut i, end) = (self.len, self.len + count);
            while i < end && i % 8 != 0 {
                self.set(i);
                i += 1;
            }
            while i + 8 <= end {
                unsafe { *self.bits.ptr_at_index(i / 8) = 0xFF };
                i += 8;
            }
            while i < end {
                self.set(i);
                i += 1;
            }
        }
        self.len += count;
    }

    #[inline]
    fn set(&mut self, i: usize) {
        unsafe { *self.bits.ptr_at_index(i / 8) |= 1 << (i % 8) };
    }

    /// Hand the bitmap to Arrow as a zero-copy slab-backed [`Buffer`].
    pub fn into_buffer(self) -> Buffer {
        slab_into_buffer(self.bits, self.len.div_ceil(8))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;

    /// Append `runs` of (count, present) and read the resulting bitmap back.
    fn validity(runs: &[(usize, bool)]) -> Vec<bool> {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let len: usize = runs.iter().map(|&(n, _)| n).sum();
        let mut builder = ValidityBuilder::with_capacity(&mut allocator, len);
        for &(n, present) in runs {
            builder.append_n(n, present);
        }
        let bits = BooleanBuffer::new(builder.into_buffer(), 0, len);
        (0..len).map(|i| bits.value(i)).collect()
    }

    #[test]
    fn present_runs_cross_byte_boundaries() {
        let got = validity(&[(3, true), (8, false), (5, true)]);

        let mut want = vec![false; 16];
        for i in [0, 1, 2, 11, 12, 13, 14, 15] {
            want[i] = true;
        }
        assert_eq!(got, want);
    }

    #[test]
    fn whole_byte_present_runs() {
        let got = validity(&[(16, true), (8, false)]);

        assert_eq!(got, [vec![true; 16], vec![false; 8]].concat());
    }

    #[test]
    fn single_bit_runs_alternate() {
        let runs: Vec<(usize, bool)> = (0..11).map(|i| (1, i % 2 == 0)).collect();

        let got = validity(&runs);

        assert_eq!(got, (0..11).map(|i| i % 2 == 0).collect::<Vec<_>>());
    }
}
