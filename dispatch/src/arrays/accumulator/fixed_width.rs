//! Accumulating a column whose values are all the same number of bytes.
//!
//! The simplest case: the values are the array's one buffer, so appending is a
//! copy of raw bytes and emitting is that buffer with the accumulated validity
//! beside it. Nothing here reads a value, so one implementation covers every
//! width Arrow packs this way, from a boolean-free 1-byte type up to a 16-byte
//! decimal.

use arrow::array::ArrayData;
use arrow_array::{ArrayRef, make_array};
use arrow_schema::{ArrowError, DataType};

use super::chunked::PreparedColumn;
use super::column::{ColumnAccumulator, SourceSelection};
use super::validity::ValidityMask;
use crate::arrays::slab_into_buffer;
use crate::memory::{SlabAllocator, SlabBuffer};

pub(super) struct FixedWidthColumn {
    data_type: DataType,
    width: usize,
    capacity: usize,
    /// The values, u128-backed so the buffer start is aligned for any
    /// fixed-width value type Arrow reads through it.
    slab: SlabBuffer<u128>,
    validity: ValidityMask,
}

impl FixedWidthColumn {
    pub(super) fn new(
        data_type: &DataType,
        width: usize,
        capacity: usize,
        allocator: &mut SlabAllocator,
    ) -> Self {
        Self {
            data_type: data_type.clone(),
            width,
            capacity,
            slab: allocate_values_slab(capacity, width, allocator),
            validity: ValidityMask::new(capacity),
        }
    }
}

impl ColumnAccumulator for FixedWidthColumn {
    fn append(
        &mut self,
        source: &ArrayRef,
        selection: SourceSelection<'_>,
        destination_start: usize,
        _allocator: &mut SlabAllocator,
    ) {
        let data = source.to_data();
        self.validity
            .append(data.nulls(), selection, destination_start);
        let width = self.width;
        // SAFETY: the source holds `offset + len` values, the rows are in-bounds
        // positions, and the slab has capacity for `at` plus the appended rows
        // (checked by the caller).
        unsafe {
            let src = data.buffers()[0].as_ptr().add(data.offset() * width);
            let dst = (self.slab.ptr_at_index(0) as *mut u8).add(destination_start * width);
            match selection {
                SourceSelection::Range { start, len } => {
                    std::ptr::copy_nonoverlapping(src.add(start * width), dst, len * width)
                }
                SourceSelection::Indices(indices) => match width {
                    1 => gather_fixed_width::<u8>(src, dst, indices),
                    2 => gather_fixed_width::<u16>(src, dst, indices),
                    4 => gather_fixed_width::<u32>(src, dst, indices),
                    8 => gather_fixed_width::<u64>(src, dst, indices),
                    16 => gather_fixed_width::<u128>(src, dst, indices),
                    _ => unreachable!("built only for the widths above"),
                },
            }
        }
    }

    fn append_chunked(
        &mut self,
        source: &PreparedColumn,
        ids: &[u32],
        shift: u32,
        destination_start: usize,
        _allocator: &mut SlabAllocator,
    ) {
        let PreparedColumn::FixedWidth {
            width,
            values,
            nulls,
        } = source
        else {
            unreachable!("a fixed-width accumulator receives a fixed-width prepared column");
        };
        debug_assert_eq!(*width, self.width);
        self.validity
            .append_by_ids(nulls, ids, shift, destination_start);
        // SAFETY: each id names an in-bounds row of its chunk, and the slab has
        // capacity for `destination_start` plus the appended rows (checked by
        // the caller).
        unsafe {
            let dst = (self.slab.ptr_at_index(0) as *mut u8).add(destination_start * self.width);
            match self.width {
                1 => gather_chunked::<u8>(values, ids, shift, dst),
                2 => gather_chunked::<u16>(values, ids, shift, dst),
                4 => gather_chunked::<u32>(values, ids, shift, dst),
                8 => gather_chunked::<u64>(values, ids, shift, dst),
                16 => gather_chunked::<u128>(values, ids, shift, dst),
                _ => unreachable!("built only for the widths above"),
            }
        }
    }

    fn take_array(
        &mut self,
        len: usize,
        allocator: &mut SlabAllocator,
    ) -> Result<ArrayRef, ArrowError> {
        let fresh = allocate_values_slab(self.capacity, self.width, allocator);
        let slab = std::mem::replace(&mut self.slab, fresh);
        let buffer = slab_into_buffer(slab, len * self.width);
        let out = ArrayData::builder(self.data_type.clone())
            .len(len)
            .add_buffer(buffer)
            .nulls(self.validity.take(len));
        // SAFETY: a fixed-width array is a single values buffer plus optional
        // validity, and both were copied verbatim.
        Ok(make_array(unsafe { out.build_unchecked() }))
    }
}

/// A slab holding `capacity` values of `width` bytes, as the `u128` elements the
/// buffer is aligned by.
fn allocate_values_slab(
    capacity: usize,
    width: usize,
    allocator: &mut SlabAllocator,
) -> SlabBuffer<u128> {
    allocator.create_slab_buffer((capacity * width).div_ceil(size_of::<u128>()), false)
}

/// Gather `indices` rows from `src` to `dst`, as elements of `T`. A flat loop
/// with independent iterations, so the CPU overlaps the source cache misses of
/// many gathers. Shared with [`view`](super::view), whose views are gathered the
/// same way when they need no rebasing.
///
/// Four rows are copied per step rather than left to the compiler to unroll. How
/// far it unrolls this loop depends on the size of the function the loop ends up
/// in, so left implicit the cost of a copy moves with the shape of the code
/// around it rather than staying a property of the copy.
///
/// # Safety
/// `src` must hold every indexed row, `dst` must have room for `indices.len()`
/// elements, and both must be valid for unaligned `T` access.
pub(super) unsafe fn gather_fixed_width<T: Copy>(src: *const u8, dst: *mut u8, indices: &[u32]) {
    let src = src as *const T;
    let dst = dst as *mut T;
    /// Rows copied per step of the loop below.
    const STEP: usize = 4;
    unsafe {
        let mut destination = 0;
        let mut rows = indices.chunks_exact(STEP);
        for step in &mut rows {
            for (offset, &row) in step.iter().enumerate() {
                dst.add(destination + offset)
                    .write_unaligned(src.add(row as usize).read_unaligned());
            }
            destination += STEP;
        }
        for (offset, &row) in rows.remainder().iter().enumerate() {
            dst.add(destination + offset)
                .write_unaligned(src.add(row as usize).read_unaligned());
        }
    }
}

/// Gather the rows at the encoded `ids` from per-chunk value pointers to
/// `dst`, as elements of `T`. The two-level read stays a flat loop with
/// independent iterations, like [`gather_fixed_width`].
///
/// # Safety
/// Every id must name an in-bounds row of an in-bounds chunk, `dst` must have
/// room for `ids.len()` elements, and all pointers must be valid for unaligned
/// `T` access.
unsafe fn gather_chunked<T: Copy>(chunks: &[*const u8], ids: &[u32], shift: u32, dst: *mut u8) {
    let mask = (1u32 << shift) - 1;
    let dst = dst as *mut T;
    unsafe {
        for (out_idx, &id) in ids.iter().enumerate() {
            let src = *chunks.get_unchecked((id >> shift) as usize) as *const T;
            dst.add(out_idx)
                .write_unaligned(src.add((id & mask) as usize).read_unaligned());
        }
    }
}
