//! How the rows of a [`PartitionAllocator`] are shaped, and how big the slabs
//! holding them are.
//!
//! The allocator itself only moves bytes around. Everything it needs to know
//! about a row comes from a [`Layout`]: the row's size and alignment, what
//! type to hand a reserved row back as, and how many rows the next slab of a
//! partition should hold. The join knows its row type at compile time and uses
//! [`Fixed`]; the group-by learns its row width from the query and uses
//! [`Dynamic`].
//!
//! [`PartitionAllocator`]: super::partition_allocator::PartitionAllocator

use std::marker::PhantomData;
use std::mem::{MaybeUninit, align_of, size_of};

pub trait Layout {
    /// What a reserved row is handed to the caller as, to be filled in place.
    type Row: ?Sized;

    fn row_size(&self) -> usize;

    fn row_align(&self) -> usize;

    /// How many rows the next slab of a partition holds, given how many slabs
    /// that partition has already filled. Lets a layout start partitions on
    /// small slabs and grow them as a partition proves to receive many rows.
    fn slab_rows(&self, slabs_so_far: usize) -> usize;

    /// View the row at `row` as [`Self::Row`].
    ///
    /// # Safety
    /// `row` must point at [`row_size`](Self::row_size) bytes aligned to
    /// [`row_align`](Self::row_align) that stay valid for `'a`.
    unsafe fn row_mut<'a>(&self, row: *mut u8) -> &'a mut Self::Row;
}

/// Rows of one compile-time type, in slabs of a fixed number of rows.
pub struct Fixed<T> {
    rows_per_slab: usize,
    _row_type: PhantomData<T>,
}

impl<T> Fixed<T> {
    /// Slabs of about `slab_bytes`, rounded down to whole rows.
    pub fn new(slab_bytes: usize) -> Self {
        let rows_per_slab = slab_bytes / size_of::<T>();
        assert!(rows_per_slab > 0, "a slab must hold at least one row");
        Self {
            rows_per_slab,
            _row_type: PhantomData,
        }
    }
}

impl<T> Layout for Fixed<T> {
    type Row = MaybeUninit<T>;

    fn row_size(&self) -> usize {
        size_of::<T>()
    }

    fn row_align(&self) -> usize {
        align_of::<T>()
    }

    fn slab_rows(&self, _slabs_so_far: usize) -> usize {
        self.rows_per_slab
    }

    unsafe fn row_mut<'a>(&self, row: *mut u8) -> &'a mut MaybeUninit<T> {
        unsafe { &mut *(row as *mut MaybeUninit<T>) }
    }
}

/// Rows of a width known only when the operator is built, handed out as byte
/// slices. A partition's first slab is small, and each slab after it holds
/// twice the rows of the one before, up to a cap, so a partition with a
/// handful of rows wastes little and a partition with many rows soon gets
/// slabs that are cheap to spill.
pub struct Dynamic {
    row_size: usize,
    row_align: usize,
    first_slab_rows: usize,
    max_slab_rows: usize,
}

impl Dynamic {
    /// Rows of `row_size` bytes aligned to `row_align`, in slabs starting at
    /// about `first_slab_bytes` and doubling up to about `max_slab_bytes`.
    pub fn new(
        row_size: usize,
        row_align: usize,
        first_slab_bytes: usize,
        max_slab_bytes: usize,
    ) -> Self {
        assert!(row_size > 0, "a row must have at least one byte");
        assert!(
            row_size % row_align == 0,
            "a row's size must be a multiple of its alignment so rows stay aligned back to back"
        );
        let first_slab_rows = (first_slab_bytes / row_size).max(1);
        let max_slab_rows = (max_slab_bytes / row_size).max(first_slab_rows);
        Self {
            row_size,
            row_align,
            first_slab_rows,
            max_slab_rows,
        }
    }
}

impl Layout for Dynamic {
    type Row = [u8];

    fn row_size(&self) -> usize {
        self.row_size
    }

    fn row_align(&self) -> usize {
        self.row_align
    }

    fn slab_rows(&self, slabs_so_far: usize) -> usize {
        let doubled = self
            .first_slab_rows
            .checked_shl(slabs_so_far as u32)
            .unwrap_or(self.max_slab_rows);
        doubled.min(self.max_slab_rows)
    }

    unsafe fn row_mut<'a>(&self, row: *mut u8) -> &'a mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(row, self.row_size) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fixed_layout_fits_whole_rows_into_the_slab_size() {
        let layout = Fixed::<u64>::new(100);

        assert_eq!(layout.slab_rows(0), 12);
        assert_eq!(layout.slab_rows(5), 12);
    }

    #[test]
    fn a_dynamic_layout_doubles_each_slab_up_to_its_cap() {
        let layout = Dynamic::new(16, 8, 32, 200);

        let rows: Vec<usize> = (0..5).map(|slabs| layout.slab_rows(slabs)).collect();

        assert_eq!(rows, [2, 4, 8, 12, 12]);
    }
}
