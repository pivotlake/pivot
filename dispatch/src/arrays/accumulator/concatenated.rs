//! Accumulating any column the other strategies do not cover, by keeping the
//! appended rows as zero-copy slices and concatenating them on emit.
//!
//! This is what makes the accumulator total: booleans, offset strings,
//! dictionaries, lists and anything else the engine produces still work, they
//! just go through Arrow's `concat` instead of a slab. The other strategies are
//! the fast paths.
//!
//! Two consequences follow from holding slices, and a caller that cares about
//! either should know which of its columns land here. The rows are held as
//! references to the batches they came from, so a
//! [`CopyValues`](super::ValueStorage::CopyValues) accumulation does *not* let
//! go of its inputs for these columns; and the concatenation allocates through
//! Arrow's own allocator, so their values leave the ring.

use arrow::compute::kernels::concat::concat;
use arrow_array::{Array, ArrayRef};
use arrow_schema::ArrowError;

use super::column::{ColumnAccumulator, SourceSelection};
use crate::memory::SlabAllocator;

/// A column held as the arrays it was appended from, concatenated on emit.
pub(super) struct ConcatenatedColumn {
    arrays: Vec<ArrayRef>,
}

impl ConcatenatedColumn {
    pub(super) fn new() -> Self {
        Self { arrays: Vec::new() }
    }
}

impl ColumnAccumulator for ConcatenatedColumn {
    fn append(
        &mut self,
        source: &ArrayRef,
        selection: SourceSelection<'_>,
        _destination_start: usize,
        _allocator: &mut SlabAllocator,
    ) {
        match selection {
            SourceSelection::Range { start, len } => self.arrays.push(source.slice(start, len)),
            // Consecutive positions become one slice, so a selection that keeps
            // a run of rows costs one array rather than one per row.
            SourceSelection::Indices(indices) => {
                let mut run: Option<(usize, usize)> = None;
                for &row in indices {
                    let row = row as usize;
                    match run {
                        Some((start, end)) if row == end => run = Some((start, end + 1)),
                        Some((start, end)) => {
                            self.arrays.push(source.slice(start, end - start));
                            run = Some((row, row + 1));
                        }
                        None => run = Some((row, row + 1)),
                    }
                }
                if let Some((start, end)) = run {
                    self.arrays.push(source.slice(start, end - start));
                }
            }
        }
    }

    fn take_array(
        &mut self,
        _len: usize,
        _allocator: &mut SlabAllocator,
    ) -> Result<ArrayRef, ArrowError> {
        let refs: Vec<&dyn Array> = self.arrays.iter().map(|array| array.as_ref()).collect();
        let out = concat(&refs)?;
        self.arrays.clear();
        Ok(out)
    }
}
