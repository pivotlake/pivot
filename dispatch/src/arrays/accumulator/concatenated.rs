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

use arrow_array::make_array;

use super::column::{ChunkedColumn, ColumnAccumulator, SourceSelection};
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
    fn append_from_single_batch(
        &mut self,
        column: &ArrayRef,
        selection: SourceSelection<'_>,
        _destination_start: usize,
        _allocator: &mut SlabAllocator,
    ) {
        match selection {
            SourceSelection::Range { start, len } => self.arrays.push(column.slice(start, len)),
            // Consecutive positions become one slice, so a selection that keeps
            // a run of rows costs one array rather than one per row.
            SourceSelection::Indices(indices) => {
                let mut run: Option<(usize, usize)> = None;
                for &row in indices {
                    let row = row as usize;
                    match run {
                        Some((start, end)) if row == end => run = Some((start, end + 1)),
                        Some((start, end)) => {
                            self.arrays.push(column.slice(start, end - start));
                            run = Some((row, row + 1));
                        }
                        None => run = Some((row, row + 1)),
                    }
                }
                if let Some((start, end)) = run {
                    self.arrays.push(column.slice(start, end - start));
                }
            }
        }
    }

    fn append_from_batches(
        &mut self,
        column: &ChunkedColumn,
        ids: &[u32],
        shift: u32,
        _destination_start: usize,
        _allocator: &mut SlabAllocator,
    ) {
        let mask = (1u32 << shift) - 1;
        // Consecutive same-batch rows become one slice, as above.
        let mut push = |batch: usize, start: usize, len: usize| {
            self.arrays
                .push(make_array(column.data[batch].slice(start, len)));
        };
        let mut run: Option<(usize, usize, usize)> = None;
        for &id in ids {
            let batch = (id >> shift) as usize;
            let row = (id & mask) as usize;
            match run {
                Some((run_batch, start, end)) if run_batch == batch && row == end => {
                    run = Some((run_batch, start, end + 1));
                }
                Some((run_batch, start, end)) => {
                    push(run_batch, start, end - start);
                    run = Some((batch, row, row + 1));
                }
                None => run = Some((batch, row, row + 1)),
            }
        }
        if let Some((run_batch, start, end)) = run {
            push(run_batch, start, end - start);
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
