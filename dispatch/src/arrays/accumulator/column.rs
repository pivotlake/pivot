//! The contract every column of a [`BatchAccumulator`](super::BatchAccumulator)
//! is accumulated through. The module docs of [`accumulator`](super) map out
//! which implementation a column type gets and why.

use arrow::array::ArrayData;
use arrow_array::ArrayRef;
use arrow_schema::ArrowError;

use crate::memory::SlabAllocator;

/// One column of an accumulation. Rows are copied in with the append methods
/// and handed to Arrow with [`take_array`](ColumnAccumulator::take_array).
///
/// An implementation is built for one Arrow type and one capacity and keeps
/// both, so neither is passed back in on every call. It must tolerate being
/// taken from and appended to again: an accumulator is reused for as many
/// batches as its owner emits.
pub(super) trait ColumnAccumulator {
    /// Append selected rows from one batch's column.
    ///
    /// `allocator` is drawn on only by an implementation that copies values
    /// (see [`ValueStorage`](super::ValueStorage)); one that retains the
    /// source's buffers never touches it.
    fn append_from_single_batch(
        &mut self,
        column: &ArrayRef,
        selection: SourceSelection<'_>,
        destination_start: usize,
        allocator: &mut SlabAllocator,
    );

    /// Append rows from a column stored across batches. Each `id` encodes
    /// `batch << shift | row`, and `column` contains one [`ArrayData`] per
    /// batch.
    fn append_from_batches(
        &mut self,
        column: &[ArrayData],
        ids: &[u32],
        shift: u32,
        destination_start: usize,
        allocator: &mut SlabAllocator,
    );

    /// Hand the `len` accumulated rows to Arrow and reset for the next batch.
    /// `len` is the accumulation's row count, which every column shares.
    fn take_array(
        &mut self,
        len: usize,
        allocator: &mut SlabAllocator,
    ) -> Result<ArrayRef, ArrowError>;
}

/// Which rows of a single source column an append takes.
///
/// One implementation serves both: a scattered selection reads its rows through
/// the indices, while a contiguous run copies them in bulk. The difference costs
/// a branch per column per append, not per row.
#[derive(Clone, Copy)]
pub(super) enum SourceSelection<'a> {
    /// The rows at these ascending positions, which is what a filter's
    /// survivors or a join's matches look like.
    Indices(&'a [u32]),
    /// The `len` rows from `start`: a whole input batch, or the part of one that
    /// fits before the accumulator fills.
    Range { start: usize, len: usize },
}

impl SourceSelection<'_> {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Indices(indices) => indices.len(),
            Self::Range { len, .. } => *len,
        }
    }

    /// The source row positions, in the order they are appended.
    pub(super) fn iter_positions(&self) -> impl Iterator<Item = usize> + '_ {
        let (indices, range) = match self {
            Self::Indices(indices) => (Some(indices.iter()), 0..0),
            Self::Range { start, len } => (None, *start..*start + *len),
        };
        indices
            .into_iter()
            .flatten()
            .map(|&row| row as usize)
            .chain(range)
    }
}
