//! The join table's row arena: one build row id per stored tuple, kept in the
//! narrowest width that addresses the build side.
//!
//! The probe reads an arena entry for every candidate its bloom tag lets
//! through, so the entry's width is what the candidate loop pays for. A build
//! side whose row id space fits in 32 bits gets a `u32` arena; a larger one
//! gets a `u64` arena and the same probe compiled for that width. The matches
//! a probe records and the gathers that resolve them run in the same width,
//! so no id is ever widened on the way to the output.

use crate::arrays::accumulator::EncodedRowIds;
use crate::memory::{MultiSlabBuffer, SlabAllocator};

/// The row id space a narrow arena addresses.
pub(crate) const NARROW_ROW_ID_SPACE: usize = u32::MAX as usize + 1;

/// A row id as an arena of one width stores it.
pub(crate) trait RowId: Copy + Into<u64> + Send + Sync + 'static {
    /// Narrow `row_id` to this width; the build chose the width so that every
    /// id of its row id space fits.
    fn from_row_id(row_id: u64) -> Self;

    /// Hand a run of ids to the accumulators' gathers.
    fn encode(ids: &[Self]) -> EncodedRowIds<'_>;
}

impl RowId for u32 {
    #[inline(always)]
    fn from_row_id(row_id: u64) -> Self {
        debug_assert!(row_id < NARROW_ROW_ID_SPACE as u64);
        row_id as u32
    }

    fn encode(ids: &[Self]) -> EncodedRowIds<'_> {
        EncodedRowIds::Narrow(ids)
    }
}

impl RowId for u64 {
    #[inline(always)]
    fn from_row_id(row_id: u64) -> Self {
        row_id
    }

    fn encode(ids: &[Self]) -> EncodedRowIds<'_> {
        EncodedRowIds::Wide(ids)
    }
}

/// The arena in the width the build chose.
pub(crate) enum RowArena {
    Narrow(MultiSlabBuffer<u32>),
    Wide(MultiSlabBuffer<u64>),
}

impl RowArena {
    /// The empty arena a [`super::JoinTable`] starts with; the build replaces
    /// it once the table's size is known.
    pub(crate) fn empty() -> Self {
        Self::Narrow(MultiSlabBuffer::new(Vec::new()))
    }

    /// Allocate `len` entries, narrow when `row_id_space` fits 32 bits and
    /// wide otherwise.
    pub(crate) fn allocate(allocator: &mut SlabAllocator, len: usize, row_id_space: usize) -> Self {
        if row_id_space <= NARROW_ROW_ID_SPACE {
            Self::Narrow(allocator.create_multi_slab_buffer::<u32>(len, false))
        } else {
            Self::Wide(allocator.create_multi_slab_buffer::<u64>(len, false))
        }
    }
}
