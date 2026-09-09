//! Accumulating a byte-view column (`Utf8View` or `BinaryView`).
//!
//! A view column is two things: 16-byte views, one per row, and the data blocks
//! the values too long to inline live in. The views always accumulate into a
//! slab of their own. Where the bytes behind them come from is the one place the
//! accumulator's two modes actually differ, so both live here:
//!
//! - Retaining the source's buffers copies no bytes at all. The source's data
//!   buffers join the emitted array's buffer list and each view's buffer index
//!   is rebased onto it, which is why the emitted array keeps its inputs alive.
//! - Copying the values gives the accumulator blocks of its own and copies each
//!   value into them, so the emitted array names nothing the inputs own.
//!
//! A value of up to [`INLINE_VIEW_LEN`] bytes needs neither: it lives inside its
//! own view, and both modes copy that view across as it lies.

use arrow::array::ArrayData;
use arrow_array::builder::make_view;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, make_array};
use arrow_buffer::{Buffer, NullBuffer};
use arrow_schema::{ArrowError, DataType};

use super::column::{ChunkedColumn, ColumnAccumulator};
use super::fixed_width::gather_fixed_width;
use super::validity::ValidityMask;
use super::{BUFFER_SIZE, ValueStorage};
use crate::arrays::slab_into_buffer;
use crate::memory::{SlabAllocator, SlabBuffer};

/// A byte-view value up to this length lives inside its view, with no data
/// buffer behind it.
pub(in crate::arrays) const INLINE_VIEW_LEN: u32 = 12;

/// Bytes a copying view column takes for its values at a time. Well under a
/// slab, so blocks pack into the buffers the allocator is bumping through rather
/// than taking one apiece and leaving the rest unused.
const DATA_BLOCK_SIZE: usize = 256 * 1024;

/// A column of byte views, with the bytes behind them held as [`ViewValues`] says.
pub(super) struct ViewColumn {
    data_type: DataType,
    capacity: usize,
    views: SlabBuffer<u128>,
    values: ViewValues,
    validity: ValidityMask,
    /// Per-batch rebase cache for multi-batch appends: the buffer-list position
    /// a batch's data buffers were registered at. An entry is live only while
    /// its generation matches `generation`, which
    /// [`take_array`](ColumnAccumulator::take_array) bumps because the buffer
    /// list resets with every emitted batch.
    batch_bases: Vec<(u64, u32)>,
    generation: u64,
}

/// Where a view column's bytes live, which is what
/// [`ValueStorage`] decides for it.
enum ViewValues {
    /// The source batches' own data buffers, cloned into one list that the
    /// appended views are rebased onto.
    SourceBuffers {
        buffers: Vec<Buffer>,
        /// Where the previous append's source data buffers were rebased onto, as
        /// their (start, length) range in `buffers`. Consecutive appends from the
        /// same source (a join draining against one build payload) reuse that
        /// range instead of adding its buffers again.
        last_source: Option<(usize, usize)>,
    },
    /// Blocks this column fills itself, so the emitted array references nothing
    /// of the batches its values came from.
    OwnedBlocks { blocks: Vec<DataBlock> },
}

/// One data block of a copying view column, shared with the chunked take
/// ([`crate::arrays::take::take_chunked`]), whose view gather copies values the
/// same way.
///
/// A block is slab memory, except for the one case the ring cannot serve: a
/// value's bytes must be one contiguous run, and the ring hands out runs of at
/// most [`BUFFER_SIZE`]. A single value larger than that gets a heap buffer of
/// its own, one allocation of exactly that value's size. Retaining the source
/// buffer it came from would copy nothing, but it would pin however much else
/// that buffer holds, which is the retention this mode exists to avoid.
pub(in crate::arrays) enum DataBlock {
    Slab {
        slab: SlabBuffer<u8>,
        /// Bytes the block can hold.
        size: usize,
        /// Bytes written so far.
        used: usize,
    },
    /// A single value too large for any slab.
    OversizedValue(Vec<u8>),
}

impl DataBlock {
    /// Hand the block's bytes to Arrow. A slab block rides in the buffer's
    /// allocation `Arc`, so it returns to the pool with the last reference to
    /// the array.
    pub(in crate::arrays) fn into_buffer(self) -> Buffer {
        match self {
            Self::Slab { slab, used, .. } => slab_into_buffer(slab, used),
            Self::OversizedValue(bytes) => Buffer::from_vec(bytes),
        }
    }
}

impl ViewColumn {
    pub(super) fn new(
        data_type: &DataType,
        capacity: usize,
        storage: ValueStorage,
        allocator: &mut SlabAllocator,
    ) -> Self {
        Self {
            data_type: data_type.clone(),
            capacity,
            views: allocator.create_slab_buffer(capacity, false),
            values: match storage {
                ValueStorage::RetainSourceBuffers => ViewValues::SourceBuffers {
                    buffers: Vec::new(),
                    last_source: None,
                },
                ValueStorage::CopyValues => ViewValues::OwnedBlocks { blocks: Vec::new() },
            },
            validity: ValidityMask::new(capacity),
            batch_bases: Vec::new(),
            generation: 1,
        }
    }
}

impl ViewColumn {
    /// Read the view at `row` through a batch's resolved values pointer, whose
    /// array offset is already applied.
    #[inline(always)]
    fn view_at(values: *const u8, row: usize) -> u128 {
        // SAFETY: `row` is an in-bounds position of the batch the pointer
        // resolves, which holds one 16-byte view per element.
        unsafe { (values as *const u128).add(row).read_unaligned() }
    }

    fn append_batches(
        &mut self,
        column: &ChunkedColumn,
        ids: &[u64],
        shift: u32,
        destination_start: usize,
        allocator: &mut SlabAllocator,
    ) {
        self.validity
            .append_by_ids(ids, shift, destination_start, |batch| {
                column.data[batch]
                    .nulls()
                    .filter(|nulls| nulls.null_count() > 0)
            });
        if self.batch_bases.len() < column.data.len() {
            self.batch_bases.resize(column.data.len(), (0, 0));
        }
        let Self {
            views,
            values,
            batch_bases,
            generation,
            ..
        } = self;
        let mask = (1u64 << shift) - 1;
        match values {
            ViewValues::SourceBuffers { buffers, .. } => {
                // SAFETY: the views slab has capacity for `destination_start`
                // plus the appended rows (checked by the caller).
                unsafe {
                    let mut dst = views.ptr_at_index(destination_start);
                    for &id in ids {
                        let batch_idx = (id >> shift) as usize;
                        let mut view = Self::view_at(
                            *column.values.get_unchecked(batch_idx),
                            (id & mask) as usize,
                        );
                        // A view longer than the inline limit points into its
                        // batch's data buffers (buffers 1 onward); register
                        // those once per emitted batch and rebase the buffer
                        // index onto the list.
                        if view as u32 > INLINE_VIEW_LEN {
                            let entry = batch_bases.get_unchecked_mut(batch_idx);
                            if entry.0 != *generation {
                                *entry = (*generation, buffers.len() as u32);
                                buffers
                                    .extend(column.data[batch_idx].buffers()[1..].iter().cloned());
                            }
                            view += (entry.1 as u128) << 64;
                        }
                        dst.write(view);
                        dst = dst.add(1);
                    }
                }
            }
            ViewValues::OwnedBlocks { blocks } => {
                for (destination, &id) in (destination_start..).zip(ids) {
                    let batch_idx = (id >> shift) as usize;
                    let view = Self::view_at(column.values[batch_idx], (id & mask) as usize);
                    let length = view as u32;
                    let copied = if length <= INLINE_VIEW_LEN {
                        view
                    } else {
                        let buffer = (view >> 64) as u32 as usize;
                        let offset = (view >> 96) as u32 as usize;
                        let value = &column.data[batch_idx].buffers()[1 + buffer]
                            [offset..offset + length as usize];
                        copy_value(blocks, value, allocator)
                    };
                    // SAFETY: the slab has room for `destination_start` plus
                    // the appended rows, which the caller checked.
                    unsafe { *views.ptr_at_index(destination) = copied };
                }
            }
        }
    }
}

impl ColumnAccumulator for ViewColumn {
    fn append_from_indices(
        &mut self,
        column: &ArrayRef,
        indices: &[u32],
        destination_start: usize,
        allocator: &mut SlabAllocator,
    ) {
        let (views, source_buffers, nulls) = source_parts(column);
        self.validity
            .append_indices(nulls, indices, destination_start);
        match &mut self.values {
            ViewValues::SourceBuffers {
                buffers,
                last_source,
            } => append_rebasing_indices(
                &mut self.views,
                buffers,
                last_source,
                views,
                source_buffers,
                indices,
                destination_start,
            ),
            ViewValues::OwnedBlocks { blocks } => append_copying_values(
                &mut self.views,
                blocks,
                views,
                source_buffers,
                indices.iter().map(|&row| row as usize),
                destination_start,
                allocator,
            ),
        }
    }

    fn append_from_range(
        &mut self,
        column: &ArrayRef,
        start: usize,
        len: usize,
        destination_start: usize,
        allocator: &mut SlabAllocator,
    ) {
        let (views, source_buffers, nulls) = source_parts(column);
        self.validity
            .append_range(nulls, start, len, destination_start);
        match &mut self.values {
            ViewValues::SourceBuffers {
                buffers,
                last_source,
            } => append_rebasing_range(
                &mut self.views,
                buffers,
                last_source,
                views,
                source_buffers,
                start,
                len,
                destination_start,
            ),
            ViewValues::OwnedBlocks { blocks } => append_copying_values(
                &mut self.views,
                blocks,
                views,
                source_buffers,
                start..start + len,
                destination_start,
                allocator,
            ),
        }
    }

    fn append_from_batches(
        &mut self,
        column: &ChunkedColumn,
        ids: &[u64],
        shift: u32,
        destination_start: usize,
        allocator: &mut SlabAllocator,
    ) {
        self.append_batches(column, ids, shift, destination_start, allocator)
    }

    fn take_array(
        &mut self,
        len: usize,
        allocator: &mut SlabAllocator,
    ) -> Result<ArrayRef, ArrowError> {
        self.generation += 1;
        let fresh = allocator.create_slab_buffer(self.capacity, false);
        let views = slab_into_buffer(
            std::mem::replace(&mut self.views, fresh),
            len * size_of::<u128>(),
        );
        let buffers: Vec<Buffer> = match &mut self.values {
            ViewValues::SourceBuffers {
                buffers,
                last_source,
            } => {
                *last_source = None;
                std::mem::take(buffers)
            }
            ViewValues::OwnedBlocks { blocks } => std::mem::take(blocks)
                .into_iter()
                .map(DataBlock::into_buffer)
                .collect(),
        };
        let out = ArrayData::builder(self.data_type.clone())
            .len(len)
            .add_buffer(views)
            .add_buffers(buffers)
            .nulls(self.validity.take(len));
        // SAFETY: every appended view was rebased onto the buffer list emitted
        // with it, or built over the block its bytes were copied into, so all
        // view references stay valid. Views of null rows hold whatever bytes the
        // source held, which the validity marks as not a value.
        Ok(make_array(unsafe { out.build_unchecked() }))
    }
}

/// Borrow view metadata through the concrete array type. `to_data` would clone
/// and drop an Arc per data buffer on every append.
fn source_parts(source: &ArrayRef) -> (&[u128], &[Buffer], Option<&NullBuffer>) {
    match source.data_type() {
        DataType::Utf8View => {
            let array = source.as_string_view();
            (array.views(), array.data_buffers(), array.nulls())
        }
        DataType::BinaryView => {
            let array = source.as_binary_view();
            (array.views(), array.data_buffers(), array.nulls())
        }
        _ => unreachable!("built only for view types"),
    }
}

fn source_buffer_base(
    buffers: &mut Vec<Buffer>,
    last_source: &mut Option<(usize, usize)>,
    source_buffers: &[Buffer],
) -> u128 {
    match *last_source {
        Some((start, count))
            if holds_same_buffers(&buffers[start..start + count], source_buffers) =>
        {
            start as u128
        }
        _ => {
            let start = buffers.len();
            buffers.extend(source_buffers.iter().cloned());
            *last_source = Some((start, source_buffers.len()));
            start as u128
        }
    }
}

/// Append indexed views that keep pointing at the source's data buffers.
#[allow(clippy::too_many_arguments)]
fn append_rebasing_indices(
    slab: &mut SlabBuffer<u128>,
    buffers: &mut Vec<Buffer>,
    last_source: &mut Option<(usize, usize)>,
    views: &[u128],
    source_buffers: &[Buffer],
    indices: &[u32],
    destination_start: usize,
) {
    let base = source_buffer_base(buffers, last_source, source_buffers);
    unsafe {
        let src = views.as_ptr();
        let dst = slab.ptr_at_index(destination_start);
        if base == 0 {
            gather_fixed_width::<u128>(src.cast(), dst.cast(), indices);
        } else {
            for (destination, &row) in indices.iter().enumerate() {
                let mut view = src.add(row as usize).read_unaligned();
                if view as u32 > INLINE_VIEW_LEN {
                    view += base << 64;
                }
                dst.add(destination).write(view);
            }
        }
    }
}

/// Append a contiguous range of views that keep pointing at source buffers.
#[allow(clippy::too_many_arguments)]
fn append_rebasing_range(
    slab: &mut SlabBuffer<u128>,
    buffers: &mut Vec<Buffer>,
    last_source: &mut Option<(usize, usize)>,
    views: &[u128],
    source_buffers: &[Buffer],
    start: usize,
    len: usize,
    destination_start: usize,
) {
    let base = source_buffer_base(buffers, last_source, source_buffers);
    unsafe {
        let src = views.as_ptr();
        let dst = slab.ptr_at_index(destination_start);
        if base == 0 {
            std::ptr::copy_nonoverlapping(src.add(start), dst, len);
        } else {
            for (destination, row) in (start..start + len).enumerate() {
                let mut view = src.add(row).read_unaligned();
                if view as u32 > INLINE_VIEW_LEN {
                    view += base << 64;
                }
                dst.add(destination).write(view);
            }
        }
    }
}

/// Append views over bytes copied into this column's own blocks, so the emitted
/// array holds nothing of the source. An inlined value is its own view and comes
/// over as it lies; a longer one's bytes are copied and re-viewed against the
/// block they land in.
#[allow(clippy::too_many_arguments)]
fn append_copying_values<I>(
    slab: &mut SlabBuffer<u128>,
    blocks: &mut Vec<DataBlock>,
    views: &[u128],
    source_buffers: &[Buffer],
    rows: I,
    destination_start: usize,
    allocator: &mut SlabAllocator,
) where
    I: IntoIterator<Item = usize>,
{
    for (destination, row) in (destination_start..).zip(rows) {
        let view = views[row];
        let length = view as u32;
        let copied = if length <= INLINE_VIEW_LEN {
            view
        } else {
            // The bytes a long view names: its buffer, at its offset.
            let buffer = (view >> 64) as u32 as usize;
            let offset = (view >> 96) as u32 as usize;
            let value = &source_buffers[buffer][offset..offset + length as usize];
            copy_value(blocks, value, allocator)
        };
        // SAFETY: the slab has room for `destination_start` plus the appended
        // rows, which the caller checked against the capacity.
        unsafe { *slab.ptr_at_index(destination) = copied };
    }
}

/// Copy `value` into the block being filled, opening one (or a larger one for an
/// outsized value) as needed, and return the view naming where it landed.
pub(in crate::arrays) fn copy_value(
    blocks: &mut Vec<DataBlock>,
    value: &[u8],
    allocator: &mut SlabAllocator,
) -> u128 {
    if value.len() > BUFFER_SIZE {
        blocks.push(DataBlock::OversizedValue(value.to_vec()));
        return make_view(value, blocks.len() as u32 - 1, 0);
    }
    let free = match blocks.last() {
        Some(DataBlock::Slab { size, used, .. }) => size - used,
        _ => 0,
    };
    if free < value.len() {
        // A value too large for a standard block takes a block sized to it,
        // which still fits one slab (anything larger took the branch above).
        let size = DATA_BLOCK_SIZE.max(value.len());
        blocks.push(DataBlock::Slab {
            slab: allocator.create_slab_buffer(size, false),
            size,
            used: 0,
        });
    }
    let block = blocks.len() as u32 - 1;
    let DataBlock::Slab { slab, used, .. } = blocks.last_mut().expect("a block was just ensured")
    else {
        unreachable!("the block being filled is always a slab block")
    };
    let offset = *used;
    // SAFETY: the block holds `size` bytes and `offset + value.len()` is within
    // it, which is what the free-space check above establishes.
    unsafe {
        std::ptr::copy_nonoverlapping(value.as_ptr(), slab.ptr_at_index(offset), value.len());
    }
    *used += value.len();
    make_view(value, block, offset as u32)
}

/// Whether `accumulated` holds exactly the buffers of `source`.
///
/// Comparing by address is sound because `accumulated` holds a clone of every
/// buffer it names: those clones keep the allocations alive, so an address that
/// still matches cannot have been freed and handed to a different buffer in the
/// meantime.
fn holds_same_buffers(accumulated: &[Buffer], source: &[Buffer]) -> bool {
    accumulated.len() == source.len()
        && accumulated.iter().zip(source).all(|(held, incoming)| {
            held.as_ptr() == incoming.as_ptr() && held.len() == incoming.len()
        })
}
