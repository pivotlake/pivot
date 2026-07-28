//! Storage for an aggregation signature whose arity is known only at query
//! construction time.
//!
//! A table entry contains the cells inline as `EntryState = [A]`. The table
//! computes the slice length and entry stride once from
//! [`RuntimeAggregationContext`]; individual entries store no length.
//!
//! The sized [`RuntimeAggregation`] type is used only when a value must outlive
//! its table entry, currently for top-k output. In that case [`copy_out`]
//! allocates a cell block in the value arena and returns a pointer-sized handle
//! to it. Copying the handle aliases the same immutable block.
//!
//! `A` is either the narrow `i64` cell or the wide `i128` cell.
//! `ALL_ADDITIVE` selects the specialized path for signatures containing only
//! `COUNT` and integer `SUM`.

use super::{BoundSlot, finish_slot, merge_slot, seed_slot, update_slot};
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::{PersistedKey, StridedScatterRows};
use crate::operations::unary::group::values::cell::{F64Cell, IntCell, StringCell, WideCell};
use crate::operations::unary::group::values::{
    AggregationColumnBuilders, AggregationContext, AggregationSlot, AggregationValue,
};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::sync::Arc;

/// Query-wide information for a [`RuntimeAggregation`].
///
/// `slots` defines both the number and meaning of the inline cells. `arena`
/// owns string states and any cell blocks copied out for top-k output.
#[derive(Clone)]
pub struct RuntimeAggregationContext {
    slots: Arc<[AggregationSlot]>,
    arena: Arc<SharedArena>,
}

impl AggregationContext for RuntimeAggregationContext {
    type Worker = WorkerArena;

    fn build(slots: &[AggregationSlot], arena: &Arc<SharedArena>) -> Self {
        Self {
            slots: Arc::from(slots),
            arena: arena.clone(),
        }
    }

    fn worker(&self) -> WorkerArena {
        WorkerArena::new(self.arena.clone())
    }
}

/// A sized handle to runtime aggregation cells copied into the value arena.
///
/// Normal table entries store their `[A]` state inline and do not contain this
/// handle.
pub struct RuntimeAggregation<
    A: IntCell + StringCell + F64Cell + WideCell = i64,
    const ALL_ADDITIVE: bool = false,
> {
    /// The copied cells in the value arena. This is null only for `Default`,
    /// which is never read; usable handles come from
    /// [`copy_out`](AggregationValue::copy_out).
    cells: *mut A,
}

// SAFETY: The pointed-to block is owned by the shared arena, remains allocated
// for the query, and is immutable after this handle is published. Copying the
// handle may alias that block, but no copy mutates it.
unsafe impl<A: IntCell + StringCell + F64Cell + WideCell, const ALL_ADDITIVE: bool> Send
    for RuntimeAggregation<A, ALL_ADDITIVE>
{
}
unsafe impl<A: IntCell + StringCell + F64Cell + WideCell, const ALL_ADDITIVE: bool> Sync
    for RuntimeAggregation<A, ALL_ADDITIVE>
{
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ALL_ADDITIVE: bool> Copy
    for RuntimeAggregation<A, ALL_ADDITIVE>
{
}
impl<A: IntCell + StringCell + F64Cell + WideCell, const ALL_ADDITIVE: bool> Clone
    for RuntimeAggregation<A, ALL_ADDITIVE>
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<A: IntCell + StringCell + F64Cell + WideCell, const ALL_ADDITIVE: bool> Default
    for RuntimeAggregation<A, ALL_ADDITIVE>
{
    fn default() -> Self {
        Self {
            cells: std::ptr::null_mut(),
        }
    }
}

/// Output builders for a runtime signature, with one [`SlabColumn`] per slot.
pub struct RuntimeAggregationColumnBuilders<
    A: IntCell + StringCell + F64Cell + WideCell,
    const ALL_ADDITIVE: bool,
> {
    cols: Vec<SlabColumn<A>>,
}

// SAFETY: `[A]` is laid out as `slot_count` contiguous, aligned cells. Both
// supported cell types accept an all-zero value, are `Copy`, and need no drop.
// The metadata and view methods below consistently use the same slot count.
unsafe impl<A: IntCell + StringCell + F64Cell + WideCell, const ALL_ADDITIVE: bool> AggregationValue
    for RuntimeAggregation<A, ALL_ADDITIVE>
{
    type EntryState = [A];
    /// The number of cells in every entry for this table.
    type EntryStateMeta = usize;
    /// Scatter rows carry the cells inline at the entry stride.
    type ScatterBuffer<KP: PersistedKey> = StridedScatterRows<KP, Self>;
    type Reader<'b> = Box<[BoundSlot<'b>]>;
    type Context = RuntimeAggregationContext;
    type Columns = RuntimeAggregationColumnBuilders<A, ALL_ADDITIVE>;
    type SortKey = i128;
    type WorkerState = WorkerArena;

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Box<[BoundSlot<'b>]> {
        slots
            .iter()
            .map(|slot| BoundSlot::bind(batch, slot))
            .collect()
    }

    fn entry_state_meta(ctx: &Self::Context) -> usize {
        ctx.slots.len()
    }

    fn entry_state_size(meta: usize) -> usize {
        meta * size_of::<A>()
    }

    fn entry_state_align(_meta: usize) -> usize {
        align_of::<A>()
    }

    #[inline(always)]
    unsafe fn entry_state_ref<'a>(ptr: *const u8, meta: usize) -> &'a [A] {
        unsafe { std::slice::from_raw_parts(ptr as *const A, meta) }
    }

    #[inline(always)]
    unsafe fn entry_state_mut<'a>(ptr: *mut u8, meta: usize) -> &'a mut [A] {
        unsafe { std::slice::from_raw_parts_mut(ptr as *mut A, meta) }
    }

    #[inline(always)]
    fn seed_entry(dst: &mut [A], reader: &Self::Reader<'_>, idx: usize, wc: &mut WorkerArena) {
        debug_assert_eq!(dst.len(), reader.len());
        for (cell, slot) in dst.iter_mut().zip(reader.iter()) {
            *cell = seed_slot::<A, ALL_ADDITIVE>(slot, idx, wc);
        }
    }

    #[inline(always)]
    fn update_entry(
        dst: &mut [A],
        reader: &Self::Reader<'_>,
        idx: usize,
        wc: &mut WorkerArena,
        ctx: &Self::Context,
    ) {
        for (cell, slot) in dst.iter_mut().zip(reader.iter()) {
            *cell = update_slot::<A, ALL_ADDITIVE>(*cell, slot, idx, wc, &ctx.arena);
        }
    }

    #[inline(always)]
    fn merge_entries(dst: &mut [A], src: &[A], ctx: &Self::Context) {
        for ((a, &b), slot) in dst.iter_mut().zip(src.iter()).zip(ctx.slots.iter()) {
            *a = if ALL_ADDITIVE {
                // All-additive: `Count` and every `Sum` merge by `+`, so skip
                // the per-slot kind dispatch entirely.
                *a + b
            } else {
                merge_slot(*a, b, slot, &ctx.arena)
            };
        }
    }

    #[inline(always)]
    fn copy_entry(dst: &mut [A], src: &[A]) {
        dst.copy_from_slice(src);
    }

    #[inline(always)]
    fn entry_sort_key(state: &[A], slot: usize) -> i128 {
        // Integer cells widen to their `ORDER BY` key. A string extreme never
        // feeds a top-k (the planner doesn't push one), so its raw bits here
        // are inert.
        state[slot].into()
    }

    fn copy_out(state: &[A], ctx: &Self::Context, wc: &mut Option<WorkerArena>) -> Self {
        let wc = wc.get_or_insert_with(|| ctx.worker());
        let cells: *mut A = wc.alloc_cells(state.len());
        unsafe { std::ptr::copy_nonoverlapping(state.as_ptr(), cells, state.len()) };
        Self { cells }
    }
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ALL_ADDITIVE: bool>
    RuntimeAggregation<A, ALL_ADDITIVE>
{
    /// Borrow the copied cells. The handle itself stores no length.
    ///
    /// # Safety
    /// `len` must not exceed the block's allocated cell count (the signature's
    /// slot count).
    unsafe fn cells(&self, len: usize) -> &[A] {
        unsafe { std::slice::from_raw_parts(self.cells, len) }
    }
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ALL_ADDITIVE: bool>
    AggregationColumnBuilders for RuntimeAggregationColumnBuilders<A, ALL_ADDITIVE>
{
    type Value = RuntimeAggregation<A, ALL_ADDITIVE>;
    type Context = RuntimeAggregationContext;

    fn with_capacity(allocator: &mut SlabAllocator, rows: usize, context: &Self::Context) -> Self {
        Self {
            cols: context
                .slots
                .iter()
                .map(|_| SlabColumn::with_capacity(allocator, rows))
                .collect(),
        }
    }

    #[inline(always)]
    fn push_owned(&mut self, value: &Self::Value) {
        // The builder count is the length of the arena block created by copy_out.
        let cells = unsafe { value.cells(self.cols.len()) };
        for (col, &cell) in self.cols.iter_mut().zip(cells.iter()) {
            col.push(cell);
        }
    }

    #[inline(always)]
    fn push_entry(&mut self, state: &[A]) {
        for (col, &cell) in self.cols.iter_mut().zip(state.iter()) {
            col.push(cell);
        }
    }

    fn finish(self, context: &Self::Context) -> (Vec<Field>, Vec<ArrayRef>) {
        let mut fields = Vec::with_capacity(self.cols.len());
        let mut arrays = Vec::with_capacity(self.cols.len());
        for (s, col) in self.cols.into_iter().enumerate() {
            let (f, a) = finish_slot(&format!("v{s}"), &context.slots[s], col, &context.arena);
            fields.push(f);
            arrays.push(a);
        }
        (fields, arrays)
    }
}
