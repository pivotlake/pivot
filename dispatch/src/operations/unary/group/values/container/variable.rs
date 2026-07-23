//! [`Variable`] — the runtime-*arity* value: a signature whose slot count is
//! fixed for a given GROUP BY but not known at compile time, so no arity is
//! monomorphised for it.
//!
//! Where [`Dynamic`](super::Dynamic) stores its `N` cells as a `[A; N]` (`N` a
//! const generic, one instantiation per arity), `Variable`'s table-resident
//! form is a runtime-length cell slice, `Stored = [A]`, living inline in the
//! hash entry at a stride the table computes at query build. The entry layout
//! is therefore the same `hash | key | cells` a `Dynamic` entry has; only who
//! knows the cell count differs. That count is deliberately stored **nowhere**
//! in the data: consume reads it off the per-batch reader, merge and output off
//! the shared slot list, and the column builders off their own length.
//!
//! Each slot folds exactly as in [`Dynamic`], through the shared per-slot
//! helpers ([`seed_slot`]/[`update_slot`]/[`merge_slot`]/[`finish_slot`]), so
//! the two containers agree on every op. The generics mirror [`Dynamic`]'s:
//! the accumulator width `A` (`i64` narrow / `i128` wide) and the
//! `ONLY_ADDITIVE` branch-free fast path.
//!
//! The owned `Variable` value itself is a thin pointer to a cell block in the
//! value arena; it exists only for the side paths that need an owned, `Sized`
//! value (a top-k heap row, whose cells must outlive the table they came from,
//! copied out via [`to_owned`](AggregationValue::to_owned)). `Variable` opts
//! out of the radix scatter
//! ([`RADIX_COMPATIBLE`](AggregationValue::RADIX_COMPATIBLE) is `false`): the
//! scatter path materialises an owned value per *row*, which here would
//! allocate an arena block per row instead of per group. Such a worker
//! aggregates in place for the whole consume, as a string-extreme signature
//! already does.

use super::super::cell::{F64Cell, IntCell, StringCell, WideCell};
use super::super::{AggregationSlot, AggregationValue, SharedContext, ValueColumns};
use super::dynamic::{BoundSlot, finish_slot, merge_slot, seed_slot, update_slot};
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::sync::Arc;

/// A runtime-arity aggregation value in its *owned* form: one thin pointer to a
/// cell block in the value arena. The table-resident form is the cell slice
/// itself (see the module docs); this owned handle appears only in the top-k
/// heap, whose rows outlive the table entries they were copied from.
pub struct Variable<
    A: IntCell + StringCell + F64Cell + WideCell = i64,
    const ONLY_ADDITIVE: bool = false,
> {
    /// The owned copy's cells, in the value arena. Null only in the `Default`
    /// value, which is never read (every real value comes from
    /// [`to_owned`](AggregationValue::to_owned) or
    /// [`value`](AggregationValue::value)).
    cells: *mut A,
}

/// An owned `Variable` moves freely across threads like any other aggregation
/// value: its block lives in the shared arena's ring buffers (kept alive by the
/// `Arc<SharedArena>` every phase holds), and each owned copy has its own
/// block, so whoever holds the value owns the cells. Cross-thread visibility of
/// the cell writes rides the same handoffs that publish the tables themselves.
unsafe impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Send
    for Variable<A, ONLY_ADDITIVE>
{
}
unsafe impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Sync
    for Variable<A, ONLY_ADDITIVE>
{
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Copy
    for Variable<A, ONLY_ADDITIVE>
{
}
impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Clone
    for Variable<A, ONLY_ADDITIVE>
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Default
    for Variable<A, ONLY_ADDITIVE>
{
    fn default() -> Self {
        Self {
            cells: std::ptr::null_mut(),
        }
    }
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool>
    Variable<A, ONLY_ADDITIVE>
{
    /// The owned copy's cells. `len` comes from the caller's context (the slot
    /// count), since the value stores no length.
    ///
    /// # Safety
    /// `len` must not exceed the block's allocated cell count (the signature's
    /// slot count).
    unsafe fn cells(&self, len: usize) -> &[A] {
        unsafe { std::slice::from_raw_parts(self.cells, len) }
    }
}

/// The output-column builders for a [`Variable`] signature: one [`SlabColumn`]
/// per slot, the count taken from the shared slot list at construction. The
/// value-side counterpart to a key extractor's
/// [`KeyColumns`](crate::operations::unary::group::keys::KeyColumns).
pub struct VariableColumns<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool>
{
    cols: Vec<SlabColumn<A>>,
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> AggregationValue
    for Variable<A, ONLY_ADDITIVE>
{
    type Stored = [A];
    /// The signature's slot count: the one runtime fact the table needs to size
    /// entries and view their cell run.
    type StoredMeta = usize;
    type Reader<'b> = Box<[BoundSlot<'b>]>;
    /// The per-slot kinds (which op folds/renders each cell, and how many cells
    /// an entry holds) and the value arena (string extremes and owned top-k
    /// copies live there).
    type SharedContext = (Arc<[AggregationSlot]>, Arc<SharedArena>);
    type Columns = VariableColumns<A, ONLY_ADDITIVE>;
    type SortKey = i128;
    /// The per-worker arena write handle a string extreme persists winners
    /// into (and owned top-k copies allocate from).
    type WorkerContext = WorkerArena;

    /// The scatter path materialises an owned value per row, which for this
    /// value would allocate an arena cell block per *row* rather than per
    /// group, so a runtime-arity signature always aggregates in place.
    const RADIX_COMPATIBLE: bool = false;

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Box<[BoundSlot<'b>]> {
        slots
            .iter()
            .map(|slot| BoundSlot::bind(batch, slot))
            .collect()
    }

    fn stored_meta(ctx: &Self::SharedContext) -> usize {
        ctx.0.len()
    }

    fn stored_size(meta: usize) -> usize {
        meta * size_of::<A>()
    }

    fn stored_align(_meta: usize) -> usize {
        align_of::<A>()
    }

    #[inline(always)]
    unsafe fn stored_ref<'a>(ptr: *const u8, meta: usize) -> &'a [A] {
        unsafe { std::slice::from_raw_parts(ptr as *const A, meta) }
    }

    #[inline(always)]
    unsafe fn stored_mut<'a>(ptr: *mut u8, meta: usize) -> &'a mut [A] {
        unsafe { std::slice::from_raw_parts_mut(ptr as *mut A, meta) }
    }

    #[inline(always)]
    fn seed_stored(dst: &mut [A], reader: &Self::Reader<'_>, idx: usize, wc: &mut WorkerArena) {
        debug_assert_eq!(dst.len(), reader.len());
        for (cell, slot) in dst.iter_mut().zip(reader.iter()) {
            *cell = seed_slot::<A, ONLY_ADDITIVE>(slot, idx, wc);
        }
    }

    #[inline(always)]
    fn update_stored(
        dst: &mut [A],
        reader: &Self::Reader<'_>,
        idx: usize,
        wc: &mut WorkerArena,
        ctx: &Self::SharedContext,
    ) {
        let (_, shared) = ctx;
        for (cell, slot) in dst.iter_mut().zip(reader.iter()) {
            *cell = update_slot::<A, ONLY_ADDITIVE>(*cell, slot, idx, wc, shared);
        }
    }

    #[inline(always)]
    fn merge_stored(dst: &mut [A], src: &[A], ctx: &Self::SharedContext) {
        let (slots, shared) = ctx;
        for ((a, &b), slot) in dst.iter_mut().zip(src.iter()).zip(slots.iter()) {
            *a = if ONLY_ADDITIVE {
                // All-additive: `Count` and every `Sum` merge by `+`, so skip
                // the per-slot kind dispatch entirely.
                *a + b
            } else {
                merge_slot(*a, b, slot, shared)
            };
        }
    }

    #[inline(always)]
    fn clone_stored(dst: &mut [A], src: &[A]) {
        dst.copy_from_slice(src);
    }

    #[inline(always)]
    fn sort_key_stored(stored: &[A], slot: usize) -> i128 {
        // Integer cells widen to their `ORDER BY` key. A string extreme never
        // feeds a top-k (the planner doesn't push one), so its raw bits here
        // are inert.
        stored[slot].into()
    }

    fn value(reader: &Self::Reader<'_>, idx: usize, wc: &mut WorkerArena) -> Self {
        // Only the radix scatter materialises per-row owned values, and this
        // value opts out of radix, so this is never on a hot path.
        let cells: *mut A = wc.alloc_cells(reader.len());
        for (s, slot) in reader.iter().enumerate() {
            unsafe {
                cells
                    .add(s)
                    .write(seed_slot::<A, ONLY_ADDITIVE>(slot, idx, wc))
            };
        }
        Self { cells }
    }

    fn store(dst: &mut [A], value: Self) {
        dst.copy_from_slice(unsafe { value.cells(dst.len()) });
    }

    fn merge_value(dst: &mut [A], value: Self, ctx: &Self::SharedContext) {
        Self::merge_stored(dst, unsafe { value.cells(dst.len()) }, ctx);
    }

    fn to_owned(stored: &[A], ctx: &Self::SharedContext, wc: &mut Option<WorkerArena>) -> Self {
        let wc = wc.get_or_insert_with(|| ctx.worker());
        let cells: *mut A = wc.alloc_cells(stored.len());
        unsafe { std::ptr::copy_nonoverlapping(stored.as_ptr(), cells, stored.len()) };
        Self { cells }
    }
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> ValueColumns
    for VariableColumns<A, ONLY_ADDITIVE>
{
    type Value = Variable<A, ONLY_ADDITIVE>;
    type Context = (Arc<[AggregationSlot]>, Arc<SharedArena>);

    fn with_capacity(allocator: &mut SlabAllocator, rows: usize, context: &Self::Context) -> Self {
        let (slots, _) = context;
        Self {
            cols: slots
                .iter()
                .map(|_| SlabColumn::with_capacity(allocator, rows))
                .collect(),
        }
    }

    #[inline(always)]
    fn push(&mut self, value: &Self::Value) {
        // An owned (top-k) value; its arena block holds one cell per slot, and
        // these columns were built with one builder per slot.
        let cells = unsafe { value.cells(self.cols.len()) };
        for (col, &cell) in self.cols.iter_mut().zip(cells.iter()) {
            col.push(cell);
        }
    }

    #[inline(always)]
    fn push_stored(&mut self, stored: &[A]) {
        for (col, &cell) in self.cols.iter_mut().zip(stored.iter()) {
            col.push(cell);
        }
    }

    fn finish(self, context: &Self::Context) -> (Vec<Field>, Vec<ArrayRef>) {
        let (slots, arena) = context;
        let mut fields = Vec::with_capacity(self.cols.len());
        let mut arrays = Vec::with_capacity(self.cols.len());
        for (s, col) in self.cols.into_iter().enumerate() {
            let (f, a) = finish_slot(&format!("v{s}"), &slots[s], col, arena);
            fields.push(f);
            arrays.push(a);
        }
        (fields, arrays)
    }
}
