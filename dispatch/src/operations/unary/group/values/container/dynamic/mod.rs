//! Runtime aggregation container.
//!
//! [`Dynamic`] handles signatures whose number of aggregation slots is known
//! while building a query, but not encoded in a Rust type.
//!
//! A table entry stores the accumulator cells inline:
//!
//! ```text
//! hash | key | cell 0 | cell 1 | ... | cell n
//! ```
//!
//! The shared query context supplies `n`; it is not repeated in every entry.
//! `A` selects `i64` or `i128` accumulator storage. `ONLY_ADDITIVE` removes the
//! per-slot operation dispatch when every slot merges with addition.
//!
//! Scatter rows use the same inline cell layout. [`Dynamic`] is the stored
//! cell run itself (an unsized slice newtype); [`OwnedDynamic`] is the sized
//! handle top-k rows hold, pointing at a copy in the value arena.

use super::super::cell::{F64Cell, IntCell, StringCell, WideCell};
use super::super::{
    AggregationSlot, AggregationValue, ArityBody, SharedContext, ValueColumnBuilder,
};
mod operations;
mod readers;

use super::super::AggregationKind;
use super::super::fold::{Count, F64Max, F64Min, F64Sum, Fold, Max, Min, StrMax, StrMin, Sum};
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::DataType;
use arrow_schema::Field;
use operations::{MergeCells, OperationReader, OperationsReader};
use std::sync::Arc;

/// One group's dynamic aggregation value: the cell run itself, one cell per
/// slot, living inline in a hash entry or scatter row. Unsized; the slot
/// count comes from the storage metadata.
#[repr(transparent)]
pub struct Dynamic<
    A: IntCell + StringCell + F64Cell + WideCell = i64,
    const ONLY_ADDITIVE: bool = false,
>([A]);

/// Owned handle to a dynamic aggregation value copied into the value arena.
///
/// Exists only for top-k heap rows, which must outlive their source table.
pub struct OwnedDynamic<
    A: IntCell + StringCell + F64Cell + WideCell = i64,
    const ONLY_ADDITIVE: bool = false,
> {
    /// Arena allocation containing one cell per slot. The default value is
    /// null and is never read.
    cells: *mut A,
}

// SAFETY: every non-null pointer refers to an independent allocation in the
// shared arena. The arena outlives every handle, and normal table handoffs
// provide synchronization between writers and readers.
unsafe impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Send
    for OwnedDynamic<A, ONLY_ADDITIVE>
{
}
unsafe impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Sync
    for OwnedDynamic<A, ONLY_ADDITIVE>
{
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Copy
    for OwnedDynamic<A, ONLY_ADDITIVE>
{
}
impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Clone
    for OwnedDynamic<A, ONLY_ADDITIVE>
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Default
    for OwnedDynamic<A, ONLY_ADDITIVE>
{
    fn default() -> Self {
        Self {
            cells: std::ptr::null_mut(),
        }
    }
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool>
    OwnedDynamic<A, ONLY_ADDITIVE>
{
    /// Returns the owned cell block using the slot count from the query context.
    ///
    /// # Safety
    /// `len` must equal the allocation's cell count.
    unsafe fn cells(&self, len: usize) -> &[A] {
        unsafe { std::slice::from_raw_parts(self.cells, len) }
    }
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> AggregationValue
    for Dynamic<A, ONLY_ADDITIVE>
{
    type Owned = OwnedDynamic<A, ONLY_ADDITIVE>;
    /// Number of cells stored in each table entry.
    type StorageMetadata = usize;
    type Reader<'b> = OperationsReader<'b>;
    /// Slot descriptors and the arena used by strings and owned top-k copies.
    type SharedContext = (Arc<[AggregationSlot]>, Arc<SharedArena>);
    type ColumnBuilder = DynamicColumnBuilder<A, ONLY_ADDITIVE>;
    type SortKey = i128;
    /// Per-worker arena handle for winning strings and owned copies.
    type WorkerContext = WorkerArena;

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> OperationsReader<'b> {
        OperationsReader::bind(batch, slots)
    }

    fn storage_metadata(ctx: &Self::SharedContext) -> usize {
        ctx.0.len()
    }

    fn metadata_for_arity<const N: usize>() -> usize {
        N
    }

    #[inline(always)]
    fn dispatch_arity<R>(metadata: usize, body: impl ArityBody<R>) -> R {
        // Specialize common arities. The fallback keeps code size bounded for
        // larger signatures.
        match metadata {
            1 => body.run::<1>(),
            2 => body.run::<2>(),
            3 => body.run::<3>(),
            4 => body.run::<4>(),
            _ => body.run::<0>(),
        }
    }

    fn stored_size(metadata: usize) -> usize {
        metadata * size_of::<A>()
    }

    fn stored_align() -> usize {
        align_of::<A>()
    }

    #[inline(always)]
    unsafe fn from_entry<'a>(ptr: *const u8, metadata: usize) -> &'a Self {
        // SAFETY of the cast: `repr(transparent)` over `[A]` gives Self the
        // same layout and slice metadata as the cell run.
        unsafe { &*(std::ptr::slice_from_raw_parts(ptr as *const A, metadata) as *const Self) }
    }

    #[inline(always)]
    unsafe fn from_entry_mut<'a>(ptr: *mut u8, metadata: usize) -> &'a mut Self {
        unsafe { &mut *(std::ptr::slice_from_raw_parts_mut(ptr as *mut A, metadata) as *mut Self) }
    }

    #[inline(always)]
    fn seed(
        &mut self,
        operations: &Self::Reader<'_>,
        idx: usize,
        worker_context: &mut WorkerArena,
    ) {
        let operations = operations.as_slice();
        debug_assert_eq!(self.0.len(), operations.len());
        // Constant lengths let LLVM unroll the common small signatures.
        // Larger signatures use the ordinary slice loop to limit generated
        // code and register pressure.
        match self.0.len() {
            1 => self.seed_fixed::<1>(operations, idx, worker_context),
            2 => self.seed_fixed::<2>(operations, idx, worker_context),
            3 => self.seed_fixed::<3>(operations, idx, worker_context),
            4 => self.seed_fixed::<4>(operations, idx, worker_context),
            _ => {
                for (cell, operation) in self.0.iter_mut().zip(operations.iter()) {
                    *cell = operation.seed::<A, ONLY_ADDITIVE>(idx, worker_context);
                }
            }
        }
    }

    #[inline(always)]
    fn update(
        &mut self,
        operations: &Self::Reader<'_>,
        idx: usize,
        worker_context: &mut WorkerArena,
        context: &Self::SharedContext,
    ) {
        let (_, shared) = context;
        let operations = operations.as_slice();
        // Match `seed` so repeated-key updates also unroll.
        match self.0.len() {
            1 => self.update_fixed::<1>(operations, idx, worker_context, shared),
            2 => self.update_fixed::<2>(operations, idx, worker_context, shared),
            3 => self.update_fixed::<3>(operations, idx, worker_context, shared),
            4 => self.update_fixed::<4>(operations, idx, worker_context, shared),
            _ => {
                for (cell, operation) in self.0.iter_mut().zip(operations.iter()) {
                    *cell =
                        operation.update::<A, ONLY_ADDITIVE>(*cell, idx, worker_context, shared);
                }
            }
        }
    }

    #[inline(always)]
    fn merge_from(&mut self, source: &Self, context: &Self::SharedContext) {
        // Match `seed` so common merge signatures unroll.
        match self.0.len() {
            1 => self.merge_fixed::<1>(source, context),
            2 => self.merge_fixed::<2>(source, context),
            3 => self.merge_fixed::<3>(source, context),
            4 => self.merge_fixed::<4>(source, context),
            _ => {
                let (slots, shared) = context;
                for ((destination_cell, &source_cell), slot) in
                    self.0.iter_mut().zip(source.0.iter()).zip(slots.iter())
                {
                    if ONLY_ADDITIVE {
                        *destination_cell = *destination_cell + source_cell;
                    } else {
                        destination_cell.merge_cells(source_cell, slot, shared);
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn copy_from(&mut self, source: &Self) {
        // Constant-size copies compile to direct loads and stores. Larger
        // values use the slice implementation.
        match self.0.len() {
            1 => self.clone_fixed::<1>(source),
            2 => self.clone_fixed::<2>(source),
            3 => self.clone_fixed::<3>(source),
            4 => self.clone_fixed::<4>(source),
            _ => self.0.copy_from_slice(&source.0),
        }
    }

    #[inline(always)]
    fn sort_key(&self, slot: usize) -> i128 {
        // The planner only requests top-k sort keys for numeric slots.
        self.0[slot].into()
    }

    fn to_owned(
        &self,
        context: &Self::SharedContext,
        worker_context: &mut Option<WorkerArena>,
    ) -> OwnedDynamic<A, ONLY_ADDITIVE> {
        let stored = &self.0;
        let worker_context = worker_context.get_or_insert_with(|| context.worker());
        let cells: *mut A = worker_context.alloc_cells(stored.len());
        unsafe { std::ptr::copy_nonoverlapping(stored.as_ptr(), cells, stored.len()) };
        OwnedDynamic { cells }
    }
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool>
    Dynamic<A, ONLY_ADDITIVE>
{
    /// Seeds a fixed-size cell array so the slot loop can unroll.
    #[inline(always)]
    fn seed_fixed<const N: usize>(
        &mut self,
        operations: &[OperationReader<'_>],
        idx: usize,
        worker_context: &mut WorkerArena,
    ) {
        // SAFETY: the caller dispatched after matching both lengths to `N`.
        debug_assert!(self.0.len() == N && operations.len() == N);
        let destination = unsafe { &mut *(self.0.as_mut_ptr() as *mut [A; N]) };
        let operations = unsafe { &*(operations.as_ptr() as *const [OperationReader<'_>; N]) };
        for operation_index in 0..N {
            destination[operation_index] =
                operations[operation_index].seed::<A, ONLY_ADDITIVE>(idx, worker_context);
        }
    }

    /// Updates a fixed-size cell array so the slot loop can unroll.
    #[inline(always)]
    fn update_fixed<const N: usize>(
        &mut self,
        operations: &[OperationReader<'_>],
        idx: usize,
        worker_context: &mut WorkerArena,
        shared: &Arc<SharedArena>,
    ) {
        // SAFETY: the caller dispatched after matching both lengths to `N`.
        debug_assert!(self.0.len() == N && operations.len() == N);
        let destination = unsafe { &mut *(self.0.as_mut_ptr() as *mut [A; N]) };
        let operations = unsafe { &*(operations.as_ptr() as *const [OperationReader<'_>; N]) };
        for operation_index in 0..N {
            destination[operation_index] = operations[operation_index].update::<A, ONLY_ADDITIVE>(
                destination[operation_index],
                idx,
                worker_context,
                shared,
            );
        }
    }

    /// Merges fixed-size cell arrays so the slot loop can unroll.
    #[inline(always)]
    fn merge_fixed<const N: usize>(
        &mut self,
        source: &Self,
        context: &(Arc<[AggregationSlot]>, Arc<SharedArena>),
    ) {
        let (slots, shared) = context;
        // SAFETY: all three cell runs come from the same `N`-slot signature.
        debug_assert!(self.0.len() == N && source.0.len() == N && slots.len() == N);
        let destination = unsafe { &mut *(self.0.as_mut_ptr() as *mut [A; N]) };
        let source = unsafe { &*(source.0.as_ptr() as *const [A; N]) };
        for slot_index in 0..N {
            if ONLY_ADDITIVE {
                // COUNT and SUM both merge by addition.
                destination[slot_index] = destination[slot_index] + source[slot_index];
            } else {
                destination[slot_index].merge_cells(source[slot_index], &slots[slot_index], shared);
            }
        }
    }

    /// Copies a fixed-size cell array with an array assignment.
    #[inline(always)]
    fn clone_fixed<const N: usize>(&mut self, source: &Self) {
        // SAFETY: the caller dispatched after matching both lengths to `N`.
        debug_assert!(self.0.len() == N && source.0.len() == N);
        let destination = unsafe { &mut *(self.0.as_mut_ptr() as *mut [A; N]) };
        let source = unsafe { &*(source.0.as_ptr() as *const [A; N]) };
        *destination = *source;
    }
}

/// One output-column builder per dynamic aggregation slot.
pub struct DynamicColumnBuilder<
    A: IntCell + StringCell + F64Cell + WideCell,
    const ONLY_ADDITIVE: bool,
> {
    builders: Vec<SlabColumn<A>>,
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> ValueColumnBuilder
    for DynamicColumnBuilder<A, ONLY_ADDITIVE>
{
    type Value = Dynamic<A, ONLY_ADDITIVE>;
    type Context = (Arc<[AggregationSlot]>, Arc<SharedArena>);

    fn with_capacity(allocator: &mut SlabAllocator, rows: usize, context: &Self::Context) -> Self {
        let (slots, _) = context;
        Self {
            builders: slots
                .iter()
                .map(|_| SlabColumn::with_capacity(allocator, rows))
                .collect(),
        }
    }

    #[inline(always)]
    fn push(&mut self, value: &OwnedDynamic<A, ONLY_ADDITIVE>) {
        // The query context gives both the allocation and columns the same
        // slot count.
        let cells = unsafe { value.cells(self.builders.len()) };
        for (builder, &cell) in self.builders.iter_mut().zip(cells.iter()) {
            builder.push(cell);
        }
    }

    #[inline(always)]
    fn push_stored(&mut self, stored: &Dynamic<A, ONLY_ADDITIVE>) {
        for (builder, &cell) in self.builders.iter_mut().zip(stored.0.iter()) {
            builder.push(cell);
        }
    }

    fn finish(self, context: &Self::Context) -> (Vec<Field>, Vec<ArrayRef>) {
        let (slots, arena) = context;
        let mut fields = Vec::with_capacity(self.builders.len());
        let mut arrays = Vec::with_capacity(self.builders.len());
        for (slot_index, builder) in self.builders.into_iter().enumerate() {
            let descriptor = &slots[slot_index];
            let name = &format!("v{slot_index}");
            let ty = &descriptor.output_type;
            let (field, array) = match descriptor.kind {
                AggregationKind::CountStar | AggregationKind::Count => {
                    Count::<A>::finish(name, builder)
                }
                AggregationKind::Sum if ty.is_floating() => F64Sum::<A>::finish(name, builder),
                AggregationKind::Sum => Sum::<A>::finish(name, builder),
                AggregationKind::Min if *ty == DataType::Utf8View => {
                    StrMin::<A>::finish(name, builder, arena)
                }
                AggregationKind::Min if ty.is_floating() => F64Min::<A>::finish(name, builder),
                AggregationKind::Min => Min::<A>::finish(name, builder),
                AggregationKind::Max if *ty == DataType::Utf8View => {
                    StrMax::<A>::finish(name, builder, arena)
                }
                AggregationKind::Max if ty.is_floating() => F64Max::<A>::finish(name, builder),
                AggregationKind::Max => Max::<A>::finish(name, builder),
            };
            fields.push(field);
            arrays.push(array);
        }
        (fields, arrays)
    }
}
