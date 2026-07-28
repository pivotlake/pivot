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
//! Scatter rows use the same inline cell layout. [`Dynamic`] itself is only an
//! owned handle used by top-k, where its pointer refers to a copy in the value
//! arena.

use super::super::cell::{F64Cell, IntCell, StringCell, WideCell};
use super::super::{
    AggregationSlot, AggregationValue, ArityBody, SharedContext, ValueColumnBuilder,
};
mod operations;
mod readers;

use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::{PersistedKey, StridedScatterRows};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use operations::{Operation, Operations, finish_operation, merge_operation};
use std::sync::Arc;

/// Owned handle to a dynamic aggregation value copied into the value arena.
///
/// Hash tables and scatter buffers store the cells inline. This handle exists
/// for top-k rows, which must outlive their source table.
pub struct Dynamic<
    A: IntCell + StringCell + F64Cell + WideCell = i64,
    const ONLY_ADDITIVE: bool = false,
> {
    /// Arena allocation containing one cell per slot. The default value is
    /// null and is never read.
    cells: *mut A,
}

// SAFETY: every non-null pointer refers to an independent allocation in the
// shared arena. The arena outlives every Dynamic handle, and normal table
// handoffs provide synchronization between writers and readers.
unsafe impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Send
    for Dynamic<A, ONLY_ADDITIVE>
{
}
unsafe impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Sync
    for Dynamic<A, ONLY_ADDITIVE>
{
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Copy
    for Dynamic<A, ONLY_ADDITIVE>
{
}
impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Clone
    for Dynamic<A, ONLY_ADDITIVE>
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Default
    for Dynamic<A, ONLY_ADDITIVE>
{
    fn default() -> Self {
        Self {
            cells: std::ptr::null_mut(),
        }
    }
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> AggregationValue
    for Dynamic<A, ONLY_ADDITIVE>
{
    type Stored = [A];
    /// Number of cells stored in each table entry.
    type StorageMetadata = usize;
    /// Scatter rows carry cells inline at a query-specific stride.
    type Scatter<KP: PersistedKey> = StridedScatterRows<KP, Self>;
    type Reader<'b> = Operations<'b>;
    /// Slot descriptors and the arena used by strings and owned top-k copies.
    type SharedContext = (Arc<[AggregationSlot]>, Arc<SharedArena>);
    type ColumnBuilder = DynamicColumnBuilder<A, ONLY_ADDITIVE>;
    type SortKey = i128;
    /// Per-worker arena handle for winning strings and owned copies.
    type WorkerContext = WorkerArena;

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Operations<'b> {
        Operations::bind(batch, slots)
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
    unsafe fn stored_ref<'a>(ptr: *const u8, metadata: usize) -> &'a [A] {
        unsafe { std::slice::from_raw_parts(ptr as *const A, metadata) }
    }

    #[inline(always)]
    unsafe fn stored_mut<'a>(ptr: *mut u8, metadata: usize) -> &'a mut [A] {
        unsafe { std::slice::from_raw_parts_mut(ptr as *mut A, metadata) }
    }

    #[inline(always)]
    fn seed_stored(
        destination: &mut [A],
        operations: &Self::Reader<'_>,
        idx: usize,
        worker_context: &mut WorkerArena,
    ) {
        let operations = operations.as_slice();
        debug_assert_eq!(destination.len(), operations.len());
        // Constant lengths let LLVM unroll the common small signatures.
        // Larger signatures use the ordinary slice loop to limit generated
        // code and register pressure.
        match destination.len() {
            1 => Self::seed_fixed::<1>(destination, operations, idx, worker_context),
            2 => Self::seed_fixed::<2>(destination, operations, idx, worker_context),
            3 => Self::seed_fixed::<3>(destination, operations, idx, worker_context),
            4 => Self::seed_fixed::<4>(destination, operations, idx, worker_context),
            _ => {
                for (cell, operation) in destination.iter_mut().zip(operations.iter()) {
                    *cell = operation.seed::<A, ONLY_ADDITIVE>(idx, worker_context);
                }
            }
        }
    }

    #[inline(always)]
    fn update_stored(
        destination: &mut [A],
        operations: &Self::Reader<'_>,
        idx: usize,
        worker_context: &mut WorkerArena,
        context: &Self::SharedContext,
    ) {
        let (_, shared) = context;
        let operations = operations.as_slice();
        // Match `seed_stored` so repeated-key updates also unroll.
        match destination.len() {
            1 => Self::update_fixed::<1>(destination, operations, idx, worker_context, shared),
            2 => Self::update_fixed::<2>(destination, operations, idx, worker_context, shared),
            3 => Self::update_fixed::<3>(destination, operations, idx, worker_context, shared),
            4 => Self::update_fixed::<4>(destination, operations, idx, worker_context, shared),
            _ => {
                for (cell, operation) in destination.iter_mut().zip(operations.iter()) {
                    *cell =
                        operation.update::<A, ONLY_ADDITIVE>(*cell, idx, worker_context, shared);
                }
            }
        }
    }

    #[inline(always)]
    fn merge_stored(destination: &mut [A], source: &[A], context: &Self::SharedContext) {
        // Match `seed_stored` so common merge signatures unroll.
        match destination.len() {
            1 => Self::merge_fixed::<1>(destination, source, context),
            2 => Self::merge_fixed::<2>(destination, source, context),
            3 => Self::merge_fixed::<3>(destination, source, context),
            4 => Self::merge_fixed::<4>(destination, source, context),
            _ => {
                let (slots, shared) = context;
                for ((destination_cell, &source_cell), slot) in
                    destination.iter_mut().zip(source.iter()).zip(slots.iter())
                {
                    *destination_cell = if ONLY_ADDITIVE {
                        *destination_cell + source_cell
                    } else {
                        merge_operation(*destination_cell, source_cell, slot, shared)
                    };
                }
            }
        }
    }

    #[inline(always)]
    fn clone_stored(destination: &mut [A], source: &[A]) {
        // Constant-size copies compile to direct loads and stores. Larger
        // values use the slice implementation.
        match destination.len() {
            1 => Self::clone_fixed::<1>(destination, source),
            2 => Self::clone_fixed::<2>(destination, source),
            3 => Self::clone_fixed::<3>(destination, source),
            4 => Self::clone_fixed::<4>(destination, source),
            _ => destination.copy_from_slice(source),
        }
    }

    #[inline(always)]
    fn sort_key_stored(stored: &[A], slot: usize) -> i128 {
        // The planner only requests top-k sort keys for numeric slots.
        stored[slot].into()
    }

    fn to_owned(
        stored: &[A],
        context: &Self::SharedContext,
        worker_context: &mut Option<WorkerArena>,
    ) -> Self {
        let worker_context = worker_context.get_or_insert_with(|| context.worker());
        let cells: *mut A = worker_context.alloc_cells(stored.len());
        unsafe { std::ptr::copy_nonoverlapping(stored.as_ptr(), cells, stored.len()) };
        Self { cells }
    }
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool>
    Dynamic<A, ONLY_ADDITIVE>
{
    /// Returns the owned cell block using the slot count from the query context.
    ///
    /// # Safety
    /// `len` must equal the allocation's cell count.
    unsafe fn cells(&self, len: usize) -> &[A] {
        unsafe { std::slice::from_raw_parts(self.cells, len) }
    }

    /// Seeds a fixed-size cell array so the slot loop can unroll.
    #[inline(always)]
    fn seed_fixed<const N: usize>(
        destination: &mut [A],
        operations: &[Operation<'_>],
        idx: usize,
        worker_context: &mut WorkerArena,
    ) {
        // SAFETY: the caller dispatched after matching both lengths to `N`.
        debug_assert!(destination.len() == N && operations.len() == N);
        let destination = unsafe { &mut *(destination.as_mut_ptr() as *mut [A; N]) };
        let operations = unsafe { &*(operations.as_ptr() as *const [Operation<'_>; N]) };
        for operation_index in 0..N {
            destination[operation_index] =
                operations[operation_index].seed::<A, ONLY_ADDITIVE>(idx, worker_context);
        }
    }

    /// Updates a fixed-size cell array so the slot loop can unroll.
    #[inline(always)]
    fn update_fixed<const N: usize>(
        destination: &mut [A],
        operations: &[Operation<'_>],
        idx: usize,
        worker_context: &mut WorkerArena,
        shared: &Arc<SharedArena>,
    ) {
        // SAFETY: the caller dispatched after matching both lengths to `N`.
        debug_assert!(destination.len() == N && operations.len() == N);
        let destination = unsafe { &mut *(destination.as_mut_ptr() as *mut [A; N]) };
        let operations = unsafe { &*(operations.as_ptr() as *const [Operation<'_>; N]) };
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
        destination: &mut [A],
        source: &[A],
        context: &(Arc<[AggregationSlot]>, Arc<SharedArena>),
    ) {
        let (slots, shared) = context;
        // SAFETY: all three slices come from the same `N`-slot signature.
        debug_assert!(destination.len() == N && source.len() == N && slots.len() == N);
        let destination = unsafe { &mut *(destination.as_mut_ptr() as *mut [A; N]) };
        let source = unsafe { &*(source.as_ptr() as *const [A; N]) };
        for slot_index in 0..N {
            destination[slot_index] = if ONLY_ADDITIVE {
                // COUNT and SUM both merge by addition.
                destination[slot_index] + source[slot_index]
            } else {
                merge_operation(
                    destination[slot_index],
                    source[slot_index],
                    &slots[slot_index],
                    shared,
                )
            };
        }
    }

    /// Copies a fixed-size cell array with an array assignment.
    #[inline(always)]
    fn clone_fixed<const N: usize>(destination: &mut [A], source: &[A]) {
        // SAFETY: the caller dispatched after matching both lengths to `N`.
        debug_assert!(destination.len() == N && source.len() == N);
        let destination = unsafe { &mut *(destination.as_mut_ptr() as *mut [A; N]) };
        let source = unsafe { &*(source.as_ptr() as *const [A; N]) };
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
    fn push(&mut self, value: &Self::Value) {
        // The query context gives both the allocation and columns the same
        // slot count.
        let cells = unsafe { value.cells(self.builders.len()) };
        for (builder, &cell) in self.builders.iter_mut().zip(cells.iter()) {
            builder.push(cell);
        }
    }

    #[inline(always)]
    fn push_stored(&mut self, stored: &[A]) {
        for (builder, &cell) in self.builders.iter_mut().zip(stored.iter()) {
            builder.push(cell);
        }
    }

    fn finish(self, context: &Self::Context) -> (Vec<Field>, Vec<ArrayRef>) {
        let (slots, arena) = context;
        let mut fields = Vec::with_capacity(self.builders.len());
        let mut arrays = Vec::with_capacity(self.builders.len());
        for (slot_index, builder) in self.builders.into_iter().enumerate() {
            let (field, array) = finish_operation(
                &format!("v{slot_index}"),
                &slots[slot_index],
                builder,
                arena,
            );
            fields.push(field);
            arrays.push(array);
        }
        (fields, arrays)
    }
}
