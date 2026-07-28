//! Per-group aggregation state and operations.
//!
//! [`AggregationValue`] is the value half of a grouped hash-table entry. It
//! defines how to read an input row, initialize or update one group's state,
//! merge partial states, and build the result columns. The key half is described
//! independently by [`KeyExtractor`](super::keys::KeyExtractor), so any supported
//! key representation can be paired with any aggregation representation.
//!
//! There are two main representations:
//!
//! - [`Compiled`] stores a fixed, statically typed tuple of numeric aggregates.
//! - [`RuntimeAggregation`] stores a query-defined number of homogeneous cells.
//!   It supports signatures whose arity is not known to Rust's type system and
//!   is also the representation used for string `MIN` and `MAX`.

use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::{PersistedKey, ScatterRows, SizedScatterRows};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field};
use std::sync::Arc;

pub mod cell;
pub mod container;
pub mod distinct;
pub mod fold;
pub mod read;

pub use cell::{Cell, F64Cell, IntCell};
pub use container::{
    Compiled, CountSlot, MaxSlot, MinSlot, OpTuple, RuntimeAggregation, RuntimeAggregationContext,
    SumSlot,
};
pub use distinct::Distinct;
pub use fold::{
    Count, F64Max, F64Min, F64Sum, Fold, Max, Min, StrMax, StrMin, Sum, U128Max, U128Min, U128Sum,
    WideSum,
};
pub use read::{IntRead, NoRead, Read, StrRead};

/// The operation computed by one aggregate output slot.
///
/// `Avg` is not represented: `AVG(c)` is lowered to `sum(c)` + `count(c)` with a
/// division projection downstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregationKind {
    /// `COUNT(*)` — +1 per row, ignores the column.
    CountStar,
    /// `COUNT(col)` — +1 per non-null row.
    Count,
    /// `SUM(col)`.
    Sum,
    /// `MIN(col)`. The input type determines whether this is numeric or textual.
    Min,
    /// `MAX(col)`. See [`Min`](AggregationKind::Min).
    Max,
}

/// A bound aggregate output: its operation, input column, and result type.
///
/// A runtime aggregation may choose one cell width for the whole signature.
/// That storage width is independent of the SQL result type. For example,
/// `COUNT` still produces `BIGINT` when it shares an `i128` cell array with a
/// wide `SUM`. The output phase therefore renders the cell and then casts it to
/// `output_type` when necessary.
#[derive(Clone, Debug)]
pub struct AggregationSlot {
    pub kind: AggregationKind,
    pub column: usize,
    pub output_type: DataType,
}

impl AggregationSlot {
    /// Bind `kind` to `column` with the declared result `output_type`.
    pub fn new(kind: AggregationKind, column: usize, output_type: DataType) -> Self {
        Self {
            kind,
            column,
            output_type,
        }
    }

    /// Whether this is string `MIN` or `MAX`.
    ///
    /// String states contain an [`ArenaKey`](super::ArenaKey), which requires a
    /// 128-bit cell. Such signatures also avoid raw radix scatter because scatter
    /// would persist every candidate string before knowing whether it wins.
    #[inline(always)]
    pub fn is_string_extreme(&self) -> bool {
        matches!(self.kind, AggregationKind::Min | AggregationKind::Max)
            && self.output_type == DataType::Utf8View
    }
}

/// Cast a rendered accumulator column to its declared result type.
///
/// When the types already match, this only clones the column's `Arc`.
pub(crate) fn cast_value_column(
    field: Field,
    column: ArrayRef,
    output_type: &DataType,
) -> (Field, ArrayRef) {
    if field.data_type() == output_type {
        return (field, column);
    }
    let casted = arrow::compute::cast(&column, output_type).expect("aggregate output column cast");
    let field = Field::new(field.name(), output_type.clone(), field.is_nullable());
    (field, casted)
}

/// Immutable, query-wide information needed by an aggregation representation.
///
/// The context is built once and cheaply cloned into workers and merge jobs.
/// Fixed, numeric representations use `()`. A runtime representation carries
/// its slot descriptors and the arena used to resolve string states.
pub trait AggregationContext: Clone + Send + Sync + 'static {
    /// Mutable worker-local state created from this context.
    type Worker: AggregationWorkerState;
    /// Build the query-wide context.
    fn build(slots: &[AggregationSlot], arena: &Arc<SharedArena>) -> Self;
    /// Create state for one worker.
    fn worker(&self) -> Self::Worker;
}

/// Worker-local resources used while aggregation states are being written.
///
/// Numeric representations use `()`. String-capable representations use a
/// [`WorkerArena`] and return its active buffer when the worker finishes.
pub trait AggregationWorkerState {
    /// Release worker-local resources back to their shared owner.
    fn flush(self);
}

impl AggregationContext for () {
    type Worker = ();
    fn build(_slots: &[AggregationSlot], _arena: &Arc<SharedArena>) {}
    fn worker(&self) {}
}
impl AggregationWorkerState for () {
    fn flush(self) {}
}

impl AggregationWorkerState for WorkerArena {
    fn flush(self) {
        WorkerArena::flush(self)
    }
}

/// Builds the aggregate columns of a grouped result.
///
/// Values are appended one group at a time. [`finish`](Self::finish) converts
/// the builders into Arrow fields and arrays.
pub trait AggregationColumnBuilders {
    /// The per-group value these columns accumulate.
    type Value: AggregationValue;
    /// Query-wide information used while building the columns.
    type Context;

    /// Allocate builders for at most `rows` groups.
    fn with_capacity(allocator: &mut SlabAllocator, rows: usize, context: &Self::Context) -> Self;
    /// Append a value copied out of table storage.
    fn push_owned(&mut self, value: &Self::Value);
    /// Append a value directly from a table entry.
    fn push_entry(&mut self, state: &<Self::Value as AggregationValue>::EntryState);
    /// Materialize the completed Arrow columns and their fields.
    fn finish(self, context: &Self::Context) -> (Vec<Field>, Vec<ArrayRef>);
}

/// A fixed-size aggregation whose table state is the value itself.
///
/// [`Compiled`] and [`Distinct`] use this simpler, by-value interface.
/// [`AggregationValue`] has a blanket implementation that adapts it to in-place
/// table operations. [`RuntimeAggregation`] cannot use this interface because
/// its entry state is a dynamically sized slice.
///
/// # Safety
///
/// The all-zero bit pattern must be a valid value of `Self`. Hash-table slabs
/// start zeroed and may be viewed as `Self` before a new group is initialized.
/// `Copy` ensures the value has no destructor and can be relocated byte-for-byte.
pub unsafe trait ByValueAggregation: Copy + Default + Send + Sync + 'static {
    /// Per-batch reader holding the downcast value columns.
    type Reader<'b>;
    /// Immutable query-wide information used by merge and output.
    type Context: AggregationContext<Worker = Self::WorkerState>;
    /// The trailing value columns these groups emit (the value-side counterpart
    /// to [`KeyExtractor::Columns`](super::keys::KeyExtractor::Columns)).
    type Columns: AggregationColumnBuilders<Value = Self, Context = Self::Context>;
    /// The scalar an `ORDER BY <slot> DESC LIMIT k` sorts on — widened to `i128`
    /// so a wide sum compares at full precision.
    type SortKey: Ord + Copy;
    /// Mutable resources used by one consume worker.
    type WorkerState: AggregationWorkerState;

    /// Bind `batch`'s value columns for the configured `slots`.
    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self::Reader<'b>;

    /// Create a new group value from row `idx`.
    fn value(reader: &Self::Reader<'_>, idx: usize, wc: &mut Self::WorkerState) -> Self;

    /// Fold row `idx` into an existing group.
    ///
    /// The default creates a one-row value and merges it.
    #[inline(always)]
    fn update_from_reader(
        self,
        reader: &Self::Reader<'_>,
        idx: usize,
        wc: &mut Self::WorkerState,
        ctx: &Self::Context,
    ) -> Self {
        self.merge(Self::value(reader, idx, wc), ctx)
    }

    /// Combine two partial group values — the partition merge and the radix fold.
    fn merge(self, other: Self, ctx: &Self::Context) -> Self;

    /// This group's value for slot `slot`, as an `ORDER BY` sort key.
    fn sort_key(&self, slot: usize) -> Self::SortKey;
}

/// Describes the aggregation state stored in each grouped hash-table entry.
///
/// `Self` is the sized value used by paths that must own a result, such as a
/// top-k heap. [`EntryState`](Self::EntryState) is the representation stored
/// inline in the table and may be dynamically sized. They are the same type for
/// fixed aggregations; runtime aggregation uses `[A]` as its entry state and a
/// small arena-backed handle as `Self`.
///
/// # Safety
///
/// Implementations define references into raw, zero-filled table storage and
/// must uphold all of these requirements:
///
/// - `entry_state_size`, `entry_state_align`, and the two view methods must
///   describe the same layout for a given `EntryStateMeta`.
/// - An all-zero region of that layout must be a valid `EntryState`. Empty slots
///   may be viewed before their hash is checked.
/// - The state must require no destructor and remain valid when its raw bytes are
///   relocated during table growth.
/// - `entry_state_ref` and `entry_state_mut` must construct references with the
///   exact extent and alignment described by the metadata.
///
/// `Self: Copy` supplies the corresponding no-destructor and relocation
/// guarantee for values copied out of the table.
pub unsafe trait AggregationValue: Copy + Default + Send + Sync + 'static {
    /// The entry-resident form of one group's value; what the fold methods
    /// mutate in place. `Self` for a fixed-arity value; a runtime-length cell
    /// slice for [`RuntimeAggregation`].
    type EntryState: ?Sized + Send + Sync;
    /// Per-table metadata needed to size and view an entry state.
    type EntryStateMeta: Copy + Send + Sync + 'static;
    /// The radix scatter's per-partition row buffer for this value: typed
    /// tuples for a fixed-arity value, runtime-strided rows (cells inline,
    /// like a hash entry) for [`RuntimeAggregation`]. Rows are seeded in place, so
    /// scattering allocates nothing per row.
    type ScatterBuffer<KP: PersistedKey>: ScatterRows<KP, Self>;
    /// Per-batch reader holding the downcast value columns.
    type Reader<'b>;
    /// The shared, read-side context the merge and output phases resolve
    /// through; see [`ByValueAggregation::Context`].
    type Context: AggregationContext<Worker = Self::WorkerState>;
    /// The trailing value columns these groups emit.
    type Columns: AggregationColumnBuilders<Value = Self, Context = Self::Context>;
    /// The scalar an `ORDER BY <slot> DESC LIMIT k` sorts on.
    type SortKey: Ord + Copy;
    /// The per-worker write state consume folds into; see
    /// [`ByValueAggregation::WorkerState`].
    type WorkerState: AggregationWorkerState;

    /// Bind `batch`'s value columns for the configured `slots`.
    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self::Reader<'b>;

    /// The runtime metadata for this query's signature, read off the shared
    /// context once per table.
    fn entry_state_meta(ctx: &Self::Context) -> Self::EntryStateMeta;

    /// The byte size of one entry state.
    fn entry_state_size(meta: Self::EntryStateMeta) -> usize;

    /// The required alignment of an entry state.
    ///
    /// Unlike its size, the alignment belongs to the state type itself and
    /// therefore does not depend on per-table metadata.
    fn entry_state_align() -> usize;

    /// View an entry's state bytes.
    ///
    /// # Safety
    /// `ptr` must point at [`entry_state_size`](Self::entry_state_size) bytes aligned to
    /// [`entry_state_align`](Self::entry_state_align), valid for the returned lifetime,
    /// and `meta` must be the table's own.
    unsafe fn entry_state_ref<'a>(
        ptr: *const u8,
        meta: Self::EntryStateMeta,
    ) -> &'a Self::EntryState;

    /// Mutable counterpart of [`entry_state_ref`](Self::entry_state_ref).
    ///
    /// # Safety
    /// As [`entry_state_ref`](Self::entry_state_ref), plus `ptr` must be exclusive for
    /// the returned lifetime.
    unsafe fn entry_state_mut<'a>(
        ptr: *mut u8,
        meta: Self::EntryStateMeta,
    ) -> &'a mut Self::EntryState;

    /// Materialise a brand-new group from row `idx` into `dst` — the consume
    /// path's new-key case.
    fn seed_entry(
        dst: &mut Self::EntryState,
        reader: &Self::Reader<'_>,
        idx: usize,
        wc: &mut Self::WorkerState,
    );

    /// Fold row `idx` into the existing group at `dst`.
    fn update_entry(
        dst: &mut Self::EntryState,
        reader: &Self::Reader<'_>,
        idx: usize,
        wc: &mut Self::WorkerState,
        ctx: &Self::Context,
    );

    /// Combine two partial group values in place — the partition merge.
    fn merge_entries(dst: &mut Self::EntryState, src: &Self::EntryState, ctx: &Self::Context);

    /// Copy a finished partial into a freshly claimed entry (the merge's
    /// new-key case; `dst` is zeroed).
    fn copy_entry(dst: &mut Self::EntryState, src: &Self::EntryState);

    /// The group's value for slot `slot`, as an `ORDER BY` sort key.
    fn entry_sort_key(state: &Self::EntryState, slot: usize) -> Self::SortKey;

    /// Copy a table-resident value out into an owned one that survives its
    /// table — a top-k heap row. A fixed-arity value is its own owned form; a
    /// runtime-arity value copies its cells into the value arena, spawning the
    /// per-worker write handle into `wc` on first use (the caller flushes it).
    fn copy_out(
        state: &Self::EntryState,
        ctx: &Self::Context,
        wc: &mut Option<Self::WorkerState>,
    ) -> Self;
}

/// Adapt a fixed-size by-value aggregation to the table's in-place interface.
///
/// The entry state is `T`, so each operation loads the value, applies the
/// by-value operation, and writes it back.
// SAFETY: ByValueAggregation requires zero-valid, byte-relocatable state. This
// adapter uses exactly T's size and alignment and casts only suitably laid-out
// table storage to T.
unsafe impl<T: ByValueAggregation> AggregationValue for T {
    type EntryState = T;
    type EntryStateMeta = ();
    type ScatterBuffer<KP: PersistedKey> = SizedScatterRows<KP, T>;
    type Reader<'b> = <T as ByValueAggregation>::Reader<'b>;
    type Context = <T as ByValueAggregation>::Context;
    type Columns = <T as ByValueAggregation>::Columns;
    type SortKey = <T as ByValueAggregation>::SortKey;
    type WorkerState = <T as ByValueAggregation>::WorkerState;

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self::Reader<'b> {
        <T as ByValueAggregation>::make_reader(batch, slots)
    }

    fn entry_state_meta(_ctx: &Self::Context) {}

    fn entry_state_size(_meta: ()) -> usize {
        size_of::<T>()
    }

    fn entry_state_align() -> usize {
        align_of::<T>()
    }

    #[inline(always)]
    unsafe fn entry_state_ref<'a>(ptr: *const u8, _meta: ()) -> &'a T {
        unsafe { &*(ptr as *const T) }
    }

    #[inline(always)]
    unsafe fn entry_state_mut<'a>(ptr: *mut u8, _meta: ()) -> &'a mut T {
        unsafe { &mut *(ptr as *mut T) }
    }

    #[inline(always)]
    fn seed_entry(dst: &mut T, reader: &Self::Reader<'_>, idx: usize, wc: &mut Self::WorkerState) {
        *dst = <T as ByValueAggregation>::value(reader, idx, wc);
    }

    #[inline(always)]
    fn update_entry(
        dst: &mut T,
        reader: &Self::Reader<'_>,
        idx: usize,
        wc: &mut Self::WorkerState,
        ctx: &Self::Context,
    ) {
        *dst = dst.update_from_reader(reader, idx, wc, ctx);
    }

    #[inline(always)]
    fn merge_entries(dst: &mut T, src: &T, ctx: &Self::Context) {
        *dst = dst.merge(*src, ctx);
    }

    #[inline(always)]
    fn copy_entry(dst: &mut T, src: &T) {
        *dst = *src;
    }

    #[inline(always)]
    fn entry_sort_key(state: &T, slot: usize) -> Self::SortKey {
        state.sort_key(slot)
    }

    #[inline(always)]
    fn copy_out(state: &T, _ctx: &Self::Context, _wc: &mut Option<Self::WorkerState>) -> Self {
        *state
    }
}
