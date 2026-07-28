//! Aggregation values for GROUP BY.
//!
//! [`AggregationValue`] defines the payload stored for each group. The planner
//! chooses one of two containers:
//!
//! - [`Compiled`] for selected, fixed numeric signatures.
//! - [`Dynamic`] for any number or mix of supported aggregation slots.
//!
//! Both containers implement the same lifecycle:
//!
//! ```text
//! input row -> seed or update table entry -> merge partial entries -> output
//! ```

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
pub use container::{Compiled, CountSlot, Dynamic, MaxSlot, MinSlot, OpTuple, SumSlot};
pub use distinct::Distinct;
pub use fold::{
    Count, F64Max, F64Min, F64Sum, Fold, Max, Min, StrMax, StrMin, Sum, U128Max, U128Min, U128Sum,
    WideSum,
};
pub use read::{IntRead, NoRead, Read, StrRead};

/// Operation computed by one aggregation slot.
///
/// `Avg` is not represented: `AVG(c)` is lowered to `sum(c)` + `count(c)` with a
/// divide projection, so a grouped average arrives as a `Sum` slot plus a `Count`
/// slot and the division happens downstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregationKind {
    /// `COUNT(*)`: adds one for every row and ignores the column.
    CountStar,
    /// `COUNT(col)`: adds one for every non-null value.
    Count,
    /// `SUM(col)`.
    Sum,
    /// `MIN(col)`. The input column type selects the numeric or string fold.
    Min,
    /// `MAX(col)`. See [`Min`](AggregationKind::Min).
    Max,
}

/// One aggregate output slot: which aggregate, over which input column, and the
/// Arrow type its output column is declared as.
///
/// Accumulator width and output type are independent. For example, a COUNT may
/// share an `i128` dynamic cell array but must still produce an Int64 column.
#[derive(Clone, Debug)]
pub struct AggregationSlot {
    pub kind: AggregationKind,
    pub column: usize,
    pub output_type: DataType,
}

impl AggregationSlot {
    /// Creates a slot over an input column with the declared output type.
    pub fn new(kind: AggregationKind, column: usize, output_type: DataType) -> Self {
        Self {
            kind,
            column,
            output_type,
        }
    }

    /// Returns whether this is MIN or MAX over strings.
    #[inline(always)]
    pub fn is_string_extreme(&self) -> bool {
        matches!(self.kind, AggregationKind::Min | AggregationKind::Max)
            && self.output_type == DataType::Utf8View
    }
}

/// Casts an accumulator column to the output type declared by the planner.
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

/// Shared aggregation state used by workers, merge jobs, and output.
///
/// Compiled numeric values use `()`. Dynamic values carry their slot
/// descriptors and value arena.
pub trait SharedContext: Clone + Send + Sync + 'static {
    /// Per-worker state created from this context.
    type Worker: WorkerContext;
    /// Builds shared state for the aggregation slots.
    fn build(slots: &[AggregationSlot], arena: &Arc<SharedArena>) -> Self;
    /// Creates state for one worker.
    fn worker(&self) -> Self::Worker;
}

/// Per-worker aggregation state used while consuming rows.
pub trait WorkerContext {
    /// Returns owned resources to the shared context after consuming input.
    fn flush(self);
}

impl SharedContext for () {
    type Worker = ();
    fn build(_slots: &[AggregationSlot], _arena: &Arc<SharedArena>) {}
    fn worker(&self) {}
}
impl WorkerContext for () {
    fn flush(self) {}
}

impl SharedContext for (Arc<[AggregationSlot]>, Arc<SharedArena>) {
    type Worker = WorkerArena;
    fn build(slots: &[AggregationSlot], arena: &Arc<SharedArena>) -> Self {
        (Arc::from(slots), arena.clone())
    }
    fn worker(&self) -> WorkerArena {
        WorkerArena::new(self.1.clone())
    }
}
impl WorkerContext for WorkerArena {
    fn flush(self) {
        WorkerArena::flush(self)
    }
}

/// Builds the aggregation columns in a GROUP BY result.
pub trait ValueColumnBuilder {
    /// Value type appended to these columns.
    type Value: AggregationValue + ?Sized;
    /// Shared context needed while finishing columns.
    type Context;

    /// Allocates builders for at most `rows` output groups.
    fn with_capacity(allocator: &mut SlabAllocator, rows: usize, context: &Self::Context) -> Self;
    /// Appends an owned value, as used by top-k output.
    fn push(&mut self, value: &<Self::Value as AggregationValue>::Owned);
    /// Appends a value directly from a table entry.
    fn push_stored(&mut self, stored: &Self::Value);
    /// Finishes the Arrow fields and arrays.
    fn finish(self, context: &Self::Context) -> (Vec<Field>, Vec<ArrayRef>);
}

/// Fixed-size aggregation value stored directly in a table entry.
///
/// [`Compiled`] and [`Distinct`] implement this trait. A blanket implementation
/// adapts them to [`AggregationValue`]. [`Dynamic`] implements
/// `AggregationValue` directly because its stored cell slice is unsized.
pub trait OwnedValue: Copy + Default + Send + Sync + 'static {
    /// Bound input readers for one record batch.
    type Reader<'b>;
    /// Shared state used during merging and output.
    type SharedContext: SharedContext<Worker = Self::WorkerContext>;
    /// Output-column builders for this value.
    type ColumnBuilder: ValueColumnBuilder<Value = Self, Context = Self::SharedContext>;
    /// Scalar used by pushed-down ORDER BY and LIMIT.
    type SortKey: Ord + Copy;
    /// State owned by one consume worker.
    type WorkerContext: WorkerContext;

    /// Binds input columns for one batch.
    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self::Reader<'b>;

    /// Creates a group value from its first input row.
    fn value(
        reader: &Self::Reader<'_>,
        idx: usize,
        worker_context: &mut Self::WorkerContext,
    ) -> Self;

    /// Folds another row into an existing group.
    #[inline(always)]
    fn update_from_reader(
        self,
        reader: &Self::Reader<'_>,
        idx: usize,
        worker_context: &mut Self::WorkerContext,
        context: &Self::SharedContext,
    ) -> Self {
        self.merge(Self::value(reader, idx, worker_context), context)
    }

    /// Combines two partial group values.
    fn merge(self, other: Self, context: &Self::SharedContext) -> Self;

    /// Returns one slot as an ORDER BY key.
    fn sort_key(&self, slot: usize) -> Self::SortKey;
}

/// Operation that can be specialized for a compile-time slot count.
///
/// `N == 0` selects the runtime fallback. Other values promise exactly `N`
/// aggregation slots.
pub trait ArityBody<R> {
    fn run<const N: usize>(self) -> R;
}

/// Aggregation payload stored for one group, implemented by the
/// entry-resident representation itself:
///
/// ```text
/// Compiled: implements this directly (its stored form is itself)
/// Dynamic:  an unsized cell run, with the slot count in StorageMetadata
/// ```
///
/// The fold methods mutate `self` in place inside a table entry or scatter
/// row. The one path that needs a sized, copyable value that outlives its
/// table (a top-k heap row) goes through [`Owned`](Self::Owned).
pub trait AggregationValue: Send + Sync + 'static {
    /// Sized handle to a value that outlives its table, used only by top-k
    /// heap rows. `Self` for a fixed-size value; an arena pointer for
    /// [`Dynamic`].
    type Owned: Copy + Default + Send + Sync + 'static;
    /// Metadata needed to size and view a stored value.
    type StorageMetadata: Copy + Send + Sync + 'static;
    /// Per-partition scatter storage selected by the value representation.
    type Scatter<KP: PersistedKey>: ScatterRows<KP, Self>;
    /// Bound input readers for one record batch.
    type Reader<'b>;
    /// Shared state used during consume, merge, and output.
    type SharedContext: SharedContext<Worker = Self::WorkerContext>;
    /// Builders for the result's aggregation columns.
    type ColumnBuilder: ValueColumnBuilder<Value = Self, Context = Self::SharedContext>;
    /// Scalar used by pushed-down ORDER BY and LIMIT.
    type SortKey: Ord + Copy;
    /// Per-worker consume state.
    type WorkerContext: WorkerContext;

    /// Binds input columns for one batch.
    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self::Reader<'b>;

    /// Returns storage metadata for this query.
    fn storage_metadata(ctx: &Self::SharedContext) -> Self::StorageMetadata;

    /// Returns storage metadata for a specialized slot count.
    fn metadata_for_arity<const N: usize>() -> Self::StorageMetadata;

    /// Runs a body specialized for common slot counts.
    fn dispatch_arity<R>(metadata: Self::StorageMetadata, body: impl ArityBody<R>) -> R;

    /// The byte size of one stored value.
    fn stored_size(metadata: Self::StorageMetadata) -> usize;

    /// The alignment of one stored value.
    fn stored_align() -> usize;

    /// Views an entry's value bytes as a value reference.
    ///
    /// # Safety
    /// `ptr` must point at [`stored_size`](Self::stored_size) bytes aligned to
    /// [`stored_align`](Self::stored_align), valid for the returned lifetime,
    /// and `metadata` must describe this table's value layout.
    unsafe fn from_entry<'a>(ptr: *const u8, metadata: Self::StorageMetadata) -> &'a Self;

    /// Mutable counterpart of [`from_entry`](Self::from_entry).
    ///
    /// # Safety
    /// As [`from_entry`](Self::from_entry), plus `ptr` must be exclusive for
    /// the returned lifetime.
    unsafe fn from_entry_mut<'a>(ptr: *mut u8, metadata: Self::StorageMetadata) -> &'a mut Self;

    /// Seeds this newly claimed, zeroed entry from one input row.
    fn seed(
        &mut self,
        reader: &Self::Reader<'_>,
        idx: usize,
        worker_context: &mut Self::WorkerContext,
    );

    /// Folds one input row into this existing group.
    fn update(
        &mut self,
        reader: &Self::Reader<'_>,
        idx: usize,
        worker_context: &mut Self::WorkerContext,
        context: &Self::SharedContext,
    );

    /// Merges a source partial into this group.
    fn merge_from(&mut self, source: &Self, context: &Self::SharedContext);

    /// Copies a partial into this newly claimed, zeroed entry.
    fn copy_from(&mut self, source: &Self);

    /// Returns one slot as an ORDER BY key.
    fn sort_key(&self, slot: usize) -> Self::SortKey;

    /// Copies this table value into an owned handle for top-k.
    fn to_owned(
        &self,
        context: &Self::SharedContext,
        worker_context: &mut Option<Self::WorkerContext>,
    ) -> Self::Owned;
}

/// Adapts every fixed-size [`OwnedValue`] to in-place table storage.
impl<T: OwnedValue> AggregationValue for T {
    type Owned = T;
    type StorageMetadata = ();
    type Scatter<KP: PersistedKey> = SizedScatterRows<KP, T>;
    type Reader<'b> = <T as OwnedValue>::Reader<'b>;
    type SharedContext = <T as OwnedValue>::SharedContext;
    type ColumnBuilder = <T as OwnedValue>::ColumnBuilder;
    type SortKey = <T as OwnedValue>::SortKey;
    type WorkerContext = <T as OwnedValue>::WorkerContext;

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self::Reader<'b> {
        <T as OwnedValue>::make_reader(batch, slots)
    }

    fn storage_metadata(_ctx: &Self::SharedContext) {}

    fn metadata_for_arity<const N: usize>() {}

    #[inline(always)]
    fn dispatch_arity<R>(_metadata: (), body: impl ArityBody<R>) -> R {
        // Fixed-size values need no additional arity specialization.
        body.run::<0>()
    }

    fn stored_size(_metadata: ()) -> usize {
        size_of::<T>()
    }

    fn stored_align() -> usize {
        align_of::<T>()
    }

    #[inline(always)]
    unsafe fn from_entry<'a>(ptr: *const u8, _metadata: ()) -> &'a T {
        unsafe { &*(ptr as *const T) }
    }

    #[inline(always)]
    unsafe fn from_entry_mut<'a>(ptr: *mut u8, _metadata: ()) -> &'a mut T {
        unsafe { &mut *(ptr as *mut T) }
    }

    #[inline(always)]
    fn seed(
        &mut self,
        reader: &Self::Reader<'_>,
        idx: usize,
        worker_context: &mut Self::WorkerContext,
    ) {
        *self = <T as OwnedValue>::value(reader, idx, worker_context);
    }

    #[inline(always)]
    fn update(
        &mut self,
        reader: &Self::Reader<'_>,
        idx: usize,
        worker_context: &mut Self::WorkerContext,
        context: &Self::SharedContext,
    ) {
        *self = self.update_from_reader(reader, idx, worker_context, context);
    }

    #[inline(always)]
    fn merge_from(&mut self, source: &T, context: &Self::SharedContext) {
        *self = OwnedValue::merge(*self, *source, context);
    }

    #[inline(always)]
    fn copy_from(&mut self, source: &T) {
        *self = *source;
    }

    #[inline(always)]
    fn sort_key(&self, slot: usize) -> Self::SortKey {
        OwnedValue::sort_key(self, slot)
    }

    #[inline(always)]
    fn to_owned(
        &self,
        _context: &Self::SharedContext,
        _worker_context: &mut Option<Self::WorkerContext>,
    ) -> Self {
        *self
    }
}
