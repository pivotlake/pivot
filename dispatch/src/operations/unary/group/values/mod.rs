//! Value extraction strategies for GROUP BY aggregation.
//!
//! A [`ValueExtractor`] is the value-side counterpart to a
//! [`KeyExtractor`](super::keys::KeyExtractor): it reads the per-row aggregate
//! value(s) from an input batch and emits the trailing value columns of the
//! result. Splitting it from the key extractor lets any key shape pair with any
//! aggregate shape (e.g. `COUNT(*)` or a multi-slot `SUM`) without an
//! `O(keys × values)` explosion of monolithic extractors.

use crate::memory::SlabAllocator;
use crate::operations::unary::group::hashtables::Value;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;

mod accumulator;
mod aggregate;
mod aggregation_row;
mod compiled;
mod distinct;

pub use accumulator::Accumulator;
pub use aggregate::{Aggregate, Count, Sum};
pub use aggregation_row::DynamicValueExtractor;
pub use compiled::Compiled;
pub use distinct::DistinctValueExtractor;

/// Which per-group aggregate a value slot accumulates during the consume phase.
///
/// `Avg` is not represented here: `AVG(c)` is lowered to `sum(c)` + `count(c)`
/// with a divide projection, so a grouped average arrives as a `Sum` slot plus a
/// `Count` slot and the division happens in the downstream projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregationKind {
    /// `COUNT(*)` — +1 per row, ignores the column.
    CountStar,
    /// `COUNT(col)` — +1 per non-null row.
    Count,
    /// `SUM(col)` — += the (widened) column value.
    Sum,
    /// `MIN(col)` over an integer column — keep the smallest (widened) value.
    Min,
    /// `MAX(col)` over an integer column — keep the largest (widened) value.
    Max,
}

impl AggregationKind {
    /// Combine two accumulators of this kind. The op is associative, so the same
    /// function folds a row's contribution into a group *and* merges two partial
    /// groups: additive kinds add, the extremes take the min/max. This is the one
    /// place the per-kind merge logic lives — `Compiled` calls it with a `const`
    /// kind (folded away at compile time), the dynamic path with a runtime kind.
    ///
    /// Generic over the accumulator width so a wide (`i128`) sum combines at full
    /// width; both `i64` and `i128` are `Copy + Ord + AddAssign` (the
    /// [`Accumulator`](crate::operations::unary::group::values::Accumulator) bound).
    #[inline(always)]
    pub fn combine<A: Copy + Ord + std::ops::AddAssign>(self, mut a: A, b: A) -> A {
        match self {
            AggregationKind::CountStar | AggregationKind::Count | AggregationKind::Sum => {
                a += b;
                a
            }
            AggregationKind::Min => a.min(b),
            AggregationKind::Max => a.max(b),
        }
    }

    /// The neutral element for this kind: combining it with any value `v` yields
    /// `v`. Additive kinds start at `0`; a running `MIN` starts at the width's
    /// maximum and a running `MAX` at its minimum. A fold seeds an accumulator
    /// with this and then [`combine`](Self::combine)s each contribution, so the
    /// first real value replaces the identity. (The grouped hash-table path never
    /// needs it — a new entry stores the first row's value directly — but the
    /// global fold over batches and the cross-worker partial merge both do.)
    #[inline(always)]
    pub fn identity<A: Accumulator>(self) -> A {
        match self {
            AggregationKind::CountStar | AggregationKind::Count | AggregationKind::Sum => {
                A::default()
            }
            AggregationKind::Min => A::MAX,
            AggregationKind::Max => A::MIN,
        }
    }
}

/// One aggregate output slot: which aggregate, over which input column.
#[derive(Clone, Copy, Debug)]
pub struct AggregationSlot {
    pub kind: AggregationKind,
    pub column: usize,
}

impl AggregationSlot {
    pub fn new(kind: AggregationKind, column: usize) -> Self {
        Self { kind, column }
    }
}

/// Reads the per-row aggregate value for a GROUP BY and emits the value columns.
pub trait ValueExtractor: Send + 'static {
    /// The aggregation value stored alongside each key in the hash table.
    type Value: Value + Send;
    /// Per-batch reader holding downcast value-column accessors.
    type Reader<'b>;
    /// Accumulates per-group values into the result's value column(s).
    type Columns: ValueColumns<Value = Self::Value>;
    /// The scalar an `ORDER BY <slot> DESC LIMIT k` sorts on — the slot's own
    /// accumulator type, so a wide (`i128`) sum is compared at full width with no
    /// lossy narrowing.
    type SortKey: Ord + Copy;

    /// Build a reader over `batch` for the configured aggregate `value_slots`.
    fn make_reader<'b>(batch: &'b RecordBatch, value_slots: &[AggregationSlot])
    -> Self::Reader<'b>;

    /// Build the per-row aggregate value at row `idx`.
    fn value(reader: &Self::Reader<'_>, idx: usize) -> Self::Value;

    /// Combine two group accumulators. The hash table calls this both to fold a
    /// row's [`value`](Self::value) into an existing entry during consume and to
    /// combine two partials in the partition merge (the op is associative). It is
    /// the kind-aware replacement for the old blanket additive merge: additive
    /// slots add, MIN/MAX slots take the extreme. `slots` carries the per-slot
    /// kinds for the runtime path; [`Compiled`] ignores it (its ops are static).
    fn merge(a: Self::Value, b: Self::Value, slots: &[AggregationSlot]) -> Self::Value;

    /// The value an `ORDER BY <slot> DESC LIMIT k` sorts on, pulled from an
    /// otherwise-opaque [`Value`]. Used only when the group feeds a top-k.
    fn sort_key(value: &Self::Value, slot: usize) -> Self::SortKey;
}

/// Builds the trailing value column(s) of a GROUP BY result, one group at a time.
///
/// The value-side analog of
/// [`KeyColumns`](super::keys::KeyColumns): the output combinator
/// pushes each surviving group's value, then `finish` materialises the Arrow
/// columns and their fields.
pub trait ValueColumns {
    type Value;

    /// Allocate column builders over engine memory, sized for `rows` (one
    /// output chunk; must fit a single 2MB slab — callers chunk to
    /// `RECORD_BATCH_SIZE`).
    fn with_capacity(allocator: &mut SlabAllocator, rows: usize) -> Self;
    fn push(&mut self, value: &Self::Value);
    fn finish(self) -> (Vec<Field>, Vec<ArrayRef>);
}
