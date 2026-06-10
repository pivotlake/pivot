//! Value extraction strategies for GROUP BY aggregation.
//!
//! A [`ValueExtractor`] is the value-side counterpart to a
//! [`KeyExtractor`](super::keys::KeyExtractor): it reads the per-row aggregate
//! value(s) from an input batch and emits the trailing value columns of the
//! result. Splitting it from the key extractor lets any key shape pair with any
//! aggregate shape (e.g. `COUNT(*)` or a multi-slot `SUM`) without an
//! `O(keys × values)` explosion of monolithic extractors.

use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::Value;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;

mod accumulator;
mod aggregate;
mod aggregation_row;
mod compiled;
mod distinct;
mod mixed;
mod mixed_compiled;

pub use accumulator::Accumulator;
pub use aggregate::{Aggregate, Count, Sum};
pub use aggregation_row::AggregationRowValueExtractor;
pub use compiled::Compiled;
pub use distinct::DistinctValueExtractor;
pub use mixed::MixedRowValueExtractor;
pub use mixed_compiled::{
    CompiledMixed, CountOp, MaxIntOp, MaxStrOp, MinIntOp, MinStrOp, MixedOp, SumOp,
};

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
    /// `MIN(col)` over an integer column.
    Min,
    /// `MAX(col)` over an integer column.
    Max,
    /// `MIN(col)` over a string column — arena-backed; only the mixed-slot
    /// extractor supports it.
    MinStr,
    /// `MAX(col)` over a string column.
    MaxStr,
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
///
/// Beyond the basic per-row [`value`](Self::value), extractors whose aggregates
/// carry side effects or need context get three richer hooks, each with a
/// default that composes `value` + [`Value::merge`]:
/// - [`init`](Self::init) — build a fresh entry's value, with arena access
///   (e.g. a string `MIN` persists its first candidate);
/// - [`fold`](Self::fold) — fold a row into an existing entry, so the side
///   effect only happens when the row improves the entry;
/// - [`combine`](Self::combine) — merge two already-persisted values in the
///   partition merge, with the shared arena for resolving persisted strings.
pub trait ValueExtractor: Send + 'static {
    /// Whether per-row values may be persisted into radix scatter buffers at
    /// high cardinality. `false` for aggregates whose value building has side
    /// effects per row (e.g. a string `MIN` writing candidates to the arena) —
    /// those must stay in-place, where folding skips non-improving rows.
    const SUPPORTS_RADIX: bool = true;

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

    /// Build a fresh entry's value at row `idx`, with arena access.
    #[inline(always)]
    fn init(reader: &Self::Reader<'_>, idx: usize, arena: &mut WorkerArena) -> Self::Value {
        let _ = arena;
        Self::value(reader, idx)
    }

    /// Fold row `idx` into an existing entry's `current` value.
    #[inline(always)]
    fn fold(
        current: Self::Value,
        reader: &Self::Reader<'_>,
        idx: usize,
        arena: &mut WorkerArena,
    ) -> Self::Value {
        let _ = arena;
        current.merge(Self::value(reader, idx))
    }

    /// Combine two already-built values during the partition merge. `slots`
    /// carries the query's slot kinds for extractors whose value doesn't
    /// (e.g. the mixed-slot extractor's min-vs-sum distinction).
    #[inline(always)]
    fn combine(
        current: Self::Value,
        incoming: Self::Value,
        arena: &SharedArena,
        slots: &[AggregationSlot],
    ) -> Self::Value {
        let _ = (arena, slots);
        current.merge(incoming)
    }

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
    fn with_capacity(
        allocator: &mut SlabAllocator,
        rows: usize,
        value_slots: &[AggregationSlot],
    ) -> Self;
    fn push(&mut self, value: &Self::Value);
    /// Materialise the value columns. Takes the shared arena so arena-backed
    /// values (string `MIN`/`MAX`) can emit zero-copy views into its buffers.
    fn finish(self, arena: &std::sync::Arc<SharedArena>) -> (Vec<Field>, Vec<ArrayRef>);
}
