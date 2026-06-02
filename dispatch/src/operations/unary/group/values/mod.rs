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

mod aggregate;
mod aggregation_row;
mod compiled;

pub use aggregate::{Count, Sum};
pub use aggregation_row::AggregationRowValueExtractor;
pub use compiled::Compiled;

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

    /// Build a reader over `batch` for the configured aggregate `value_slots`.
    fn make_reader<'b>(batch: &'b RecordBatch, value_slots: &[AggregationSlot])
    -> Self::Reader<'b>;

    /// Build the per-row aggregate value at row `idx`.
    fn value(reader: &Self::Reader<'_>, idx: usize) -> Self::Value;

    /// The scalar that an `ORDER BY <slot> DESC LIMIT k` sorts on, pulled from an
    /// otherwise-opaque [`Value`]. Used only when the group feeds a top-k.
    fn sort_key(value: &Self::Value, slot: usize) -> i64;
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
