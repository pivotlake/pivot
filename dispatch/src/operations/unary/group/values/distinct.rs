//! Keys-only GROUP BY value extractor (DISTINCT / count-distinct dedup).
//!
//! A [`ValueExtractor`] that carries no aggregate: the per-entry value is a
//! zero-sized type, so a hash-table `Entry` shrinks to just hash + key (e.g.
//! 24→16 bytes for an `Int64` key, 48→32 for a `u128` pair key). That is one
//! third less memory to write, zero, and merge per group — which matters for
//! high-cardinality grouping where the table build and the page-zeroing of its
//! slabs dominate.
//!
//! Used for the dedup stage of `COUNT(DISTINCT x)` (and, in future, `SELECT
//! DISTINCT`): only the *set* of distinct keys matters, no per-group
//! accumulator. Emits no value columns — the output is the key column(s) alone.

use crate::memory::SlabAllocator;
use crate::operations::unary::group::hashtables::Value;
use crate::operations::unary::group::values::{AggregationSlot, ValueColumns, ValueExtractor};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;

/// Zero-sized per-entry value: distinctness needs no accumulator, so merging two
/// occurrences of the same key is a no-op.
#[derive(Copy, Clone, Default)]
pub struct DistinctValue;

impl Value for DistinctValue {
    #[inline(always)]
    fn merge(self, _v: Self) -> Self {
        self
    }
}

/// A [`ValueExtractor`] that stores nothing per group and emits no value
/// columns — the group's output is its key column(s) only.
pub struct DistinctValueExtractor;

impl ValueExtractor for DistinctValueExtractor {
    type Value = DistinctValue;
    type Reader<'b> = ();
    type Columns = DistinctValueColumns;
    type SortKey = i64;

    #[inline(always)]
    fn make_reader(_batch: &RecordBatch, _value_slots: &[AggregationSlot]) {}

    #[inline(always)]
    fn value(_reader: &(), _idx: usize) -> DistinctValue {
        DistinctValue
    }

    #[inline(always)]
    fn sort_key(_value: &DistinctValue, _slot: usize) -> i64 {
        // A keys-only group never feeds an ORDER BY <agg> top-k.
        0
    }
}

/// The value-side of a keys-only group: contributes no columns to the output.
pub struct DistinctValueColumns;

impl ValueColumns for DistinctValueColumns {
    type Value = DistinctValue;

    fn with_capacity(_allocator: &mut SlabAllocator, _rows: usize) -> Self {
        DistinctValueColumns
    }

    #[inline(always)]
    fn push(&mut self, _value: &DistinctValue) {}

    fn finish(self) -> (Vec<Field>, Vec<ArrayRef>) {
        (Vec::new(), Vec::new())
    }
}
