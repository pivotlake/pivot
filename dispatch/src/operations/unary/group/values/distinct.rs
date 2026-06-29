//! [`Distinct`] — the keys-only aggregation value (DISTINCT / count-distinct dedup).
//!
//! A zero-sized value: a hash-table `Entry` shrinks to just hash + key (e.g.
//! 24→16 bytes for an `Int64` key), which is a third less memory to write, zero,
//! and merge per group — and that build cost dominates high-cardinality grouping.
//! Used for the dedup stage of `COUNT(DISTINCT x)` (and, later, `SELECT DISTINCT`):
//! only the *set* of distinct keys matters, so it emits no value columns.

use super::{AggregationSlot, AggregationValue, ValueColumns};
use crate::memory::SlabAllocator;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;

/// Zero-sized keys-only value: distinctness needs no accumulator, so merging two
/// occurrences of the same key is a no-op and the result has no value columns.
#[derive(Clone, Copy, Default)]
pub struct Distinct;

/// The (empty) output columns of a [`Distinct`] value: a keys-only group emits no
/// value columns, so this builds nothing.
pub struct DistinctColumns;

impl AggregationValue for Distinct {
    type Reader<'b> = ();
    type SharedContext = ();
    type Columns = DistinctColumns;
    type SortKey = i64;
    type WorkerContext = ();

    #[inline(always)]
    fn make_reader(_batch: &RecordBatch, _slots: &[AggregationSlot]) {}

    #[inline(always)]
    fn value(_reader: &(), _idx: usize, _wc: &mut ()) -> Self {
        Distinct
    }

    #[inline(always)]
    fn merge(self, _other: Self, _ctx: &()) -> Self {
        // No accumulator — both sides are the same (distinct) key.
        self
    }

    #[inline(always)]
    fn sort_key(&self, _slot: usize) -> i64 {
        // A keys-only group never feeds an ORDER BY <agg> top-k.
        0
    }
}

impl ValueColumns for DistinctColumns {
    type Value = Distinct;
    type Context = ();

    fn with_capacity(_allocator: &mut SlabAllocator, _rows: usize) -> Self {
        DistinctColumns
    }

    #[inline(always)]
    fn push(&mut self, _value: &Distinct) {}

    fn finish(self, _context: &()) -> (Vec<Field>, Vec<ArrayRef>) {
        (Vec::new(), Vec::new())
    }
}
