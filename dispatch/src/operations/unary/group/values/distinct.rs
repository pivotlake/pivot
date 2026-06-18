//! [`Distinct`] — the keys-only aggregation value (DISTINCT / count-distinct dedup).
//!
//! A zero-sized value: a hash-table `Entry` shrinks to just hash + key (e.g.
//! 24→16 bytes for an `Int64` key), which is a third less memory to write, zero,
//! and merge per group — and that build cost dominates high-cardinality grouping.
//! Used for the dedup stage of `COUNT(DISTINCT x)` (and, later, `SELECT DISTINCT`):
//! only the *set* of distinct keys matters, so it emits no value columns.

use super::{AggregationSlot, AggregationValue};
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::sync::Arc;

/// Zero-sized keys-only value: distinctness needs no accumulator, so merging two
/// occurrences of the same key is a no-op and the result has no value columns.
#[derive(Clone, Copy, Default)]
pub struct Distinct;

impl AggregationValue for Distinct {
    type Reader<'b> = ();
    type MergeConfig = ();
    type Columns = ();
    type SortKey = i64;

    #[inline(always)]
    fn merge_config(_slots: &[AggregationSlot], _arena: &Arc<SharedArena>) {}

    #[inline(always)]
    fn make_reader(_batch: &RecordBatch, _slots: &[AggregationSlot]) {}

    #[inline(always)]
    fn value(_reader: &(), _idx: usize, _arena: &mut WorkerArena) -> Self {
        Distinct
    }

    #[inline(always)]
    fn merge(self, _other: Self, _cfg: &()) -> Self {
        // No accumulator — both sides are the same (distinct) key.
        self
    }

    #[inline(always)]
    fn sort_key(&self, _slot: usize) -> i64 {
        // A keys-only group never feeds an ORDER BY <agg> top-k.
        0
    }

    fn new_columns(_allocator: &mut SlabAllocator, _rows: usize) {}

    #[inline(always)]
    fn push_to(&self, _cols: &mut ()) {}

    fn finish_columns(
        _cols: (),
        _arena: &Arc<SharedArena>,
        _cfg: &(),
    ) -> (Vec<Field>, Vec<ArrayRef>) {
        (Vec::new(), Vec::new())
    }
}
