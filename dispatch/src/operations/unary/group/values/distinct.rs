//! [`Distinct`] — the keys-only aggregation value (DISTINCT / count-distinct dedup).
//!
//! A zero-sized value: a hash-table `Entry` shrinks to just hash + key (e.g.
//! 24→16 bytes for an `Int64` key), which is a third less memory to write, zero,
//! and merge per group — and that build cost dominates high-cardinality grouping.
//! Used for the dedup stage of `COUNT(DISTINCT x)` (and, later, `SELECT DISTINCT`):
//! only the *set* of distinct keys matters, so it emits no value columns.

use super::{AggregationSlot, AggregationValue, ArityBody, ValueColumnBuilder};
use crate::memory::SlabAllocator;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;

/// Zero-sized keys-only value: distinctness needs no accumulator, so merging two
/// occurrences of the same key is a no-op and the result has no value columns.
#[derive(Clone, Copy, Default)]
pub struct Distinct;

impl AggregationValue for Distinct {
    type Owned = Self;
    type StorageMetadata = ();
    type Reader<'b> = ();
    type SharedContext = ();
    type ColumnBuilder = DistinctColumnBuilder;
    type SortKey = i64;
    type WorkerContext = ();

    #[inline(always)]
    fn make_reader(_batch: &RecordBatch, _slots: &[AggregationSlot]) {}

    fn storage_metadata(_ctx: &()) {}

    fn metadata_for_arity<const N: usize>() {}

    #[inline(always)]
    fn dispatch_arity<Ret>(_metadata: (), body: impl ArityBody<Ret>) -> Ret {
        body.run::<0>()
    }

    fn stored_size(_metadata: ()) -> usize {
        0
    }

    fn stored_align() -> usize {
        1
    }

    #[inline(always)]
    unsafe fn from_entry<'a>(ptr: *const u8, _metadata: ()) -> &'a Self {
        unsafe { &*(ptr as *const Self) }
    }

    #[inline(always)]
    unsafe fn from_entry_mut<'a>(ptr: *mut u8, _metadata: ()) -> &'a mut Self {
        unsafe { &mut *(ptr as *mut Self) }
    }

    #[inline(always)]
    fn seed(&mut self, _reader: &(), _idx: usize, _wc: &mut ()) {}

    #[inline(always)]
    fn update(&mut self, _reader: &(), _idx: usize, _wc: &mut (), _ctx: &()) {}

    #[inline(always)]
    fn merge_from(&mut self, _source: &Self, _ctx: &()) {
        // No accumulator: both sides are the same (distinct) key.
    }

    #[inline(always)]
    fn copy_from(&mut self, _source: &Self) {}

    #[inline(always)]
    fn sort_key(&self, _slot: usize) -> i64 {
        // A keys-only group never feeds an ORDER BY <agg> top-k.
        0
    }

    #[inline(always)]
    fn to_owned(&self, _ctx: &(), _wc: &mut Option<()>) -> Self {
        Distinct
    }
}

/// Empty output builder for a keys-only value.
pub struct DistinctColumnBuilder;

impl ValueColumnBuilder for DistinctColumnBuilder {
    type Value = Distinct;
    type Context = ();

    fn with_capacity(_allocator: &mut SlabAllocator, _rows: usize, _context: &()) -> Self {
        DistinctColumnBuilder
    }

    #[inline(always)]
    fn push(&mut self, _value: &Distinct) {}

    #[inline(always)]
    fn push_stored(&mut self, _stored: &Distinct) {}

    fn finish(self, _context: &()) -> (Vec<Field>, Vec<ArrayRef>) {
        (Vec::new(), Vec::new())
    }
}
