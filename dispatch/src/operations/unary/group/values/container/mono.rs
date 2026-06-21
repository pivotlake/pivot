//! [`Mono`] — a homogeneous *additive* value: `N` cells of width `A`, every slot a
//! `COUNT` or `SUM`, all folded by `+`.
//!
//! Three things it does that [`Dynamic`](super::Dynamic) doesn't, all of which
//! matter on the high-cardinality two-level `COUNT(DISTINCT)` path (all-additive:
//! dedup counts + re-summed partials), whose merge dominates:
//! - **branch-free merge** — one `+` per cell, no per-slot `match kind`;
//! - **no widen** — [`SortKey`](AggregationValue::SortKey) is `A` itself, so the
//!   cell never sign-extends to `i128`;
//! - **no string arms** — additive-only, so the body carries no `ArenaKey`/arena
//!   machinery.

use super::super::cell::Numeric;
use super::super::{AggregationKind, AggregationSlot, AggregationValue};
use super::dynamic::NumReader;
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;

/// One additive slot's reader: a `COUNT` (no column, contributes `1`) or a `SUM`
/// (a numeric column, contributes the row's value). Built once per batch.
pub enum AddReader<'b> {
    Count,
    Sum(NumReader<'b>),
}

impl<'b> AddReader<'b> {
    fn bind(batch: &'b RecordBatch, slot: &AggregationSlot) -> Self {
        match slot.kind {
            AggregationKind::CountStar | AggregationKind::Count => AddReader::Count,
            AggregationKind::Sum => AddReader::Sum(NumReader::bind(batch, slot.column)),
            other => panic!("Mono is additive-only (COUNT/SUM); got {other:?}"),
        }
    }
    #[inline(always)]
    fn read<A: Numeric>(&self, idx: usize) -> A {
        match self {
            AddReader::Count => A::from(1),
            AddReader::Sum(r) => A::from(r.read(idx)),
        }
    }
}

/// `N` additive cells of width `A` — every slot folded by `+`.
pub struct Mono<const N: usize, A: Numeric = i64> {
    cells: [A; N],
}

impl<const N: usize, A: Numeric> Copy for Mono<N, A> {}
impl<const N: usize, A: Numeric> Clone for Mono<N, A> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<const N: usize, A: Numeric> Default for Mono<N, A> {
    fn default() -> Self {
        Self {
            cells: [A::default(); N],
        }
    }
}

impl<const N: usize, A: Numeric> AggregationValue for Mono<N, A> {
    type Reader<'b> = [AddReader<'b>; N];
    type SharedContext = ();
    type Columns = [SlabColumn<A>; N];
    type SortKey = A;
    type WorkerContext = ();

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> [AddReader<'b>; N] {
        assert_eq!(slots.len(), N, "slot count must match N");
        std::array::from_fn(|s| AddReader::bind(batch, &slots[s]))
    }

    #[inline(always)]
    fn value(reader: &[AddReader<'_>; N], idx: usize, _wc: &mut ()) -> Self {
        let mut cells = [A::default(); N];
        #[allow(clippy::needless_range_loop)]
        for s in 0..N {
            cells[s] = reader[s].read(idx);
        }
        Self { cells }
    }

    #[inline(always)]
    fn merge(self, other: Self, _ctx: &()) -> Self {
        // Branch-free: one `+` per cell, no per-slot dispatch.
        let mut cells = [A::default(); N];
        #[allow(clippy::needless_range_loop)]
        for s in 0..N {
            cells[s] = self.cells[s] + other.cells[s];
        }
        Self { cells }
    }

    #[inline(always)]
    fn sort_key(&self, slot: usize) -> A {
        self.cells[slot]
    }

    fn new_columns(allocator: &mut SlabAllocator, rows: usize) -> [SlabColumn<A>; N] {
        std::array::from_fn(|_| SlabColumn::with_capacity(allocator, rows))
    }

    #[inline(always)]
    fn push_to(&self, cols: &mut [SlabColumn<A>; N]) {
        for (c, cell) in cols.iter_mut().zip(self.cells.iter()) {
            c.push(*cell);
        }
    }

    fn finish_columns(cols: [SlabColumn<A>; N], _ctx: &()) -> (Vec<Field>, Vec<ArrayRef>) {
        let fields = Vec::with_capacity(N);
        let arrays = Vec::with_capacity(N);
        for _col in cols {
            // let (f, a) = <A as NumericArrow>::finish(&format!("v{s}"), col);
            todo!()
            // fields.push(f);
            // arrays.push(a);
        }
        (fields, arrays)
    }
}
