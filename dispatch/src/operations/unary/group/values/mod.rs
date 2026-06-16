//! Aggregation values for GROUP BY.
//!
//! An [`AggregationValue`] is the value-side counterpart to a
//! [`KeyExtractor`](super::keys::KeyExtractor): it *is* the per-group payload
//! stored in the hash table, and it knows how to be read from an input batch,
//! folded with another row or partial, and emitted as the result's trailing value
//! column(s). Any key shape pairs with any aggregation value.
//!
//! There are three shapes, chosen by the planner:
//! - [`Mono<F, N, A>`](Mono) — every slot folds the same way (`F` = [`Add`]/
//!   [`Min`]/[`Max`]); the common all-`SUM`/`COUNT` query is `Mono<Add>`. Its
//!   [`MergeConfig`](AggregationValue::MergeConfig) is `()`, so the fold is
//!   branch-free with nothing to carry.
//! - [`DynamicMixed<N, A>`](DynamicMixed) — a heterogeneous signature (e.g.
//!   `COUNT(*), MIN(x)`); folds per slot on the runtime kind, carried in its
//!   `MergeConfig`.
//! - [`CompiledMixed<Ops>`](CompiledMixed) — a fixed signature monomorphised over
//!   a tuple of [`Aggregate`] ops (straight-line, no per-row dispatch).

use crate::memory::SlabAllocator;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;

mod aggregate;
mod cell;
mod columns;
mod compiled;
mod distinct;
mod dynamic;
mod fold;
mod mono;
mod reader;
mod row;

pub use aggregate::{Aggregate, Count, Sum};
pub use cell::Cell;
pub use compiled::CompiledMixed;
pub use distinct::Distinct;
pub use dynamic::DynamicMixed;
pub use fold::{Add, Max, Min};
pub use mono::Mono;

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
    /// groups: additive kinds add, the extremes take the min/max. Used by
    /// [`DynamicMixed`] (per-slot) and the global aggregate operator.
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
    /// maximum and a running `MAX` at its minimum. The grouped hash-table path
    /// never needs it (a new entry stores the first row's value directly), but the
    /// global fold over batches and the cross-worker partial merge both do.
    #[inline(always)]
    pub fn identity<A: Cell>(self) -> A {
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

/// The per-group value stored in a GROUP BY hash table — read from input rows,
/// folded with other rows and partials, and emitted as the result's value columns.
///
/// Reading and folding are separate so a string extreme can persist lazily: a new
/// group materialises with [`value`](Self::value), an existing group folds the
/// next row with [`update_from_reader`](Self::update_from_reader) (which can skip
/// persisting a row that doesn't win), and two finished partials combine with
/// [`merge`](Self::merge) (no new materialisation). For integers all three are the
/// same elementwise fold, so `update_from_reader` defaults to merging the row in.
pub trait AggregationValue: Copy + Default + Send + Sync + 'static {
    /// Per-batch reader holding the downcast value columns.
    type Reader<'b>;
    /// Runtime data [`merge`](Self::merge) needs that the type can't carry (the
    /// slot kinds for [`DynamicMixed`]; `()` otherwise). Built once at `Group`
    /// creation, like [`KeyExtractor::Config`](super::keys::KeyExtractor::Config).
    type MergeConfig: Clone + Send + Sync + 'static;
    /// The result value columns under construction.
    type Columns;
    /// The scalar an `ORDER BY <slot> DESC LIMIT k` sorts on — the slot's own
    /// width, so a wide (`i128`) sum compares at full precision.
    type SortKey: Ord + Copy;

    /// Build the [`MergeConfig`](Self::MergeConfig) for these `slots`, once, at
    /// `Group` creation. Homogeneous/compiled values need nothing (`()`); the
    /// dynamic one keeps the slot kinds.
    fn merge_config(slots: &[AggregationSlot]) -> Self::MergeConfig;

    /// Bind `batch`'s value columns for the configured `slots`.
    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self::Reader<'b>;

    /// Materialise a brand-new group from row `idx` — the consume path's new-key
    /// case, and the radix scatter.
    fn value(reader: &Self::Reader<'_>, idx: usize) -> Self;

    /// Fold row `idx` into this (existing) group. Defaults to merging the row's
    /// [`value`](Self::value) in; a string extreme overrides it to read the cell
    /// lazily and persist only when it beats the current extreme.
    #[inline(always)]
    fn update_from_reader(
        self,
        reader: &Self::Reader<'_>,
        idx: usize,
        cfg: &Self::MergeConfig,
    ) -> Self {
        self.merge(Self::value(reader, idx), cfg)
    }

    /// Combine two partial group values — the partition merge and the radix fold.
    fn merge(self, other: Self, cfg: &Self::MergeConfig) -> Self;

    /// This group's value for slot `slot`, as an `ORDER BY` sort key.
    fn sort_key(&self, slot: usize) -> Self::SortKey;

    /// Allocate the result value columns over engine memory, sized for `rows`
    /// (one output chunk; must fit a single 2 MB slab).
    fn new_columns(allocator: &mut SlabAllocator, rows: usize) -> Self::Columns;
    /// Append this group to the columns.
    fn push_to(&self, cols: &mut Self::Columns);
    /// Materialise the columns and their fields.
    fn finish_columns(cols: Self::Columns) -> (Vec<Field>, Vec<ArrayRef>);
}
