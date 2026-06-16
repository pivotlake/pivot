//! Aggregation values for GROUP BY.
//!
//! An [`AggregationValue`] is the value-side counterpart to a
//! [`KeyExtractor`](super::keys::KeyExtractor): it *is* the per-group payload
//! stored in the hash table — read from input rows, folded with other rows and
//! partials, and emitted as the result's trailing value column(s). Any key shape
//! pairs with any aggregation value.
//!
//! It is built from two orthogonal axes and three containers:
//!
//! - **[`read`]** — what a slot pulls from a row ([`One`](read::One) /
//!   [`Col<T>`](read::Col) / [`Str`](read::Str)).
//! - **[`fold`]** — how cells combine and render ([`Add`](fold::Add) /
//!   [`Min`](fold::Min) / [`Max`](fold::Max) / [`StrMin`](fold::StrMin) /
//!   [`StrMax`](fold::StrMax)).
//! - an **[`op`] atom** = a read paired with a fold ([`Op<R, F, A>`](op::Op);
//!   aliases [`Count`](op::Count), [`Sum<T>`](op::Sum), …).
//! - **containers** ([`Mono`](container::Mono) / [`Compiled`](container::Compiled)
//!   / [`Dynamic`](container::Dynamic)) — three ways to assemble `N` slots.

use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::sync::Arc;

pub mod cell;
pub mod container;
pub mod distinct;
pub mod fold;
pub mod op;
pub mod read;

pub use cell::{Cell, NumericCell};
pub use container::{Compiled, Dynamic, Mono, OpTuple};
pub use distinct::Distinct;
pub use fold::{Add, Max, Min, StrMax, StrMin};
pub use op::{Aggregate, Op};
// Op atoms, suffixed where they would clash with the same-named fold (the planner
// uses the folds for `Mono`, the atoms for `Compiled`).
pub use op::{
    Count, Max as MaxOp, Min as MinOp, StrMax as StrMaxOp, StrMin as StrMinOp, Sum, WideSum,
};

/// Which per-group aggregate a value slot computes during consume — a pure
/// descriptor the planner attaches to each slot. It tells [`SlotReader`](read::SlotReader)
/// what to read (a `COUNT` reads no column; everything else reads its column,
/// numeric or string by the column's type) and the numeric [`Dynamic`](container::Dynamic)
/// fallback how to fold.
///
/// `Avg` is not represented: `AVG(c)` is lowered to `sum(c)` + `count(c)` with a
/// divide projection, so a grouped average arrives as a `Sum` slot plus a `Count`
/// slot and the division happens downstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregationKind {
    /// `COUNT(*)` — +1 per row, ignores the column.
    CountStar,
    /// `COUNT(col)` — +1 per non-null row.
    Count,
    /// `SUM(col)`.
    Sum,
    /// `MIN(col)`.
    Min,
    /// `MAX(col)`.
    Max,
}

impl AggregationKind {
    /// Combine two numeric cells of this kind. Associative, so it folds a row's
    /// contribution into a group *and* merges two partial groups: additive kinds
    /// add, the extremes take the min/max. Used by [`Dynamic`](container::Dynamic)
    /// (per slot) and the global aggregate operator.
    #[inline(always)]
    pub fn combine<A: NumericCell>(self, acc: A, incoming: A) -> A {
        match self {
            AggregationKind::CountStar | AggregationKind::Count | AggregationKind::Sum => {
                acc.add(incoming)
            }
            AggregationKind::Min => acc.min(incoming),
            AggregationKind::Max => acc.max(incoming),
        }
    }

    /// The neutral seed for this kind: additive kinds start at `0`, a running
    /// `MIN` at the width's `MAX`, a running `MAX` at its `MIN`. Only the global
    /// aggregate needs it (it folds from nothing); the grouped path stores a new
    /// group's first row directly.
    #[inline(always)]
    pub fn identity<A: NumericCell>(self) -> A {
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
/// [`merge`](Self::merge) (no new materialisation). For values whose fold is the
/// same elementwise op for rows and partials, `update_from_reader` defaults to
/// merging the row in.
pub trait AggregationValue: Copy + Default + Send + Sync + 'static {
    /// Per-batch reader holding the downcast value columns.
    type Reader<'b>;
    /// Runtime data [`merge`](Self::merge) needs that the type can't carry (slot
    /// kinds for [`Dynamic`](container::Dynamic); the value arena for a string
    /// extreme; `()` otherwise). Built once at `Group` creation, like
    /// [`KeyExtractor::Config`](super::keys::KeyExtractor::Config).
    type MergeConfig: Clone + Send + Sync + 'static;
    /// The result value columns under construction.
    type Columns;
    /// The scalar an `ORDER BY <slot> DESC LIMIT k` sorts on — widened to `i128`
    /// so a wide sum compares at full precision.
    type SortKey: Ord + Copy;

    /// Build the [`MergeConfig`](Self::MergeConfig) for these `slots`, once, at
    /// `Group` creation. `arena` is the *value* arena (a string extreme keeps it
    /// to resolve `ArenaKey`s during the partition merge).
    fn merge_config(slots: &[AggregationSlot], arena: &Arc<SharedArena>) -> Self::MergeConfig;

    /// Bind `batch`'s value columns for the configured `slots`.
    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self::Reader<'b>;

    /// Materialise a brand-new group from row `idx` — the consume path's new-key
    /// case, and the radix scatter. `arena` is the value arena a string extreme
    /// persists its winning string into; numeric cells ignore it.
    fn value(reader: &Self::Reader<'_>, idx: usize, arena: &mut WorkerArena) -> Self;

    /// Fold row `idx` into this (existing) group. Defaults to merging the row's
    /// [`value`](Self::value) in; a string extreme overrides it to compare against
    /// the current extreme (resolved via `arena`) and persist only when it wins.
    #[inline(always)]
    fn update_from_reader(
        self,
        reader: &Self::Reader<'_>,
        idx: usize,
        arena: &mut WorkerArena,
        cfg: &Self::MergeConfig,
    ) -> Self {
        self.merge(Self::value(reader, idx, arena), cfg)
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
    /// Materialise the columns and their fields. `arena` backs the zero-copy
    /// `StringView` output of a string extreme; numeric columns ignore it.
    fn finish_columns(cols: Self::Columns, arena: &Arc<SharedArena>)
    -> (Vec<Field>, Vec<ArrayRef>);
}
