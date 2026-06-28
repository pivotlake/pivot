//! Aggregation values for GROUP BY.
//!
//! An [`AggregationValue`] is the value-side counterpart to a
//! [`KeyExtractor`](super::keys::KeyExtractor): it *is* the per-group payload
//! stored in the hash table — read from input rows, folded with other rows and
//! partials, and emitted as the result's trailing value column(s). Any key shape
//! pairs with any aggregation value.
//!
//! It is built from one trait and two containers:
//!
//! - an **`Aggregation`** op — fully typed to its own input array and cell
//!   ([`Count`], [`Sum<T>`](Sum), [`Min<T>`](Min), [`Max<T>`](Max), [`StrMin`],
//!   [`StrMax`]).
//! - **containers** — [`Compiled`] (a fixed *numeric* tuple
//!   of ops, branch-free) and [`Dynamic`] (a runtime
//!   signature folded per slot, generic over the width — the path for any string
//!   extreme).

use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field};
use std::sync::Arc;

pub mod cell;
pub mod container;
pub mod distinct;
pub mod fold;
pub mod read;

pub use cell::{Cell, Numeric};
pub use container::{Compiled, CountSlot, Dynamic, MaxSlot, MinSlot, OpTuple, SumSlot};
pub use distinct::Distinct;
pub use fold::{Count, Fold, Max, Min, StrMax, StrMin, Sum, WideSum};
pub use read::{IntRead, NoRead, Read, StrRead};

/// Which per-group aggregate a value slot computes during consume — a pure
/// descriptor the planner attaches to each slot. It tells the numeric
/// [`Dynamic`] fallback what to read (a `COUNT` reads no
/// column; everything else reads its column) and how to fold.
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
    /// `MIN(col)` over an integer column.
    Min,
    /// `MAX(col)` over an integer column.
    Max,
    /// `MIN(col)` over a string (`Utf8`) column — its cell is an `ArenaKey` and
    /// its fold ([`StrMin`]) compares the raw bytes through the value arena, not
    /// the numeric path.
    StrMin,
    /// `MAX(col)` over a string (`Utf8`) column.
    StrMax,
}

impl AggregationKind {
    /// Whether this slot is a string extreme (`MIN`/`MAX` over a `Utf8` column),
    /// whose cell is an [`ArenaKey`](super::ArenaKey). A value signature with any
    /// such slot must use 128-bit (`i128`) cells and stays off the radix scatter
    /// path (which would eagerly persist every row's string, winner or not).
    #[inline(always)]
    pub fn is_string_extreme(self) -> bool {
        matches!(self, AggregationKind::StrMin | AggregationKind::StrMax)
    }
}

/// One aggregate output slot: which aggregate, over which input column, and the
/// Arrow type its output column is declared as.
///
/// The accumulator's *storage* width (`i64` / `i128`) is an execution detail
/// independent of this declared type: a `COUNT` may sit in an `i128` cell (forced
/// by a `SUM` sharing a [`Dynamic`] cell array) yet is always a `BIGINT`, and a
/// narrow `SUM` accumulates in `i64` yet is always a `HUGEINT`. The slot carries
/// the type the result column must have, so the output phase renders the
/// accumulator at its storage width then casts each column to its `output_type`
/// (a no-op when they already match).
#[derive(Clone, Debug)]
pub struct AggregationSlot {
    pub kind: AggregationKind,
    pub column: usize,
    pub output_type: DataType,
}

impl AggregationSlot {
    /// A slot for `kind` over input `column`, whose result column is declared as
    /// `output_type` (the planner passes DuckDB's result type for the call).
    pub fn new(kind: AggregationKind, column: usize, output_type: DataType) -> Self {
        Self {
            kind,
            column,
            output_type,
        }
    }
}

/// Cast a rendered value column to its slot's declared `output_type`, rebuilding
/// the field to match. A no-op (a cheap `Arc` clone) when the rendered type
/// already equals the declared one, so a slot whose accumulator width is its
/// output type pays nothing; a `COUNT` rendered from an `i128` cell narrows to
/// `Int64` here, a narrow `SUM` widens to `Decimal128`.
pub(crate) fn cast_value_column(
    field: Field,
    column: ArrayRef,
    output_type: &DataType,
) -> (Field, ArrayRef) {
    if field.data_type() == output_type {
        return (field, column);
    }
    let casted = arrow::compute::cast(&column, output_type).expect("aggregate output column cast");
    let field = Field::new(field.name(), output_type.clone(), field.is_nullable());
    (field, casted)
}

/// The shared, read-side context the merge + output phase resolves through — the
/// counterpart to a value's per-worker [`WorkerContext`]. Built once from the
/// slots and the value arena (cloned across workers and into the merge jobs).
/// `()` for a numeric value (it stores nothing); a string-capable value carries
/// the slot layout and the `Arc<SharedArena>` its keys resolve through.
pub trait SharedContext: Clone + Send + Sync + 'static {
    /// The per-worker write side this context spawns for the consume phase.
    type Worker: WorkerContext;
    /// Build the context for `slots` over the value `arena`.
    fn build(slots: &[AggregationSlot], arena: &Arc<SharedArena>) -> Self;
    /// Spawn a fresh per-worker write context (called once per worker).
    fn worker(&self) -> Self::Worker;
}

/// The per-worker, exclusive write side of a value's string storage during
/// consume. `()` for a numeric value; a [`WorkerArena`] for a string extreme.
pub trait WorkerContext {
    /// Hand any active arena buffer back to the shared arena at end of consume.
    fn flush(self);
}

impl SharedContext for () {
    type Worker = ();
    fn build(_slots: &[AggregationSlot], _arena: &Arc<SharedArena>) {}
    fn worker(&self) {}
}
impl WorkerContext for () {
    fn flush(self) {}
}

impl SharedContext for (Arc<[AggregationSlot]>, Arc<SharedArena>) {
    type Worker = WorkerArena;
    fn build(slots: &[AggregationSlot], arena: &Arc<SharedArena>) -> Self {
        (Arc::from(slots), arena.clone())
    }
    fn worker(&self) -> WorkerArena {
        WorkerArena::new(self.1.clone())
    }
}
impl WorkerContext for WorkerArena {
    fn flush(self) {
        WorkerArena::flush(self)
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
    /// The shared, read-side context [`merge`](Self::merge)/[`finish_columns`](Self::finish_columns)
    /// resolve through (slot kinds for [`Dynamic`] + the value
    /// arena for a string extreme; `()` otherwise). It builds the per-worker
    /// [`WorkerContext`](Self::WorkerContext); see [`SharedContext`].
    type SharedContext: SharedContext<Worker = Self::WorkerContext>;
    /// The result value columns under construction.
    type Columns;
    /// The scalar an `ORDER BY <slot> DESC LIMIT k` sorts on — widened to `i128`
    /// so a wide sum compares at full precision.
    type SortKey: Ord + Copy;
    /// The per-worker write state consume folds into — `()` for an all-numeric
    /// signature (so consume threads `&mut ()`, free: a `()` reference can't alias
    /// the table the probe loop mutates), a real [`WorkerArena`] for a string
    /// extreme. Spawned from [`SharedContext`](Self::SharedContext) per worker.
    type WorkerContext: WorkerContext;

    /// Bind `batch`'s value columns for the configured `slots`.
    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self::Reader<'b>;

    /// Materialise a brand-new group from row `idx` — the consume path's new-key
    /// case, and the radix scatter. `wc` is the per-worker write state a string
    /// extreme persists its winning string into; numeric cells ignore it.
    fn value(reader: &Self::Reader<'_>, idx: usize, wc: &mut Self::WorkerContext) -> Self;

    /// Fold row `idx` into this (existing) group. Defaults to merging the row's
    /// [`value`](Self::value) in; a string extreme overrides it to compare against
    /// the current extreme (resolved via `ctx`) and persist only when it wins.
    #[inline(always)]
    fn update_from_reader(
        self,
        reader: &Self::Reader<'_>,
        idx: usize,
        wc: &mut Self::WorkerContext,
        ctx: &Self::SharedContext,
    ) -> Self {
        self.merge(Self::value(reader, idx, wc), ctx)
    }

    /// Combine two partial group values — the partition merge and the radix fold.
    fn merge(self, other: Self, ctx: &Self::SharedContext) -> Self;

    /// This group's value for slot `slot`, as an `ORDER BY` sort key.
    fn sort_key(&self, slot: usize) -> Self::SortKey;

    /// Allocate the result value columns over engine memory, sized for `rows`
    /// (one output chunk; must fit a single 2 MB slab).
    fn new_columns(allocator: &mut SlabAllocator, rows: usize) -> Self::Columns;
    /// Append this group to the columns.
    fn push_to(&self, cols: &mut Self::Columns);
    /// Materialise the columns and their fields. `ctx` backs the zero-copy
    /// `StringView` output of a string extreme (numeric columns ignore it) and
    /// carries the per-slot descriptor a runtime value ([`Dynamic`])
    /// needs to pick each slot's output type.
    fn finish_columns(
        cols: Self::Columns,
        ctx: &Self::SharedContext,
    ) -> (Vec<Field>, Vec<ArrayRef>);
}
