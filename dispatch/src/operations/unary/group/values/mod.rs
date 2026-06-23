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
//! - an **[`Aggregation`]** op — fully typed to its own input array and cell
//!   ([`Count`], [`Sum<T>`](Sum), [`Min<T>`](Min), [`Max<T>`](Max), [`StrMin`],
//!   [`StrMax`]).
//! - **containers** — [`Compiled`](container::Compiled) (a fixed *numeric* tuple
//!   of ops, branch-free) and [`Dynamic`](container::Dynamic) (a runtime
//!   signature folded per slot, generic over the width — the path for any string
//!   extreme).

use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::Field;
use std::sync::Arc;

pub mod cell;
pub mod container;
pub mod distinct;
pub mod fold;
pub mod read;

pub use cell::{Cell, Numeric};
pub use container::{Compiled, CountSlot, Dynamic, MaxSlot, MinSlot, OpTuple, SumSlot};
pub use distinct::Distinct;
pub use fold::{Count, Fold, FoldAcc, Max, Min, StrMax, StrMin, Sum, WideSum};
pub use read::{IntRead, NoRead, Read, StrRead};

/// Which per-group aggregate a value slot computes during consume — a pure
/// descriptor the planner attaches to each slot. It tells the numeric
/// [`Dynamic`](container::Dynamic) fallback what to read (a `COUNT` reads no
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
/// The in-place consume path folds through [`consume_seed`](Self::consume_seed)
/// (the probe's empty-slot branch) and [`consume_update`](Self::consume_update)
/// (its key-match branch), then [`finalize_batch`](Self::finalize_batch) at the end
/// of each batch. An *eager* value folds straight into the cell in those two calls
/// and leaves `finalize_batch` empty; a *deferred* value (see
/// [`Dynamic`](container::Dynamic)) instead records the matched cell into its
/// [`WorkerContext`](Self::WorkerContext) and applies a whole batch's seeds/updates
/// in `finalize_batch`, once per slot rather than once per row.
///
/// [`value`](Self::value) materialises a standalone group value from one row (the
/// radix scatter path, and an eager container's `consume_seed`); two finished
/// partials combine with [`merge`](Self::merge).
pub trait AggregationValue: Copy + Default + Send + Sync + 'static {
    /// Whether a worker holding this value may switch to the radix scatter path.
    /// `true` for an eager value; a deferred value sets it `false` (its scatter
    /// path is not ported yet), so the worker always folds in place. This is the
    /// value-side counterpart to [`KeyExtractor::SUPPORTS_RADIX`](super::keys::KeyExtractor::SUPPORTS_RADIX).
    const RADIX_SCATTER: bool = true;

    /// Per-batch reader holding the downcast value columns.
    type Reader<'b>;
    /// The shared, read-side context [`merge`](Self::merge)/[`finish_columns`](Self::finish_columns)
    /// resolve through (slot kinds for [`Dynamic`](container::Dynamic) + the value
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

    /// Materialise a brand-new group from row `idx` — the radix scatter, and an
    /// eager container's [`consume_seed`](Self::consume_seed). `wc` is the
    /// per-worker write state a string extreme persists its winning string into;
    /// numeric cells ignore it.
    fn value(reader: &Self::Reader<'_>, idx: usize, wc: &mut Self::WorkerContext) -> Self;

    /// Seed a freshly inserted group `cell` from row `idx` (the probe's empty-slot
    /// branch). An eager value folds in place now; a deferred value records
    /// `(cell, idx)` into `wc` for [`finalize_batch`](Self::finalize_batch).
    fn consume_seed(cell: &mut Self, reader: &Self::Reader<'_>, idx: usize, wc: &mut Self::WorkerContext);

    /// Fold row `idx` into the existing group `cell` (the probe's key-match
    /// branch). Eager or deferred, as [`consume_seed`](Self::consume_seed).
    fn consume_update(
        cell: &mut Self,
        reader: &Self::Reader<'_>,
        idx: usize,
        wc: &mut Self::WorkerContext,
        ctx: &Self::SharedContext,
    );

    /// Apply a batch's recorded seeds/updates to the table, once per slot (the
    /// per-row kind dispatch hoisted out of the probe loop). Empty for an eager
    /// value, which already folded each row in place.
    fn finalize_batch(wc: &mut Self::WorkerContext, reader: &Self::Reader<'_>, ctx: &Self::SharedContext);

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
    /// carries the per-slot descriptor a runtime value ([`Dynamic`](container::Dynamic))
    /// needs to pick each slot's output type.
    fn finish_columns(
        cols: Self::Columns,
        ctx: &Self::SharedContext,
    ) -> (Vec<Field>, Vec<ArrayRef>);
}
