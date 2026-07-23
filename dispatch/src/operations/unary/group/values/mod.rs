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

pub use cell::{Cell, F64Cell, IntCell};
pub use container::{Compiled, CountSlot, Dynamic, MaxSlot, MinSlot, OpTuple, SumSlot, Variable};
pub use distinct::Distinct;
pub use fold::{
    Count, F64Max, F64Min, F64Sum, Fold, Max, Min, StrMax, StrMin, Sum, U128Max, U128Min, U128Sum,
    WideSum,
};
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
    /// `MIN(col)`. The value's family (integer / float / string / wide) is decided
    /// by the column type at bind, not by the kind — a `Utf8` column reads the
    /// string extreme, a numeric column the numeric one.
    Min,
    /// `MAX(col)`. See [`Min`](AggregationKind::Min).
    Max,
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

    /// Whether this slot is a string extreme (`MIN`/`MAX` over a `Utf8` column),
    /// whose cell is an [`ArenaKey`](super::ArenaKey). Read off the declared
    /// `output_type` (`Utf8View`), since the `MIN`/`MAX` kind alone doesn't say — the
    /// value family is decided by the column type. A signature with any such slot
    /// must use 128-bit (`i128`) cells and stays off the radix scatter path (which
    /// would eagerly persist every row's string, winner or not).
    #[inline(always)]
    pub fn is_string_extreme(&self) -> bool {
        matches!(self.kind, AggregationKind::Min | AggregationKind::Max)
            && self.output_type == DataType::Utf8View
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

/// Builds the trailing value column(s) of a GROUP BY result, one group at a time:
/// the value-side counterpart to [`KeyColumns`](super::keys::KeyColumns).
///
/// The output combinator pushes each surviving group's value, then
/// [`finish`](Self::finish) materialises the Arrow columns and their fields.
/// `finish` takes the value's [`SharedContext`] so a string extreme can emit
/// zero-copy `StringView`s into the value arena and a runtime [`Dynamic`] value
/// can read each slot's render kind; an all-numeric value ignores it.
pub trait ValueColumns {
    /// The per-group value these columns accumulate.
    type Value: AggregationValue;
    /// The owning value's read-side context (see [`AggregationValue::SharedContext`]).
    type Context;

    /// Allocate the value-column builders over engine memory, sized for `rows`
    /// (one output chunk; must fit a single 2 MB slab). `context` carries the
    /// slot list a runtime-arity value ([`Variable`]) sizes its builder count
    /// from; fixed-arity values ignore it.
    fn with_capacity(allocator: &mut SlabAllocator, rows: usize, context: &Self::Context) -> Self;
    /// Append one finished group's owned value (the top-k heap drain).
    fn push(&mut self, value: &Self::Value);
    /// Append one finished group's value straight from its table entry.
    fn push_stored(&mut self, stored: &<Self::Value as AggregationValue>::Stored);
    /// Materialise the value columns and their fields. `context` backs the
    /// zero-copy `StringView` output of a string extreme (numeric columns ignore
    /// it) and carries the per-slot descriptor a runtime value ([`Dynamic`]) needs
    /// to pick each slot's output type.
    fn finish(self, context: &Self::Context) -> (Vec<Field>, Vec<ArrayRef>);
}

/// A fixed-arity aggregation value that is moved around *by value*: what a hash
/// entry stores for it is the value itself, so every fold produces a new value
/// from owned inputs. All compile-time-shaped containers ([`Compiled`],
/// [`Dynamic`], [`Distinct`]) implement this; the blanket impl below lifts any
/// `OwnedValue` into the storage-generic [`AggregationValue`] the group operator
/// actually runs on. The runtime-arity [`Variable`] cannot (its entry payload is
/// a runtime-length cell slice), so it implements [`AggregationValue`] directly.
///
/// Reading and folding are separate so a string extreme can persist lazily: a new
/// group materialises with [`value`](Self::value), an existing group folds the
/// next row with [`update_from_reader`](Self::update_from_reader) (which can skip
/// persisting a row that doesn't win), and two finished partials combine with
/// [`merge`](Self::merge) (no new materialisation). For values whose fold is the
/// same elementwise op for rows and partials, `update_from_reader` defaults to
/// merging the row in.
pub trait OwnedValue: Copy + Default + Send + Sync + 'static {
    /// Per-batch reader holding the downcast value columns.
    type Reader<'b>;
    /// The shared, read-side context [`merge`](Self::merge)/[`ValueColumns::finish`]
    /// resolve through (slot kinds for [`Dynamic`] + the value
    /// arena for a string extreme; `()` otherwise). It builds the per-worker
    /// [`WorkerContext`](Self::WorkerContext); see [`SharedContext`].
    type SharedContext: SharedContext<Worker = Self::WorkerContext>;
    /// The trailing value columns these groups emit (the value-side counterpart
    /// to [`KeyExtractor::Columns`](super::keys::KeyExtractor::Columns)).
    type Columns: ValueColumns<Value = Self, Context = Self::SharedContext>;
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
}

/// The per-group aggregation value as the GROUP BY operator sees it: read from
/// input rows, folded *in place* inside a hash-table entry, and emitted as the
/// result's value columns.
///
/// The table never holds `Self` directly; it holds [`Stored`](Self::Stored), the
/// entry-resident form the fold methods mutate through a reference. For every
/// fixed-arity value `Stored = Self` (the blanket impl over [`OwnedValue`]
/// forwards each in-place op to the by-value fold). For the runtime-arity
/// [`Variable`], `Stored = [A]` — a cell slice living inline in the entry at a
/// stride fixed per query — and `Self` is a thin owned handle used only where an
/// owned, `Sized` value is unavoidable (a radix scatter row, a top-k heap row).
///
/// The table's entries are raw bytes at a per-table stride, so the value also
/// tells the table how large its stored form is and how to view an entry's
/// value bytes as `Stored`, via [`StoredMeta`](Self::StoredMeta) (the one
/// runtime fact needed: nothing for a fixed-arity value, the slot count for
/// [`Variable`]).
///
/// `Self` (not `Stored`) still travels through the owned side paths, which is
/// why the trait keeps the `Copy` bound.
pub trait AggregationValue: Copy + Default + Send + Sync + 'static {
    /// The entry-resident form of one group's value; what the fold methods
    /// mutate in place. `Self` for a fixed-arity value; a runtime-length cell
    /// slice for [`Variable`].
    type Stored: ?Sized + Send + Sync;
    /// The runtime fact needed to size and view a stored value: `()` for a
    /// fixed-arity value, the slot count for [`Variable`]. Held once per hash
    /// table, never per value.
    type StoredMeta: Copy + Send + Sync + 'static;
    /// Per-batch reader holding the downcast value columns.
    type Reader<'b>;
    /// The shared, read-side context the merge and output phases resolve
    /// through; see [`OwnedValue::SharedContext`].
    type SharedContext: SharedContext<Worker = Self::WorkerContext>;
    /// The trailing value columns these groups emit.
    type Columns: ValueColumns<Value = Self, Context = Self::SharedContext>;
    /// The scalar an `ORDER BY <slot> DESC LIMIT k` sorts on.
    type SortKey: Ord + Copy;
    /// The per-worker write state consume folds into; see
    /// [`OwnedValue::WorkerContext`].
    type WorkerContext: WorkerContext;

    /// Whether the radix scatter path may materialise this value once per *row*
    /// (it scatters `(hash, key, value)` triples raw, deferring aggregation to
    /// the merge). `false` for a value whose materialisation allocates per-group
    /// state (a [`Variable`] arena block), which per row would grow the arena
    /// with the row count; such a value always aggregates in place.
    const RADIX_COMPATIBLE: bool = true;

    /// Bind `batch`'s value columns for the configured `slots`.
    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self::Reader<'b>;

    /// The runtime metadata for this query's signature, read off the shared
    /// context once per table.
    fn stored_meta(ctx: &Self::SharedContext) -> Self::StoredMeta;

    /// The byte size of one stored value.
    fn stored_size(meta: Self::StoredMeta) -> usize;

    /// The alignment of one stored value.
    fn stored_align(meta: Self::StoredMeta) -> usize;

    /// View an entry's value bytes as the stored form.
    ///
    /// # Safety
    /// `ptr` must point at [`stored_size`](Self::stored_size) bytes aligned to
    /// [`stored_align`](Self::stored_align), valid for the returned lifetime,
    /// and `meta` must be the table's own.
    unsafe fn stored_ref<'a>(ptr: *const u8, meta: Self::StoredMeta) -> &'a Self::Stored;

    /// Mutable counterpart of [`stored_ref`](Self::stored_ref).
    ///
    /// # Safety
    /// As [`stored_ref`](Self::stored_ref), plus `ptr` must be exclusive for
    /// the returned lifetime.
    unsafe fn stored_mut<'a>(ptr: *mut u8, meta: Self::StoredMeta) -> &'a mut Self::Stored;

    /// Materialise a brand-new group from row `idx` into `dst` — the consume
    /// path's new-key case.
    fn seed_stored(
        dst: &mut Self::Stored,
        reader: &Self::Reader<'_>,
        idx: usize,
        wc: &mut Self::WorkerContext,
    );

    /// Fold row `idx` into the existing group at `dst`.
    fn update_stored(
        dst: &mut Self::Stored,
        reader: &Self::Reader<'_>,
        idx: usize,
        wc: &mut Self::WorkerContext,
        ctx: &Self::SharedContext,
    );

    /// Combine two partial group values in place — the partition merge.
    fn merge_stored(dst: &mut Self::Stored, src: &Self::Stored, ctx: &Self::SharedContext);

    /// Copy a finished partial into a freshly claimed entry (the merge's
    /// new-key case; `dst` is zeroed).
    fn clone_stored(dst: &mut Self::Stored, src: &Self::Stored);

    /// The group's value for slot `slot`, as an `ORDER BY` sort key.
    fn sort_key_stored(stored: &Self::Stored, slot: usize) -> Self::SortKey;

    /// Materialise one row's contribution as an owned value — the radix scatter
    /// row. Only called when [`RADIX_COMPATIBLE`](Self::RADIX_COMPATIBLE).
    fn value(reader: &Self::Reader<'_>, idx: usize, wc: &mut Self::WorkerContext) -> Self;

    /// Write an owned value into a freshly claimed entry (the scatter merge's
    /// new-key case).
    fn store(dst: &mut Self::Stored, value: Self);

    /// Fold an owned value into the existing group at `dst` (the scatter merge).
    fn merge_value(dst: &mut Self::Stored, value: Self, ctx: &Self::SharedContext);

    /// Copy a table-resident value out into an owned one that survives its
    /// table — a top-k heap row. A fixed-arity value is its own owned form; a
    /// runtime-arity value copies its cells into the value arena, spawning the
    /// per-worker write handle into `wc` on first use (the caller flushes it).
    fn to_owned(
        stored: &Self::Stored,
        ctx: &Self::SharedContext,
        wc: &mut Option<Self::WorkerContext>,
    ) -> Self;
}

/// Every fixed-arity (by-value) container is an [`AggregationValue`] whose
/// stored form is itself: each in-place op reads the entry, runs the by-value
/// fold, and writes the result back — which is exactly what the pre-storage
/// table did, so codegen is unchanged.
impl<T: OwnedValue> AggregationValue for T {
    type Stored = T;
    type StoredMeta = ();
    type Reader<'b> = <T as OwnedValue>::Reader<'b>;
    type SharedContext = <T as OwnedValue>::SharedContext;
    type Columns = <T as OwnedValue>::Columns;
    type SortKey = <T as OwnedValue>::SortKey;
    type WorkerContext = <T as OwnedValue>::WorkerContext;

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self::Reader<'b> {
        <T as OwnedValue>::make_reader(batch, slots)
    }

    fn stored_meta(_ctx: &Self::SharedContext) {}

    fn stored_size(_meta: ()) -> usize {
        size_of::<T>()
    }

    fn stored_align(_meta: ()) -> usize {
        align_of::<T>()
    }

    #[inline(always)]
    unsafe fn stored_ref<'a>(ptr: *const u8, _meta: ()) -> &'a T {
        unsafe { &*(ptr as *const T) }
    }

    #[inline(always)]
    unsafe fn stored_mut<'a>(ptr: *mut u8, _meta: ()) -> &'a mut T {
        unsafe { &mut *(ptr as *mut T) }
    }

    #[inline(always)]
    fn seed_stored(
        dst: &mut T,
        reader: &Self::Reader<'_>,
        idx: usize,
        wc: &mut Self::WorkerContext,
    ) {
        *dst = <T as OwnedValue>::value(reader, idx, wc);
    }

    #[inline(always)]
    fn update_stored(
        dst: &mut T,
        reader: &Self::Reader<'_>,
        idx: usize,
        wc: &mut Self::WorkerContext,
        ctx: &Self::SharedContext,
    ) {
        *dst = dst.update_from_reader(reader, idx, wc, ctx);
    }

    #[inline(always)]
    fn merge_stored(dst: &mut T, src: &T, ctx: &Self::SharedContext) {
        *dst = dst.merge(*src, ctx);
    }

    #[inline(always)]
    fn clone_stored(dst: &mut T, src: &T) {
        *dst = *src;
    }

    #[inline(always)]
    fn sort_key_stored(stored: &T, slot: usize) -> Self::SortKey {
        stored.sort_key(slot)
    }

    #[inline(always)]
    fn value(reader: &Self::Reader<'_>, idx: usize, wc: &mut Self::WorkerContext) -> Self {
        <T as OwnedValue>::value(reader, idx, wc)
    }

    #[inline(always)]
    fn store(dst: &mut T, value: T) {
        *dst = value;
    }

    #[inline(always)]
    fn merge_value(dst: &mut T, value: T, ctx: &Self::SharedContext) {
        *dst = dst.merge(value, ctx);
    }

    #[inline(always)]
    fn to_owned(
        stored: &T,
        _ctx: &Self::SharedContext,
        _wc: &mut Option<Self::WorkerContext>,
    ) -> Self {
        *stored
    }
}
