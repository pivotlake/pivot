//! [`Dynamic`] — the runtime-signature value: `N` cells of a uniform width `A`,
//! each folded by its slot's op.
//!
//! Where [`Compiled`](super::Compiled) names its `(`[`Read`]`, `[`Fold`]`)` slots
//! in the type, `Dynamic` resolves them at runtime: [`make_reader`](AggregationValue::make_reader)
//! binds each slot to a [`BoundSlot`] (one variant *per op*, the column width a
//! [`NumReader`] payload — so the variants are `ops`, not `ops × widths`), and the
//! fold drives every slot through the *same* `Op::<A>::update(cell, read, arena,
//! cfg)`. The cell is `A` in and `A` out for every op — a string extreme's
//! `ArenaKey` is just the 128 bits of `A` (`= i128`), viewed as a key *inside*
//! [`StrMin`]/[`StrMax`] (via [`StringCell`]), so the container never reinterprets
//! and never branches string-vs-int.
//!
//! Generic over `A` (`i64` narrow / `i128` wide). A string extreme rides the wide
//! (`i128`) instantiation; the `i64` [`StringCell`] arms are the fail-out (the
//! planner always widens a string signature to `i128`).

use super::super::cell::{Numeric, StringCell};
use super::super::fold::{Fold, FoldAcc};
use super::super::read::{IntRead, Read, StrRead};
use super::super::{
    AggregationKind, AggregationSlot, AggregationValue, Count, Max, Min, SharedContext, StrMax,
    StrMin, Sum, WorkerContext,
};
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int16Type, Int32Type, Int64Type};
use arrow_array::{ArrayRef, PrimitiveArray, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field};
use std::sync::Arc;

/// An integer column bound at one of the three widths, read as `i64`. The width
/// is a *payload* here, not a cross-product with the op: adding a width is one
/// more variant in this enum, shared by every numeric op.
pub enum NumReader<'b> {
    I16(&'b PrimitiveArray<Int16Type>),
    I32(&'b PrimitiveArray<Int32Type>),
    I64(&'b PrimitiveArray<Int64Type>),
}

impl<'b> NumReader<'b> {
    pub(crate) fn bind(batch: &'b RecordBatch, column: usize) -> Self {
        let col = batch.column(column);
        match col.data_type() {
            DataType::Int16 => NumReader::I16(col.as_primitive::<Int16Type>()),
            DataType::Int32 => NumReader::I32(col.as_primitive::<Int32Type>()),
            DataType::Int64 => NumReader::I64(col.as_primitive::<Int64Type>()),
            other => panic!("numeric aggregate over unsupported column type {other:?}"),
        }
    }
    #[inline(always)]
    pub(crate) fn read(&self, idx: usize) -> i64 {
        match self {
            NumReader::I16(a) => IntRead::<Int16Type>::read(a, idx),
            NumReader::I32(a) => IntRead::<Int32Type>::read(a, idx),
            NumReader::I64(a) => IntRead::<Int64Type>::read(a, idx),
        }
    }
}

/// One slot's bound reader for a batch — one variant *per op*, the numeric column
/// width carried inside [`NumReader`]. Built once per batch by [`bind`](BoundSlot::bind).
pub enum BoundSlot<'b> {
    Count,
    Sum(NumReader<'b>),
    Min(NumReader<'b>),
    Max(NumReader<'b>),
    StrMin(&'b StringViewArray),
    StrMax(&'b StringViewArray),
}

impl<'b> BoundSlot<'b> {
    fn bind(batch: &'b RecordBatch, slot: &AggregationSlot) -> Self {
        use AggregationKind::*;
        match slot.kind {
            CountStar | Count => BoundSlot::Count,
            Sum => BoundSlot::Sum(NumReader::bind(batch, slot.column)),
            Min => BoundSlot::Min(NumReader::bind(batch, slot.column)),
            Max => BoundSlot::Max(NumReader::bind(batch, slot.column)),
            StrMin => BoundSlot::StrMin(StrRead::bind(batch, slot.column)),
            StrMax => BoundSlot::StrMax(StrRead::bind(batch, slot.column)),
        }
    }
}

/// `N` cells of width `A`, each folded by its slot op.
///
/// `ONLY_ADDITIVE` is a fast-path promise: when `true`, every slot is guaranteed
/// (by the planner) to be a `COUNT` or `SUM`, so the per-slot folds (`update`,
/// `merge`) drop their `Min`/`Max`/`StrMin`/`StrMax` arms to `unreachable!()`. With
/// the non-additive arms gone the compiler collapses the per-slot dispatch and the
/// `StringCell`/`i128` machinery to a branch-free additive loop — recovering the
/// hand-written `Mono` codegen from this same generic container. The two-level
/// `COUNT(DISTINCT)` path (all additive) sets it; everything else leaves it `false`.
///
/// `EAGER` selects the consume strategy at compile time: `false` (default) records
/// each row's cell into the worker's seed/update buffers and folds them in
/// [`finalize_batch`](AggregationValue::finalize_batch) once per slot per batch;
/// `true` is the MEASUREMENT baseline that folds each row in place during the probe
/// (the original path). The `if EAGER` branches const-fold, so each value compiles
/// to one strategy with no runtime test.
pub struct Dynamic<
    const N: usize,
    A: Numeric + StringCell = i64,
    const ONLY_ADDITIVE: bool = false,
    const EAGER: bool = false,
> {
    cells: [A; N],
}

impl<const N: usize, A: Numeric + StringCell, const ONLY_ADDITIVE: bool, const EAGER: bool> Copy
    for Dynamic<N, A, ONLY_ADDITIVE, EAGER>
{
}
impl<const N: usize, A: Numeric + StringCell, const ONLY_ADDITIVE: bool, const EAGER: bool> Clone
    for Dynamic<N, A, ONLY_ADDITIVE, EAGER>
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<const N: usize, A: Numeric + StringCell, const ONLY_ADDITIVE: bool, const EAGER: bool> Default
    for Dynamic<N, A, ONLY_ADDITIVE, EAGER>
{
    fn default() -> Self {
        Self {
            cells: [A::default(); N],
        }
    }
}

/// The read-side context for a [`Dynamic`]: the per-slot kinds (which op renders
/// each cell) and the value arena (which a string extreme resolves keys through).
/// Type-erased of the value's shape, so one type serves every `Dynamic`.
#[derive(Clone)]
pub struct DynShared {
    slots: Arc<[AggregationSlot]>,
    arena: Arc<SharedArena>,
}

impl SharedContext for DynShared {
    type Worker = DynWorker;
    fn build(slots: &[AggregationSlot], arena: &Arc<SharedArena>) -> Self {
        Self {
            slots: Arc::from(slots),
            arena: arena.clone(),
        }
    }
    fn worker(&self) -> DynWorker {
        DynWorker {
            arena: WorkerArena::new(self.arena.clone()),
            seeds: Vec::new(),
            updates: Vec::new(),
        }
    }
}

/// The per-worker write state for the deferred [`Dynamic`] consume: the string
/// arena, plus the two batch-scoped buffers of `(cell, row)` the probe records
/// into. The cell pointers are type-erased (`*mut u8`);
/// [`finalize_batch`](AggregationValue::finalize_batch) casts each back to the
/// concrete value, which it alone knows. Drained and cleared each batch, so the
/// capacity is reused.
///
/// The recorded pointers are stable: a hash-table entry lives in slab memory, so
/// growing the table stack never moves it, and the buffers are always emptied
/// before the batch's reader is dropped.
pub struct DynWorker {
    arena: WorkerArena,
    seeds: Vec<(*mut u8, u32)>,
    updates: Vec<(*mut u8, u32)>,
}

// The raw cell pointers live and die within a single worker's batch (recorded and
// drained on one thread, never read across threads); the buffers are empty when
// the worker is first handed to its thread.
unsafe impl Send for DynWorker {}

impl WorkerContext for DynWorker {
    fn flush(self) {
        self.arena.flush();
    }
}

impl<const N: usize, A: Numeric + StringCell, const ONLY_ADDITIVE: bool, const EAGER: bool>
    AggregationValue for Dynamic<N, A, ONLY_ADDITIVE, EAGER>
{
    // Neither consume strategy is ported to the radix scatter path, so the worker
    // always folds in place.
    const RADIX_SCATTER: bool = false;

    type Reader<'b> = [BoundSlot<'b>; N];
    /// Type-erased read-side context (slot kinds + value arena), shared by every
    /// `Dynamic`. It spawns the per-worker [`DynWorker`] via [`SharedContext::worker`].
    type SharedContext = DynShared;
    type Columns = [SlabColumn<A>; N];
    type SortKey = i128;
    /// Carries the string arena plus the deferred-consume seed/update buffers.
    type WorkerContext = DynWorker;

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> [BoundSlot<'b>; N] {
        assert_eq!(slots.len(), N, "slot count must match N");
        std::array::from_fn(|s| BoundSlot::bind(batch, &slots[s]))
    }

    #[inline(always)]
    fn value(reader: &[BoundSlot<'_>; N], idx: usize, wc: &mut DynWorker) -> Self {
        // A plain loop, not `std::array::from_fn`: the per-slot match is large, so
        // as a `from_fn` closure it exceeds the inline threshold and is emitted
        // out-of-line through the `Wrapped`/try-trait machinery — measured at ~40%
        // of a q09 merge regression. The loop keeps the op `seed`s inlined.
        // Numeric arms take a throwaway `&mut ()` (their `Arena` is `()`); only a
        // string arm touches the real `WorkerArena`.
        let mut na = ();
        let mut cells = [A::default(); N];
        #[allow(clippy::needless_range_loop)]
        for s in 0..N {
            // `ONLY_ADDITIVE` prunes each non-additive arm *in its body* — `if
            // ONLY_ADDITIVE { unreachable!() } else { .. }` const-folds to a bare
            // `unreachable!()` arm. A `_ if ONLY_ADDITIVE` guard arm instead lowers to
            // a worse Count/Sum dispatch — measured ~2.3B more in `consume_window`.
            cells[s] = match &reader[s] {
                BoundSlot::Count => Count::<A>::seed((), &mut na),
                BoundSlot::Sum(r) => Sum::<A>::seed(r.read(idx), &mut na),
                BoundSlot::Min(r) => {
                    if ONLY_ADDITIVE {
                        unreachable!()
                    } else {
                        Min::<A>::seed(r.read(idx), &mut na)
                    }
                }
                BoundSlot::Max(r) => {
                    if ONLY_ADDITIVE {
                        unreachable!()
                    } else {
                        Max::<A>::seed(r.read(idx), &mut na)
                    }
                }
                BoundSlot::StrMin(a) => {
                    if ONLY_ADDITIVE {
                        unreachable!()
                    } else {
                        StrMin::<A>::seed(StrRead::read(a, idx), &mut wc.arena)
                    }
                }
                BoundSlot::StrMax(a) => {
                    if ONLY_ADDITIVE {
                        unreachable!()
                    } else {
                        StrMax::<A>::seed(StrRead::read(a, idx), &mut wc.arena)
                    }
                }
            };
        }
        Self { cells }
    }

    /// Deferred: just record the matched cell and the source row. The fold is hoisted
    /// to [`finalize_batch`](Self::finalize_batch), once per slot per batch. With the
    /// MEASUREMENT `eager` baseline set, fold the row in place instead (the original
    /// per-row path).
    #[inline(always)]
    fn consume_seed(
        cell: &mut Self,
        reader: &[BoundSlot<'_>; N],
        idx: usize,
        wc: &mut DynWorker,
    ) {
        if EAGER {
            *cell = Self::value(reader, idx, wc);
        } else {
            wc.seeds.push((cell as *mut Self as *mut u8, idx as u32));
        }
    }

    #[inline(always)]
    fn consume_update(
        cell: &mut Self,
        reader: &[BoundSlot<'_>; N],
        idx: usize,
        wc: &mut DynWorker,
        ctx: &Self::SharedContext,
    ) {
        if !EAGER {
            wc.updates.push((cell as *mut Self as *mut u8, idx as u32));
            return;
        }
        // MEASUREMENT eager baseline: the original per-row, per-slot fold.
        let shared = &ctx.arena;
        let mut na = ();
        #[allow(clippy::needless_range_loop)]
        for s in 0..N {
            let c = cell.cells[s];
            cell.cells[s] = match &reader[s] {
                BoundSlot::Count => Count::<A>::update(c, (), &mut na, &()),
                BoundSlot::Sum(r) => Sum::<A>::update(c, r.read(idx), &mut na, &()),
                BoundSlot::Min(r) => {
                    if ONLY_ADDITIVE {
                        unreachable!()
                    } else {
                        Min::<A>::update(c, r.read(idx), &mut na, &())
                    }
                }
                BoundSlot::Max(r) => {
                    if ONLY_ADDITIVE {
                        unreachable!()
                    } else {
                        Max::<A>::update(c, r.read(idx), &mut na, &())
                    }
                }
                BoundSlot::StrMin(a) => {
                    if ONLY_ADDITIVE {
                        unreachable!()
                    } else {
                        StrMin::<A>::update(c, StrRead::read(a, idx), &mut wc.arena, shared)
                    }
                }
                BoundSlot::StrMax(a) => {
                    if ONLY_ADDITIVE {
                        unreachable!()
                    } else {
                        StrMax::<A>::update(c, StrRead::read(a, idx), &mut wc.arena, shared)
                    }
                }
            };
        }
    }

    /// Fold the batch's recorded seeds then updates, one slot at a time: the
    /// per-slot kind match runs once here instead of once per row, leaving each
    /// slot's inner loop a monomorphic gather over the recorded cells.
    fn finalize_batch(
        wc: &mut DynWorker,
        reader: &[BoundSlot<'_>; N],
        ctx: &Self::SharedContext,
    ) {
        // The eager baseline folds in place during the probe and records nothing.
        if EAGER {
            return;
        }
        // Split-borrow: the string arena and the two buffers are distinct fields,
        // so the seed/update loops can write through the arena while iterating.
        let DynWorker {
            arena,
            seeds,
            updates,
        } = wc;
        let shared = &ctx.arena;
        let mut na = ();
        // Cache-blocking: a tile's touched value cells (at most `TILE` lines) stay
        // L1-resident across the N per-aggregation passes, so each cell line loads
        // once per phase instead of once per slot — the difference between eager's
        // 1x and a naive per-slot finalize's Nx entry touches. `TILE` lines is
        // ~32 KB, about half of L1.
        //
        // SAFETY (both phases): each pointer addresses a live hash-table entry in
        // slab memory (stable across table growth) and the buffers are drained
        // before the batch's reader is dropped. A cell is seeded exactly once (first
        // occurrence) and the seed phase fully precedes the update phase, so every
        // update folds onto a materialised cell; seeds touch distinct cells, updates
        // to one cell stay in row order.
        const TILE: usize = 512;

        // Seed phase: materialise each first-seen group's cells.
        let mut start = 0;
        while start < seeds.len() {
            let tile = &seeds[start..(start + TILE).min(seeds.len())];
            #[allow(clippy::needless_range_loop)]
            for s in 0..N {
                match &reader[s] {
                    BoundSlot::Count => {
                        for &(ptr, _) in tile {
                            unsafe { (*(ptr as *mut Self)).cells[s] = Count::<A>::seed((), &mut na) };
                        }
                    }
                    BoundSlot::Sum(r) => {
                        for &(ptr, row) in tile {
                            unsafe {
                                (*(ptr as *mut Self)).cells[s] = Sum::<A>::seed(r.read(row as usize), &mut na)
                            };
                        }
                    }
                    BoundSlot::Min(r) => {
                        if ONLY_ADDITIVE {
                            unreachable!()
                        } else {
                            for &(ptr, row) in tile {
                                unsafe {
                                    (*(ptr as *mut Self)).cells[s] = Min::<A>::seed(r.read(row as usize), &mut na)
                                };
                            }
                        }
                    }
                    BoundSlot::Max(r) => {
                        if ONLY_ADDITIVE {
                            unreachable!()
                        } else {
                            for &(ptr, row) in tile {
                                unsafe {
                                    (*(ptr as *mut Self)).cells[s] = Max::<A>::seed(r.read(row as usize), &mut na)
                                };
                            }
                        }
                    }
                    BoundSlot::StrMin(a) => {
                        if ONLY_ADDITIVE {
                            unreachable!()
                        } else {
                            for &(ptr, row) in tile {
                                unsafe {
                                    (*(ptr as *mut Self)).cells[s] = StrMin::<A>::seed(StrRead::read(a, row as usize), arena)
                                };
                            }
                        }
                    }
                    BoundSlot::StrMax(a) => {
                        if ONLY_ADDITIVE {
                            unreachable!()
                        } else {
                            for &(ptr, row) in tile {
                                unsafe {
                                    (*(ptr as *mut Self)).cells[s] = StrMax::<A>::seed(StrRead::read(a, row as usize), arena)
                                };
                            }
                        }
                    }
                }
            }
            start += TILE;
        }

        // Update phase: fold repeat occurrences onto their (already seeded) cells.
        let mut start = 0;
        while start < updates.len() {
            let tile = &updates[start..(start + TILE).min(updates.len())];
            #[allow(clippy::needless_range_loop)]
            for s in 0..N {
                match &reader[s] {
                    BoundSlot::Count => {
                        for &(ptr, _) in tile {
                            unsafe {
                                let c = (*(ptr as *mut Self)).cells[s];
                                (*(ptr as *mut Self)).cells[s] = Count::<A>::update(c, (), &mut na, &());
                            }
                        }
                    }
                    BoundSlot::Sum(r) => {
                        for &(ptr, row) in tile {
                            unsafe {
                                let c = (*(ptr as *mut Self)).cells[s];
                                (*(ptr as *mut Self)).cells[s] = Sum::<A>::update(c, r.read(row as usize), &mut na, &());
                            }
                        }
                    }
                    BoundSlot::Min(r) => {
                        if ONLY_ADDITIVE {
                            unreachable!()
                        } else {
                            for &(ptr, row) in tile {
                                unsafe {
                                    let c = (*(ptr as *mut Self)).cells[s];
                                    (*(ptr as *mut Self)).cells[s] = Min::<A>::update(c, r.read(row as usize), &mut na, &());
                                }
                            }
                        }
                    }
                    BoundSlot::Max(r) => {
                        if ONLY_ADDITIVE {
                            unreachable!()
                        } else {
                            for &(ptr, row) in tile {
                                unsafe {
                                    let c = (*(ptr as *mut Self)).cells[s];
                                    (*(ptr as *mut Self)).cells[s] = Max::<A>::update(c, r.read(row as usize), &mut na, &());
                                }
                            }
                        }
                    }
                    BoundSlot::StrMin(a) => {
                        if ONLY_ADDITIVE {
                            unreachable!()
                        } else {
                            for &(ptr, row) in tile {
                                unsafe {
                                    let c = (*(ptr as *mut Self)).cells[s];
                                    (*(ptr as *mut Self)).cells[s] = StrMin::<A>::update(c, StrRead::read(a, row as usize), arena, shared);
                                }
                            }
                        }
                    }
                    BoundSlot::StrMax(a) => {
                        if ONLY_ADDITIVE {
                            unreachable!()
                        } else {
                            for &(ptr, row) in tile {
                                unsafe {
                                    let c = (*(ptr as *mut Self)).cells[s];
                                    (*(ptr as *mut Self)).cells[s] = StrMax::<A>::update(c, StrRead::read(a, row as usize), arena, shared);
                                }
                            }
                        }
                    }
                }
            }
            start += TILE;
        }
        seeds.clear();
        updates.clear();
    }

    #[inline(always)]
    fn merge(self, other: Self, ctx: &Self::SharedContext) -> Self {
        let (slots, shared) = (&ctx.slots, &ctx.arena);
        // A plain loop, not `std::array::from_fn`, for the same inlining reason as
        // `value` — this runs per matched entry in the partition merge, the hottest
        // path for a high-cardinality `COUNT(DISTINCT)`. Each slot combines via its
        // op's own `merge` — same `Op::<A>::merge(a, b, cfg)` shape, no reader.
        let mut cells = [A::default(); N];
        #[allow(clippy::needless_range_loop)]
        for s in 0..N {
            let (a, b) = (self.cells[s], other.cells[s]);
            cells[s] = if ONLY_ADDITIVE {
                // All-additive: `Count` and `Sum` both merge by `+`, so skip the
                // per-slot `slots[s].kind` load and Count/Sum branch entirely — a
                // pure add, identical to Mono's branch-free merge.
                a + b
            } else {
                match slots[s].kind {
                    AggregationKind::CountStar | AggregationKind::Count => {
                        Count::<A>::merge(a, b, &())
                    }
                    AggregationKind::Sum => Sum::<A>::merge(a, b, &()),
                    AggregationKind::Min => Min::<A>::merge(a, b, &()),
                    AggregationKind::Max => Max::<A>::merge(a, b, &()),
                    AggregationKind::StrMin => StrMin::<A>::merge(a, b, shared),
                    AggregationKind::StrMax => StrMax::<A>::merge(a, b, shared),
                }
            };
        }
        Self { cells }
    }

    #[inline(always)]
    fn sort_key(&self, slot: usize) -> i128 {
        // Numeric cells widen to their `ORDER BY` key. A string extreme never feeds
        // a top-k (the planner doesn't push one), so its raw bits here are inert.
        self.cells[slot].into()
    }

    fn new_columns(allocator: &mut SlabAllocator, rows: usize) -> [SlabColumn<A>; N] {
        std::array::from_fn(|_| SlabColumn::with_capacity(allocator, rows))
    }

    #[inline(always)]
    fn push_to(&self, cols: &mut [SlabColumn<A>; N]) {
        for (col, cell) in cols.iter_mut().zip(self.cells.iter()) {
            col.push(*cell);
        }
    }

    fn finish_columns(
        cols: [SlabColumn<A>; N],
        ctx: &Self::SharedContext,
    ) -> (Vec<Field>, Vec<ArrayRef>) {
        let (slots, arena) = (&ctx.slots, &ctx.arena);
        let mut fields = Vec::with_capacity(N);
        let mut arrays = Vec::with_capacity(N);
        for (s, col) in cols.into_iter().enumerate() {
            let name = format!("v{s}");
            // Each op renders its own column (numeric → its width's Arrow type via
            // `&()`, string → `Utf8View` resolved through the arena).
            let (f, a) = match slots[s].kind {
                AggregationKind::CountStar | AggregationKind::Count => {
                    Count::<A>::finish(&name, col, &())
                }
                AggregationKind::Sum => Sum::<A>::finish(&name, col, &()),
                AggregationKind::Min => Min::<A>::finish(&name, col, &()),
                AggregationKind::Max => Max::<A>::finish(&name, col, &()),
                AggregationKind::StrMin => StrMin::<A>::finish(&name, col, arena),
                AggregationKind::StrMax => StrMax::<A>::finish(&name, col, arena),
            };
            fields.push(f);
            arrays.push(a);
        }
        (fields, arrays)
    }
}
