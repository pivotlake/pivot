//! [`Patched`] — an additive numeric value folded by copy-and-patch machine code.
//!
//! Same `[i64; N]` cells as the additive [`Dynamic`](super::Dynamic), but the
//! per-row fold and per-cell merge run as branch-free tiles assembled at the
//! query's first batch (see [`cap`](super::super::cap)) instead of the per-slot
//! `match`. The probe/merge loops stay in Rust (they own the multi-slab table,
//! overflow and radix); the tile is called via a hoisted raw fn pointer — on the
//! memory-bound merge it hides behind the cache miss, and the fold is the same
//! `slots[]`-load-free code the hand-written `Compiled`/`ONLY_ADDITIVE` paths emit.
//!
//! Off aarch64, or for a signature the tiles don't cover, [`compile_additive`]
//! returns `None` and the same Rust scalar fold runs — so `Patched` is always
//! correct, just JIT-accelerated where it can be.

use super::super::cap::{self, AdditiveCompiled, FoldOp};
use super::super::fold::{Count, FoldAcc, Sum};
use super::super::{AggregationKind, AggregationSlot, AggregationValue};
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int16Type, Int32Type, Int64Type};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field};
use std::sync::{Arc, OnceLock};

/// `N` additive `i64` cells folded by JIT tiles. `repr(transparent)` so a
/// `*mut Self` is exactly the `*mut i64` cell base the tiles write through.
#[repr(transparent)]
pub struct Patched<const N: usize> {
    cells: [i64; N],
}

impl<const N: usize> Copy for Patched<N> {}
impl<const N: usize> Clone for Patched<N> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<const N: usize> Default for Patched<N> {
    fn default() -> Self {
        Self { cells: [0; N] }
    }
}

/// Per-batch reader: each slot's column base pointer and element width (`0` for
/// `COUNT`). The tiles index `cols[s]` by row; the widths also drive the one-time
/// tile compile and the scalar fallback.
pub struct PatchedReader<const N: usize> {
    cols: [*const u8; N],
    widths: [u8; N],
}

/// `(slots, value-arena, lazily-compiled tiles)`. The tiles need column widths,
/// known only at the first batch, so they're filled then via the `OnceLock`
/// (shared across workers — compiled once).
pub type PatchedConfig = (
    Arc<[AggregationSlot]>,
    Arc<SharedArena>,
    Arc<OnceLock<Option<AdditiveCompiled>>>,
);

/// One slot's `(col_ptr, width)` for the reader.
fn col_ptr_width(batch: &RecordBatch, slot: &AggregationSlot) -> (*const u8, u8) {
    match slot.kind {
        AggregationKind::CountStar | AggregationKind::Count => (std::ptr::null(), 0),
        AggregationKind::Sum => {
            let col = batch.column(slot.column);
            match col.data_type() {
                DataType::Int16 => (col.as_primitive::<Int16Type>().values().as_ptr() as *const u8, 2),
                DataType::Int32 => (col.as_primitive::<Int32Type>().values().as_ptr() as *const u8, 4),
                DataType::Int64 => (col.as_primitive::<Int64Type>().values().as_ptr() as *const u8, 8),
                other => panic!("Patched SUM over unsupported column {other:?}"),
            }
        }
        k => panic!("Patched is additive-only, got {k:?}"),
    }
}

/// Scalar fallback fold of row `idx` into `cells` (when no tiles compiled).
#[inline(always)]
fn scalar_fold<const N: usize>(cells: &mut [i64; N], reader: &PatchedReader<N>, idx: usize) {
    for s in 0..N {
        let v = match reader.widths[s] {
            0 => 1,
            2 => unsafe { (reader.cols[s] as *const i16).add(idx).read() as i64 },
            4 => unsafe { (reader.cols[s] as *const i32).add(idx).read() as i64 },
            8 => unsafe { (reader.cols[s] as *const i64).add(idx).read() },
            _ => unreachable!(),
        };
        cells[s] += v;
    }
}

impl<const N: usize> Patched<N> {
    #[inline(always)]
    fn fold_one(cells: &mut [i64; N], reader: &PatchedReader<N>, idx: usize, cfg: &PatchedConfig) {
        let compiled = cfg.2.get_or_init(|| build_tiles::<N>(reader, &cfg.0));
        if let Some(c) = compiled {
            // SAFETY: cells is [i64; N] (the layout the tiles were built for);
            // cols has N valid bases; idx is in range for the bound columns.
            unsafe { c.fold_row(cells.as_mut_ptr(), reader.cols.as_ptr(), idx as u64) }
        } else {
            scalar_fold(cells, reader, idx);
        }
    }
}

/// Compile the additive tiles for `slots` at `reader`'s widths (`None` if any slot
/// isn't additive or the buffers can't be mapped).
fn build_tiles<const N: usize>(
    reader: &PatchedReader<N>,
    slots: &[AggregationSlot],
) -> Option<AdditiveCompiled> {
    let mut ops: Vec<FoldOp> = Vec::with_capacity(N);
    for s in 0..N {
        ops.push(FoldOp::from_slot(slots[s].kind, reader.widths[s])?);
    }
    cap::compile_additive(&ops)
}

impl<const N: usize> AggregationValue for Patched<N> {
    type Reader<'b> = PatchedReader<N>;
    type MergeConfig = PatchedConfig;
    type Columns = [SlabColumn<i64>; N];
    type SortKey = i128;

    const CAP: bool = true;

    fn cap_fold_ops(
        slots: &[AggregationSlot],
        reader: &PatchedReader<N>,
    ) -> Option<Vec<FoldOp>> {
        let mut ops = Vec::with_capacity(N);
        for s in 0..N {
            ops.push(FoldOp::from_slot(slots[s].kind, reader.widths[s])?);
        }
        Some(ops)
    }

    #[inline(always)]
    fn cap_cols(reader: &PatchedReader<N>) -> *const *const u8 {
        reader.cols.as_ptr()
    }

    #[inline(always)]
    fn cap_nslots() -> usize {
        N
    }

    #[inline(always)]
    unsafe fn cap_from_cells(p: *const i64) -> Self {
        // `Self` is `repr(transparent)` over `[i64; N]`.
        Self {
            cells: unsafe { (p as *const [i64; N]).read() },
        }
    }

    fn merge_config(slots: &[AggregationSlot], arena: &Arc<SharedArena>) -> Self::MergeConfig {
        (Arc::from(slots), arena.clone(), Arc::new(OnceLock::new()))
    }

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> PatchedReader<N> {
        assert_eq!(slots.len(), N, "slot count must match N");
        let mut cols = [std::ptr::null(); N];
        let mut widths = [0u8; N];
        for s in 0..N {
            let (p, w) = col_ptr_width(batch, &slots[s]);
            cols[s] = p;
            widths[s] = w;
        }
        PatchedReader { cols, widths }
    }

    /// No-cfg materialise — only the scalar path is reachable here (the consume
    /// seed and radix scatter go through [`value_cfg`](AggregationValue::value_cfg)).
    #[inline(always)]
    fn value(reader: &PatchedReader<N>, idx: usize, _arena: &mut WorkerArena) -> Self {
        let mut cells = [0i64; N];
        scalar_fold(&mut cells, reader, idx);
        Self { cells }
    }

    #[inline(always)]
    fn value_cfg(
        reader: &PatchedReader<N>,
        idx: usize,
        _arena: &mut WorkerArena,
        cfg: &Self::MergeConfig,
    ) -> Self {
        let mut cells = [0i64; N];
        Self::fold_one(&mut cells, reader, idx, cfg);
        Self { cells }
    }

    #[inline(always)]
    fn update_from_reader(
        mut self,
        reader: &PatchedReader<N>,
        idx: usize,
        _arena: &mut WorkerArena,
        cfg: &Self::MergeConfig,
    ) -> Self {
        Self::fold_one(&mut self.cells, reader, idx, cfg);
        self
    }

    #[inline(always)]
    fn merge(mut self, other: Self, cfg: &Self::MergeConfig) -> Self {
        self.merge_in_place(&other, cfg);
        self
    }

    #[inline(always)]
    fn merge_in_place(&mut self, other: &Self, cfg: &Self::MergeConfig) {
        // Tiles compiled during consume; if present, merge via native add tiles.
        if let Some(Some(c)) = cfg.2.get() {
            // SAFETY: both are [i64; N] (transparent); the merge tile adds N cells.
            unsafe { c.merge_pair(self.cells.as_mut_ptr(), other.cells.as_ptr()) }
        } else {
            for s in 0..N {
                self.cells[s] += other.cells[s];
            }
        }
    }

    #[inline(always)]
    fn sort_key(&self, slot: usize) -> i128 {
        self.cells[slot] as i128
    }

    fn new_columns(allocator: &mut SlabAllocator, rows: usize) -> Self::Columns {
        std::array::from_fn(|_| SlabColumn::<i64>::with_capacity(allocator, rows))
    }

    #[inline(always)]
    fn push_to(&self, cols: &mut Self::Columns) {
        for (col, cell) in cols.iter_mut().zip(self.cells.iter()) {
            col.push(*cell);
        }
    }

    fn finish_columns(
        cols: Self::Columns,
        arena: &Arc<SharedArena>,
        cfg: &Self::MergeConfig,
    ) -> (Vec<Field>, Vec<ArrayRef>) {
        let (slots, _, _) = cfg;
        let mut fields = Vec::with_capacity(N);
        let mut arrays = Vec::with_capacity(N);
        for (s, col) in cols.into_iter().enumerate() {
            let name = format!("v{s}");
            // Count and Sum both render Int64 — identical to the additive `Dynamic`.
            let (f, a) = match slots[s].kind {
                AggregationKind::CountStar | AggregationKind::Count => {
                    Count::<i64>::finish(&name, col, arena)
                }
                AggregationKind::Sum => Sum::<i64>::finish(&name, col, arena),
                k => panic!("Patched is additive-only, got {k:?}"),
            };
            fields.push(f);
            arrays.push(a);
        }
        (fields, arrays)
    }
}
