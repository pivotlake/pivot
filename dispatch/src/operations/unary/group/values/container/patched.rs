//! [`Patched`] — an additive numeric value whose per-batch fold runs as a single
//! copy-and-patch machine-code loop.
//!
//! Same `[i64; N]` cells as the additive [`Dynamic`](super::Dynamic), but instead
//! of dispatching on each slot's kind for every row, the consume path assembles a
//! signature-specialised loop once (see [`cap`](super::super::cap)) and folds a
//! whole batch in one pass with no per-row call — driven by
//! [`batch_fold`](AggregationValue::batch_fold), which the table calls after
//! probing a run of rows to their cells.
//!
//! The merge phase needs no JIT: merging additive partials is an elementwise add,
//! and `Patched<N>`'s `const N` lets the compiler unroll [`merge`](Patched::merge)
//! per signature. Off aarch64 (or for a signature the stencils don't cover)
//! `batch_fold` returns `None` and the per-row scalar fold runs, so `Patched` is
//! always correct — JIT-accelerated where it can be.

use super::super::cap::{self, CompiledFold};
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

/// `N` additive `i64` cells. `repr(transparent)` so a `*mut Self` is exactly the
/// `*mut i64` cell base the assembled fold loop writes through.
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

/// Per-batch reader: each slot's column base pointer and element byte width (`0`
/// for `COUNT`). The widths drive the one-time fold compile and the scalar fallback.
pub struct PatchedReader<const N: usize> {
    cols: [*const u8; N],
    widths: [u8; N],
}

/// `(slots, value-arena, query-lifetime compiled fold)`. The fold needs the column
/// widths, known only at the first batch, so it's compiled then via the `OnceLock`
/// (shared across workers — assembled once, its executable page reused by all).
pub type PatchedConfig = (
    Arc<[AggregationSlot]>,
    Arc<SharedArena>,
    Arc<OnceLock<Option<CompiledFold>>>,
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

/// Scalar fold of row `idx` into `cells` — the correctness fallback when no loop
/// was assembled (non-aarch64, or an uncovered signature).
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

/// Assemble the fold loop for `slots` at these column widths (`None` if any slot
/// isn't an additive numeric the stencils cover).
fn compile_fold_for<const N: usize>(
    slots: &[AggregationSlot],
    widths: &[u8; N],
) -> Option<CompiledFold> {
    let mut cap_slots = Vec::with_capacity(N);
    for s in 0..N {
        let op = cap::FoldOp::from_slot(slots[s].kind, widths[s])?;
        // The loop is handed `reader.cols` (one base per slot, indexed 0..N), so a
        // slot reads column index `s`; its cell is the `s`-th `i64`.
        cap_slots.push(cap::Slot {
            op,
            cell_offset: (s * 8) as u32,
            column: s as u32,
        });
    }
    cap::compile_fold(&cap_slots)
}

impl<const N: usize> AggregationValue for Patched<N> {
    type Reader<'b> = PatchedReader<N>;
    type MergeConfig = PatchedConfig;
    type Columns = [SlabColumn<i64>; N];
    type SortKey = i128;

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

    fn batch_fold(reader: &PatchedReader<N>, cfg: &Self::MergeConfig) -> Option<cap::BatchFold> {
        let (slots, _, cache) = cfg;
        let compiled = cache.get_or_init(|| compile_fold_for::<N>(slots, &reader.widths));
        let func = compiled.as_ref()?.func();
        Some(cap::BatchFold::new(func, reader.cols.to_vec(), reader.widths.to_vec()))
    }

    /// Materialise a new group from one row — the radix scatter and the scalar
    /// consume fallback; the JIT consume path seeds zeroes and folds in bulk.
    #[inline(always)]
    fn value(reader: &PatchedReader<N>, idx: usize, _arena: &mut WorkerArena) -> Self {
        let mut cells = [0i64; N];
        scalar_fold(&mut cells, reader, idx);
        Self { cells }
    }

    #[inline(always)]
    fn update_from_reader(
        mut self,
        reader: &PatchedReader<N>,
        idx: usize,
        _arena: &mut WorkerArena,
        _cfg: &Self::MergeConfig,
    ) -> Self {
        scalar_fold(&mut self.cells, reader, idx);
        self
    }

    /// Merge two additive partials — elementwise add. `const N` lets the compiler
    /// unroll this, so the partition merge and radix fold need no JIT.
    #[inline(always)]
    fn merge(mut self, other: Self, _cfg: &Self::MergeConfig) -> Self {
        for s in 0..N {
            self.cells[s] += other.cells[s];
        }
        self
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
