//! Copy-and-patch compilation of the GROUP BY value fold.
//!
//! Instead of interpreting a runtime aggregate signature per row (the
//! [`Dynamic`](super::container::Dynamic) `match`) or monomorphising it ahead of
//! time (the [`Compiled`](super::container::Compiled) tuple — which can't cover an
//! arbitrary runtime shape), we assemble the inlined fold/merge at plan time by
//! copying pre-compiled machine-code *tiles* into an [`exec::ExecBuffer`] and
//! patching their holes (the slot index). The result is the same branch-free,
//! `slots[]`-load-free machine code `Compiled` emits, produced at runtime for a
//! signature the type system never saw.
//!
//! Scope: the additive ops that cover ClickBench's grouped numerics
//! (`COUNT`/`SUM`), where a zeroed-on-insert cell makes `+=` serve both seed and
//! update — one uniform tile sequence, no `is_new`, no branches. Other ops / arches
//! return `None` from [`compile_additive`] and the caller folds via the interpreter.

/// One additive fold slot, reduced to the tile it needs. Arch-independent so the
/// planner can build a signature anywhere; the tiles themselves are aarch64-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldOp {
    /// `COUNT` — `cell[s] += 1`, reads no column.
    Count,
    /// `SUM` over a 2/4/8-byte signed integer column.
    SumI16,
    SumI32,
    SumI64,
}

impl FoldOp {
    /// Map a planner slot (`kind` + the column's element width in bytes) to a tile,
    /// or `None` if it isn't an additive op the tiles cover.
    pub fn from_slot(kind: super::AggregationKind, col_width: u8) -> Option<FoldOp> {
        use super::AggregationKind::*;
        match kind {
            CountStar | Count => Some(FoldOp::Count),
            Sum => match col_width {
                2 => Some(FoldOp::SumI16),
                4 => Some(FoldOp::SumI32),
                8 => Some(FoldOp::SumI64),
                _ => None,
            },
            Min | Max | StrMin | StrMax => None,
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod consume;
#[cfg(target_arch = "aarch64")]
mod exec;
#[cfg(target_arch = "aarch64")]
mod fold;
#[cfg(target_arch = "aarch64")]
mod merge;
#[cfg(target_arch = "aarch64")]
mod merge_run;
#[cfg(target_arch = "aarch64")]
mod seed;
#[cfg(target_arch = "aarch64")]
pub use merge_run::{CompiledMergeRun, MergeCtx, compile_merge_run};

/// The additive merge loop for `n` slots, compiled **once** and cached — the merge
/// runs per partition, so re-`mmap`-ing it each call floods the kernel (page faults
/// dominated a profile). `merge_run` depends only on `n` (offsets come via the ctx
/// at run time), so one compile per arity serves every partition/query/worker.
#[cfg(target_arch = "aarch64")]
pub fn cached_merge_run(n: usize) -> Option<&'static CompiledMergeRun> {
    use std::sync::OnceLock;
    const MAX: usize = 7;
    static CACHE: [OnceLock<Option<CompiledMergeRun>>; MAX] = [const { OnceLock::new() }; MAX];
    if n >= MAX {
        return None;
    }
    CACHE[n]
        .get_or_init(|| compile_merge_run(&vec![FoldOp::Count; n]))
        .as_ref()
}
#[cfg(not(target_arch = "aarch64"))]
pub fn cached_merge_run(_n: usize) -> Option<&'static CompiledMergeRun> {
    None
}
#[cfg(not(target_arch = "aarch64"))]
pub struct CompiledMergeRun(());
#[cfg(target_arch = "aarch64")]
pub use seed::{CompiledSeed, compile_seed};

/// The whole-loop probe+fold consume function (aarch64); a stub elsewhere so the
/// caller's `Option<CompiledConsume>` typing is uniform across targets.
#[cfg(target_arch = "aarch64")]
pub use consume::{CompiledConsume, ConsumeCtx, compile_consume};
#[cfg(not(target_arch = "aarch64"))]
pub struct CompiledConsume(());
#[cfg(not(target_arch = "aarch64"))]
pub fn compile_consume(_ops: &[FoldOp]) -> Option<CompiledConsume> {
    None
}

/// A compiled additive value: the per-row fold and the per-cell merge, branch-free
/// machine code for one signature. Built once per query (`compile_additive`), then
/// called from the hot consume/merge loops.
#[cfg(target_arch = "aarch64")]
pub struct AdditiveCompiled {
    fold: fold::CompiledFoldRow,
    merge: merge::CompiledMergePair,
}

#[cfg(target_arch = "aarch64")]
impl AdditiveCompiled {
    /// Fold one input row's columns into one group's cells (`cell[s] += …`).
    #[inline(always)]
    pub unsafe fn fold_row(&self, cell: *mut i64, cols: *const *const u8, row: u64) {
        unsafe { self.fold.run(cell, cols, row) }
    }
    /// Combine two partial cell arrays (`tgt[s] += src[s]`).
    #[inline(always)]
    pub unsafe fn merge_pair(&self, tgt: *mut i64, src: *const i64) {
        unsafe { self.merge.run(tgt, src) }
    }
}

/// Compile the additive `ops` to native fold + merge, or `None` to fall back to the
/// interpreter (non-additive signature, or buffers can't be mapped).
#[cfg(target_arch = "aarch64")]
pub fn compile_additive(ops: &[FoldOp]) -> Option<AdditiveCompiled> {
    Some(AdditiveCompiled {
        fold: fold::compile_fold_row(ops)?,
        merge: merge::compile_merge_pair(ops.len())?,
    })
}

// ---- non-aarch64: no tiles; everything falls back to the interpreter ----
#[cfg(not(target_arch = "aarch64"))]
pub struct AdditiveCompiled(());
#[cfg(not(target_arch = "aarch64"))]
impl AdditiveCompiled {
    #[inline(always)]
    pub unsafe fn fold_row(&self, _: *mut i64, _: *const *const u8, _: u64) {
        unreachable!()
    }
    #[inline(always)]
    pub unsafe fn merge_pair(&self, _: *mut i64, _: *const i64) {
        unreachable!()
    }
}
#[cfg(not(target_arch = "aarch64"))]
pub fn compile_additive(_ops: &[FoldOp]) -> Option<AdditiveCompiled> {
    None
}
