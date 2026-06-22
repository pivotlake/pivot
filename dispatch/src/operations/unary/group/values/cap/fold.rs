//! Additive fold **tiles** and a copy-and-patch compiler for the per-row fold.
//!
//! Each tile is one slot's `cell[s] += <value>` as pre-compiled machine code with
//! the slot index left as a hole. For the additive ops that cover ClickBench's
//! grouped numerics (`COUNT`/`SUM`), a freshly inserted cell is zeroed, so `+=`
//! serves both seed and update — one uniform tile sequence, no `is_new`, no
//! branches.
//!
//! Calling convention the tiles assume (AArch64 AAPCS):
//! ```text
//!   x0 = cell  : *mut i64   (base of this group's [i64; N] cell array)
//!   x1 = cols  : *const *const u8   (N column base pointers, slot order)
//!   x2 = row   : u64        (row index into the columns)
//! ```
//! For slot `s`: the cell lives at `x0 + s*8`, its column base at `cols[s] = x1 +
//! s*8`. Both are 64-bit unsigned-offset loads, so both encode to `imm12 = s`
//! (the field is scaled by the 8-byte access size) — a single patch value per slot.

#![cfg(target_arch = "aarch64")]

use super::FoldOp;
use super::exec::{ExecBuffer, RET, TERMINATOR, stencil_words};

/// `cell[#OFF] += 1` — words 0 and 2 carry the cell-offset hole.
#[unsafe(naked)]
unsafe extern "C" fn tile_count() {
    std::arch::naked_asm!("ldr x10, [x0]", "add x10, x10, #1", "str x10, [x0]", "brk #0xCAFE");
}

/// `cell[#OFF] += sext16(cols[#COL][row])` — word 0 = col hole, words 2,4 = cell hole.
#[unsafe(naked)]
unsafe extern "C" fn tile_sum_i16() {
    std::arch::naked_asm!(
        "ldr   x9,  [x1]",            // x9 = cols[s]
        "ldrsh x11, [x9, x2, lsl #1]", // x11 = sext i16 at row
        "ldr   x10, [x0]",
        "add   x10, x10, x11",
        "str   x10, [x0]",
        "brk #0xCAFE",
    );
}

#[unsafe(naked)]
unsafe extern "C" fn tile_sum_i32() {
    std::arch::naked_asm!(
        "ldr   x9,  [x1]",
        "ldrsw x11, [x9, x2, lsl #2]",
        "ldr   x10, [x0]",
        "add   x10, x10, x11",
        "str   x10, [x0]",
        "brk #0xCAFE",
    );
}

#[unsafe(naked)]
unsafe extern "C" fn tile_sum_i64() {
    std::arch::naked_asm!(
        "ldr x9,  [x1]",
        "ldr x11, [x9, x2, lsl #3]",
        "ldr x10, [x0]",
        "add x10, x10, x11",
        "str x10, [x0]",
        "brk #0xCAFE",
    );
}

/// The tile entry plus the indices of its words that hold a patchable `imm12`
/// (all patched to the slot index `s`).
struct Tile {
    f: usize,
    holes: &'static [usize],
}

impl FoldOp {
    fn tile(self) -> Tile {
        match self {
            FoldOp::Count => Tile { f: tile_count as *const () as usize, holes: &[0, 2] },
            FoldOp::SumI16 => Tile { f: tile_sum_i16 as *const () as usize, holes: &[0, 2, 4] },
            FoldOp::SumI32 => Tile { f: tile_sum_i32 as *const () as usize, holes: &[0, 2, 4] },
            FoldOp::SumI64 => Tile { f: tile_sum_i64 as *const () as usize, holes: &[0, 2, 4] },
        }
    }
}

/// Set the `imm12` (bits [21:10]) of a load/store word to `v`.
#[inline]
fn patch_imm12(word: u32, v: u32) -> u32 {
    (word & !(0xFFF << 10)) | ((v & 0xFFF) << 10)
}

/// The compiled per-row fold: `fn(cell, cols, row)`, folding one row's columns
/// into one group's cells, branch-free, fully unrolled over the signature.
pub type FoldRowFn = unsafe extern "C" fn(*mut i64, *const *const u8, u64);

/// A finalized per-row fold and the buffer that owns its code.
pub struct CompiledFoldRow {
    f: FoldRowFn,
    _buf: ExecBuffer,
}
impl CompiledFoldRow {
    #[inline(always)]
    pub unsafe fn run(&self, cell: *mut i64, cols: *const *const u8, row: u64) {
        unsafe { (self.f)(cell, cols, row) }
    }
}

/// The patched fold tile sequence for `ops` (each slot's tile, holes set to its
/// index), no terminator — ready to fall through. Used standalone (with a `ret`,
/// [`compile_fold_row`]) and spliced into the consume skeleton's gap.
pub fn fold_body_words(ops: &[FoldOp]) -> Vec<u32> {
    let mut words: Vec<u32> = Vec::new();
    for (s, op) in ops.iter().enumerate() {
        let tile = op.tile();
        // SAFETY: each tile is a naked fn ending in TERMINATOR.
        let body = unsafe { stencil_words(tile.f, TERMINATOR) };
        let body = &body[..body.len() - 1]; // drop terminator → fall through
        let base = words.len();
        words.extend_from_slice(body);
        for &h in tile.holes {
            words[base + h] = patch_imm12(words[base + h], s as u32);
        }
    }
    words
}

/// Stitch the additive `ops` into a callable per-row fold (the tile body + `ret`).
/// `None` if the buffer can't be mapped.
pub fn compile_fold_row(ops: &[FoldOp]) -> Option<CompiledFoldRow> {
    let mut words: Vec<u32> = fold_body_words(ops);
    words.push(RET);
    let bytes =
        unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
    let buf = ExecBuffer::new(bytes)?;
    let f: FoldRowFn = unsafe { std::mem::transmute(buf.entry()) };
    Some(CompiledFoldRow { f, _buf: buf })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_sum_per_row_matches_scalar() {
        // signature: COUNT(*), SUM(i32 col)
        let fold = compile_fold_row(&[FoldOp::Count, FoldOp::SumI32]).expect("compile");
        let col: Vec<i32> = vec![10, 20, 30];
        let cols: Vec<*const u8> = vec![std::ptr::null(), col.as_ptr() as *const u8];

        let mut g = [0i64; 2];
        // fold rows 0,2 into the same group
        unsafe {
            fold.run(g.as_mut_ptr(), cols.as_ptr(), 0);
            fold.run(g.as_mut_ptr(), cols.as_ptr(), 2);
        }
        assert_eq!(g, [2, 40]); // count 2, sum 10+30
    }

    #[test]
    fn three_slots_sum_i16() {
        // q09-inner shape: SUM(i16), COUNT, SUM(i16)
        let fold =
            compile_fold_row(&[FoldOp::SumI16, FoldOp::Count, FoldOp::SumI16]).expect("compile");
        let a: Vec<i16> = vec![3, 5];
        let b: Vec<i16> = vec![7, 11];
        let cols: Vec<*const u8> =
            vec![a.as_ptr() as *const u8, std::ptr::null(), b.as_ptr() as *const u8];

        let mut g = [0i64; 3];
        unsafe {
            fold.run(g.as_mut_ptr(), cols.as_ptr(), 0);
            fold.run(g.as_mut_ptr(), cols.as_ptr(), 1);
        }
        assert_eq!(g, [8, 2, 18]); // sum a=3+5, count=2, sum b=7+11
    }
}
