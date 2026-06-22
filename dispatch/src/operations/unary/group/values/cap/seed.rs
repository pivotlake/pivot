//! Batch value materialisation for the radix scatter (no probe).
//!
//! On the high-cardinality path each input row scatters its *own* partial value
//! into a radix buffer; the interpreted path calls `value` per row. This compiles
//! one loop that materialises a whole batch's values into a temp `[[i64; N]; n]`
//! array — the fold tiles (`cell[s] += …`) over a zeroed cell, so the result is
//! the row's contribution. The scatter then reads the temp array, no per-row call.
//!
//! Signature: `fn(temp: *mut i64, cols: *const *const u8, n, vstride)` where
//! `vstride = N*8` is the per-row cell stride in the temp array.

#![cfg(target_arch = "aarch64")]

use super::FoldOp;
use super::exec::{ExecBuffer, RET};

const GAP_START: u32 = 0xD435_5420; // brk #0xAAA1
const GAP_END: u32 = 0xD435_5440; // brk #0xAAA2
const NOP: u32 = 0xD503_201F;

/// `fn(temp, cols, n, vstride, start)` — temp must be zeroed by the caller (`+=`
/// tiles). Writes `temp[i]` for `i in 0..n`, reading the columns at row `start+i`.
#[unsafe(naked)]
unsafe extern "C" fn skel_seed() {
    std::arch::naked_asm!(
        "stp x19, x20, [sp, #-48]!",
        "stp x21, x22, [sp, #16]",
        "stp x23, x24, [sp, #32]",
        "mov x19, x0",  // temp base
        "mov x20, x1",  // cols
        "mov x21, x2",  // n
        "mov x22, x3",  // vstride (bytes per row's cells)
        "mov x23, x4",  // start (column row offset)
        "mov x12, #0",  // i = 0 (temp slot)
        "0:",
        "cmp x12, x21",
        "b.ge 4f",
        "madd x0, x12, x22, x19", // cell = temp + i*vstride
        "mov x1, x20",            // cols
        "add x2, x12, x23",       // row = start + i
        "brk #0xAAA1",            // fold gap
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "brk #0xAAA2",
        "add x12, x12, #1",
        "b 0b",
        "4:",
        "ldp x23, x24, [sp, #32]",
        "ldp x21, x22, [sp, #16]",
        "ldp x19, x20, [sp], #48",
        "ret",
    );
}

/// `fn(temp, cols, n, vstride, start)`.
pub type SeedFn = unsafe extern "C" fn(*mut i64, *const *const u8, u64, u64, u64);

pub struct CompiledSeed {
    f: SeedFn,
    _buf: ExecBuffer,
}
impl CompiledSeed {
    #[inline(always)]
    pub unsafe fn run(
        &self,
        temp: *mut i64,
        cols: *const *const u8,
        n: u64,
        vstride: u64,
        start: u64,
    ) {
        unsafe { (self.f)(temp, cols, n, vstride, start) }
    }
}

/// Splice the additive `ops` fold into the seed loop. `None` if it won't fit.
pub fn compile_seed(ops: &[FoldOp]) -> Option<CompiledSeed> {
    let skel = unsafe { super::exec::stencil_words(skel_seed as *const () as usize, RET) };
    let mut words: Vec<u32> = skel.to_vec();
    let gstart = words.iter().position(|&w| w == GAP_START)?;
    let gend = words.iter().position(|&w| w == GAP_END)?;
    let fold = super::fold::fold_body_words(ops);
    if fold.len() > gend - gstart + 1 {
        return None;
    }
    for (k, slot) in (gstart..=gend).enumerate() {
        words[slot] = if k < fold.len() { fold[k] } else { NOP };
    }
    let bytes =
        unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
    let buf = ExecBuffer::new(bytes)?;
    let f: SeedFn = unsafe { std::mem::transmute(buf.entry()) };
    Some(CompiledSeed { f, _buf: buf })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_batch_materialises_per_row_values() {
        // COUNT(*), SUM(i32) over 3 rows → temp[i] = [1, col[i]].
        let seed = compile_seed(&[FoldOp::Count, FoldOp::SumI32]).expect("compile");
        let col: Vec<i32> = vec![10, 20, 30];
        let cols: Vec<*const u8> = vec![std::ptr::null(), col.as_ptr() as *const u8];
        let mut temp = vec![0i64; 3 * 2];
        unsafe { seed.run(temp.as_mut_ptr(), cols.as_ptr(), 3, (2 * 8) as u64, 0) };
        assert_eq!(temp, vec![1, 10, 1, 20, 1, 30]);
        // start=1 → temp[0] reads row 1.
        let mut t2 = vec![0i64; 2 * 2];
        unsafe { seed.run(t2.as_mut_ptr(), cols.as_ptr(), 2, 16, 1) };
        assert_eq!(t2, vec![1, 20, 1, 30]);
    }
}
