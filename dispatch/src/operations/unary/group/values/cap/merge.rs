//! Additive merge **tile** and its copy-and-patch compiler.
//!
//! Partition-merge counterpart to [`fold`](super::fold): combine two `[i64; N]`
//! cell arrays slot-by-slot. For the additive ops (`COUNT`/`SUM`) the combine is a
//! plain add, so one tile (`tgt[s] += src[s]`) repeated per slot covers the whole
//! signature — branch-free, no `slots[s].kind` load (the interpreted `Dynamic`
//! merge's cost). Called once per matched cell from the existing Rust probe (which
//! handles the multi-slab table, overflow and radix); on the memory-bound merge
//! the call hides behind the target cache miss.
//!
//! Convention: `x0 = tgt: *mut i64`, `x1 = src: *const i64`.

#![cfg(target_arch = "aarch64")]

use super::exec::{ExecBuffer, RET, TERMINATOR, stencil_words};

/// `tgt[#OFF] += src[#OFF]` — words 0,1,3 carry the slot-offset hole.
#[unsafe(naked)]
unsafe extern "C" fn tile_add() {
    std::arch::naked_asm!(
        "ldr x11, [x1]", // src[s]
        "ldr x10, [x0]", // tgt[s]
        "add x10, x10, x11",
        "str x10, [x0]", // tgt[s]
        "brk #0xCAFE",
    );
}

/// Set the `imm12` (bits [21:10]) of a load/store word to `v`.
#[inline]
fn patch_imm12(word: u32, v: u32) -> u32 {
    (word & !(0xFFF << 10)) | ((v & 0xFFF) << 10)
}

/// The patched add-tile sequence for `n` slots (`tgt[s] += src[s]`), no terminator
/// — ready to splice into the whole-loop merge skeleton's gap.
pub fn merge_body_words(n: usize) -> Vec<u32> {
    let body = unsafe { stencil_words(tile_add as *const () as usize, TERMINATOR) };
    let body = &body[..body.len() - 1];
    let mut words: Vec<u32> = Vec::with_capacity(body.len() * n);
    for s in 0..n {
        let base = words.len();
        words.extend_from_slice(body);
        for &h in &[0usize, 1, 3] {
            words[base + h] = patch_imm12(words[base + h], s as u32);
        }
    }
    words
}

/// The compiled per-cell merge: `fn(tgt, src)` folding `src` into `tgt`.
pub type MergePairFn = unsafe extern "C" fn(*mut i64, *const i64);

/// A finalized merge and the buffer owning its code.
pub struct CompiledMergePair {
    f: MergePairFn,
    _buf: ExecBuffer,
}
impl CompiledMergePair {
    #[inline(always)]
    pub unsafe fn run(&self, tgt: *mut i64, src: *const i64) {
        unsafe { (self.f)(tgt, src) }
    }
}

/// Compile the additive merge for `n` slots (`n` add-tiles, each patched to its
/// index, then `ret`). `None` if the buffer can't be mapped.
pub fn compile_merge_pair(n: usize) -> Option<CompiledMergePair> {
    let body = unsafe { stencil_words(tile_add as *const () as usize, TERMINATOR) };
    let body = &body[..body.len() - 1]; // drop terminator
    let mut words: Vec<u32> = Vec::with_capacity(body.len() * n + 1);
    for s in 0..n {
        let base = words.len();
        words.extend_from_slice(body);
        for &h in &[0usize, 1, 3] {
            words[base + h] = patch_imm12(words[base + h], s as u32);
        }
    }
    words.push(RET);
    let bytes =
        unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
    let buf = ExecBuffer::new(bytes)?;
    let f: MergePairFn = unsafe { std::mem::transmute(buf.entry()) };
    Some(CompiledMergePair { f, _buf: buf })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_adds_cells() {
        let m = compile_merge_pair(3).expect("compile");
        let mut tgt = [1i64, 10, 100];
        let src = [2i64, 20, 200];
        unsafe { m.run(tgt.as_mut_ptr(), src.as_ptr()) };
        assert_eq!(tgt, [3, 30, 300]);
    }
}
