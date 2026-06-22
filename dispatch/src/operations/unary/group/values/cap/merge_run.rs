//! Whole-loop copy-and-patch for the partition **merge** with a 16-byte (int-pair)
//! key — the path q32/q30/q31's high-cardinality merge takes.
//!
//! The merge driver hands a contiguous run of scattered source rows
//! `(hash, key, value)`; this owns the whole loop: for each source row it probes
//! the single-slab target (slot `(hash<<pre_shift)>>shift`, 16-byte key compare),
//! inserts on an empty slot, then adds the source value into the target cells via
//! the add-tiles spliced into its gap. No per-element call, no `slots[]` load.
//!
//! Layout-agnostic: every offset (source `hash`/`key`/`value`, target `key`/`value`,
//! strides, mask, shift, pre_shift) comes in via [`MergeCtx`], read from the ctx
//! pointer per use (cache-hot, hidden behind the memory-bound probe) to keep the
//! register file small. Returns rows-done (`< n` ⇒ the target filled; the caller
//! grows it and re-enters).

#![cfg(target_arch = "aarch64")]

use super::FoldOp;
use super::exec::{ExecBuffer, RET, stencil_words};

/// Passed by pointer in `x0`; `#[repr(C)]` for stable field offsets.
#[repr(C)]
pub struct MergeCtx {
    pub tbuf: *mut u8,       // 0   target entry array (single slab, zeroed)
    pub tstride: u64,        // 8   target entry size
    pub tvaloff: u64,        // 16  target value offset
    pub tkeyoff: u64,        // 24  target key offset
    pub mask: u64,           // 32  capacity-1
    pub shift: u64,          // 40
    pub pre_shift: u64,      // 48  partition pre-shift
    pub tlen: *mut u64,      // 56  target length (bumped on insert)
    pub max_load: u64,       // 64  stop inserting past this
    pub sbase: *const u8,    // 72  source row array
    pub sstride: u64,        // 80  source row size
    pub shashoff: u64,       // 88  source hash offset
    pub skeyoff: u64,        // 96  source key offset
    pub svaloff: u64,        // 104 source value offset
    pub n: u64,              // 112 source row count
    pub start: u64,          // 120 first source row (re-call after a grow)
}

const GAP_START: u32 = 0xD435_5420; // brk #0xAAA1
const GAP_END: u32 = 0xD435_5440; // brk #0xAAA2
const NOP: u32 = 0xD503_201F;

#[unsafe(naked)]
unsafe extern "C" fn skel_merge_key16() {
    std::arch::naked_asm!(
        // Hoist the hot fields into callee-saved regs so the probe inner loop is
        // register-only (no per-iter ctx reloads); the rare per-row offsets are read
        // from the ctx ptr (x19), and only on a hash match / insert.
        "stp x19, x20, [sp, #-80]!",
        "stp x21, x22, [sp, #16]",
        "stp x23, x24, [sp, #32]",
        "stp x25, x26, [sp, #48]",
        "stp x27, x28, [sp, #64]",
        "mov x19, x0",          // ctx
        "ldr x20, [x19, #120]", // i = start
        "ldr x21, [x19, #0]",   // tbuf
        "ldr x22, [x19, #8]",   // tstride
        "ldr x23, [x19, #32]",  // mask
        "ldr x24, [x19, #40]",  // shift
        "ldr x25, [x19, #48]",  // pre_shift
        "ldr x26, [x19, #72]",  // sbase
        "ldr x27, [x19, #80]",  // sstride
        "ldr x28, [x19, #56]",  // tlen ptr
        "0:",                   // row loop
        "ldr x9,  [x28]",       // *tlen
        "ldr x10, [x19, #64]",  // max_load
        "cmp x9, x10",
        // strict `>` matches `undersized()` so the caller's grow_if_full fires when
        // we stop (a `>=` stalls at len == max_load: we stop but it won't grow).
        "b.gt 4f",
        "ldr x10, [x19, #112]", // n
        "cmp x20, x10",
        "b.ge 4f",              // done
        "madd x12, x20, x27, x26",  // src row ptr (x12)
        "ldr x9,  [x19, #88]",  // shashoff
        "ldr x13, [x12, x9]",   // x13 = src.hash
        "lsl x14, x13, x25",    // slot = (hash << pre_shift) >> shift
        "lsr x14, x14, x24",    // x14 = slot
        "1:",                   // probe (register-only on the common path)
        "madd x15, x14, x22, x21", // tentry = tbuf + slot*tstride (x15)
        "ldr x16, [x15]",       // target.hash
        "cbz x16, 2f",          // empty -> insert
        "cmp x16, x13",
        "b.ne 3f",              // hash mismatch -> next
        // 16-byte key compare (only on a hash match)
        "ldr x9,  [x19, #24]",  // tkeyoff
        "add x16, x15, x9",     // target key ptr
        "ldr x10, [x19, #96]",  // skeyoff
        "add x17, x12, x10",    // src key ptr
        "ldr x9,  [x16]",
        "ldr x10, [x17]",
        "cmp x9, x10",
        "b.ne 3f",
        "ldr x9,  [x16, #8]",
        "ldr x10, [x17, #8]",
        "cmp x9, x10",
        "b.ne 3f",
        // match: x0 = target value
        "ldr x9, [x19, #16]",   // tvaloff
        "add x0, x15, x9",
        "b 5f",
        "2:",                   // insert (empty)
        "str x13, [x15]",       // target.hash = src.hash
        "ldr x9,  [x19, #24]",  // tkeyoff
        "add x16, x15, x9",
        "ldr x10, [x19, #96]",  // skeyoff
        "add x17, x12, x10",
        "ldr x9,  [x17]",
        "str x9,  [x16]",
        "ldr x9,  [x17, #8]",
        "str x9,  [x16, #8]",
        "ldr x9,  [x28]",
        "add x9, x9, #1",
        "str x9,  [x28]",       // *tlen += 1
        "ldr x9, [x19, #16]",   // tvaloff
        "add x0, x15, x9",      // x0 = target value (zeroed → add = copy)
        "b 5f",
        "3:",                   // next slot
        "add x14, x14, #1",
        "and x14, x14, x23",
        "b 1b",
        "5:",                   // merge: x0 = target value, x1 = src value
        "ldr x9, [x19, #104]",  // svaloff
        "add x1, x12, x9",
        "brk #0xAAA1",          // add-tile gap
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "brk #0xAAA2",
        "add x20, x20, #1",
        "b 0b",
        "4:",
        "mov x0, x20",          // return rows done
        "ldp x27, x28, [sp, #64]",
        "ldp x25, x26, [sp, #48]",
        "ldp x23, x24, [sp, #32]",
        "ldp x21, x22, [sp, #16]",
        "ldp x19, x20, [sp], #80",
        "ret",
    );
}

/// `fn(ctx: *mut MergeCtx) -> rows_done`.
pub type MergeRunFn = unsafe extern "C" fn(*mut MergeCtx) -> u64;

pub struct CompiledMergeRun {
    f: MergeRunFn,
    _buf: ExecBuffer,
}
impl CompiledMergeRun {
    #[inline(always)]
    pub unsafe fn run(&self, ctx: *mut MergeCtx) -> u64 {
        unsafe { (self.f)(ctx) }
    }
}

/// Splice the additive `ops` add-tiles into the key16 merge skeleton.
pub fn compile_merge_run(ops: &[FoldOp]) -> Option<CompiledMergeRun> {
    let skel = unsafe { stencil_words(skel_merge_key16 as *const () as usize, RET) };
    let mut words: Vec<u32> = skel.to_vec();
    let gstart = words.iter().position(|&w| w == GAP_START)?;
    let gend = words.iter().position(|&w| w == GAP_END)?;
    let body = super::merge::merge_body_words(ops.len());
    if body.len() > gend - gstart + 1 {
        return None;
    }
    for (k, slot) in (gstart..=gend).enumerate() {
        words[slot] = if k < body.len() { body[k] } else { NOP };
    }
    let bytes =
        unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
    let buf = ExecBuffer::new(bytes)?;
    let f: MergeRunFn = unsafe { std::mem::transmute(buf.entry()) };
    Some(CompiledMergeRun { f, _buf: buf })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Source rows are (u128 key, [count,sum] value) with an explicit hash; merge
    // into a target table and compare to a HashMap reference.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Row {
        hash: u64,
        key: u128,
        val: [i64; 2],
    }

    // The real target entry layout (repr(C), u128 key forces 16-align: size 48).
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct TEntry {
        hash: u64,
        key: u128,
        val: [i64; 2],
    }

    #[test]
    fn merge_key16_matches_reference() {
        let cap = 1024usize;
        let tstride = std::mem::size_of::<TEntry>() as u64;
        let tkeyoff = std::mem::offset_of!(TEntry, key) as u64;
        let tvaloff = std::mem::offset_of!(TEntry, val) as u64;
        let mut tbuf = vec![0u8; cap * tstride as usize];
        let mut tlen = 0u64;

        let keys: Vec<u128> = vec![5, 9, 5, 9, 5, 7];
        let vals: Vec<[i64; 2]> = vec![[1, 10], [1, 1], [1, 20], [1, 2], [1, 30], [1, 100]];
        let rows: Vec<Row> = keys
            .iter()
            .zip(&vals)
            .map(|(&k, &v)| Row {
                hash: (k as u64).wrapping_mul(0x9E3779B97F4A7C15).max(1),
                key: k,
                val: v,
            })
            .collect();

        let mut ctx = MergeCtx {
            tbuf: tbuf.as_mut_ptr(),
            tstride,
            tvaloff,
            tkeyoff,
            mask: (cap - 1) as u64,
            shift: (cap as u64).leading_zeros() as u64 + 1,
            pre_shift: 0,
            tlen: &mut tlen,
            max_load: cap as u64,
            sbase: rows.as_ptr() as *const u8,
            sstride: std::mem::size_of::<Row>() as u64,
            shashoff: std::mem::offset_of!(Row, hash) as u64,
            skeyoff: std::mem::offset_of!(Row, key) as u64,
            svaloff: std::mem::offset_of!(Row, val) as u64,
            n: rows.len() as u64,
            start: 0,
        };
        let compiled = compile_merge_run(&[FoldOp::Count, FoldOp::SumI64]).expect("compile");
        let done = unsafe { compiled.run(&mut ctx) };
        assert_eq!(done, rows.len() as u64);

        let mut got: HashMap<u128, [i64; 2]> = HashMap::new();
        for slot in 0..cap {
            let e = unsafe { tbuf.as_ptr().add(slot * tstride as usize) };
            let h = unsafe { (e as *const u64).read() };
            if h == 0 {
                continue;
            }
            let k = unsafe { (e.add(tkeyoff as usize) as *const u128).read_unaligned() };
            let v0 = unsafe { (e.add(tvaloff as usize) as *const i64).read() };
            let v1 = unsafe { (e.add(tvaloff as usize + 8) as *const i64).read() };
            got.insert(k, [v0, v1]);
        }
        let mut want: HashMap<u128, [i64; 2]> = HashMap::new();
        for (k, v) in keys.iter().zip(&vals) {
            let e = want.entry(*k).or_insert([0, 0]);
            e[0] += v[0];
            e[1] += v[1];
        }
        assert_eq!(got, want);
        assert_eq!(tlen, want.len() as u64);
    }

    #[test]
    fn merge_returns_rows_done_on_overflow() {
        // max_load=1 → stops once len > max_load (strict), returning rows done so the
        // caller grows and re-enters. Rows: key 5 (×2), then key 9.
        let cap = 64usize;
        let tstride = std::mem::size_of::<TEntry>() as u64;
        let keys: Vec<u128> = vec![5, 5, 9, 11];
        let rows: Vec<Row> = keys
            .iter()
            .map(|&k| Row {
                hash: (k as u64).wrapping_mul(0x9E3779B97F4A7C15).max(1),
                key: k,
                val: [1, 0],
            })
            .collect();
        let mut tbuf = vec![0u8; cap * tstride as usize];
        let mut tlen = 0u64;
        let mut ctx = MergeCtx {
            tbuf: tbuf.as_mut_ptr(),
            tstride,
            tvaloff: std::mem::offset_of!(TEntry, val) as u64,
            tkeyoff: std::mem::offset_of!(TEntry, key) as u64,
            mask: (cap - 1) as u64,
            shift: (cap as u64).leading_zeros() as u64 + 1,
            pre_shift: 0,
            tlen: &mut tlen,
            max_load: 1,
            sbase: rows.as_ptr() as *const u8,
            sstride: std::mem::size_of::<Row>() as u64,
            shashoff: std::mem::offset_of!(Row, hash) as u64,
            skeyoff: std::mem::offset_of!(Row, key) as u64,
            svaloff: std::mem::offset_of!(Row, val) as u64,
            n: rows.len() as u64,
            start: 0,
        };
        let compiled = compile_merge_run(&[FoldOp::Count, FoldOp::SumI64]).expect("compile");
        let done = unsafe { compiled.run(&mut ctx) };
        // rows 0,1→key5 (len 1), row 2→key9 (len 2 > max_load); row 3's row-top stops.
        // Returns rows done = 3 (< n=4): the overflow signal, with real progress (the
        // caller grows + re-enters from 3 — never stalls).
        assert_eq!(done, 3);
    }
}
