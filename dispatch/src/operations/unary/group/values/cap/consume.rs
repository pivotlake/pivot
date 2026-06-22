//! Whole-loop copy-and-patch for the consume probe+fold (no per-row call).
//!
//! The probe loop is authored once as an assembly **skeleton** (a `#[naked]` fn —
//! the compiler assembles it, so its internal relative branches are correct and
//! survive a verbatim byte-copy). At the fold site it carries a fixed-size NOP
//! **gap** bracketed by `brk` markers. To compile a signature we copy the skeleton
//! and overwrite the gap (same byte length) with the additive [`fold`](super::fold)
//! tiles — so no branch displacement ever changes. The result is one inlined,
//! branch-free, call-free probe+fold loop.
//!
//! This skeleton handles a `u64` key (the single-int key levels — e.g. q09's outer
//! `RegionID`); the int-pair (`u128`) skeleton is a sibling with a two-word key
//! compare. Entry layout is `{ hash:u64 @0, key:u64 @8, value:[i64;N] @VALOFF }`;
//! `VALOFF`/stride/mask/shift come in via [`ConsumeCtx`] so the asm bakes no sizes.

#![cfg(target_arch = "aarch64")]
use super::FoldOp;
use super::exec::{ExecBuffer, RET, stencil_words};

/// Everything the patched loop needs, passed by pointer in `x0`. `#[repr(C)]` so
/// the field offsets the asm loads (`[x0, #8*k]`) are stable.
#[repr(C)]
pub struct ConsumeCtx {
    pub buf: *mut u8,            // 0  table entry array (zeroed; hash==0 = empty)
    pub stride: u64,             // 8  entry size in bytes
    pub valoff: u64,             // 16 value offset within an entry
    pub mask: u64,               // 24 capacity-1
    pub shift: u64,              // 32 64 - log2(capacity)
    pub hashes: *const u64,      // 40
    pub keys: *const u64,        // 48 one pre-extracted key per row
    pub cols: *const *const u8,  // 56 N column bases (slot order)
    pub n: u64,                  // 64 row count (exclusive end index)
    pub len: *mut u64,           // 72 table length (bumped on insert)
    pub max_load: u64,           // 80 stop inserting past this — return rows done
    pub start: u64,              // 88 first row to process (for re-call after a grow)
}

/// Gap markers (`brk #0xAAA1` / `brk #0xAAA2`) bracketing the fold NOP-sled.
const GAP_START: u32 = 0xD435_5420; // brk #0xAAA1
const GAP_END: u32 = 0xD435_5440; // brk #0xAAA2
const NOP: u32 = 0xD503_201F;

/// The probe+fold skeleton (u64 key). Branch labels are local; the fold gap is 64
/// NOPs (256 bytes — fits 6 additive slots) between the markers.
#[unsafe(naked)]
unsafe extern "C" fn skel_consume_u64() {
    std::arch::naked_asm!(
        // save callee-saved we use for loop-carried ctx fields
        "stp x19, x20, [sp, #-112]!",
        "stp x21, x22, [sp, #16]",
        "stp x23, x24, [sp, #32]",
        "stp x25, x26, [sp, #48]",
        "stp x27, x28, [sp, #64]",
        // load ctx → callee-saved
        "ldr x19, [x0, #0]",   // buf
        "ldr x20, [x0, #8]",   // stride
        "ldr x21, [x0, #16]",  // valoff
        "ldr x22, [x0, #24]",  // mask
        "ldr x23, [x0, #32]",  // shift
        "ldr x24, [x0, #40]",  // hashes
        "ldr x25, [x0, #48]",  // keys
        "ldr x26, [x0, #56]",  // cols
        "ldr x27, [x0, #64]",  // n
        "ldr x28, [x0, #72]",  // len ptr
        "ldr x9,  [x0, #80]",  // max_load → stash on stack
        "str x9,  [sp, #80]",
        "ldr x12, [x0, #88]",  // i = start (absolute row index)
        // ---- row loop ----
        "0:",                  // row_top
        "ldr x9,  [x28]",      // *len
        "ldr x10, [sp, #80]",  // max_load
        "cmp x9, x10",
        "b.ge 6f",             // table full -> return rows done (i)
        "cmp x12, x27",
        "b.ge 4f",             // i >= n -> done (return n)
        "ldr x13, [x24, x12, lsl #3]", // h = hashes[i]
        "ldr x14, [x25, x12, lsl #3]", // k = keys[i]
        "lsr x15, x13, x23",   // slot = h >> shift
        // ---- probe ----
        "1:",                  // probe
        "madd x16, x15, x20, x19", // entry = buf + slot*stride
        "ldr x17, [x16]",      // entry.hash
        "cbz x17, 2f",         // empty -> insert
        "cmp x17, x13",
        "b.ne 3f",             // hash mismatch -> next
        "ldr x17, [x16, #8]",  // entry.key
        "cmp x17, x14",
        "b.ne 3f",             // key mismatch -> next
        // match: x0 = &entry.value
        "add x0, x16, x21",
        "b 5f",                // -> fold
        "2:",                  // insert (empty slot)
        "str x13, [x16]",      // entry.hash = h
        "str x14, [x16, #8]",  // entry.key = k
        "ldr x9, [x28]",       // *len += 1
        "add x9, x9, #1",
        "str x9, [x28]",
        "add x0, x16, x21",    // x0 = &entry.value
        "b 5f",
        "3:",                  // next probe slot
        "add x15, x15, #1",
        "and x15, x15, x22",
        "b 1b",
        // ---- fold (x0 = cell) ----
        "5:",
        "mov x1, x26",         // cols
        "mov x2, x12",         // row = i
        "brk #0xAAA1",         // GAP_START marker (overwritten by fold tiles)
        // gap: 64 NOPs
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "nop","nop","nop","nop","nop","nop","nop","nop",
        "brk #0xAAA2",         // GAP_END marker
        // advance
        "add x12, x12, #1",
        "b 0b",
        // ---- returns ----
        "4:",                  // all rows consumed
        "mov x0, x27",         // return n
        "b 7f",
        "6:",                  // table full mid-batch
        "mov x0, x12",         // return rows done so far (i)
        "7:",                  // shared epilogue
        "ldp x27, x28, [sp, #64]",
        "ldp x25, x26, [sp, #48]",
        "ldp x23, x24, [sp, #32]",
        "ldp x21, x22, [sp, #16]",
        "ldp x19, x20, [sp], #112",
        "ret",
    );
}

/// `fn(ctx: *mut ConsumeCtx) -> rows_consumed`.
pub type ConsumeFn = unsafe extern "C" fn(*mut ConsumeCtx) -> u64;

/// A finalized consume loop and the buffer owning its code.
pub struct CompiledConsume {
    f: ConsumeFn,
    _buf: ExecBuffer,
}
impl CompiledConsume {
    #[inline(always)]
    pub unsafe fn run(&self, ctx: *mut ConsumeCtx) -> u64 {
        unsafe { (self.f)(ctx) }
    }
}

/// Splice the additive `ops` fold into the u64-key probe skeleton. `None` if the
/// fold doesn't fit the gap or the buffer can't be mapped.
pub fn compile_consume(ops: &[FoldOp]) -> Option<CompiledConsume> {
    // Skeleton bytes (verbatim — keeps every relative branch valid on copy).
    let skel = unsafe { stencil_words(skel_consume_u64 as *const () as usize, RET) };
    let mut words: Vec<u32> = skel.to_vec();

    let gstart = words.iter().position(|&w| w == GAP_START)?;
    let gend = words.iter().position(|&w| w == GAP_END)?;
    let gap_len = gend - gstart + 1; // inclusive of both markers

    // Build the fold body (tiles, patched to slot index), no terminator/ret.
    let fold = super::fold::fold_body_words(ops);
    if fold.len() > gap_len {
        return None; // signature too wide for this gap
    }
    // Overwrite [gstart..=gend] with fold + NOP pad (preserves total size).
    for (k, slot) in (gstart..=gend).enumerate() {
        words[slot] = if k < fold.len() { fold[k] } else { NOP };
    }

    let bytes =
        unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
    let buf = ExecBuffer::new(bytes)?;
    let f: ConsumeFn = unsafe { std::mem::transmute(buf.entry()) };
    Some(CompiledConsume { f, _buf: buf })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Reference: open-addressing fold of (key -> [count, sum]) for COUNT,SUM(i32).
    #[test]
    fn consume_u64_count_sum_matches_reference() {
        const N: usize = 2;
        let valoff = 16u64; // hash8 + key8
        let stride = (16 + N * 8) as u64; // + value
        let cap = 1024usize;
        let mask = (cap - 1) as u64;
        let shift = (cap as u64).leading_zeros() as u64 + 1; // 64 - log2(cap)

        // rows: keys with dups; col = i32 values
        let keys: Vec<u64> = vec![5, 9, 5, 9, 5, 7];
        let vals: Vec<i32> = vec![10, 1, 20, 2, 30, 100];
        let n = keys.len();
        // hash = key * golden (just needs to be a deterministic spread)
        let hashes: Vec<u64> = keys
            .iter()
            .map(|&k| (k.wrapping_mul(0x9E3779B97F4A7C15)).max(1))
            .collect();
        let cols: Vec<*const u8> = vec![std::ptr::null(), vals.as_ptr() as *const u8];

        let mut table = vec![0u8; cap * stride as usize];
        let mut len = 0u64;
        let mut ctx = ConsumeCtx {
            buf: table.as_mut_ptr(),
            stride,
            valoff,
            mask,
            shift,
            hashes: hashes.as_ptr(),
            keys: keys.as_ptr(),
            cols: cols.as_ptr(),
            n: n as u64,
            len: &mut len,
            max_load: cap as u64, // no overflow in this test
            start: 0,
        };

        let compiled = compile_consume(&[FoldOp::Count, FoldOp::SumI32]).expect("compile");
        let consumed = unsafe { compiled.run(&mut ctx) };
        assert_eq!(consumed, n as u64);

        // Read back: scan table, collect key -> [count, sum].
        let mut got: HashMap<u64, [i64; 2]> = HashMap::new();
        for slot in 0..cap {
            let e = unsafe { table.as_ptr().add(slot * stride as usize) };
            let h = unsafe { (e as *const u64).read() };
            if h == 0 {
                continue;
            }
            let k = unsafe { (e.add(8) as *const u64).read() };
            let v0 = unsafe { (e.add(valoff as usize) as *const i64).read() };
            let v1 = unsafe { (e.add(valoff as usize + 8) as *const i64).read() };
            got.insert(k, [v0, v1]);
        }

        // Reference
        let mut want: HashMap<u64, [i64; 2]> = HashMap::new();
        for i in 0..n {
            let e = want.entry(keys[i]).or_insert([0, 0]);
            e[0] += 1;
            e[1] += vals[i] as i64;
        }
        assert_eq!(got, want);
        assert_eq!(len, want.len() as u64);
    }

    #[test]
    fn consume_returns_rows_done_on_overflow() {
        // max_load = 1 → after the first distinct key is inserted, the next row
        // that finds the table "full" stops and returns rows-done.
        const N: usize = 1;
        let stride = (16 + N * 8) as u64;
        let cap = 64usize;
        let keys: Vec<u64> = vec![5, 5, 9, 9]; // key 5 (rows 0,1), then key 9
        let hashes: Vec<u64> = keys.iter().map(|&k| k.wrapping_mul(0x9E3779B97F4A7C15).max(1)).collect();
        let cols: Vec<*const u8> = vec![std::ptr::null()];
        let mut table = vec![0u8; cap * stride as usize];
        let mut len = 0u64;
        let mut ctx = ConsumeCtx {
            buf: table.as_mut_ptr(),
            stride,
            valoff: 16,
            mask: (cap - 1) as u64,
            shift: (cap as u64).leading_zeros() as u64 + 1,
            hashes: hashes.as_ptr(),
            keys: keys.as_ptr(),
            cols: cols.as_ptr(),
            n: keys.len() as u64,
            len: &mut len,
            max_load: 1, // room for exactly one distinct key
            start: 0,
        };
        let compiled = compile_consume(&[FoldOp::Count]).expect("compile");
        let done = unsafe { compiled.run(&mut ctx) };
        // row 0 inserts key 5 (len→1); row 1's row-top sees len>=max_load and stops
        // (conservative: caller grows and re-processes from row 1). Returns rows done.
        assert_eq!(done, 1, "stops once the table is full");
        assert_eq!(len, 1);
    }
}
