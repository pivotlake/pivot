//! Copy-and-patch runtime for the GROUP BY value fold.
//!
//! At plan time we know a group-by's value signature — a list of `(op, column)`
//! aggregate slots — but not at *compile* time, so today the [`Dynamic`] container
//! dispatches on each slot's kind for every row. This module removes that per-row
//! dispatch: it assembles a single, signature-specialised machine-code loop that
//! folds a whole batch in one pass with no per-row call.
//!
//! It does so by **copy-and-patch** (see [`stencils`](mod@self) source notes):
//! `build.rs` compiles the stencils in `stencils.rs` and hands us, per stencil,
//! its raw aarch64 bytes plus a tiny relocation table. To assemble a fold we
//!
//! 1. copy the [`FOLD_LOOP`] frame,
//! 2. **splice** the per-slot op stencils into its `bl body` site, patching each
//!    op's `OFF`/`COL` holes (the cell byte offset and input column index), and
//! 3. fix up the frame's two internal branches that the splice shifted.
//!
//! The result is a `extern "C" fn(cells, cols, n)` where `cells[row]` is the
//! Rust-probe-resolved pointer to row `row`'s group cell and `cols[slot]` is slot
//! `slot`'s input column base. Only aarch64 is supported; elsewhere the generated
//! table sets [`SUPPORTED`] to `false` and callers keep the interpreted fold.
//!
//! [`Dynamic`]: super::container::Dynamic

/// One absolute-`movz/movk` hole: write the 16-bit chunk `value >> shift` into the
/// instruction at byte `at` (relative to the stencil's first byte).
#[derive(Clone, Copy)]
pub struct Movw {
    pub at: u32,
    /// `0` = `OFF` (cell byte offset), `1` = `COL` (input column index).
    pub hole: u8,
    pub shift: u8,
}

/// A leaf body-op stencil: straight-line aarch64 with its trailing `ret` already
/// stripped, ready to splice. Reads `cell`/`cols`/`row` from `x0`/`x1`/`x2`.
pub struct OpStencil {
    pub code: &'static [u8],
    pub holes: &'static [Movw],
}

/// The loop-frame stencil: a counted batch loop with one `bl body` splice site.
pub struct Frame {
    pub code: &'static [u8],
    pub splice_at: u32,
}

include!(concat!(env!("OUT_DIR"), "/cap_stencils.rs"));

/// The fold a value signature needs for one slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldOp {
    /// `COUNT(*)` / `COUNT(col)` — `+1` per row, reads no column.
    Count,
    /// `SUM` over an `Int16` / `Int32` / `Int64` column.
    SumI16,
    SumI32,
    SumI64,
}

impl FoldOp {
    /// The fold a value slot of `kind` over a column of byte-`width` (`0` for
    /// `COUNT`) needs — `None` if it isn't an additive numeric the stencils cover.
    pub fn from_slot(kind: super::AggregationKind, width: u8) -> Option<FoldOp> {
        use super::AggregationKind::{Count, CountStar, Sum};
        Some(match (kind, width) {
            (CountStar | Count, _) => FoldOp::Count,
            (Sum, 2) => FoldOp::SumI16,
            (Sum, 4) => FoldOp::SumI32,
            (Sum, 8) => FoldOp::SumI64,
            _ => return None,
        })
    }

    fn stencil(self) -> &'static OpStencil {
        match self {
            FoldOp::Count => &OP_COUNT,
            FoldOp::SumI16 => &OP_SUM_I16,
            FoldOp::SumI32 => &OP_SUM_I32,
            FoldOp::SumI64 => &OP_SUM_I64,
        }
    }
}

/// One slot of an assembled fold: an op, the cell byte offset it accumulates into,
/// and (for sums) the input column index it reads.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub op: FoldOp,
    pub cell_offset: u32,
    pub column: u32,
}

/// The C ABI of an assembled fold loop.
pub type FoldFn = unsafe extern "C" fn(cells: *const *mut i64, cols: *const *const u8, n: u64);

/// An assembled fold: owns its executable page and exposes the entry point.
pub struct CompiledFold {
    exec: ExecBuffer,
}

impl CompiledFold {
    /// The assembled entry point. `cells[i]` must point to row `i`'s group cell
    /// (additive slots pre-seeded to 0); `cols[slot.column]` to that column's base.
    #[inline(always)]
    pub fn func(&self) -> FoldFn {
        unsafe { std::mem::transmute::<*const u8, FoldFn>(self.exec.ptr()) }
    }
}

/// A per-batch fold ready to run: the assembled loop's entry point plus this
/// batch's per-slot column bases and widths.
///
/// The entry point is a plain function pointer; the executable memory it lives in
/// is owned elsewhere (a [`CompiledFold`] cached for the query), so a `BatchFold`
/// is cheap to build per batch. [`run`](Self::run) folds a contiguous run of rows,
/// advancing the column bases so the loop's row index lines up with the columns.
pub struct BatchFold {
    func: FoldFn,
    bases: Vec<*const u8>,
    widths: Vec<u8>,
}

impl BatchFold {
    pub fn new(func: FoldFn, bases: Vec<*const u8>, widths: Vec<u8>) -> Self {
        Self { func, bases, widths }
    }

    /// Fold rows `[start, start + n)`. `cells` must point at row `start`'s cell
    /// pointer (i.e. already offset by `start`); each `cells[j]` is row
    /// `start + j`'s group cell. The loop reads column element `j`, so we advance
    /// each base by `start` elements first.
    ///
    /// # Safety
    /// `cells[0..n]` must be valid `*mut i64` group-cell pointers and the bound
    /// columns must have at least `start + n` elements.
    #[inline]
    pub unsafe fn run(&self, cells: *const *mut i64, start: usize, n: usize) {
        let cols: Vec<*const u8> = self
            .bases
            .iter()
            .zip(&self.widths)
            .map(|(&b, &w)| if w == 0 { b } else { unsafe { b.add(start * w as usize) } })
            .collect();
        unsafe { (self.func)(cells, cols.as_ptr(), n as u64) };
    }
}

/// Assemble a fold loop for `slots`. Returns `None` on a target without stencils.
pub fn compile_fold(slots: &[Slot]) -> Option<CompiledFold> {
    if !SUPPORTED {
        return None;
    }
    let code = assemble(slots);
    Some(CompiledFold {
        exec: ExecBuffer::new(&code),
    })
}

/// Build the machine code: frame with the per-slot ops spliced into its body.
fn assemble(slots: &[Slot]) -> Vec<u8> {
    let frame = &FOLD_LOOP;
    let splice = frame.splice_at as usize;

    // The spliced body: each op's bytes, holes patched for that slot.
    let mut body: Vec<u8> = Vec::new();
    for slot in slots {
        let st = slot.op.stencil();
        let base = body.len();
        body.extend_from_slice(st.code);
        for h in st.holes {
            let value = match h.hole {
                0 => slot.cell_offset,
                _ => slot.column,
            };
            let chunk = (((value as u64) >> h.shift) & 0xFFFF) as u16;
            patch_movw(&mut body[base + h.at as usize..], chunk);
        }
    }

    // Lay out: frame head | spliced body | frame tail (the `bl` is dropped).
    let mut out = Vec::with_capacity(frame.code.len() + body.len());
    out.extend_from_slice(&frame.code[..splice]);
    out.extend_from_slice(&body);
    out.extend_from_slice(&frame.code[splice + 4..]);

    // The splice shifted every frame byte after the site by `growth`; re-encode the
    // frame's PC-relative branches against the new layout. `map` sends an original
    // frame offset to its position in `out`.
    let growth = body.len() as i64 - 4;
    let map = |x: usize| if x <= splice { x } else { (x as i64 + growth) as usize };
    let mut o = 0usize;
    while o + 4 <= frame.code.len() {
        if o != splice {
            let instr = u32::from_le_bytes(frame.code[o..o + 4].try_into().unwrap());
            if let Some((kind, off)) = branch_offset(instr) {
                let target = (o as i64 + off) as usize;
                let new_off = map(target) as i64 - map(o) as i64;
                let reenc = set_branch_offset(instr, kind, new_off);
                let at = map(o);
                out[at..at + 4].copy_from_slice(&reenc.to_le_bytes());
            }
        }
        o += 4;
    }
    out
}

/// Write a 16-bit immediate into a `movz`/`movk` (imm16 occupies bits `[20:5]`).
fn patch_movw(slot: &mut [u8], value: u16) {
    let mut w = u32::from_le_bytes(slot[..4].try_into().unwrap());
    w &= !(0xFFFF << 5);
    w |= (value as u32) << 5;
    slot[..4].copy_from_slice(&w.to_le_bytes());
}

#[derive(Clone, Copy)]
enum BrKind {
    /// Unconditional `b` — imm26.
    B,
    /// `b.cond` / `cbz` / `cbnz` — imm19.
    Imm19,
    /// `tbz` / `tbnz` — imm14.
    Imm14,
}

/// If `instr` is a PC-relative branch we relocate, return its kind and signed byte
/// displacement. (`bl` is excluded — the only `bl` is the splice site, handled
/// separately.)
fn branch_offset(instr: u32) -> Option<(BrKind, i64)> {
    let sext = |v: u32, bits: u32| -> i64 {
        let s = 32 - bits;
        (((v << s) as i32) >> s) as i64
    };
    if instr & 0xFC00_0000 == 0x1400_0000 {
        Some((BrKind::B, sext(instr & 0x03FF_FFFF, 26) << 2))
    } else if instr & 0xFF00_0010 == 0x5400_0000 {
        Some((BrKind::Imm19, sext((instr >> 5) & 0x7_FFFF, 19) << 2))
    } else if instr & 0x7E00_0000 == 0x3400_0000 {
        Some((BrKind::Imm19, sext((instr >> 5) & 0x7_FFFF, 19) << 2))
    } else if instr & 0x7E00_0000 == 0x3600_0000 {
        Some((BrKind::Imm14, sext((instr >> 5) & 0x3FFF, 14) << 2))
    } else {
        None
    }
}

/// Re-encode `instr`'s displacement to `off` bytes.
fn set_branch_offset(instr: u32, kind: BrKind, off: i64) -> u32 {
    let w = (off >> 2) as u32;
    match kind {
        BrKind::B => (instr & !0x03FF_FFFF) | (w & 0x03FF_FFFF),
        BrKind::Imm19 => (instr & !0x00FF_FFE0) | ((w & 0x7_FFFF) << 5),
        BrKind::Imm14 => (instr & !0x0007_FFE0) | ((w & 0x3FFF) << 5),
    }
}

// ---- executable memory -----------------------------------------------------

/// A page of executable memory holding assembled code, freed on drop.
struct ExecBuffer {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for ExecBuffer {}
unsafe impl Sync for ExecBuffer {}

impl ExecBuffer {
    fn ptr(&self) -> *const u8 {
        self.ptr
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
impl ExecBuffer {
    fn new(code: &[u8]) -> Self {
        // Apple silicon: a single MAP_JIT page is toggled W^X per thread; the
        // icache must be invalidated after the writable window closes.
        const MAP_JIT: i32 = 0x0800;
        unsafe extern "C" {
            fn pthread_jit_write_protect_np(enabled: i32);
            fn sys_icache_invalidate(start: *mut std::ffi::c_void, len: usize);
        }
        let len = code.len();
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANON | MAP_JIT,
                -1,
                0,
            )
        };
        assert!(ptr != libc::MAP_FAILED, "MAP_JIT mmap failed");
        let ptr = ptr as *mut u8;
        unsafe {
            pthread_jit_write_protect_np(0);
            std::ptr::copy_nonoverlapping(code.as_ptr(), ptr, len);
            pthread_jit_write_protect_np(1);
            sys_icache_invalidate(ptr as *mut std::ffi::c_void, len);
        }
        Self { ptr, len }
    }
}

#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
impl ExecBuffer {
    fn new(code: &[u8]) -> Self {
        // Linux/other: write to an RW page, flip to RX, then flush the icache.
        unsafe extern "C" {
            fn __clear_cache(start: *mut std::ffi::c_void, end: *mut std::ffi::c_void);
        }
        let len = code.len();
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert!(ptr != libc::MAP_FAILED, "mmap failed");
        let ptr = ptr as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(code.as_ptr(), ptr, len);
            assert_eq!(
                libc::mprotect(ptr as *mut _, len, libc::PROT_READ | libc::PROT_EXEC),
                0,
                "mprotect RX failed"
            );
            __clear_cache(ptr as *mut _, ptr.add(len) as *mut _);
        }
        Self { ptr, len }
    }
}

// On non-aarch64 targets `SUPPORTED` is false and `compile_fold` returns `None`
// before ever constructing an `ExecBuffer`, so this stub is never reached.
#[cfg(not(target_arch = "aarch64"))]
impl ExecBuffer {
    fn new(_code: &[u8]) -> Self {
        unreachable!("compile_fold returns None when stencils are unsupported")
    }
}

impl Drop for ExecBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut _, self.len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spliced_fold_matches_scalar() {
        // A fold of COUNT(*) into cell[0] and SUM(i32 col0) into cell[8], over a
        // batch whose rows alternate between two groups.
        let col0: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
        let mut g0 = [0i64; 2];
        let mut g1 = [0i64; 2];
        let cells: Vec<*mut i64> = vec![
            g0.as_mut_ptr(),
            g1.as_mut_ptr(),
            g0.as_mut_ptr(),
            g1.as_mut_ptr(),
            g0.as_mut_ptr(),
            g1.as_mut_ptr(),
        ];
        let cols: Vec<*const u8> = vec![col0.as_ptr() as *const u8];
        let slots = [
            Slot { op: FoldOp::Count, cell_offset: 0, column: 0 },
            Slot { op: FoldOp::SumI32, cell_offset: 8, column: 0 },
        ];

        let fold = compile_fold(&slots).expect("aarch64 build assembles");
        unsafe { fold.func()(cells.as_ptr(), cols.as_ptr(), cells.len() as u64) };

        // g0 = rows 0,2,4 -> count 3, sum 1+3+5; g1 = rows 1,3,5 -> count 3, sum 2+4+6.
        assert_eq!(g0, [3, 9]);
        assert_eq!(g1, [3, 12]);
    }

    #[test]
    fn empty_batch_is_a_noop() {
        let cells: Vec<*mut i64> = vec![];
        let cols: Vec<*const u8> = vec![];
        let slots = [Slot { op: FoldOp::Count, cell_offset: 0, column: 0 }];

        let fold = compile_fold(&slots).expect("aarch64 build assembles");
        unsafe { fold.func()(cells.as_ptr(), cols.as_ptr(), 0) };
        // No crash, nothing to assert beyond reaching here.
    }
}
