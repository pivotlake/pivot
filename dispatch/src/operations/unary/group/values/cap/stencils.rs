//! Copy-and-patch **stencils** for the GROUP BY value fold.
//!
//! This file is **not** a normal module — `build.rs` compiles it standalone with
//! `rustc --emit obj -C code-model=large` (the large code model materialises every
//! symbol address with an absolute `movz/movk` quad, which is what makes the holes
//! patchable), then extracts each function's machine code + relocations into a
//! table the runtime ([`super`]) copies and patches. See `build.rs` and `cap/mod.rs`.
//!
//! The technique is "Copy-and-Patch Compilation" (Xu & Kjolstad, 2021): write the
//! code templates in a high-level language, let the compiler generate the asm, and
//! at runtime `memcpy` the templates and fill the holes via relocation records.
//!
//! # The two stencil shapes
//!
//! * **Body ops** (`op_*`) — fold ONE aggregate slot for ONE row. Compiled leaf
//!   (no stack frame), they read `cell`/`cols`/`row` from `x0`/`x1`/`x2` and
//!   *preserve* them, so several can be concatenated back-to-back and each still
//!   sees the same arguments. Their two holes are the cell byte offset
//!   ([`OFF`]) and the input column index ([`COL`]); both become absolute
//!   `MOVW_UABS` relocations the runtime patches per slot.
//!
//! * **The loop frame** ([`fold_loop`]) — iterates the batch and calls `body` once
//!   per row. The single `bl body` is the *splice site* (a `CALL26` relocation):
//!   the runtime replaces it with the concatenated, patched body ops, so the whole
//!   fold runs in one pass with no per-row call.
//!
//! All loop state lives in callee-saved registers, so the spliced ops (which only
//! touch `x0`–`x2` and caller-saved temporaries) can never clobber it.
#![no_std]
#![crate_type = "lib"]

// The holes. Their *address* is the patched constant (paper trick): the runtime
// rewrites the `movz/movk` immediates so `&OFF as usize` evaluates to the slot's
// cell byte offset and `&COL as usize` to its input column index.
unsafe extern "C" {
    static OFF: u8;
    static COL: u8;
    // The splice site. `fold_loop`'s `bl body` is a `CALL26` relocation the runtime
    // overwrites with the spliced body ops; it is never actually called.
    fn body(cell: *mut i64, cols: *const *const u8, row: u64);
}

/// `COUNT(*)` / `COUNT(col)` partial: `cell[OFF] += 1`. (Null handling is the
/// caller's concern — these stencils run only for additive numeric signatures.)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_count(cell: *mut i64, _cols: *const *const u8, _row: u64) {
    unsafe {
        let slot = cell.byte_add(&OFF as *const u8 as usize);
        *slot += 1;
    }
}

/// `SUM` over an `Int16` column: `cell[OFF] += cols[COL][row] as i64` (sign-extend).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_sum_i16(cell: *mut i64, cols: *const *const u8, row: u64) {
    unsafe {
        let col = *cols.add(&COL as *const u8 as usize) as *const i16;
        let slot = cell.byte_add(&OFF as *const u8 as usize);
        *slot += *col.add(row as usize) as i64;
    }
}

/// `SUM` over an `Int32` column: `cell[OFF] += cols[COL][row] as i64` (sign-extend).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_sum_i32(cell: *mut i64, cols: *const *const u8, row: u64) {
    unsafe {
        let col = *cols.add(&COL as *const u8 as usize) as *const i32;
        let slot = cell.byte_add(&OFF as *const u8 as usize);
        *slot += *col.add(row as usize) as i64;
    }
}

/// `SUM` over an `Int64` column: `cell[OFF] += cols[COL][row]`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_sum_i64(cell: *mut i64, cols: *const *const u8, row: u64) {
    unsafe {
        let col = *cols.add(&COL as *const u8 as usize) as *const i64;
        let slot = cell.byte_add(&OFF as *const u8 as usize);
        *slot += *col.add(row as usize);
    }
}

/// The loop frame: fold every row of the batch into its pre-resolved cell.
///
/// `cells[row]` is the (Rust-probe-resolved) pointer to row `row`'s group cell;
/// `cols[slot]` is the base pointer of slot `slot`'s input column. The lone
/// `bl body` is the splice site — at assembly time it becomes the concatenated
/// per-slot ops, so this loop carries the entire fold with no per-row call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn fold_loop(cells: *const *mut i64, cols: *const *const u8, n: u64) {
    let mut row = 0u64;
    while row < n {
        unsafe {
            let cell = *cells.add(row as usize);
            body(cell, cols, row);
        }
        row += 1;
    }
}
