//! A page of own-managed executable memory, plus extraction of a pre-compiled
//! machine-code *stencil* from a `#[naked]` function.
//!
//! This is the substrate for copy-and-patch: we author tiles/skeletons as naked
//! Rust functions (so the assembler/compiler produces correct machine code for
//! the host), read their bytes back out at runtime, `memcpy` them into an
//! [`ExecBuffer`] (patching holes), and jump in. Unlike a full JIT we emit no
//! instructions ourselves — we only copy and patch ones the compiler already
//! produced.
//!
//! aarch64 only (the only place the asm tiles exist); [`compile`](super::compile)
//! gates on the arch so other targets fall back to the interpreter.

#![cfg(target_arch = "aarch64")]

/// `brk #0xCAFE` — the unique terminator each chainable stencil ends with. We scan
/// for it to find a stencil's length, then drop it when copying (the tile falls
/// through to whatever we place next).
pub const TERMINATOR: u32 = 0xD439_5FC0;
/// `ret` — terminates a *standalone* (callable) stencil, kept when copied.
pub const RET: u32 = 0xD65F_03C0;

/// The machine-code bytes of a naked stencil `f`, from its entry up to (and
/// including) the first `stop` instruction. `stop` is [`TERMINATOR`] for a
/// fall-through tile (caller drops the last word) or [`RET`] for a callable one.
///
/// # Safety
/// `f` must be a `#[naked]` function whose body is straight-line machine code
/// ending in `stop` within `MAX_WORDS` instructions.
pub unsafe fn stencil_words(f: usize, stop: u32) -> &'static [u32] {
    const MAX_WORDS: usize = 4096;
    let p = f as *const u32;
    let mut i = 0usize;
    loop {
        // SAFETY: caller guarantees `stop` appears within MAX_WORDS; the bound is
        // a backstop against a missing terminator.
        let w = unsafe { p.add(i).read() };
        if w == stop {
            return unsafe { std::slice::from_raw_parts(p, i + 1) };
        }
        i += 1;
        assert!(i < MAX_WORDS, "stencil terminator not found");
    }
}

/// Own-managed executable memory holding a finalized function. Allocated RW,
/// written, then flipped to RX with the instruction cache flushed — mandatory on
/// aarch64, where stale i-cache over freshly written code is a crash, not a stale
/// read. Dropped (unmapped) when the last holder goes away.
pub struct ExecBuffer {
    ptr: *mut u8,
    len: usize,
}

// The finalized code is immutable and re-entrant; only the raw entry is called,
// from any worker.
unsafe impl Send for ExecBuffer {}
unsafe impl Sync for ExecBuffer {}

impl ExecBuffer {
    /// Map `code` as executable memory and return the buffer. `None` if the OS
    /// rejects the mapping (caller falls back to the interpreter).
    pub fn new(code: &[u8]) -> Option<ExecBuffer> {
        if code.is_empty() {
            return None;
        }
        let len = code.len();
        unsafe {
            let ptr = map_writable(len)?;
            std::ptr::copy_nonoverlapping(code.as_ptr(), ptr, len);
            if !make_executable(ptr, len) {
                libc::munmap(ptr as *mut libc::c_void, len);
                return None;
            }
            Some(ExecBuffer { ptr, len })
        }
    }

    /// The entry address. Transmute to the concrete `extern "C"` fn pointer.
    #[inline(always)]
    pub fn entry(&self) -> *const u8 {
        self.ptr
    }
}

impl Drop for ExecBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn pthread_jit_write_protect_np(enabled: libc::c_int);
    fn sys_icache_invalidate(start: *mut libc::c_void, len: libc::size_t);
}

#[cfg(target_os = "macos")]
const MAP_JIT: libc::c_int = 0x800;

/// Map `len` bytes for writing. On macOS (Apple Silicon, strict W^X) the page is
/// `MAP_JIT` RWX and write-protection is toggled per-thread; on Linux it's a plain
/// RW page promoted to RX in [`make_executable`].
#[cfg(target_os = "macos")]
unsafe fn map_writable(len: usize) -> Option<*mut u8> {
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
    if ptr == libc::MAP_FAILED {
        return None;
    }
    unsafe { pthread_jit_write_protect_np(0) }; // writable for this thread
    Some(ptr as *mut u8)
}

#[cfg(target_os = "macos")]
unsafe fn make_executable(ptr: *mut u8, len: usize) -> bool {
    unsafe {
        pthread_jit_write_protect_np(1); // executable for this thread
        sys_icache_invalidate(ptr as *mut libc::c_void, len);
    }
    true
}

#[cfg(not(target_os = "macos"))]
unsafe fn map_writable(len: usize) -> Option<*mut u8> {
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
    if ptr == libc::MAP_FAILED {
        None
    } else {
        Some(ptr as *mut u8)
    }
}

#[cfg(not(target_os = "macos"))]
unsafe extern "C" {
    // compiler-rt / libgcc builtin: flush the i-cache for [start, end). Handles the
    // host's cache-line size (the part that's error-prone to hand-roll).
    fn __clear_cache(start: *mut libc::c_char, end: *mut libc::c_char);
}

#[cfg(not(target_os = "macos"))]
unsafe fn make_executable(ptr: *mut u8, len: usize) -> bool {
    unsafe {
        if libc::mprotect(
            ptr as *mut libc::c_void,
            len,
            libc::PROT_READ | libc::PROT_EXEC,
        ) != 0
        {
            return false;
        }
        __clear_cache(ptr as *mut libc::c_char, ptr.add(len) as *mut libc::c_char);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // A standalone stencil: `x0 += 5; ret`.
    #[unsafe(naked)]
    unsafe extern "C" fn st_add5() {
        std::arch::naked_asm!("add x0, x0, #5", "ret");
    }

    // A fall-through tile: `x0 += 7` then the terminator (dropped on copy).
    #[unsafe(naked)]
    unsafe extern "C" fn tile_add7() {
        std::arch::naked_asm!("add x0, x0, #7", "brk #0xCAFE");
    }

    #[test]
    fn extract_copy_execute_standalone() {
        let words = unsafe { stencil_words(st_add5 as *const () as usize, RET) };
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4)
        };
        let buf = ExecBuffer::new(bytes).expect("exec buffer");
        let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(buf.entry()) };
        assert_eq!(f(10), 15);
        assert_eq!(f(100), 105);
    }

    #[test]
    fn chain_two_tiles_then_ret() {
        // Stitch: tile_add7 body (terminator dropped) + a `ret`.
        let t = unsafe { stencil_words(tile_add7 as *const () as usize, TERMINATOR) };
        let body = &t[..t.len() - 1]; // drop the brk
        let mut words: Vec<u32> = body.to_vec();
        words.push(body[0]); // add x0,x0,#7 again — fold two tiles
        words.push(RET);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4)
        };
        let buf = ExecBuffer::new(bytes).expect("exec buffer");
        let f: extern "C" fn(u64) -> u64 = unsafe { std::mem::transmute(buf.entry()) };
        assert_eq!(f(0), 14); // +7 +7
    }
}
