use crate::env::MAX_INLINE_STRING_VIEW;
use crate::memory::{BUFFER_SIZE, WriteBuffer, memory_ctx};
use crate::operations::unary::group::ArenaKey;
use arrow_buffer::Buffer;
use std::cell::UnsafeCell;
use std::ptr;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Shared arena holding all string buffers across workers.
///
/// Buffer pointers are stored in a plain pointer array for zero-overhead resolution.
/// Safety: `next_idx` (atomic) guarantees each slot is written by exactly one thread.
/// Cross-thread visibility is ensured by the mpsc channel between consume and merge phases.
pub struct SharedArena {
    /// We want to have all u128 views point to a centralized place shared by all workers;
    /// this allows us in merge time to simply copy the u128s as is when creating the record batch,
    /// instead of needing to recreate pointers (if we need to shuffle pointers around). We also want
    /// all `resolve` calls (get a pointer from a u128) to be lock-free.
    ///
    /// We therefore need a ptrs array *that is initialized in advance*, i.e., that it is never moved
    ptrs: Box<[UnsafeCell<*mut u8>]>,
    next_idx: AtomicU32,
    /// Owns WriteBuffers so ring memory stays alive until the arena is dropped.
    buffers: Mutex<Vec<WriteBuffer>>,
}

unsafe impl Send for SharedArena {}
unsafe impl Sync for SharedArena {}
impl std::panic::RefUnwindSafe for SharedArena {}

impl SharedArena {
    /// Create a new shared arena with space for up to `ring()` amount of write buffers.
    pub fn new(buffers: usize) -> Arc<Self> {
        Arc::new(Self {
            ptrs: (0..buffers)
                .map(|_| UnsafeCell::new(ptr::null_mut()))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            next_idx: AtomicU32::new(0),
            buffers: Mutex::new(Vec::new()),
        })
    }

    /// Allocate a new write buffer, register its pointer, and return it with its index.
    /// The caller owns the buffer for writing; call [`Self::return_buffer`] when done.
    pub fn take_buffer(&self) -> (WriteBuffer, u32) {
        let wb = memory_ctx().get_write_buffer(false);
        let idx = self.next_idx.fetch_add(1, Ordering::Relaxed);
        // Safety: fetch_add guarantees unique idx per caller; no two threads write the same slot.
        unsafe { *self.ptrs[idx as usize].get() = wb.ptr };
        (wb, idx)
    }

    /// Transfer ownership of a completed buffer back to the arena.
    pub fn return_buffer(&self, wb: WriteBuffer) {
        self.buffers.lock().unwrap().push(wb);
    }

    /// Resolve a non-inline key's buffer slice.
    #[inline]
    pub fn resolve(&self, buffer_index: u32, offset: u32, len: u32) -> &[u8] {
        let ptr = unsafe { *self.ptrs[buffer_index as usize].get() };
        unsafe { std::slice::from_raw_parts(ptr.add(offset as usize), len as usize) }
    }

    /// Prefetch into L1 the cache line backing a non-inline key's bytes — a hint
    /// used by the row-key output decode to hide the scattered arena reads while
    /// it walks keys in (hash) slot order. A no-op hint, so the index/offset need
    /// only be valid, not currently mapped.
    #[inline(always)]
    pub fn prefetch(&self, buffer_index: u32, offset: u32) {
        let ptr = unsafe { (*self.ptrs[buffer_index as usize].get()).add(offset as usize) };
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr as *const i8);
        }
        #[cfg(target_arch = "aarch64")]
        #[allow(clippy::pointers_in_nomem_asm_block)]
        unsafe {
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) ptr, options(nomem, nostack, preserves_flags));
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let _ = ptr;
    }

    /// Convert all registered buffers into Arrow Buffers for zero-copy StringViewArray output.
    /// The `Arc<SharedArena>` keeps ring memory alive as long as any Arrow Buffer exists.
    pub fn to_arrow_buffers(self: &Arc<Self>) -> Vec<Buffer> {
        let count = self.next_idx.load(Ordering::Acquire) as usize;
        (0..count)
            .map(|i| {
                let ptr = unsafe { *self.ptrs[i].get() };
                unsafe {
                    Buffer::from_custom_allocation(
                        NonNull::new_unchecked(ptr),
                        BUFFER_SIZE,
                        self.clone(),
                    )
                }
            })
            .collect()
    }
}

/// Per-worker write handle into the [`SharedArena`].
///
/// Each worker writes to its own active buffer without locking. When the buffer
/// fills, it is returned to the shared arena and a new one is taken.
pub struct WorkerArena {
    shared: Arc<SharedArena>,
    active_buffer: WriteBuffer,
    buffer_index: u32,
    cursor: usize,
}

impl WorkerArena {
    /// Create a new worker arena, taking a fresh buffer from the shared arena.
    pub fn new(shared: Arc<SharedArena>) -> Self {
        let (wb, idx) = shared.take_buffer();
        Self {
            shared,
            active_buffer: wb,
            buffer_index: idx,
            cursor: 0,
        }
    }

    /// Borrow the underlying shared arena (used for key comparison).
    pub fn shared(&self) -> &SharedArena {
        &self.shared
    }

    /// Take a fresh buffer from the shared arena when the current one is full.
    #[cold]
    fn add_buffer(&mut self) {
        let (new_wb, idx) = self.shared.take_buffer();
        let old = std::mem::replace(&mut self.active_buffer, new_wb);
        self.shared.return_buffer(old);
        self.buffer_index = idx;
        self.cursor = 0;
    }

    /// Push a string into the arena and return its ArenaKey (inline or view).
    #[inline]
    pub fn push(&mut self, s: &str) -> ArenaKey {
        self.push_bytes(s.as_bytes())
    }

    /// Push arbitrary bytes into the arena and return their ArenaKey (inline or
    /// view). The bytes need not be valid UTF-8 — an `ArenaKey`/StringView is
    /// just length-prefixed bytes — so the row-encoded composite key uses this
    /// to store a packed key tuple.
    #[inline]
    pub fn push_bytes(&mut self, data: &[u8]) -> ArenaKey {
        if data.len() <= MAX_INLINE_STRING_VIEW {
            return ArenaKey::inline(data);
        }

        if self.cursor + data.len() > BUFFER_SIZE {
            self.add_buffer();
        }

        let offset = self.cursor;
        unsafe {
            ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.active_buffer.as_mut_ptr().add(offset),
                data.len(),
            );
        }
        self.cursor += data.len();

        ArenaKey::view(data, self.buffer_index, offset as u32)
    }

    /// Return the active buffer to the shared arena. Must be called before dropping.
    pub fn flush(self) {
        self.shared.return_buffer(self.active_buffer);
    }
}
