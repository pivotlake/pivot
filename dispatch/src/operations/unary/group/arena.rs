use crate::env::MAX_INLINE_STRING_VIEW;
use crate::memory::{BUFFER_SIZE, WriteBuffer, memory_ctx};
use crate::operations::unary::group::ArenaKey;
use arrow_buffer::Buffer;
use std::cell::UnsafeCell;
use std::ptr;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Pointer-table slots reserved beyond the ring's own buffers for input
/// buffers registered by [`SharedArena::register_input_buffer`]. The table is
/// allocated zeroed and backed lazily, so unused headroom costs nothing.
const INPUT_BUFFER_SLOTS: usize = 1 << 20;

/// One registered buffer: where its bytes start and how many there are.
#[derive(Clone, Copy)]
struct BufferSlot {
    ptr: *mut u8,
    len: usize,
}

/// Shared arena holding all string buffers across workers.
///
/// Buffer pointers are stored in a plain array for zero-overhead resolution.
/// Safety: `next_idx` (atomic) guarantees each slot is written by exactly one thread.
/// Cross-thread visibility is ensured by the gather barrier between consume and merge phases.
///
/// A slot is either a ring buffer a worker wrote strings into, or an input
/// batch's own string buffer registered as-is so scattered keys can point at
/// it without copying a byte. Keys resolve through the table the same way
/// either way.
pub struct SharedArena {
    /// We want to have all u128 views point to a centralized place shared by all workers;
    /// this allows us in merge time to simply copy the u128s as is when creating the record batch,
    /// instead of needing to recreate pointers (if we need to shuffle pointers around). We also want
    /// all `resolve` calls (get a pointer from a u128) to be lock-free.
    ///
    /// We therefore need a slot array *that is initialized in advance*, i.e., that it is never moved
    slots: Box<[UnsafeCell<BufferSlot>]>,
    next_idx: AtomicU32,
    /// Owns WriteBuffers so ring memory stays alive until the arena is dropped.
    buffers: Mutex<Vec<WriteBuffer>>,
    /// Keeps registered input buffers alive until the arena is dropped.
    input_buffers: Mutex<Vec<Buffer>>,
}

unsafe impl Send for SharedArena {}
unsafe impl Sync for SharedArena {}
impl std::panic::RefUnwindSafe for SharedArena {}

impl SharedArena {
    /// Create a new shared arena with room for `ring_buffers` write buffers
    /// plus [`INPUT_BUFFER_SLOTS`] registered input buffers.
    pub fn new(ring_buffers: usize) -> Arc<Self> {
        let capacity = ring_buffers + INPUT_BUFFER_SLOTS;
        // The slot table must never move (`resolve` reads it lock-free), so
        // it is sized up front for the worst case. Allocate it zeroed rather
        // than writing empty slots (a null pointer and zero length are the
        // all-zero pattern), so the OS backs its pages lazily: a typical query
        // touches only the first handful of entries, and eagerly writing
        // megabytes of empty slots per arena per query is a measurable fixed
        // cost on short queries.
        let layout = std::alloc::Layout::array::<UnsafeCell<BufferSlot>>(capacity).unwrap();
        let slots: Box<[UnsafeCell<BufferSlot>]> = unsafe {
            let raw = std::alloc::alloc_zeroed(layout);
            if raw.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                raw as *mut UnsafeCell<BufferSlot>,
                capacity,
            ))
        };
        Arc::new(Self {
            slots,
            next_idx: AtomicU32::new(0),
            buffers: Mutex::new(Vec::new()),
            input_buffers: Mutex::new(Vec::new()),
        })
    }

    /// Claims the next slot and stores the buffer's location in it.
    fn claim_slot(&self, ptr: *mut u8, len: usize) -> u32 {
        let idx = self.next_idx.fetch_add(1, Ordering::Relaxed);
        assert!(
            (idx as usize) < self.slots.len(),
            "arena slot table exhausted after {} buffers",
            self.slots.len()
        );
        // Safety: fetch_add guarantees unique idx per caller; no two threads write the same slot.
        unsafe { *self.slots[idx as usize].get() = BufferSlot { ptr, len } };
        idx
    }

    /// Allocate a new write buffer, register its pointer, and return it with its index.
    /// The caller owns the buffer for writing; call [`Self::return_buffer`] when done.
    pub fn take_buffer(&self) -> (WriteBuffer, u32) {
        let wb = memory_ctx().get_write_buffer(false);
        let idx = self.claim_slot(wb.ptr, BUFFER_SIZE);
        (wb, idx)
    }

    /// Transfer ownership of a completed buffer back to the arena.
    pub fn return_buffer(&self, wb: WriteBuffer) {
        self.buffers.lock().unwrap().push(wb);
    }

    /// Registers an input batch's buffer so keys can point into it directly,
    /// and returns the slot index those keys carry. The arena holds the buffer
    /// alive for its whole lifetime.
    pub fn register_input_buffer(&self, buffer: &Buffer) -> u32 {
        let idx = self.claim_slot(buffer.as_ptr().cast_mut(), buffer.len());
        self.input_buffers.lock().unwrap().push(buffer.clone());
        idx
    }

    /// Resolve a non-inline key's buffer slice.
    #[inline]
    pub fn resolve(&self, buffer_index: u32, offset: u32, len: u32) -> &[u8] {
        let ptr = unsafe { (*self.slots[buffer_index as usize].get()).ptr };
        unsafe { std::slice::from_raw_parts(ptr.add(offset as usize), len as usize) }
    }

    /// Prefetch into L1 the cache line backing a non-inline key's bytes — a hint
    /// used by the row-key output decode to hide the scattered arena reads while
    /// it walks keys in (hash) slot order. A no-op hint, so the index/offset need
    /// only be valid, not currently mapped.
    #[inline(always)]
    pub fn prefetch(&self, buffer_index: u32, offset: u32) {
        let ptr = unsafe {
            (*self.slots[buffer_index as usize].get())
                .ptr
                .add(offset as usize)
        };
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

    /// Wrap every registered buffer as a shared `Arc<[Buffer]>` for zero-copy
    /// `StringViewArray` output. Each `Buffer` holds a clone of the owning
    /// `Arc<SharedArena>`, so the memory stays alive as long as any output
    /// array points into it. Valid once consume has finished and `next_idx` is final.
    pub fn to_arrow_buffers(self: &Arc<Self>) -> Arc<[Buffer]> {
        let count = self.next_idx.load(Ordering::Acquire) as usize;
        (0..count)
            .map(|i| {
                let slot = unsafe { *self.slots[i].get() };
                // Each slot in `0..next_idx` is fully written by the time consume
                // finishes; a null here means this ran before a `claim_slot` that
                // reserved slot `i` stored its pointer (a caller-contract violation),
                // which `new_unchecked` would turn into silent UB.
                debug_assert!(
                    !slot.ptr.is_null(),
                    "arena buffer {i} read before its pointer was stored"
                );
                unsafe {
                    Buffer::from_custom_allocation(
                        NonNull::new_unchecked(slot.ptr),
                        slot.len,
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

    /// Allocates an uninitialized, naturally aligned cell block.
    ///
    /// The returned address remains stable for the shared arena's lifetime.
    #[inline]
    pub fn alloc_cells<T>(&mut self, count: usize) -> *mut T {
        let size = count * size_of::<T>();
        debug_assert!(size <= BUFFER_SIZE, "cell block exceeds one arena buffer");
        let mut offset = self.cursor.next_multiple_of(align_of::<T>());
        if offset + size > BUFFER_SIZE {
            self.add_buffer();
            // A fresh buffer starts at its 2MB-aligned base, so offset 0 is
            // aligned for any cell type.
            offset = 0;
        }
        self.cursor = offset + size;
        unsafe { self.active_buffer.as_mut_ptr().add(offset) as *mut T }
    }

    /// Return the active buffer to the shared arena. Must be called before dropping.
    pub fn flush(self) {
        self.shared.return_buffer(self.active_buffer);
    }
}
