//! Pre-allocated ring of 2 MB memory slots backed by a single `mmap` region.
//!
//! The ring reserves half of physical memory at startup and divides it into
//! fixed-size [`BUFFER_SIZE`] slots. Each slot is independently tracked by a
//! [`BufferSlot`] that records whether the memory is zeroed and how many
//! readers/writers currently hold it.
//!
//! Slots support a single-writer / multiple-reader access pattern similar to an
//! `RwLock`, but implemented lock-free with atomics:
//!
//! - **Write**: [`Ring::try_write`] atomically sets the `WRITING` bit via CAS.
//!   On success it returns a [`WriteBuffer`] that has exclusive mutable access.
//! - **Read**: [`Ring::try_read`] atomically increments the reader count. It
//!   fails if the `WRITING` bit is set, preventing reads during a write.
//!
//! Dropping a `WriteBuffer` or all `ReadBuffer`s releases the slot back (see their
//! respective modules for details).

use crate::memory::read_buffer::ReadBuffer;
use crate::memory::write_buffer::WriteBuffer;
use std::fmt::{Debug, Formatter};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering};
use std::{io, ptr};

/// Size of each slot in the ring (2 MB).
pub const BUFFER_SIZE: usize = 2 * 1024 * 1024;
/// Page alignment for the mmap region.
const BUFFER_ALIGN: usize = 4096;
/// Bit flag set in `BufferSlot::used` when a writer holds the slot.
const WRITING: u32 = 1 << 31;

/// Metadata for a single 2 MB slot in the ring.
///
/// `used` encodes both writer and reader state in a single atomic word:
/// - Bit 31 (`WRITING`): set when a writer holds the slot exclusively.
/// - Bits 0-30: reader count (incremented by each [`ReadBuffer`] handle).
///
/// A slot is free when `used == 0`.
pub struct BufferSlot {
    /// Raw pointer into the mmap region for this slot.
    pub buffer: *mut u8,
    /// Whether the slot's memory is currently zeroed.
    pub zeroed: AtomicBool,
    /// Combined writer flag + reader count (see struct docs).
    pub used: AtomicU32,
}

unsafe impl Send for BufferSlot {}
unsafe impl Sync for BufferSlot {}

/// Who is holding a slot right now.
///
/// The three cases are exactly what [`BufferSlot::used`] encodes, read as one
/// word so the writer flag and the reader count can never disagree. `Idle` is
/// a slot nobody holds, which is either free or cached and evictable - that
/// distinction is the [`Clock`](super::Clock)'s, not the ring's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SlotUsage {
    Idle,
    /// Held exclusively by a [`WriteBuffer`].
    Writing,
    /// Shared by this many live [`ReadBuffer`]s.
    Reading(u32),
}

/// A contiguous mmap-backed arena divided into fixed-size slots.
///
/// Constructed once at startup via [`Ring::new`]. Individual slots are acquired for
/// writing ([`try_write`](Self::try_write)) or reading
/// ([`try_read`](Self::try_read)) and released when the returned handle is
/// dropped.
pub struct Ring {
    pub slots: Vec<BufferSlot>,
    /// What each slot was last taken for, for the diagnostics in
    /// [`crate::memory::RingCensus`]. Written when a writer takes the slot and
    /// read only while one holds it.
    tags: Vec<AtomicU8>,
}

impl Debug for Ring {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Ring({} x {}MB)",
            self.len(),
            BUFFER_SIZE / (1024 * 1024)
        )
    }
}

impl Ring {
    /// Allocates the ring by `mmap`-ing a single contiguous region and splitting
    /// it into [`BUFFER_SIZE`]-aligned slots. Returns an error if `mmap` fails.
    pub fn new(buffers: usize) -> io::Result<Self> {
        const_assert_aligned();

        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                buffers * BUFFER_SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        // Back the ring with transparent huge pages. Slots are BUFFER_SIZE (2MB)
        // aligned and sized, so each maps to exactly one 2MB page, cutting TLB
        // entries ~512x versus the default 4KB pages — worthwhile for a multi-GB
        // region under random access. Best-effort: ignored if THP is unavailable.
        // Linux-only: `MADV_HUGEPAGE` does not exist on macOS/other targets.
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(ptr, buffers * BUFFER_SIZE, libc::MADV_HUGEPAGE);
        }

        // Verify alignment
        assert_eq!(
            ptr as usize % BUFFER_ALIGN,
            0,
            "mmap returned unaligned pointer"
        );

        let slots: Vec<BufferSlot> = (0..buffers)
            .map(|i| {
                let ptr = unsafe { ptr.add(i * BUFFER_SIZE) } as *mut u8;
                debug_assert_eq!(ptr as usize % BUFFER_ALIGN, 0);
                BufferSlot {
                    buffer: ptr,
                    zeroed: AtomicBool::new(true),
                    used: AtomicU32::new(0),
                }
            })
            .collect();

        let tags = (0..buffers).map(|_| AtomicU8::new(0)).collect();
        Ok(Self { slots, tags })
    }

    /// Returns the total number of slots in the ring.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Record what slot `idx` was taken for.
    pub fn set_tag(&self, idx: usize, tag: u8) {
        self.tags[idx].store(tag, Ordering::Relaxed);
    }

    /// What slot `idx` was last taken for.
    pub fn tag(&self, idx: usize) -> u8 {
        self.tags[idx].load(Ordering::Relaxed)
    }

    /// Whether the ring has no slots.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// How slot `idx` is held right now.
    ///
    /// One relaxed load of the word that packs the writer flag with the reader
    /// count, taking no hold of its own: peer workers keep acquiring and
    /// releasing while this reads, so the answer describes the slot at the
    /// instant it was read. Reporting memory pressure needs nothing stronger.
    pub fn slot_usage(&self, idx: usize) -> SlotUsage {
        let used = self.slots[idx].used.load(Ordering::Relaxed);
        if used & WRITING != 0 {
            SlotUsage::Writing
        } else if used == 0 {
            SlotUsage::Idle
        } else {
            SlotUsage::Reading(used)
        }
    }

    /// Overwrites the `used` word for slot `idx` (writer flag + reader count).
    #[inline(always)]
    pub fn set_slot_used(&self, idx: usize, used: u32, ordering: Ordering) {
        self.slots[idx].used.store(used, ordering)
    }

    /// Records whether slot `idx` contains zeroed memory.
    pub fn set_slot_zeroed(&self, idx: usize, zeroed: bool) {
        self.slots[idx].zeroed.store(zeroed, Ordering::Relaxed)
    }

    /// Whether slot `idx` currently contains zeroed memory.
    pub fn slot_zeroed(&self, idx: usize) -> bool {
        self.slots[idx].zeroed.load(Ordering::Relaxed)
    }

    /// Tries to acquire exclusive write access to slot `idx`.
    ///
    /// Succeeds only if the slot is completely free (`used == 0`). On success,
    /// atomically sets the `WRITING` bit and returns a [`WriteBuffer`].
    /// Returns `None` if the slot is already held by a reader or writer.
    pub fn try_write(&self, idx: usize) -> Option<WriteBuffer> {
        let slot = &self.slots[idx];
        if slot
            .used
            .compare_exchange(0, WRITING, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            return Some(WriteBuffer {
                ptr: slot.buffer, // *mut u8 from the mmap
                slot_idx: idx,
                zeroed: slot.zeroed.load(Ordering::Relaxed),
            });
        }
        None
    }

    /// Tries to acquire shared read access to slot `idx`. Will fail if currently being written to.
    ///
    /// Atomically increments the reader count. If the `WRITING` bit is set
    /// (a writer holds the slot), None` is
    /// returned. Multiple readers can hold the same slot concurrently.
    pub fn try_read(&self, idx: usize) -> Option<ReadBuffer> {
        let slot = &self.slots[idx];
        let prev = slot.used.fetch_add(1, Ordering::Acquire);
        if prev & WRITING != 0 {
            // We purposefully do NOT decrement here, as given we didn't succeed there may be a race
            // where:
            // 1. This buffer was in write mode
            //              a. We increment read, but fail
            // 2. The buffer drops, and sets slot used to 0
            //              b. We decrement read, causing underflow
            //
            // If a slot is in write mode, it will necessarily be set to a value - either 0 or 1.
            // Therefore, we don't touch it after seeing it is in write mode.
            return None;
        }
        Some(ReadBuffer {
            ptr: slot.buffer,
            slot_idx: idx,
        })
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        if let Some(first) = self.slots.first() {
            let size = self.slots.len() * BUFFER_SIZE;
            // NOTE: do not use `tracing` here. `Ring` is held in an `Arc` inside the
            // per-worker `MemoryContext` thread-local, so the last drop happens during
            // TLS destruction — and `tracing_subscriber` also relies on TLS. If its
            // thread-local has already been destroyed, the macro panics with
            // "cannot access a Thread Local Storage value during or after destruction".
            //
            // SAFETY: `first.buffer` + `size` describe the exact mmap region
            // created in `Ring::new`. `Drop` is the unique owner — readers
            // and writers can only borrow slots from a `&'static Ring`, and
            // by the time the `Ring` itself drops there's nothing left to
            // borrow.
            unsafe {
                libc::munmap(first.buffer as *mut libc::c_void, size);
            }
        }
    }
}

/// BUFFER_SIZE must be a multiple of BUFFER_ALIGN so adjacent buffers stay aligned.
const fn const_assert_aligned() {
    assert!(
        BUFFER_SIZE.is_multiple_of(BUFFER_ALIGN),
        "BUFFER_SIZE must be a multiple of BUFFER_ALIGN"
    );
    assert!(
        BUFFER_ALIGN.is_power_of_two(),
        "BUFFER_ALIGN must be a power of two"
    );
}
