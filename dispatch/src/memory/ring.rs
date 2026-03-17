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

use crate::env::get_total_memory;
use crate::memory::read_buffer::ReadBuffer;
use crate::memory::write_buffer::WriteBuffer;
use std::fmt::{Debug, Formatter};
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::{io, ptr};

/// Size of each slot in the ring (2 MB).
pub const BUFFER_SIZE: usize = 2 * 1024 * 1024;
/// Page alignment for the mmap region.
const BUFFER_ALIGN: usize = 4096;
/// Total number of slots — half of physical memory divided by slot size.
static BUFFERS: LazyLock<usize> = LazyLock::new(|| get_total_memory() / 2 / BUFFER_SIZE);
/// Bit flag set in `BufferSlot::used` when a writer holds the slot.
const WRITING: u32 = 1 << 31;
/// Global singleton ring, lazily initialized on first access.
pub static RING: LazyLock<Ring> = LazyLock::new(|| Ring::new().unwrap());

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

/// A contiguous mmap-backed arena divided into fixed-size slots.
///
/// Constructed once at startup via [`RING`]. Individual slots are acquired for
/// writing ([`try_write`](Self::try_write)) or reading
/// ([`try_read`](Self::try_read)) and released when the returned handle is
/// dropped.
pub struct Ring {
    pub slots: Vec<BufferSlot>,
}

impl Debug for Ring {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Ring({} x {}MB)", *BUFFERS, BUFFER_SIZE / (1024 * 1024))
    }
}

impl Ring {
    /// Allocates the ring by `mmap`-ing a single contiguous region and splitting
    /// it into [`BUFFER_SIZE`]-aligned slots. Returns an error if `mmap` fails.
    pub fn new() -> io::Result<Self> {
        const_assert_aligned();

        let total_size = *BUFFERS * BUFFER_SIZE;
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                total_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        // Verify alignment
        assert_eq!(
            ptr as usize % BUFFER_ALIGN,
            0,
            "mmap returned unaligned pointer"
        );

        let slots: Vec<BufferSlot> = (0..*BUFFERS)
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

        Ok(Self { slots })
    }

    /// Returns the total number of slots in the ring.
    pub fn len(&self) -> usize {
        *BUFFERS
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

    /// Tries to acquire exclusive write access to slot `idx`.
    ///
    /// Succeeds only if the slot is completely free (`used == 0`). On success,
    /// atomically sets the `WRITING` bit and returns a [`WriteBuffer`].
    /// Returns `None` if the slot is already held by a reader or writer.
    pub fn try_write(&'static self, idx: usize) -> Option<WriteBuffer> {
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
    pub fn try_read(&'static self, idx: usize) -> Option<ReadBuffer> {
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
