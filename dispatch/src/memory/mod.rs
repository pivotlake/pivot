//! The `memory` module contains all dispatch's memory management/utilities.
//!
//! There are a few targets in our memory management:
//!     1. Hit as few page faults as possible
//!     2. Know when we're hitting disk and when we have something cached
//!     3. Allow spilling to disk
//!
//! Memory usage is divided into two- [`WriteBuffer`]s (each 2 mb) for large allocations,
//! regular `malloc` for either small allocations or allocations that necessitate continuous memory.
//!
//! Our assumption is that we're the only important process on the machine; we therefore immediately
//! allocate on initialization half the system memory for [`WriteBuffer`]s, as this will be our main
//! usage of memory. Every worker upon initialization faults in all [`WriteBuffer`]s and pushes equal
//! amounts to local pools for use afterward. These [`WriteBuffer`]s will be used for any disk access
//! (and subsequently saved in the [`CompressedCache`](compressed_cache::CompressedCache)) as well as large allocations.
//!
//! [`WriteBuffer`]s can also be used for many miscellaneous things, such as Vectors and HashTables. It
//! is generally preferred to use [`WriteBuffer`]s as the memory is easily accounted for. See [`SlabAllocator`].
//! Bytes on their way out of the engine are held the same way: an encoded Parquet file is a
//! [`FileBytes`] of the slabs its pages were written into, so a large write is accounted against the
//! ring rather than growing the heap beside it.
//!
//! On a multi-NUMA-node machine the one ring is split into a contiguous region per node
//! (see [`RingLayout`]): a worker faults, acquires, and evicts only its own node's slots,
//! so every allocation is node-local memory, while data cached anywhere in the ring stays
//! readable by every worker (a remote read beats re-reading from disk).

mod ring;
pub use ring::{BUFFER_SIZE, Ring, SlotUsage};

mod tag;
pub use tag::{MemoryTag, TAG_COUNT, TagGuard, tagged};

mod status;
pub use status::{MemoryBlockState, MemoryBlockStatus, RingCensus, block_size_bytes};

mod layout;
pub use layout::RingLayout;

pub mod compressed_cache;
pub use compressed_cache::CacheLookup;

pub mod decompressed_cache;
pub use decompressed_cache::{BlockKey, DecompressedCache, Segment};

pub mod clock;
pub use clock::{Clock, Owner};

mod file_bytes;
pub use file_bytes::FileBytes;

mod fill_cursor;
pub use fill_cursor::FillCursor;

mod free_pool;

mod write_buffer;
pub use write_buffer::WriteBuffer;

mod read_buffer;
pub use read_buffer::ReadBuffer;

mod slab;
pub(crate) use slab::{HeapBuffer, MultiTopK, Ranked, SingleTopK, SlabTopK, slots_per_slab};
pub use slab::{MultiSlabBuffer, Slab, SlabAllocator, SlabBuffer, SlabVec};

mod reader;
pub use reader::{MultiBufferReader, ReaderPosition};

mod context;
pub use context::{MemoryContextFactory, has_memory_context, init_memory_context, memory_ctx};

/// Return the heap's freed pages to the operating system now rather than at
/// the end of jemalloc's decay period.
///
/// A burst of heap work that has just finished, such as a compaction merge,
/// otherwise keeps its peak resident for the seconds the decay takes. On a
/// machine whose buffer pool already holds most of the memory, that lingering
/// peak is what the next burst runs out of.
pub fn purge_heap() {
    // jemalloc's index for "every arena".
    const MALLCTL_ARENAS_ALL: usize = 4096;
    let name = format!("arena.{MALLCTL_ARENAS_ALL}.purge\0");
    // SAFETY: the name is NUL-terminated, and the purge reads and writes no
    // mallctl value, so the null pointers and zero length are what it expects.
    unsafe {
        tikv_jemalloc_sys::mallctl(
            name.as_ptr().cast(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        );
    }
}

#[cfg(any(test, feature = "test-util"))]
pub use context::init_test_free_pool;
