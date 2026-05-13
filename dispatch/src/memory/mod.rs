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
//! (and subsequently saved in [`FILE_CACHE`]) as well as large allocations.
//!
//! [`WriteBuffer`]s can also be used for many miscellaneous things, such as Vectors and HashTables. It
//! is generally preferred to use [`WriteBuffer`]s as the memory is easily accounted for. See [`SlabAllocator`].

use crate::memory::free_pool::{pop_free_idx, pop_local_dirty_idx};
use std::sync::LazyLock;

mod ring;
pub use ring::{BUFFER_SIZE, RING};

mod file_cache;
pub use file_cache::FILE_CACHE;

mod free_pool;
pub use free_pool::init_free_pool;

#[cfg(test)]
pub use free_pool::init_test_free_pool;

mod write_buffer;
pub use write_buffer::WriteBuffer;

mod read_buffer;
pub use read_buffer::ReadBuffer;

mod slab;
pub use slab::{MultiSlabBuffer, SlabAllocator, SlabBuffer};

mod reader;
use crate::env::get_env_var_with_default;
pub use reader::{MultiBufferReader, ReaderPosition};

static PANIC_ON_EVICT: LazyLock<bool> =
    LazyLock::new(|| get_env_var_with_default("PANIC_ON_EVICT", true));

/// Acquire a [`WriteBuffer`] from the free pool, falling back to eviction.
///
/// When `prefer_zeroed` is true, tries the zeroed pool first — use this when the caller
/// needs zeroed memory (e.g. `WBVec`) so we can skip a memset. When false, tries the
/// dirty pool first — use this when the caller will overwrite the buffer entirely
/// (e.g. decompression, I/O reads) to preserve zeroed buffers for those who need them.
/// Either way, the other pool is used as a fallback if the preferred one is empty.
///
/// If the popped index's ring slot is contended, retries with a fresh index rather
/// than evicting.
pub fn get_write_buffer(prefer_zeroed: bool) -> WriteBuffer {
    loop {
        if let Some(idx) = pop_free_idx(prefer_zeroed) {
            if let Some(r) = RING.try_write(idx) {
                return r;
            } else {
                // Let's try again to get a free idx, we don't want to start evicting yet
                continue;
            }
        }

        if *PANIC_ON_EVICT {
            panic!("Evicting");
        }
        // Nothing free! Let's evict from page cache
        return FILE_CACHE.evict();
    }
}

/// Pop a dirty buffer from this worker's local deque only (no stealing).
pub fn pop_local_dirty_buffer() -> Option<WriteBuffer> {
    pop_local_dirty_idx().and_then(|i| RING.try_write(i))
}
