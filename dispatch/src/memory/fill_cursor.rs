//! A worker's bump cursor over a shared ring slot being packed with cache
//! entries, shared by the compressed
//! [`CompressedCache`](super::compressed_cache::CompressedCache) and the
//! decompressed [`DecompressedCache`](super::decompressed_cache::DecompressedCache)
//! (each worker owns one cursor per cache, on its
//! [`MemoryContext`](super::context::MemoryContext)).
//!
//! The slot is held as a [`ReadBuffer`] (a reader pin) so entries already packed
//! into it stay readable and the evictor cannot take the slot while the worker
//! keeps filling its tail. Rotation to a fresh slot stays with each cache - they
//! reset different per-slot metadata and bind different owners - but the shape is
//! shared: take a write buffer (holding no map lock, since that may evict), reset
//! the slot's metadata while it is still held exclusively, bind the cache's owner
//! in the clock, then publish the pin.

use crate::memory::read_buffer::ReadBuffer;
use crate::memory::ring::BUFFER_SIZE;
use std::sync::Arc;

/// See the module doc. The bump pointer is byte-granular; the compressed cache
/// only ever advances it in whole 4 KB blocks, the decompressed cache packs
/// byte-exact page spans.
pub struct FillCursor {
    /// The slot currently being packed, pinned. `None` before the first miss and
    /// after the owning cache's `clear`.
    pub(crate) buffer: Option<Arc<ReadBuffer>>,
    /// Ring slot index of `buffer`.
    pub(crate) slot_idx: usize,
    /// Bump pointer: the next free byte in `buffer` (0..=`BUFFER_SIZE`).
    pub(crate) next_byte: usize,
}

impl FillCursor {
    /// An empty cursor - the next miss takes a fresh fill buffer.
    pub fn empty() -> Self {
        FillCursor {
            buffer: None,
            slot_idx: 0,
            next_byte: 0,
        }
    }

    /// Whether the cursor must rotate to a fresh slot before it can hand out
    /// another byte.
    pub(crate) fn is_exhausted(&self) -> bool {
        self.buffer.is_none() || self.next_byte == BUFFER_SIZE
    }
}
