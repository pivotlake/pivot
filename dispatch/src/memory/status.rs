//! The memory ring consists of fixed-size blocks.
//! Each block is cached by one of two caches, pinned by an operator, or free.
//! Because every block is the same size, counting rows measures memory usage.

use crate::memory::BUFFER_SIZE;
use crate::memory::clock::{Clock, Owner};
use crate::memory::layout::RingLayout;
use crate::memory::ring::{Ring, SlotUsage};
use crate::memory::tag::{MemoryTag, TAG_COUNT};

/// What one block of the ring is being used for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemoryBlockState {
    /// Held by no one and owned by no cache: sitting in a free pool, ready to
    /// be handed out without evicting anything.
    Free,
    /// Held by an operator as working memory - a
    /// [`WriteBuffer`](crate::memory::WriteBuffer) backing a hash table, a sort
    /// run, a read in flight. Not evictable: it is released when the work that
    /// took it finishes.
    Pinned,
    /// Lent to the compressed cache, holding file bytes as they were read.
    CompressedCache,
    /// Lent to the decompressed cache, holding pages as they were decoded.
    DecompressedCache,
}

/// What one block of the ring currently holds.
#[derive(Clone, Copy, Debug)]
pub struct MemoryBlockStatus {
    /// The block's position in the ring.
    pub slot: usize,
    /// The NUMA node whose region owns the block. Its workers are the only ones
    /// that may allocate or evict it, so pressure is a per-node property.
    pub node: usize,
    pub state: MemoryBlockState,
    /// What the block was last taken for. Meaningful for a block a writer
    /// holds; a free block's is whatever last used it.
    pub tag: MemoryTag,
    /// Live holds on the block's bytes, counting a cache's own hold on what it
    /// has filled, so a cached block at rest commonly reports one. Zero is what
    /// eviction needs: a block anyone is holding cannot be taken back yet. A
    /// block a writer has exclusively reports none, the two being mutually
    /// exclusive.
    pub readers: u32,
}

/// Bytes in one block, the unit everything in the ring is allocated in.
pub fn block_size_bytes() -> usize {
    BUFFER_SIZE
}

/// Read what one block currently holds.
///
/// Its two words are read one after the other while the workers around them
/// keep allocating, caching, and evicting, so this describes the block at the
/// instant it was read. Holding it still to answer would cost far more than
/// the reading is worth, and measuring memory pressure does not need it.
pub(crate) fn read_block_status(
    ring: &Ring,
    clock: &Clock,
    layout: &RingLayout,
    slot: usize,
) -> MemoryBlockStatus {
    let usage = ring.slot_usage(slot);
    // Ownership decides the state first: a cache that owns a block accounts for
    // it whether or not a reader happens to be holding it at this instant,
    // since the block goes on holding cached bytes either way. Only a block no
    // cache owns is the free pool's or an operator's, and there the holder
    // decides.
    let state = match (clock.owner(slot), usage) {
        (Some(Owner::Compressed), _) => MemoryBlockState::CompressedCache,
        (Some(Owner::Decompressed), _) => MemoryBlockState::DecompressedCache,
        (None, SlotUsage::Idle) => MemoryBlockState::Free,
        (None, SlotUsage::Writing | SlotUsage::Reading(_)) => MemoryBlockState::Pinned,
    };
    MemoryBlockStatus {
        slot,
        node: layout.node_of_slot(slot),
        state,
        readers: match usage {
            SlotUsage::Reading(readers) => readers,
            SlotUsage::Idle | SlotUsage::Writing => 0,
        },
        tag: MemoryTag::from_raw(ring.tag(slot)),
    }
}

/// What the ring holds, counted by state: the one line a worker prints when it
/// cannot find anything to evict, so a run that ran out of memory says where
/// the memory went rather than only that it was gone.
///
/// A pinned block with no readers is one a [`WriteBuffer`](crate::memory::WriteBuffer)
/// holds exclusively - working memory an operator is filling, slab memory
/// included. A pinned block with readers is one live [`ReadBuffer`](crate::memory::ReadBuffer)s
/// share outside any cache. A cached block with readers cannot be evicted at
/// this instant even though its cache would give it up, so the two counts
/// separate a working set that is too large from a cache the readers are
/// sitting on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RingCensus {
    pub total: usize,
    pub free: usize,
    pub pinned_writing: usize,
    pub pinned_reading: usize,
    pub compressed: usize,
    pub compressed_held: usize,
    pub decompressed: usize,
    pub decompressed_held: usize,
    /// The most readers any one block has, cached or not.
    pub max_readers: u32,
    /// Blocks a writer holds, by what it took them for.
    pub writers_by_tag: [usize; TAG_COUNT],
}

impl RingCensus {
    pub fn of(blocks: &[MemoryBlockStatus]) -> Self {
        let mut census = Self {
            total: blocks.len(),
            ..Self::default()
        };
        for block in blocks {
            let held = block.readers > 0;
            census.max_readers = census.max_readers.max(block.readers);
            match block.state {
                MemoryBlockState::Free => census.free += 1,
                MemoryBlockState::Pinned if held => census.pinned_reading += 1,
                MemoryBlockState::Pinned => {
                    census.pinned_writing += 1;
                    census.writers_by_tag[block.tag as usize] += 1;
                }
                MemoryBlockState::CompressedCache => {
                    census.compressed += 1;
                    census.compressed_held += usize::from(held);
                }
                MemoryBlockState::DecompressedCache => {
                    census.decompressed += 1;
                    census.decompressed_held += usize::from(held);
                }
            }
        }
        census
    }

    /// The blocks a sweep could take right now: cached, and nobody reading.
    pub fn evictable(&self) -> usize {
        (self.compressed - self.compressed_held) + (self.decompressed - self.decompressed_held)
    }

    /// The stages holding blocks, biggest first, as `shred=1200 widen=300`.
    /// Only what is actually held is named, so a healthy ring says little and
    /// an exhausted one says where it went.
    pub fn writers(&self) -> String {
        let mut stages: Vec<(usize, MemoryTag)> = self
            .writers_by_tag
            .iter()
            .enumerate()
            .filter(|&(_, &blocks)| blocks > 0)
            .map(|(tag, &blocks)| (blocks, MemoryTag::from_raw(tag as u8)))
            .collect();
        stages.sort_by(|a, b| b.0.cmp(&a.0));
        stages
            .iter()
            .map(|(blocks, tag)| format!("{}={blocks}", tag.name()))
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn gigabytes(&self, blocks: usize) -> f64 {
        (blocks * block_size_bytes()) as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    //! The classification, exercised through the worker-side read that serves
    //! it. Each test installs its own memory context (and its own ring) on its
    //! own thread, so nothing is shared and no lock is needed.

    use super::*;
    use crate::memory::{WriteBuffer, init_test_free_pool, memory_ctx};

    /// Every block of the ring, in ring order.
    fn blocks() -> Vec<MemoryBlockStatus> {
        memory_ctx().read_all_blocks()
    }

    fn states() -> Vec<MemoryBlockState> {
        blocks().into_iter().map(|block| block.state).collect()
    }

    /// Take block `slot` for writing, the way an operator asking for working
    /// memory ends up holding one.
    fn hold_for_writing(slot: usize) -> WriteBuffer {
        memory_ctx().ring().try_write(slot).unwrap()
    }

    #[test]
    fn every_block_of_a_fresh_ring_is_free() {
        init_test_free_pool(0);

        let blocks = blocks();

        assert_eq!(blocks.len(), 128);
        assert!(
            blocks
                .iter()
                .all(|block| block.state == MemoryBlockState::Free)
        );
    }

    #[test]
    fn the_whole_ring_is_read_in_order_with_each_block_named_by_its_position() {
        init_test_free_pool(0);

        let slots: Vec<_> = blocks().into_iter().map(|block| block.slot).collect();

        assert_eq!(slots, (0..128).collect::<Vec<_>>());
    }

    #[test]
    fn a_block_held_for_writing_is_pinned() {
        init_test_free_pool(0);

        let _held = hold_for_writing(1);

        assert_eq!(
            states()[0..2],
            [MemoryBlockState::Free, MemoryBlockState::Pinned]
        );
    }

    #[test]
    fn releasing_a_block_frees_it_again() {
        init_test_free_pool(0);
        let held = hold_for_writing(0);

        drop(held);

        assert_eq!(states()[0], MemoryBlockState::Free);
    }

    #[test]
    fn a_block_reports_the_cache_that_owns_it() {
        init_test_free_pool(0);

        memory_ctx().clock().bind(0, Owner::Compressed);
        memory_ctx().clock().bind(1, Owner::Decompressed);

        assert_eq!(
            states()[0..2],
            [
                MemoryBlockState::CompressedCache,
                MemoryBlockState::DecompressedCache,
            ],
        );
    }

    #[test]
    fn a_cached_block_being_read_still_counts_as_cached() {
        init_test_free_pool(0);
        memory_ctx().clock().bind(0, Owner::Decompressed);

        let _reader = memory_ctx().ring().try_read(0).unwrap();

        let block = blocks()[0];
        assert_eq!(block.state, MemoryBlockState::DecompressedCache);
        assert_eq!(block.readers, 1);
    }

    #[test]
    fn readers_are_counted_per_block() {
        init_test_free_pool(0);

        let _first = memory_ctx().ring().try_read(0).unwrap();
        let _second = memory_ctx().ring().try_read(0).unwrap();

        assert_eq!(blocks()[0].readers, 2);
    }

    #[test]
    fn a_block_a_cache_gives_back_is_free_again() {
        init_test_free_pool(0);
        memory_ctx().clock().bind(0, Owner::Compressed);

        memory_ctx().clock().release(0);

        assert_eq!(states()[0], MemoryBlockState::Free);
    }

    #[test]
    fn every_block_is_named_by_the_node_that_owns_it() {
        init_test_free_pool(0);

        let nodes: Vec<_> = blocks().into_iter().map(|block| block.node).collect();

        assert_eq!(nodes, vec![0; 128]);
    }
}
