//! Buffers for partitioned data, arranged so that spilling frees whole buffers.
//!
//! Chunks of partitioned data (a radix scatter's partitions, say) are small and arrive in
//! random partition order. Carving them all out of one bump allocator leaves every 2MB buffer
//! holding chunks of dozens of unrelated partitions, so no partition can be dropped without
//! the others keeping its buffer alive.
//!
//! This allocator gives each buffer a range of partitions instead. The first buffer serves
//! all partitions. When it fills, the range splits in two and each half gets a buffer of its
//! own, and so on, down to a smallest range below which a range simply takes buffer after
//! buffer. A full buffer therefore holds chunks of a narrow range of partitions. Splitting
//! stops at the smallest range because every split leaves a fresh, mostly empty buffer per
//! half: with no floor, a worker would pin about twice its data right after a round of splits.
//!
//! Full buffers are handed out for spilling in the order the merge can best do without
//! them, see [`PartitionAllocator::take_full_buffer`].

use crate::memory::{BUFFER_SIZE, SpillBuffer, memory_ctx};
use std::cmp::Reverse;
use std::sync::Arc;

/// Partitions in the narrowest range a buffer is dedicated to.
const SMALLEST_RANGE: usize = 64;

/// A chunk carved out of a spill buffer: the buffer, and where in it the chunk starts.
pub struct BufferChunk {
    pub buffer: Arc<SpillBuffer>,
    pub offset: usize,
}

/// A buffer that filled up, ready to be spilled.
pub struct FullBuffer {
    pub buffer: Arc<SpillBuffer>,
    /// Every partition with a chunk in the buffer, so a chunk still being filled can be
    /// finished before the bytes go to disk. A partition may appear more than once.
    pub partitions: Vec<u16>,
    /// Depth of the range the buffer served: 0 for the whole range, one more per split.
    depth: u32,
}

impl FullBuffer {
    /// The first partition the merge needs this buffer for.
    fn lowest_partition(&self) -> u16 {
        *self
            .partitions
            .iter()
            .min()
            .expect("a buffer fills only by holding chunks")
    }
}

/// The buffer a range is carving chunks from.
struct OpenBuffer {
    buffer: Arc<SpillBuffer>,
    /// Where the next chunk starts.
    offset: usize,
    partitions: Vec<u16>,
    depth: u32,
}

impl OpenBuffer {
    fn acquire(depth: u32) -> Self {
        Self {
            buffer: SpillBuffer::new(memory_ctx().get_write_buffer(false)),
            offset: 0,
            partitions: Vec::new(),
            depth,
        }
    }

    /// Carves `bytes` at `align` from this buffer, or returns `None` when they do not fit.
    fn carve(&mut self, partition: usize, bytes: usize, align: usize) -> Option<BufferChunk> {
        let start = self.offset.next_multiple_of(align);
        if start + bytes > BUFFER_SIZE {
            return None;
        }
        self.offset = start + bytes;
        if self.partitions.last() != Some(&(partition as u16)) {
            self.partitions.push(partition as u16);
        }
        Some(BufferChunk {
            buffer: self.buffer.clone(),
            offset: start,
        })
    }

    /// Whether the last carve used up the buffer's last byte.
    fn is_used_up(&self) -> bool {
        self.offset == BUFFER_SIZE
    }

    fn into_full(self) -> FullBuffer {
        FullBuffer {
            buffer: self.buffer,
            partitions: self.partitions,
            depth: self.depth,
        }
    }
}

/// Hands out chunks of partitioned data from buffers dedicated to partition ranges. See the
/// module documentation.
pub struct PartitionAllocator {
    partition_count: usize,
    /// How far the range tree may split: the whole range is depth 0, and a range at the
    /// greatest depth covers [`SMALLEST_RANGE`] partitions (or fewer, for small partition
    /// counts).
    max_depth: u32,
    /// Each range's open buffer, once a partition in the range has asked for a chunk. A
    /// range that split no longer opens buffers; its halves do. Indexed as a binary heap:
    /// the whole range is 1, and a range `n`'s halves are `2n` and `2n + 1`. A partition's
    /// range at depth `d` is its top `d` bits, with the depth's bit set above them.
    open: Vec<Option<OpenBuffer>>,
    /// For each smallest range, the depth of the range that currently serves it. Every
    /// partition of a smallest range is always served by the same range.
    depth_of_smallest_range: Vec<u32>,
    /// Buffers that filled up and were not spilled yet.
    full: Vec<FullBuffer>,
}

impl PartitionAllocator {
    /// An allocator for `partition_count` partitions, which must be a power of two.
    pub fn new(partition_count: usize) -> Self {
        assert!(partition_count.is_power_of_two());
        let smallest_range = SMALLEST_RANGE.min(partition_count);
        let smallest_range_count = partition_count / smallest_range;
        let max_depth = smallest_range_count.trailing_zeros();
        Self {
            partition_count,
            max_depth,
            open: (0..2 * smallest_range_count).map(|_| None).collect(),
            depth_of_smallest_range: vec![0; smallest_range_count],
            full: Vec::new(),
        }
    }

    fn range_index(&self, partition: usize, depth: u32) -> usize {
        let shift = self.partition_count.trailing_zeros() - depth;
        (1 << depth) | (partition >> shift)
    }

    fn depth_of(&self, partition: usize) -> u32 {
        let shift = self.partition_count.trailing_zeros() - self.max_depth;
        self.depth_of_smallest_range[partition >> shift]
    }

    /// Carves `bytes` at `align` for `partition` out of its range's buffer. A buffer that
    /// cannot fit the chunk, or that the chunk uses up, is set aside for spilling, and the
    /// range either splits (above the greatest depth) or opens another buffer.
    pub fn allocate(&mut self, partition: usize, bytes: usize, align: usize) -> BufferChunk {
        assert!(
            bytes + align <= BUFFER_SIZE,
            "a chunk must fit in one buffer"
        );
        loop {
            let depth = self.depth_of(partition);
            let range_index = self.range_index(partition, depth);
            let open = self.open[range_index].get_or_insert_with(|| OpenBuffer::acquire(depth));
            let carved = open.carve(partition, bytes, align);
            if carved.is_none() || open.is_used_up() {
                let full = self.open[range_index]
                    .take()
                    .expect("the buffer just carved from");
                self.full.push(full.into_full());
                if depth < self.max_depth {
                    self.split(partition, depth);
                }
            }
            if let Some(chunk) = carved {
                return chunk;
            }
        }
    }

    /// Splits the range serving `partition` at `depth` into its two halves. The halves
    /// open buffers of their own when first asked.
    fn split(&mut self, partition: usize, depth: u32) {
        let shift = self.partition_count.trailing_zeros() - self.max_depth;
        let smallest_ranges_per_range = 1 << (self.max_depth - depth);
        let first = (partition >> shift) & !(smallest_ranges_per_range - 1);
        for smallest_range in first..first + smallest_ranges_per_range {
            self.depth_of_smallest_range[smallest_range] = depth + 1;
        }
    }

    /// The next full buffer to spill, if there is one.
    ///
    /// The merge reads partitions in ascending order, and a spilled buffer is first needed
    /// at the lowest partition it holds, so the buffer whose lowest partition is highest
    /// goes first: the merge then runs from memory for as long as possible. Among buffers
    /// with the same lowest partition, the one that served the wider range goes first.
    pub fn take_full_buffer(&mut self) -> Option<FullBuffer> {
        let next = (0..self.full.len())
            .max_by_key(|&i| (self.full[i].lowest_partition(), Reverse(self.full[i].depth)))?;
        Some(self.full.swap_remove(next))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;

    const CHUNK: usize = 64 * 1024;

    fn fill_one_buffer(allocator: &mut PartitionAllocator, partition: usize) -> Vec<BufferChunk> {
        (0..BUFFER_SIZE / CHUNK)
            .map(|_| allocator.allocate(partition, CHUNK, 8))
            .collect()
    }

    fn share_a_buffer(first: &BufferChunk, second: &BufferChunk) -> bool {
        Arc::ptr_eq(&first.buffer, &second.buffer)
    }

    #[test]
    fn all_partitions_start_in_one_buffer() {
        init_test_free_pool(8);
        let mut allocator = PartitionAllocator::new(4096);

        let first = allocator.allocate(0, CHUNK, 8);
        let last = allocator.allocate(4095, CHUNK, 8);

        assert!(share_a_buffer(&first, &last));
    }

    #[test]
    fn a_full_buffer_splits_its_range_in_two() {
        init_test_free_pool(8);
        let mut allocator = PartitionAllocator::new(4096);
        let _shared = fill_one_buffer(&mut allocator, 0);

        let low = allocator.allocate(0, CHUNK, 8);
        let low_again = allocator.allocate(2047, CHUNK, 8);
        let high = allocator.allocate(2048, CHUNK, 8);

        assert!(share_a_buffer(&low, &low_again));
        assert!(!share_a_buffer(&low, &high));
    }

    #[test]
    fn a_full_buffer_remembers_its_partitions() {
        init_test_free_pool(8);
        let mut allocator = PartitionAllocator::new(4096);
        let _first = allocator.allocate(5, CHUNK, 8);
        let _rest = fill_one_buffer(&mut allocator, 9);

        let full = allocator.take_full_buffer().unwrap();

        assert_eq!(full.partitions, vec![5, 9]);
        assert!(allocator.take_full_buffer().is_none());
    }

    #[test]
    fn full_buffers_spill_in_order_of_their_lowest_partition() {
        init_test_free_pool(8);
        let mut allocator = PartitionAllocator::new(4096);
        let _shared_high = allocator.allocate(4095, CHUNK, 8);
        let _shared = fill_one_buffer(&mut allocator, 0);
        let _low = fill_one_buffer(&mut allocator, 0);
        let _high = fill_one_buffer(&mut allocator, 4095);

        let first = allocator.take_full_buffer().unwrap();
        let second = allocator.take_full_buffer().unwrap();
        let third = allocator.take_full_buffer().unwrap();

        assert_eq!(first.partitions, vec![4095]);
        assert_eq!(second.partitions, vec![4095, 0]);
        assert_eq!(third.partitions, vec![0]);
    }

    #[test]
    fn the_smallest_range_keeps_taking_buffers_without_splitting() {
        init_test_free_pool(8);
        let mut allocator = PartitionAllocator::new(64);
        let _first = fill_one_buffer(&mut allocator, 3);

        let _second = fill_one_buffer(&mut allocator, 3);
        let third = allocator.allocate(3, CHUNK, 8);
        let neighbour = allocator.allocate(60, CHUNK, 8);

        assert!(share_a_buffer(&third, &neighbour));
    }
}
