//! Per-partition rows produced by radix scatter and consumed by merge.
//!
//! Every value stores its rows at the hash table entry layout:
//!
//! ```text
//! hash | key | stored value
//! ```
//!
//! For a compiled value the layout is fully constant; for a dynamic value the
//! stride comes from the query. Values are seeded directly in the destination
//! row either way.

use super::hash_table::{PersistedKey, entry_layout};
use crate::memory::{Slab, SlabAllocator};
use crate::operations::unary::group::values::AggregationValue;
use std::marker::PhantomData;

/// Copyable layout for strided scatter rows.
///
/// The caller holds one layout value for all partitions. Keeping it out of
/// each buffer makes partition headers smaller and keeps layout fields local in
/// the scatter loop.
pub struct ScatterLayout<V: AggregationValue + ?Sized> {
    hash_offset: usize,
    key_offset: usize,
    value_offset: usize,
    stride: usize,
    align: usize,
    /// Row counts for the smaller first chunk and later full chunks.
    first_chunk_rows: usize,
    full_chunk_rows: usize,
    metadata: V::StorageMetadata,
}

impl<V: AggregationValue + ?Sized> Clone for ScatterLayout<V> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<V: AggregationValue + ?Sized> Copy for ScatterLayout<V> {}

/// Marks a bucket with no chunk, or the last chunk of its chain.
const NO_CHUNK: u32 = u32::MAX;

/// One chunk of scatter rows, linked to the next chunk of the same bucket.
struct ScatterChunk {
    slab: Slab,
    next: u32,
}

/// One worker's scatter buckets, one per radix partition, and every chunk
/// behind them.
///
/// The chunks of all buckets live in one vector, each bucket chaining its own
/// through [`ScatterChunk::next`], so a worker's whole scatter state is two
/// allocations however many buckets it has. A pool holds workers times buckets
/// of these buckets, and a heap list per bucket would cost that many
/// allocations to build and, worse, to free from whichever thread drops the
/// merge state last.
pub struct PartitionBuffers<KP: PersistedKey, V: AggregationValue + ?Sized> {
    buckets: Vec<StridedScatterRows<KP, V>>,
    chunks: Vec<ScatterChunk>,
}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> PartitionBuffers<KP, V> {
    /// Creates `bucket_count` empty buckets; chunks are allocated as rows arrive.
    pub fn new(bucket_count: usize) -> Self {
        Self {
            buckets: (0..bucket_count)
                .map(|_| StridedScatterRows::new())
                .collect(),
            // Every bucket that receives a row takes at least one chunk.
            chunks: Vec::with_capacity(bucket_count),
        }
    }

    /// Computes the row layout shared by every bucket.
    ///
    /// `N` is the specialized slot count of the surrounding
    /// [`dispatch_arity`](AggregationValue::dispatch_arity) body: a nonzero
    /// `N` derives the layout from that constant so it folds, while `N == 0`
    /// reads the runtime metadata off the shared context.
    pub fn layout<const N: usize>(ctx: &V::SharedContext) -> ScatterLayout<V> {
        let metadata = if N == 0 {
            V::storage_metadata(ctx)
        } else {
            V::metadata_for_arity::<N>()
        };
        let layout = entry_layout::<KP, V>(metadata);
        let full_chunk_rows = (StridedScatterRows::<KP, V>::CHUNK_BYTES / layout.stride).max(1);
        ScatterLayout {
            hash_offset: layout.hash_offset,
            key_offset: layout.key_offset,
            value_offset: layout.value_offset,
            stride: layout.stride,
            align: layout.align,
            first_chunk_rows: (full_chunk_rows / 8).max(1),
            full_chunk_rows,
            metadata,
        }
    }

    /// How many scatter buckets the worker holds.
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Rows appended to `bucket`.
    pub fn bucket_len(&self, bucket: usize) -> usize {
        self.buckets[bucket].len()
    }

    /// Rows appended across every bucket.
    pub fn total_rows(&self) -> usize {
        self.buckets.iter().map(|bucket| bucket.len()).sum()
    }

    /// Appends a row to `bucket` and initializes its value in place.
    #[inline(always)]
    pub fn push_with(
        &mut self,
        bucket: usize,
        layout: ScatterLayout<V>,
        allocator: &mut SlabAllocator,
        hash: u64,
        key: KP,
        seed: impl FnOnce(&mut V),
    ) {
        self.buckets[bucket].push_with(&mut self.chunks, layout, allocator, hash, key, seed)
    }

    /// Visits every row of `bucket` in insertion order.
    #[inline(always)]
    pub fn for_each(&self, bucket: usize, layout: ScatterLayout<V>, f: impl FnMut(u64, &KP, &V)) {
        self.buckets[bucket].for_each(&self.chunks, layout, f)
    }

    /// Visits every row of `bucket` and exposes a future row for software
    /// prefetching.
    #[inline(always)]
    pub fn for_each_prefetched<const AHEAD: usize>(
        &self,
        bucket: usize,
        layout: ScatterLayout<V>,
        f: impl FnMut(u64, &KP, &V, Option<(u64, &KP)>),
    ) {
        self.buckets[bucket].for_each_prefetched::<AHEAD>(&self.chunks, layout, f)
    }
}

/// One partition's scatter rows, stored at the hash table entry layout.
///
/// Serves every value: a compiled value's layout is fully constant, a dynamic
/// value's stride comes from the query. The bucket stores only its chain of
/// chunk indices and the current chunk's fill; the caller supplies
/// [`ScatterLayout`] and the chunk vector.
struct StridedScatterRows<KP: PersistedKey, V: AggregationValue + ?Sized> {
    /// Base address of the chunk being filled.
    ///
    /// The push loop hops between thousands of partition buffers in random
    /// order, so their headers are a cache-resident working set and every
    /// field the push touches costs a stalled access. The push therefore
    /// reads only this base and `last_rows` (adjacent fields), addresses the
    /// row as `base + last_rows * stride`, and writes back only the
    /// incremented `last_rows`; fullness is a compare against the caller's
    /// register-resident layout, not a stored end pointer.
    current_chunk_base: *mut u8,
    /// Rows in the chunk being filled.
    last_rows: usize,
    /// Rows in all full chunks; maintained only when a chunk fills.
    completed_rows: usize,
    /// Chunks in this bucket's chain; the chunk being filled is the last.
    chunk_count: u32,
    /// Chain ends into the worker's chunk vector.
    first_chunk: u32,
    last_chunk: u32,
    // A raw-pointer marker: a plain `(KP, V)` tuple would require `V: Sized`.
    _phantom: PhantomData<(KP, *const V)>,
}

// SAFETY: the cached row pointer refers to an address-stable slab owned by the
// worker's chunk vector, which moves between threads together with the bucket.
unsafe impl<KP: PersistedKey, V: AggregationValue + ?Sized> Send for StridedScatterRows<KP, V> {}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> StridedScatterRows<KP, V> {
    /// Target allocation size for each chunk.
    const CHUNK_BYTES: usize = 64 * 1024;

    /// Rows the chunk at `index` holds when full.
    #[inline(always)]
    fn chunk_capacity(layout: ScatterLayout<V>, index: usize) -> usize {
        if index == 0 {
            layout.first_chunk_rows
        } else {
            layout.full_chunk_rows
        }
    }

    /// Allocates the next chunk, links it onto the chain, and resets the row
    /// count. The layout is copied by value to keep the caller's copy local.
    #[cold]
    fn grow(
        &mut self,
        chunks: &mut Vec<ScatterChunk>,
        layout: ScatterLayout<V>,
        allocator: &mut SlabAllocator,
    ) {
        let rows = Self::chunk_capacity(layout, self.chunk_count as usize);
        let slab = allocator.get_aligned_slab(rows * layout.stride, layout.align, false);
        self.current_chunk_base = slab.ptr;
        self.completed_rows += self.last_rows;
        self.last_rows = 0;
        let index = u32::try_from(chunks.len()).expect("scatter chunk index fits in u32");
        assert!(
            index != NO_CHUNK,
            "scatter chunk index collides with the chain end marker"
        );
        chunks.push(ScatterChunk {
            slab,
            next: NO_CHUNK,
        });
        if self.chunk_count == 0 {
            self.first_chunk = index;
        } else {
            chunks[self.last_chunk as usize].next = index;
        }
        self.last_chunk = index;
        self.chunk_count += 1;
    }

    /// Visits each chunk with its base address and populated row count.
    #[inline(always)]
    fn for_each_chunk(
        &self,
        chunks: &[ScatterChunk],
        layout: ScatterLayout<V>,
        mut f: impl FnMut(*mut u8, usize),
    ) {
        let chunk_count = self.chunk_count as usize;
        let mut chunk = self.first_chunk;
        for index in 0..chunk_count {
            let entry = &chunks[chunk as usize];
            let rows = if index + 1 == chunk_count {
                self.last_rows
            } else {
                Self::chunk_capacity(layout, index)
            };
            f(entry.slab.ptr, rows);
            chunk = entry.next;
        }
    }
}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> StridedScatterRows<KP, V> {
    /// Creates an empty, lazily allocated bucket.
    fn new() -> Self {
        Self {
            current_chunk_base: std::ptr::null_mut(),
            last_rows: 0,
            completed_rows: 0,
            chunk_count: 0,
            first_chunk: NO_CHUNK,
            last_chunk: NO_CHUNK,
            _phantom: PhantomData,
        }
    }

    /// Number of rows appended.
    fn len(&self) -> usize {
        self.completed_rows + self.last_rows
    }

    /// Appends a row and initializes its value in place.
    #[inline(always)]
    fn push_with(
        &mut self,
        chunks: &mut Vec<ScatterChunk>,
        layout: ScatterLayout<V>,
        allocator: &mut SlabAllocator,
        hash: u64,
        key: KP,
        seed: impl FnOnce(&mut V),
    ) {
        // The capacity compare uses the caller's layout, kept in registers,
        // so an over-full check costs no buffer-header access.
        if self.chunk_count == 0
            || self.last_rows == Self::chunk_capacity(layout, self.chunk_count as usize - 1)
        {
            self.grow(chunks, layout, allocator);
        }
        // Field offsets come from the caller's shared layout snapshot.
        unsafe {
            let row = self.current_chunk_base.add(self.last_rows * layout.stride);
            *(row.add(layout.hash_offset) as *mut u64) = hash;
            (row.add(layout.key_offset) as *mut KP).write(key);
            seed(V::from_entry_mut(
                row.add(layout.value_offset),
                layout.metadata,
            ));
        }
        self.last_rows += 1;
    }

    /// Visits every row in insertion order.
    #[inline(always)]
    fn for_each(
        &self,
        chunks: &[ScatterChunk],
        layout: ScatterLayout<V>,
        mut f: impl FnMut(u64, &KP, &V),
    ) {
        self.for_each_chunk(chunks, layout, |base, rows| {
            let mut row = base;
            for _ in 0..rows {
                unsafe {
                    let hash = *(row.add(layout.hash_offset) as *const u64);
                    let key = &*(row.add(layout.key_offset) as *const KP);
                    f(
                        hash,
                        key,
                        V::from_entry(row.add(layout.value_offset), layout.metadata),
                    );
                    row = row.add(layout.stride);
                }
            }
        });
    }

    /// Visits rows and exposes a future row for software prefetching.
    #[inline(always)]
    fn for_each_prefetched<const AHEAD: usize>(
        &self,
        chunks: &[ScatterChunk],
        layout: ScatterLayout<V>,
        mut f: impl FnMut(u64, &KP, &V, Option<(u64, &KP)>),
    ) {
        self.for_each_chunk(chunks, layout, |base, rows| {
            let mut row = base;
            for i in 0..rows {
                unsafe {
                    let hash = *(row.add(layout.hash_offset) as *const u64);
                    let key = &*(row.add(layout.key_offset) as *const KP);
                    let ahead = (i + AHEAD < rows).then(|| {
                        let ahead_row = row.add(AHEAD * layout.stride);
                        (
                            *(ahead_row.add(layout.hash_offset) as *const u64),
                            &*(ahead_row.add(layout.key_offset) as *const KP),
                        )
                    });
                    f(
                        hash,
                        key,
                        V::from_entry(row.add(layout.value_offset), layout.metadata),
                        ahead,
                    );
                    row = row.add(layout.stride);
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use crate::operations::unary::group::values::{Compiled, CountSlot};

    type CountValue = Compiled<(CountSlot,), u8>;

    fn read_bucket(buffers: &PartitionBuffers<i32, CountValue>, bucket: usize) -> Vec<(u64, i32)> {
        let layout = PartitionBuffers::<i32, CountValue>::layout::<0>(&());
        let mut rows = Vec::new();
        buffers.for_each(bucket, layout, |hash, key, _| rows.push((hash, *key)));
        rows
    }

    #[test]
    fn interleaved_buckets_keep_their_rows_in_order_across_chunks() {
        init_test_free_pool(64);
        let mut allocator = SlabAllocator::new(true);
        let layout = PartitionBuffers::<i32, CountValue>::layout::<0>(&());
        let mut buffers = PartitionBuffers::<i32, CountValue>::new(4);
        let rows_past_two_chunks = layout.first_chunk_rows + layout.full_chunk_rows + 7;

        for i in 0..rows_past_two_chunks {
            for bucket in [3, 1] {
                let key = (bucket * 100_000 + i) as i32;
                buffers.push_with(bucket, layout, &mut allocator, key as u64, key, |_| {});
            }
        }

        assert_eq!(buffers.bucket_len(3), rows_past_two_chunks);
        assert_eq!(buffers.bucket_len(1), rows_past_two_chunks);
        assert_eq!(buffers.bucket_len(0), 0);
        assert_eq!(buffers.total_rows(), 2 * rows_past_two_chunks);
        assert!(read_bucket(&buffers, 0).is_empty());
        for bucket in [1, 3] {
            let expected: Vec<(u64, i32)> = (0..rows_past_two_chunks)
                .map(|i| (bucket * 100_000 + i) as i32)
                .map(|key| (key as u64, key))
                .collect();
            assert_eq!(read_bucket(&buffers, bucket), expected);
        }
    }
}
