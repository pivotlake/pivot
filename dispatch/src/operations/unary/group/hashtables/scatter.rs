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
//!
//! Chunks are carved from a [`PartitionAllocator`], whose buffers each serve a
//! range of partitions and can be spilled to disk while the rows in them stay
//! readable, until the ring takes the buffer's memory back (see
//! [`crate::memory::SpillBuffer`]).

use super::hash_table::{PersistedKey, entry_layout};
use crate::memory::{PartitionAllocator, SpillBuffer};
use crate::operations::unary::group::values::AggregationValue;
use std::marker::PhantomData;
use std::sync::Arc;

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

/// One chunk of rows.
struct ScatterChunk {
    /// The buffer the chunk was carved from, and where in it the chunk starts.
    buffer: Arc<SpillBuffer>,
    offset: usize,
    /// Rows in the chunk once it stopped being filled. While the chunk is the
    /// one being filled, the count lives in `StridedScatterRows::last_rows`.
    rows: usize,
}

/// One partition's scatter rows, stored at the hash table entry layout.
///
/// Serves every value: a compiled value's layout is fully constant, a dynamic
/// value's stride comes from the query. The buffer stores only allocation
/// state and the current chunk's fill; the caller supplies [`ScatterLayout`].
pub struct StridedScatterRows<KP: PersistedKey, V: AggregationValue + ?Sized> {
    chunks: Vec<ScatterChunk>,
    /// Base address of the chunk being filled.
    ///
    /// The push loop hops between thousands of partition buffers in random
    /// order, so their headers are a cache-resident working set and every
    /// field the push touches costs a stalled access. The push therefore
    /// reads only this base, `last_rows` and `current_chunk_capacity`
    /// (adjacent fields), addresses the row as `base + last_rows * stride`,
    /// and writes back only the incremented `last_rows`.
    current_chunk_base: *mut u8,
    /// Rows in the chunk being filled.
    last_rows: usize,
    /// Rows the chunk being filled holds when full. Zero while there is no
    /// chunk to fill, so the next push allocates one.
    current_chunk_capacity: usize,
    /// Rows in every chunk no longer being filled.
    completed_rows: usize,
    /// Chunks allocated so far; only the first is small.
    allocated_chunks: usize,
    // A raw-pointer marker: a plain `(KP, V)` tuple would require `V: Sized`.
    _phantom: PhantomData<(KP, *const V)>,
}

// SAFETY: the cached row pointer refers into a buffer owned by `chunks`, whose
// bytes stay in memory while the chunk is being filled (see `grow`).
unsafe impl<KP: PersistedKey, V: AggregationValue + ?Sized> Send for StridedScatterRows<KP, V> {}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> StridedScatterRows<KP, V> {
    /// Target allocation size for each chunk.
    const CHUNK_BYTES: usize = 64 * 1024;

    /// Rows the next chunk holds when full.
    #[inline(always)]
    fn next_chunk_capacity(&self, layout: ScatterLayout<V>) -> usize {
        if self.allocated_chunks == 0 {
            layout.first_chunk_rows
        } else {
            layout.full_chunk_rows
        }
    }

    /// Whether the last chunk in `chunks` is the one being filled.
    fn is_filling_a_chunk(&self) -> bool {
        self.current_chunk_capacity != 0
    }

    /// Stops filling the current chunk, recording its row count.
    pub fn finish_current_chunk(&mut self) {
        if !self.is_filling_a_chunk() {
            return;
        }
        let current = self.chunks.last_mut().expect("a chunk is being filled");
        current.rows = self.last_rows;
        self.completed_rows += self.last_rows;
        self.last_rows = 0;
        self.current_chunk_capacity = 0;
    }

    /// Allocates the next chunk and starts filling it.
    #[cold]
    /// The layout is copied by value to keep the caller's copy local.
    fn grow(
        &mut self,
        layout: ScatterLayout<V>,
        allocator: &mut PartitionAllocator,
        partition: usize,
    ) {
        self.finish_current_chunk();
        let rows = self.next_chunk_capacity(layout);
        let chunk = allocator.allocate(partition, rows * layout.stride, layout.align);
        // The pointer stays valid for as long as the chunk is being filled: the
        // buffer goes to disk only after `finish_chunk_in` stopped the filling,
        // and its memory is taken back only after it went to disk.
        self.current_chunk_base = unsafe { chunk.buffer.ptr().add(chunk.offset) };
        self.current_chunk_capacity = rows;
        self.allocated_chunks += 1;
        self.chunks.push(ScatterChunk {
            buffer: chunk.buffer,
            offset: chunk.offset,
            rows: 0,
        });
    }

    /// Visits each chunk with its base address and populated row count. The
    /// chunk's bytes are pinned in memory for the duration of the visit.
    #[inline(always)]
    fn for_each_chunk(&self, mut f: impl FnMut(*mut u8, usize)) {
        let chunk_count = self.chunks.len();
        for (index, chunk) in self.chunks.iter().enumerate() {
            let rows = if index + 1 == chunk_count && self.is_filling_a_chunk() {
                self.last_rows
            } else {
                chunk.rows
            };
            let in_memory = chunk.buffer.pin().expect(
                "scatter rows were evicted to disk, and reading them back is not supported yet",
            );
            f(unsafe { in_memory.ptr().add(chunk.offset) }, rows);
        }
    }

    /// Stops filling the current chunk if it lies in `buffer`, which is about to
    /// be written to disk, so that the copy on disk holds every row of every
    /// chunk in it.
    pub fn finish_chunk_in(&mut self, buffer: &Arc<SpillBuffer>) {
        let filling_in_buffer = self.is_filling_a_chunk()
            && Arc::ptr_eq(
                &self.chunks.last().expect("a chunk is being filled").buffer,
                buffer,
            );
        if filling_in_buffer {
            self.finish_current_chunk();
        }
    }
}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> StridedScatterRows<KP, V> {
    /// Creates an empty, lazily allocated buffer.
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            current_chunk_base: std::ptr::null_mut(),
            last_rows: 0,
            current_chunk_capacity: 0,
            completed_rows: 0,
            allocated_chunks: 0,
            _phantom: PhantomData,
        }
    }

    /// Computes the row layout shared by every partition buffer.
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
        let full_chunk_rows = (Self::CHUNK_BYTES / layout.stride).max(1);
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

    /// Number of rows appended.
    pub fn len(&self) -> usize {
        self.completed_rows + self.last_rows
    }

    /// Appends a row and initializes its value in place. `partition` is the
    /// partition these rows belong to, which decides the buffer a new chunk
    /// comes from.
    #[inline(always)]
    pub fn push_with(
        &mut self,
        layout: ScatterLayout<V>,
        allocator: &mut PartitionAllocator,
        partition: usize,
        hash: u64,
        key: KP,
        seed: impl FnOnce(&mut V),
    ) {
        if self.last_rows == self.current_chunk_capacity {
            self.grow(layout, allocator, partition);
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
    pub fn for_each(&self, layout: ScatterLayout<V>, mut f: impl FnMut(u64, &KP, &V)) {
        self.for_each_chunk(|base, rows| {
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
    pub fn for_each_prefetched<const AHEAD: usize>(
        &self,
        layout: ScatterLayout<V>,
        mut f: impl FnMut(u64, &KP, &V, Option<(u64, &KP)>),
    ) {
        self.for_each_chunk(|base, rows| {
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
