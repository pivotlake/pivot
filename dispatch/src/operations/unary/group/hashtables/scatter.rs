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

/// One partition's scatter rows, stored at the hash table entry layout.
///
/// Serves every value: a compiled value's layout is fully constant, a dynamic
/// value's stride comes from the query. The buffer stores only allocation
/// state and the current chunk's fill; the caller supplies [`ScatterLayout`].
pub struct StridedScatterRows<KP: PersistedKey, V: AggregationValue + ?Sized> {
    chunks: ChunkList,
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
    // A raw-pointer marker: a plain `(KP, V)` tuple would require `V: Sized`.
    _phantom: PhantomData<(KP, *const V)>,
}

/// A bucket's chunks, the first two held inline.
///
/// A pool has workers times buckets of these, and a heap-allocated list per
/// bucket costs an allocation on the scatter path and, when the buckets are
/// released, a free that contends on the allocator's locks across every
/// worker at once. The small first chunk and one full chunk cover a bucket
/// holding a worker's even share of rows, so only a skewed bucket spills to
/// the heap.
struct ChunkList {
    inline: [Option<Slab>; 2],
    spilled: Vec<Slab>,
    /// Chunks held, inline and spilled; the push path reads it per row.
    len: usize,
}

impl ChunkList {
    const fn new() -> Self {
        Self {
            inline: [None, None],
            spilled: Vec::new(),
            len: 0,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn push(&mut self, chunk: Slab) {
        match self.inline.get_mut(self.len) {
            Some(slot) => *slot = Some(chunk),
            None => self.spilled.push(chunk),
        }
        self.len += 1;
    }

    fn iter(&self) -> impl Iterator<Item = &Slab> {
        self.inline.iter().flatten().chain(self.spilled.iter())
    }
}

// SAFETY: cached row pointers refer to address-stable slabs owned by `chunks`.
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

    /// Allocates the next chunk and resets the row count.
    #[cold]
    /// The layout is copied by value to keep the caller's copy local.
    fn grow(&mut self, layout: ScatterLayout<V>, allocator: &mut SlabAllocator) {
        let rows = Self::chunk_capacity(layout, self.chunks.len());
        let chunk = allocator.get_aligned_slab(rows * layout.stride, layout.align, false);
        self.current_chunk_base = chunk.ptr;
        self.completed_rows += self.last_rows;
        self.last_rows = 0;
        self.chunks.push(chunk);
    }

    /// Visits each chunk with its base address and populated row count.
    #[inline(always)]
    fn for_each_chunk(&self, layout: ScatterLayout<V>, mut f: impl FnMut(*mut u8, usize)) {
        let chunk_count = self.chunks.len();
        for (index, chunk) in self.chunks.iter().enumerate() {
            let rows = if index + 1 == chunk_count {
                self.last_rows
            } else {
                Self::chunk_capacity(layout, index)
            };
            f(chunk.ptr, rows);
        }
    }
}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> StridedScatterRows<KP, V> {
    /// Creates an empty, lazily allocated buffer.
    pub fn new() -> Self {
        Self {
            chunks: ChunkList::new(),
            current_chunk_base: std::ptr::null_mut(),
            last_rows: 0,
            completed_rows: 0,
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

    /// Appends a row and initializes its value in place.
    #[inline(always)]
    pub fn push_with(
        &mut self,
        layout: ScatterLayout<V>,
        allocator: &mut SlabAllocator,
        hash: u64,
        key: KP,
        seed: impl FnOnce(&mut V),
    ) {
        // The capacity compare uses the caller's layout, kept in registers,
        // so an over-full check costs no buffer-header access.
        if self.chunks.is_empty()
            || self.last_rows == Self::chunk_capacity(layout, self.chunks.len() - 1)
        {
            self.grow(layout, allocator);
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
        self.for_each_chunk(layout, |base, rows| {
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
        self.for_each_chunk(layout, |base, rows| {
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
