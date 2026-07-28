//! Per-partition rows produced by radix scatter and consumed by merge.
//!
//! Compiled values use typed `(hash, key, value)` tuples. Dynamic values use a
//! query-specific stride:
//!
//! ```text
//! hash | key | cell 0 | ... | cell n
//! ```
//!
//! Both forms seed values directly in the destination row.

use super::hash_table::{PersistedKey, entry_layout};
use crate::memory::{Slab, SlabAllocator, SlabVec};
use crate::operations::unary::group::values::AggregationValue;
use std::marker::PhantomData;

/// Storage for one worker's rows in one radix partition.
pub trait ScatterRows<KP: PersistedKey, V: AggregationValue>: Send {
    /// Copyable row layout shared by every partition buffer for a signature.
    type Layout: Copy;

    /// Creates an empty, lazily allocated buffer.
    fn new() -> Self;

    /// Computes the row layout from the shared value context.
    fn layout(ctx: &V::SharedContext) -> Self::Layout {
        Self::layout_with_metadata(V::storage_metadata(ctx))
    }

    /// Computes the row layout from caller-provided value metadata.
    fn layout_with_metadata(metadata: V::StorageMetadata) -> Self::Layout;

    /// Number of rows appended.
    fn len(&self) -> usize;

    /// Appends a row and initializes its value in place.
    fn push_with(
        &mut self,
        layout: Self::Layout,
        allocator: &mut SlabAllocator,
        hash: u64,
        key: KP,
        seed: impl FnOnce(&mut V::Stored),
    );

    /// Visit every row in insertion order.
    fn for_each(&self, layout: Self::Layout, visitor: impl FnMut(u64, &KP, &V::Stored));

    /// Visits rows and exposes a future row for software prefetching.
    fn for_each_prefetched<const AHEAD: usize>(
        &self,
        layout: Self::Layout,
        visitor: impl FnMut(u64, &KP, &V::Stored, Option<(u64, &KP)>),
    );
}

/// Typed scatter rows for a compiled value.
pub struct SizedScatterRows<KP: PersistedKey, V: AggregationValue + Copy>(SlabVec<(u64, KP, V)>);

impl<KP: PersistedKey, V: AggregationValue<Stored = V>> ScatterRows<KP, V>
    for SizedScatterRows<KP, V>
{
    type Layout = ();

    fn new() -> Self {
        Self(SlabVec::new())
    }

    fn layout_with_metadata(_metadata: V::StorageMetadata) {}

    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    fn push_with(
        &mut self,
        _layout: (),
        allocator: &mut SlabAllocator,
        hash: u64,
        key: KP,
        seed: impl FnOnce(&mut V::Stored),
    ) {
        let mut value = V::default();
        seed(&mut value);
        self.0.push(allocator, (hash, key, value));
    }

    #[inline(always)]
    fn for_each(&self, _layout: (), mut visitor: impl FnMut(u64, &KP, &V::Stored)) {
        self.0
            .for_each(|(hash, key, value)| visitor(hash, &key, &value));
    }

    #[inline(always)]
    fn for_each_prefetched<const AHEAD: usize>(
        &self,
        _layout: (),
        mut visitor: impl FnMut(u64, &KP, &V::Stored, Option<(u64, &KP)>),
    ) {
        self.0
            .for_each_prefetched::<AHEAD>(|(hash, key, value), ahead| {
                visitor(
                    hash,
                    &key,
                    &value,
                    ahead.map(|(ahead_hash, ahead_key, _)| (*ahead_hash, ahead_key)),
                );
            });
    }
}

/// Copyable layout for strided scatter rows.
///
/// The caller holds one layout value for all partitions. Keeping it out of
/// each buffer makes partition headers smaller and keeps layout fields local in
/// the scatter loop.
pub struct ScatterLayout<V: AggregationValue> {
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

impl<V: AggregationValue> Clone for ScatterLayout<V> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<V: AggregationValue> Copy for ScatterLayout<V> {}

/// Runtime-strided scatter rows for a dynamic value.
///
/// Rows share the hash table entry layout. The buffer stores only allocation
/// and cursor state; the caller supplies [`ScatterLayout`].
pub struct StridedScatterRows<KP: PersistedKey, V: AggregationValue> {
    chunks: Vec<Slab>,
    /// The next row's address in the chunk being filled.
    cursor: *mut u8,
    /// End (exclusive) of the chunk being filled; `cursor == end` means full (and
    /// both start null, so the first push allocates).
    end: *mut u8,
    /// Rows pushed.
    len: usize,
    _phantom: PhantomData<(KP, V)>,
}

// SAFETY: cached row pointers refer to address-stable slabs owned by `chunks`.
unsafe impl<KP: PersistedKey, V: AggregationValue> Send for StridedScatterRows<KP, V> {}

impl<KP: PersistedKey, V: AggregationValue> StridedScatterRows<KP, V> {
    /// Target allocation size for each chunk.
    const CHUNK_BYTES: usize = 64 * 1024;

    /// Allocates the next chunk and resets the row cursor.
    #[cold]
    /// The layout is copied by value to keep the caller's copy local.
    fn grow(&mut self, layout: ScatterLayout<V>, allocator: &mut SlabAllocator) {
        let rows = if self.chunks.is_empty() {
            layout.first_chunk_rows
        } else {
            layout.full_chunk_rows
        };
        let chunk = allocator.get_aligned_slab(rows * layout.stride, layout.align, false);
        self.cursor = chunk.ptr;
        self.end = unsafe { chunk.ptr.add(rows * layout.stride) };
        self.chunks.push(chunk);
    }

    /// Visits each chunk with its base address and populated row count.
    #[inline(always)]
    fn for_each_chunk(&self, stride: usize, mut f: impl FnMut(*mut u8, usize)) {
        let mut remaining = self.len;
        for chunk in &self.chunks {
            let rows = (chunk.size / stride).min(remaining);
            f(chunk.ptr, rows);
            remaining -= rows;
        }
    }
}

impl<KP: PersistedKey, V: AggregationValue> ScatterRows<KP, V> for StridedScatterRows<KP, V> {
    type Layout = ScatterLayout<V>;

    fn new() -> Self {
        Self {
            chunks: Vec::new(),
            cursor: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
            len: 0,
            _phantom: PhantomData,
        }
    }

    fn layout_with_metadata(metadata: V::StorageMetadata) -> ScatterLayout<V> {
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

    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn push_with(
        &mut self,
        layout: ScatterLayout<V>,
        allocator: &mut SlabAllocator,
        hash: u64,
        key: KP,
        seed: impl FnOnce(&mut V::Stored),
    ) {
        if self.cursor == self.end {
            self.grow(layout, allocator);
        }
        // Field offsets come from the caller's shared layout snapshot.
        let row = self.cursor;
        unsafe {
            *(row.add(layout.hash_offset) as *mut u64) = hash;
            (row.add(layout.key_offset) as *mut KP).write(key);
            seed(V::stored_mut(row.add(layout.value_offset), layout.metadata));
            self.cursor = row.add(layout.stride);
        }
        self.len += 1;
    }

    #[inline(always)]
    fn for_each(&self, layout: ScatterLayout<V>, mut f: impl FnMut(u64, &KP, &V::Stored)) {
        self.for_each_chunk(layout.stride, |base, rows| {
            let mut row = base;
            for _ in 0..rows {
                unsafe {
                    let hash = *(row.add(layout.hash_offset) as *const u64);
                    let key = &*(row.add(layout.key_offset) as *const KP);
                    f(
                        hash,
                        key,
                        V::stored_ref(row.add(layout.value_offset), layout.metadata),
                    );
                    row = row.add(layout.stride);
                }
            }
        });
    }

    #[inline(always)]
    fn for_each_prefetched<const AHEAD: usize>(
        &self,
        layout: ScatterLayout<V>,
        mut f: impl FnMut(u64, &KP, &V::Stored, Option<(u64, &KP)>),
    ) {
        self.for_each_chunk(layout.stride, |base, rows| {
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
                        V::stored_ref(row.add(layout.value_offset), layout.metadata),
                        ahead,
                    );
                    row = row.add(layout.stride);
                }
            }
        });
    }
}
