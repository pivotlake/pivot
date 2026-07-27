//! Per-partition radix scatter rows: `(hash, key, value)` triples appended raw
//! during the consume phase and folded once by the merge.
//!
//! The row storage is chosen by the value, like the hash table's entry storage:
//! a fixed-arity value appends typed tuples to a [`SlabVec`]
//! ([`SizedScatterRows`]); the runtime-arity
//! [`Variable`](crate::operations::unary::group::values::Variable) lays rows
//! out at a per-query stride ([`StridedScatterRows`]), its cells inline in the
//! row exactly as they sit inline in a hash entry. Either way a row's value is
//! **seeded in place** ([`ScatterRows::push_with`]), so scattering never
//! materialises an owned value and never allocates per row.

use super::hash_table::{EntryLayout, PersistedKey, entry_layout};
use crate::memory::{Slab, SlabAllocator, SlabVec};
use crate::operations::unary::group::values::AggregationValue;
use std::marker::PhantomData;

/// One partition's scatter rows for value `V` keyed by `KP`. Appended by one
/// worker during consume, drained once by the merge job that owns the
/// partition.
pub trait ScatterRows<KP: PersistedKey, V: AggregationValue>: Send {
    /// An empty buffer for a signature described by `ctx` (a runtime-arity
    /// value derives its row stride from it). Allocates nothing until the
    /// first push.
    fn new(ctx: &V::SharedContext) -> Self;

    /// Number of rows appended.
    fn len(&self) -> usize;

    /// Append one row, seeding its value in place: `seed` receives the row's
    /// zero-initialised-by-write value slot and must fully initialise it (the
    /// consume path seeds from the batch reader; the abandon path copies a
    /// deduplicated entry's stored value in).
    fn push_with(
        &mut self,
        allocator: &mut SlabAllocator,
        hash: u64,
        key: KP,
        seed: impl FnOnce(&mut V::Stored),
    );

    /// Visit every row in insertion order.
    fn for_each(&self, f: impl FnMut(u64, &KP, &V::Stored));

    /// Like [`for_each`](Self::for_each), but also hands `f` the hash and key
    /// of the row `AHEAD` slots ahead (or `None` near the end of a chunk), so
    /// the merge can warm the target slot and a string key's arena blob before
    /// reaching the row.
    fn for_each_prefetched<const AHEAD: usize>(
        &self,
        f: impl FnMut(u64, &KP, &V::Stored, Option<(u64, &KP)>),
    );
}

/// Scatter rows for a fixed-arity value: typed `(hash, key, value)` tuples in a
/// [`SlabVec`].
pub struct SizedScatterRows<KP: PersistedKey, V: AggregationValue + Copy>(SlabVec<(u64, KP, V)>);

impl<KP: PersistedKey, V: AggregationValue<Stored = V>> ScatterRows<KP, V>
    for SizedScatterRows<KP, V>
{
    fn new(_ctx: &V::SharedContext) -> Self {
        Self(SlabVec::new())
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    fn push_with(
        &mut self,
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
    fn for_each(&self, mut f: impl FnMut(u64, &KP, &V::Stored)) {
        self.0.for_each(|(hash, key, value)| f(hash, &key, &value));
    }

    #[inline(always)]
    fn for_each_prefetched<const AHEAD: usize>(
        &self,
        mut f: impl FnMut(u64, &KP, &V::Stored, Option<(u64, &KP)>),
    ) {
        self.0
            .for_each_prefetched::<AHEAD>(|(hash, key, value), ahead| {
                f(hash, &key, &value, ahead.map(|(h, k, _)| (*h, k)));
            });
    }
}

/// Scatter rows at a runtime stride: each row is `hash | key | cells` with the
/// same per-query layout a hash entry has ([`entry_layout`]), chunked like a
/// [`SlabVec`] (a ~64KB chunk target, a small first chunk so the thousands of
/// mostly-tiny per-partition buffers don't each pin real memory).
///
/// The hot paths never divide or multiply by the runtime stride: chunk
/// capacities are precomputed, pushes bump a row pointer, and the scans walk
/// each chunk by row count with the layout hoisted into locals — the scatter
/// runs once per input row, so it gets the same care as a table probe.
pub struct StridedScatterRows<KP: PersistedKey, V: AggregationValue> {
    chunks: Vec<Slab>,
    /// The next row's address in the chunk being filled.
    cur: *mut u8,
    /// End (exclusive) of the chunk being filled; `cur == end` means full (and
    /// both start null, so the first push allocates).
    end: *mut u8,
    /// Rows pushed.
    len: usize,
    layout: EntryLayout,
    /// Rows per chunk, precomputed: the first chunk and every later one.
    first_chunk_rows: usize,
    full_chunk_rows: usize,
    meta: V::StoredMeta,
    _phantom: PhantomData<KP>,
}

// SAFETY: as for `SlabVec` — the cached row pointers target address-stable
// slab memory owned by `chunks`.
unsafe impl<KP: PersistedKey, V: AggregationValue> Send for StridedScatterRows<KP, V> {}

impl<KP: PersistedKey, V: AggregationValue> StridedScatterRows<KP, V> {
    /// Byte target per chunk; see [`SlabVec`]'s chunk sizing rationale.
    const CHUNK_BYTES: usize = 64 * 1024;

    /// Allocate the next chunk and point the row cursor at it.
    #[cold]
    fn grow(&mut self, allocator: &mut SlabAllocator) {
        let rows = if self.chunks.is_empty() {
            self.first_chunk_rows
        } else {
            self.full_chunk_rows
        };
        let chunk = allocator.get_aligned_slab(rows * self.layout.stride, self.layout.align, false);
        self.cur = chunk.ptr;
        self.end = unsafe { chunk.ptr.add(rows * self.layout.stride) };
        self.chunks.push(chunk);
    }

    /// Visit each chunk as `(base, rows)`. The division deriving a chunk's row
    /// count runs once per chunk, never per row.
    #[inline(always)]
    fn for_each_chunk(&self, mut f: impl FnMut(*mut u8, usize)) {
        let mut remaining = self.len;
        for chunk in &self.chunks {
            let rows = (chunk.size / self.layout.stride).min(remaining);
            f(chunk.ptr, rows);
            remaining -= rows;
        }
    }
}

impl<KP: PersistedKey, V: AggregationValue> ScatterRows<KP, V> for StridedScatterRows<KP, V> {
    fn new(ctx: &V::SharedContext) -> Self {
        let meta = V::stored_meta(ctx);
        let layout = entry_layout::<KP, V>(meta);
        let full_chunk_rows = (Self::CHUNK_BYTES / layout.stride).max(1);
        Self {
            chunks: Vec::new(),
            cur: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
            len: 0,
            first_chunk_rows: (full_chunk_rows / 8).max(1),
            full_chunk_rows,
            layout,
            meta,
            _phantom: PhantomData,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn push_with(
        &mut self,
        allocator: &mut SlabAllocator,
        hash: u64,
        key: KP,
        seed: impl FnOnce(&mut V::Stored),
    ) {
        if self.cur == self.end {
            self.grow(allocator);
        }
        let row = self.cur;
        unsafe {
            *(row.add(self.layout.hash_offset) as *mut u64) = hash;
            (row.add(self.layout.key_offset) as *mut KP).write(key);
            seed(V::stored_mut(row.add(self.layout.value_offset), self.meta));
            self.cur = row.add(self.layout.stride);
        }
        self.len += 1;
    }

    #[inline(always)]
    fn for_each(&self, mut f: impl FnMut(u64, &KP, &V::Stored)) {
        // Hoist the layout into locals so the per-row reads don't re-load it
        // around the caller's writes.
        let EntryLayout {
            hash_offset,
            key_offset,
            value_offset,
            stride,
            ..
        } = self.layout;
        let meta = self.meta;
        self.for_each_chunk(|base, rows| {
            let mut row = base;
            for _ in 0..rows {
                unsafe {
                    let hash = *(row.add(hash_offset) as *const u64);
                    let key = &*(row.add(key_offset) as *const KP);
                    f(hash, key, V::stored_ref(row.add(value_offset), meta));
                    row = row.add(stride);
                }
            }
        });
    }

    #[inline(always)]
    fn for_each_prefetched<const AHEAD: usize>(
        &self,
        mut f: impl FnMut(u64, &KP, &V::Stored, Option<(u64, &KP)>),
    ) {
        let EntryLayout {
            hash_offset,
            key_offset,
            value_offset,
            stride,
            ..
        } = self.layout;
        let meta = self.meta;
        self.for_each_chunk(|base, rows| {
            let mut row = base;
            for i in 0..rows {
                unsafe {
                    let hash = *(row.add(hash_offset) as *const u64);
                    let key = &*(row.add(key_offset) as *const KP);
                    let ahead = (i + AHEAD < rows).then(|| {
                        let a = row.add(AHEAD * stride);
                        (
                            *(a.add(hash_offset) as *const u64),
                            &*(a.add(key_offset) as *const KP),
                        )
                    });
                    f(hash, key, V::stored_ref(row.add(value_offset), meta), ahead);
                    row = row.add(stride);
                }
            }
        });
    }
}
