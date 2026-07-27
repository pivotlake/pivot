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
pub struct StridedScatterRows<KP: PersistedKey, V: AggregationValue> {
    chunks: Vec<Slab>,
    /// Base address of the chunk being filled.
    cur_base: *mut u8,
    /// Rows in the chunk being filled.
    last_len: usize,
    layout: EntryLayout,
    meta: V::StoredMeta,
    _phantom: PhantomData<KP>,
}

// SAFETY: as for `SlabVec` — the cached base pointer targets address-stable
// slab memory owned by `chunks`.
unsafe impl<KP: PersistedKey, V: AggregationValue> Send for StridedScatterRows<KP, V> {}

impl<KP: PersistedKey, V: AggregationValue> StridedScatterRows<KP, V> {
    /// Byte target per chunk; see [`SlabVec`]'s chunk sizing rationale.
    const CHUNK_BYTES: usize = 64 * 1024;

    /// Rows in chunk `ci`: the first chunk is an eighth of the target so a
    /// barely-used partition stays small.
    fn chunk_rows(&self, ci: usize) -> usize {
        let full = (Self::CHUNK_BYTES / self.layout.stride).max(1);
        if ci == 0 { (full / 8).max(1) } else { full }
    }

    /// The address of row `i` of `chunk`.
    #[inline(always)]
    fn row_ptr(base: *mut u8, stride: usize, i: usize) -> *mut u8 {
        unsafe { base.add(i * stride) }
    }

    /// Visit each chunk as `(base, rows)`.
    #[inline(always)]
    fn for_each_chunk(&self, mut f: impl FnMut(*mut u8, usize)) {
        let n = self.chunks.len();
        for (ci, chunk) in self.chunks.iter().enumerate() {
            let rows = if ci + 1 == n {
                self.last_len
            } else {
                self.chunk_rows(ci)
            };
            f(chunk.ptr, rows);
        }
    }

    /// Read row `ptr`'s header.
    #[inline(always)]
    fn header<'a>(&self, ptr: *mut u8) -> (u64, &'a KP) {
        unsafe {
            (
                *(ptr.add(self.layout.hash_offset) as *const u64),
                &*(ptr.add(self.layout.key_offset) as *const KP),
            )
        }
    }
}

impl<KP: PersistedKey, V: AggregationValue> ScatterRows<KP, V> for StridedScatterRows<KP, V> {
    fn new(ctx: &V::SharedContext) -> Self {
        let meta = V::stored_meta(ctx);
        Self {
            chunks: Vec::new(),
            cur_base: std::ptr::null_mut(),
            last_len: 0,
            layout: entry_layout::<KP, V>(meta),
            meta,
            _phantom: PhantomData,
        }
    }

    fn len(&self) -> usize {
        if self.chunks.is_empty() {
            return 0;
        }
        (0..self.chunks.len() - 1)
            .map(|ci| self.chunk_rows(ci))
            .sum::<usize>()
            + self.last_len
    }

    #[inline(always)]
    fn push_with(
        &mut self,
        allocator: &mut SlabAllocator,
        hash: u64,
        key: KP,
        seed: impl FnOnce(&mut V::Stored),
    ) {
        if self.chunks.is_empty() || self.last_len == self.chunk_rows(self.chunks.len() - 1) {
            let rows = self.chunk_rows(self.chunks.len());
            let chunk =
                allocator.get_aligned_slab(rows * self.layout.stride, self.layout.align, false);
            self.cur_base = chunk.ptr;
            self.chunks.push(chunk);
            self.last_len = 0;
        }
        let row = Self::row_ptr(self.cur_base, self.layout.stride, self.last_len);
        unsafe {
            *(row.add(self.layout.hash_offset) as *mut u64) = hash;
            (row.add(self.layout.key_offset) as *mut KP).write(key);
            seed(V::stored_mut(row.add(self.layout.value_offset), self.meta));
        }
        self.last_len += 1;
    }

    #[inline(always)]
    fn for_each(&self, mut f: impl FnMut(u64, &KP, &V::Stored)) {
        let (stride, value_offset, meta) =
            (self.layout.stride, self.layout.value_offset, self.meta);
        self.for_each_chunk(|base, rows| {
            for i in 0..rows {
                let row = Self::row_ptr(base, stride, i);
                let (hash, key) = self.header(row);
                let stored = unsafe { V::stored_ref(row.add(value_offset), meta) };
                f(hash, key, stored);
            }
        });
    }

    #[inline(always)]
    fn for_each_prefetched<const AHEAD: usize>(
        &self,
        mut f: impl FnMut(u64, &KP, &V::Stored, Option<(u64, &KP)>),
    ) {
        let (stride, value_offset, meta) =
            (self.layout.stride, self.layout.value_offset, self.meta);
        self.for_each_chunk(|base, rows| {
            for i in 0..rows {
                let row = Self::row_ptr(base, stride, i);
                let (hash, key) = self.header(row);
                let stored = unsafe { V::stored_ref(row.add(value_offset), meta) };
                let ahead =
                    (i + AHEAD < rows).then(|| self.header(Self::row_ptr(base, stride, i + AHEAD)));
                f(hash, key, stored, ahead);
            }
        });
    }
}
