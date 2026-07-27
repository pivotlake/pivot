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

use super::hash_table::{PersistedKey, entry_layout};
use crate::memory::{Slab, SlabAllocator, SlabVec};
use crate::operations::unary::group::values::AggregationValue;
use std::marker::PhantomData;

/// One partition's scatter rows for value `V` keyed by `KP`. Appended by one
/// worker during consume, drained once by the merge job that owns the
/// partition.
pub trait ScatterRows<KP: PersistedKey, V: AggregationValue>: Send {
    /// A register-resident snapshot of the row layout, derived from the
    /// signature. Every partition buffer of one scatter shares it, so the
    /// scatter loop computes it once and hands it to every call; it is *not*
    /// stored per buffer, because a scatter keeps thousands of partition
    /// buffers and touches one per row in random order — every byte of the
    /// buffer struct multiplies a cache-resident working set. `()` for typed
    /// rows.
    type Geometry: Copy;

    /// An empty buffer. Allocates nothing until the first push.
    fn new() -> Self;

    /// The shared row-layout snapshot for `ctx`'s signature (see
    /// [`Geometry`](Self::Geometry)).
    fn geometry(ctx: &V::SharedContext) -> Self::Geometry;

    /// Number of rows appended.
    fn len(&self) -> usize;

    /// Append one row, seeding its value in place: `seed` receives the row's
    /// zero-initialised-by-write value slot and must fully initialise it (the
    /// consume path seeds from the batch reader; the abandon path copies a
    /// deduplicated entry's stored value in). `geo` must be this scatter's
    /// [`geometry`](Self::geometry).
    fn push_with(
        &mut self,
        geo: Self::Geometry,
        allocator: &mut SlabAllocator,
        hash: u64,
        key: KP,
        seed: impl FnOnce(&mut V::Stored),
    );

    /// Visit every row in insertion order.
    fn for_each(&self, geo: Self::Geometry, f: impl FnMut(u64, &KP, &V::Stored));

    /// Like [`for_each`](Self::for_each), but also hands `f` the hash and key
    /// of the row `AHEAD` slots ahead (or `None` near the end of a chunk), so
    /// the merge can warm the target slot and a string key's arena blob before
    /// reaching the row.
    fn for_each_prefetched<const AHEAD: usize>(
        &self,
        geo: Self::Geometry,
        f: impl FnMut(u64, &KP, &V::Stored, Option<(u64, &KP)>),
    );
}

/// Scatter rows for a fixed-arity value: typed `(hash, key, value)` tuples in a
/// [`SlabVec`].
pub struct SizedScatterRows<KP: PersistedKey, V: AggregationValue + Copy>(SlabVec<(u64, KP, V)>);

impl<KP: PersistedKey, V: AggregationValue<Stored = V>> ScatterRows<KP, V>
    for SizedScatterRows<KP, V>
{
    type Geometry = ();

    fn new() -> Self {
        Self(SlabVec::new())
    }

    fn geometry(_ctx: &V::SharedContext) {}

    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    fn push_with(
        &mut self,
        _geo: (),
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
    fn for_each(&self, _geo: (), mut f: impl FnMut(u64, &KP, &V::Stored)) {
        self.0.for_each(|(hash, key, value)| f(hash, &key, &value));
    }

    #[inline(always)]
    fn for_each_prefetched<const AHEAD: usize>(
        &self,
        _geo: (),
        mut f: impl FnMut(u64, &KP, &V::Stored, Option<(u64, &KP)>),
    ) {
        self.0
            .for_each_prefetched::<AHEAD>(|(hash, key, value), ahead| {
                f(hash, &key, &value, ahead.map(|(h, k, _)| (*h, k)));
            });
    }
}

/// The strided rows' caller-held layout snapshot: the field offsets, the
/// stride, the chunk row capacities, and the value's view metadata, all
/// runtime values kept in registers across a scatter loop rather than in the
/// thousands of per-partition buffers the loop hops between.
pub struct ScatterGeometry<V: AggregationValue> {
    hash_offset: usize,
    key_offset: usize,
    value_offset: usize,
    stride: usize,
    align: usize,
    /// Rows in the first chunk and in every later one; see [`SlabVec`]'s
    /// chunk sizing rationale.
    first_chunk_rows: usize,
    full_chunk_rows: usize,
    meta: V::StoredMeta,
}

impl<V: AggregationValue> Clone for ScatterGeometry<V> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<V: AggregationValue> Copy for ScatterGeometry<V> {}

/// Scatter rows at a runtime stride: each row is `hash | key | cells` with the
/// same per-query layout a hash entry has ([`entry_layout`]), chunked like a
/// [`SlabVec`].
///
/// The struct holds only the chunk list and the row cursor — one cache line,
/// like a `SlabVec` — because a scatter keeps one buffer per radix partition
/// (thousands) and touches one per row in hash order: the buffer headers form
/// a cache-resident working set that every extra field inflates. Everything
/// layout-shaped lives in the caller's [`ScatterGeometry`].
pub struct StridedScatterRows<KP: PersistedKey, V: AggregationValue> {
    chunks: Vec<Slab>,
    /// The next row's address in the chunk being filled.
    cur: *mut u8,
    /// End (exclusive) of the chunk being filled; `cur == end` means full (and
    /// both start null, so the first push allocates).
    end: *mut u8,
    /// Rows pushed.
    len: usize,
    _phantom: PhantomData<(KP, V)>,
}

// SAFETY: as for `SlabVec` — the cached row pointers target address-stable
// slab memory owned by `chunks`.
unsafe impl<KP: PersistedKey, V: AggregationValue> Send for StridedScatterRows<KP, V> {}

impl<KP: PersistedKey, V: AggregationValue> StridedScatterRows<KP, V> {
    /// Byte target per chunk; see [`SlabVec`]'s chunk sizing rationale.
    const CHUNK_BYTES: usize = 64 * 1024;

    /// Allocate the next chunk and point the row cursor at it.
    #[cold]
    fn grow(&mut self, geo: &ScatterGeometry<V>, allocator: &mut SlabAllocator) {
        let rows = if self.chunks.is_empty() {
            geo.first_chunk_rows
        } else {
            geo.full_chunk_rows
        };
        let chunk = allocator.get_aligned_slab(rows * geo.stride, geo.align, false);
        self.cur = chunk.ptr;
        self.end = unsafe { chunk.ptr.add(rows * geo.stride) };
        self.chunks.push(chunk);
    }

    /// Visit each chunk as `(base, rows)`. The division deriving a chunk's row
    /// count runs once per chunk, never per row.
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
    type Geometry = ScatterGeometry<V>;

    fn new() -> Self {
        Self {
            chunks: Vec::new(),
            cur: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
            len: 0,
            _phantom: PhantomData,
        }
    }

    fn geometry(ctx: &V::SharedContext) -> ScatterGeometry<V> {
        let meta = V::stored_meta(ctx);
        let layout = entry_layout::<KP, V>(meta);
        let full_chunk_rows = (Self::CHUNK_BYTES / layout.stride).max(1);
        ScatterGeometry {
            hash_offset: layout.hash_offset,
            key_offset: layout.key_offset,
            value_offset: layout.value_offset,
            stride: layout.stride,
            align: layout.align,
            first_chunk_rows: (full_chunk_rows / 8).max(1),
            full_chunk_rows,
            meta,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn push_with(
        &mut self,
        geo: ScatterGeometry<V>,
        allocator: &mut SlabAllocator,
        hash: u64,
        key: KP,
        seed: impl FnOnce(&mut V::Stored),
    ) {
        if self.cur == self.end {
            self.grow(&geo, allocator);
        }
        // Every offset comes off the caller's snapshot (registers); the only
        // buffer state this touches is the row cursor.
        let row = self.cur;
        unsafe {
            *(row.add(geo.hash_offset) as *mut u64) = hash;
            (row.add(geo.key_offset) as *mut KP).write(key);
            seed(V::stored_mut(row.add(geo.value_offset), geo.meta));
            self.cur = row.add(geo.stride);
        }
        self.len += 1;
    }

    #[inline(always)]
    fn for_each(&self, geo: ScatterGeometry<V>, mut f: impl FnMut(u64, &KP, &V::Stored)) {
        self.for_each_chunk(geo.stride, |base, rows| {
            let mut row = base;
            for _ in 0..rows {
                unsafe {
                    let hash = *(row.add(geo.hash_offset) as *const u64);
                    let key = &*(row.add(geo.key_offset) as *const KP);
                    f(
                        hash,
                        key,
                        V::stored_ref(row.add(geo.value_offset), geo.meta),
                    );
                    row = row.add(geo.stride);
                }
            }
        });
    }

    #[inline(always)]
    fn for_each_prefetched<const AHEAD: usize>(
        &self,
        geo: ScatterGeometry<V>,
        mut f: impl FnMut(u64, &KP, &V::Stored, Option<(u64, &KP)>),
    ) {
        self.for_each_chunk(geo.stride, |base, rows| {
            let mut row = base;
            for i in 0..rows {
                unsafe {
                    let hash = *(row.add(geo.hash_offset) as *const u64);
                    let key = &*(row.add(geo.key_offset) as *const KP);
                    let ahead = (i + AHEAD < rows).then(|| {
                        let a = row.add(AHEAD * geo.stride);
                        (
                            *(a.add(geo.hash_offset) as *const u64),
                            &*(a.add(geo.key_offset) as *const KP),
                        )
                    });
                    f(
                        hash,
                        key,
                        V::stored_ref(row.add(geo.value_offset), geo.meta),
                        ahead,
                    );
                    row = row.add(geo.stride);
                }
            }
        });
    }
}
