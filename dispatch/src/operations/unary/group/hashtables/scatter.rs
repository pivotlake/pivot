//! Per-partition rows buffered for the radix merge.
//!
//! Consume appends `(hash, key, state)` rows without deduplicating them. The
//! merge phase later folds all rows for a partition into its target table.
//! Fixed-size states use typed tuples in [`SizedScatterRows`]. Runtime-sized
//! states use [`StridedScatterRows`], whose raw rows share the hash table's
//! runtime layout. In both cases [`ScatterRows::push_with`] initializes the
//! state directly in its destination.

use super::entry_layout::EntryLayout;
use super::hash_table::PersistedKey;
use crate::memory::{Slab, SlabAllocator, SlabVec};
use crate::operations::unary::group::values::AggregationValue;

/// Buffered rows for one worker and one radix partition.
pub trait ScatterRows<KP: PersistedKey, V: AggregationValue>: Send {
    /// Create an empty, lazily allocated buffer for `V`'s entry layout.
    fn new(ctx: &V::Context) -> Self;

    /// Number of rows appended.
    fn len(&self) -> usize;

    /// Append a row and initialize its state in place.
    ///
    /// `seed` receives the zero-filled destination and must leave it as a fully
    /// initialized `EntryState`.
    fn push_with(
        &mut self,
        allocator: &mut SlabAllocator,
        hash: u64,
        key: KP,
        seed: impl FnOnce(&mut V::EntryState),
    );

    /// Visit every row in insertion order.
    fn for_each(&self, f: impl FnMut(u64, &KP, &V::EntryState));

    /// Like [`for_each`](Self::for_each), but also hands `f` the hash and key
    /// of the row `AHEAD` slots ahead (or `None` near the end of a chunk), so
    /// the merge can warm the target slot and a string key's arena blob before
    /// reaching the row.
    fn for_each_prefetched<const AHEAD: usize>(
        &self,
        f: impl FnMut(u64, &KP, &V::EntryState, Option<(u64, &KP)>),
    );
}

/// Scatter rows for a fixed-arity value: typed `(hash, key, value)` tuples in a
/// [`SlabVec`].
pub struct SizedScatterRows<KP: PersistedKey, V: AggregationValue + Copy>(SlabVec<(u64, KP, V)>);

impl<KP: PersistedKey, V: AggregationValue<EntryState = V>> ScatterRows<KP, V>
    for SizedScatterRows<KP, V>
{
    fn new(_ctx: &V::Context) -> Self {
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
        seed: impl FnOnce(&mut V::EntryState),
    ) {
        let mut value = V::default();
        seed(&mut value);
        self.0.push(allocator, (hash, key, value));
    }

    #[inline(always)]
    fn for_each(&self, mut f: impl FnMut(u64, &KP, &V::EntryState)) {
        self.0.for_each(|(hash, key, value)| f(hash, &key, &value));
    }

    #[inline(always)]
    fn for_each_prefetched<const AHEAD: usize>(
        &self,
        mut f: impl FnMut(u64, &KP, &V::EntryState, Option<(u64, &KP)>),
    ) {
        self.0
            .for_each_prefetched::<AHEAD>(|(hash, key, value), ahead| {
                f(hash, &key, &value, ahead.map(|(h, k, _)| (*h, k)));
            });
    }
}

/// Raw scatter rows using the same runtime layout as hash-table entries.
///
/// Rows are split into roughly 64 KiB chunks. The first chunk is smaller so
/// thousands of lightly used partitions do not each retain a full chunk.
pub struct StridedScatterRows<KP: PersistedKey, V: AggregationValue> {
    chunks: Vec<Slab>,
    /// Base address of the current write chunk.
    current_chunk_base: *mut u8,
    /// Initialized rows in the current write chunk.
    current_chunk_len: usize,
    layout: EntryLayout<KP, V>,
}

// SAFETY: as for `SlabVec` — the cached base pointer targets address-stable
// slab memory owned by `chunks`.
unsafe impl<KP: PersistedKey, V: AggregationValue> Send for StridedScatterRows<KP, V> {}

impl<KP: PersistedKey, V: AggregationValue> StridedScatterRows<KP, V> {
    /// Target size after the deliberately smaller first chunk.
    const CHUNK_BYTES: usize = 64 * 1024;

    /// Capacity of a chunk in rows.
    fn chunk_rows(&self, chunk_index: usize) -> usize {
        let full = (Self::CHUNK_BYTES / self.layout.stride).max(1);
        if chunk_index == 0 {
            (full / 8).max(1)
        } else {
            full
        }
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
        for (chunk_index, chunk) in self.chunks.iter().enumerate() {
            let rows = if chunk_index + 1 == n {
                self.current_chunk_len
            } else {
                self.chunk_rows(chunk_index)
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
    fn new(ctx: &V::Context) -> Self {
        Self {
            chunks: Vec::new(),
            current_chunk_base: std::ptr::null_mut(),
            current_chunk_len: 0,
            layout: EntryLayout::from(ctx),
        }
    }

    fn len(&self) -> usize {
        if self.chunks.is_empty() {
            return 0;
        }
        (0..self.chunks.len() - 1)
            .map(|chunk_index| self.chunk_rows(chunk_index))
            .sum::<usize>()
            + self.current_chunk_len
    }

    #[inline(always)]
    fn push_with(
        &mut self,
        allocator: &mut SlabAllocator,
        hash: u64,
        key: KP,
        seed: impl FnOnce(&mut V::EntryState),
    ) {
        if self.chunks.is_empty()
            || self.current_chunk_len == self.chunk_rows(self.chunks.len() - 1)
        {
            let rows = self.chunk_rows(self.chunks.len());
            let chunk =
                allocator.get_aligned_slab(rows * self.layout.stride, self.layout.align, false);
            self.current_chunk_base = chunk.ptr;
            self.chunks.push(chunk);
            self.current_chunk_len = 0;
        }
        let row = Self::row_ptr(
            self.current_chunk_base,
            self.layout.stride,
            self.current_chunk_len,
        );
        unsafe {
            *(row.add(self.layout.hash_offset) as *mut u64) = hash;
            (row.add(self.layout.key_offset) as *mut KP).write(key);
            seed(V::entry_state_mut(
                row.add(self.layout.state_offset),
                self.layout.state_meta,
            ));
        }
        self.current_chunk_len += 1;
    }

    #[inline(always)]
    fn for_each(&self, mut f: impl FnMut(u64, &KP, &V::EntryState)) {
        let (stride, state_offset, state_meta) = (
            self.layout.stride,
            self.layout.state_offset,
            self.layout.state_meta,
        );
        self.for_each_chunk(|base, rows| {
            for i in 0..rows {
                let row = Self::row_ptr(base, stride, i);
                let (hash, key) = self.header(row);
                let state = unsafe { V::entry_state_ref(row.add(state_offset), state_meta) };
                f(hash, key, state);
            }
        });
    }

    #[inline(always)]
    fn for_each_prefetched<const AHEAD: usize>(
        &self,
        mut f: impl FnMut(u64, &KP, &V::EntryState, Option<(u64, &KP)>),
    ) {
        let (stride, state_offset, state_meta) = (
            self.layout.stride,
            self.layout.state_offset,
            self.layout.state_meta,
        );
        self.for_each_chunk(|base, rows| {
            for i in 0..rows {
                let row = Self::row_ptr(base, stride, i);
                let (hash, key) = self.header(row);
                let state = unsafe { V::entry_state_ref(row.add(state_offset), state_meta) };
                let ahead =
                    (i + AHEAD < rows).then(|| self.header(Self::row_ptr(base, stride, i + AHEAD)));
                f(hash, key, state, ahead);
            }
        });
    }
}
