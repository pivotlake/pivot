//! A worker's consolidated, hash-ordered run of merged group entries.
//!
//! At flush, every worker folds its stack of sealed tables into one of these:
//! entries packed back to back in ascending hash order, deduplicated within
//! the worker. The partition merge then reads one run per worker, so its
//! fan-in stays at the worker count no matter how many tables the input's
//! cardinality forced a worker to stack.
//!
//! Entries reuse the table entry layout (hash, key, stored value at the same
//! offsets and stride) and never straddle a slab boundary, so a
//! [`TableReader`] decodes a dense run the same way it decodes a table.
//!
//! Like a sorted run, a dense run records where each of [`BUCKET_COUNT`] hash
//! ranges starts, so a merge partition's slice is two array reads.

use crate::memory::{BUFFER_SIZE, Slab, SlabAllocator};
use crate::operations::unary::group::hashtables::hash_table::{
    EntryLayout, adjusted_bases, entry_layout,
};
use crate::operations::unary::group::hashtables::sorted_run::BUCKET_COUNT;
use crate::operations::unary::group::hashtables::{AggregationValue, PersistedKey, TableReader};
use std::marker::PhantomData;

/// One worker's merged output: entries in ascending hash order.
pub struct DenseRun<KP: PersistedKey, V: AggregationValue + ?Sized> {
    /// Slab addresses adjusted by their first global entry offset.
    adjusted_slab_bases: Vec<usize>,
    /// Owns the run's memory; only ever read through `adjusted_slab_bases`.
    _slabs: Vec<Slab>,
    entries_per_slab: usize,
    len: usize,
    /// `bucket_starts[b]` is the first entry whose hash prefix is >= `b`; the
    /// extra final element is `len`.
    bucket_starts: Vec<u32>,
    metadata: V::StorageMetadata,
    _phantom: PhantomData<(KP, *const V)>,
}

unsafe impl<KP: PersistedKey, V: AggregationValue + ?Sized> Send for DenseRun<KP, V> {}
unsafe impl<KP: PersistedKey, V: AggregationValue + ?Sized> Sync for DenseRun<KP, V> {}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> DenseRun<KP, V> {
    /// Number of entries in the run.
    pub fn len(&self) -> usize {
        self.len
    }

    /// This run's entry range within the half-open bucket range `[lo, hi)`.
    #[inline(always)]
    pub(crate) fn bucket_range(&self, lo: usize, hi: usize) -> (usize, usize) {
        (
            self.bucket_starts[lo] as usize,
            self.bucket_starts[hi] as usize,
        )
    }

    /// Returns a reader over this run's entries.
    ///
    /// Dense entries are addressed by their index, so the reader's slot
    /// mapping fields are inert; only its entry addressing and layout offsets
    /// are meaningful.
    #[inline(always)]
    pub(crate) fn reader<const N: usize>(&self) -> TableReader<'_, KP, V> {
        let metadata = if N == 0 {
            self.metadata
        } else {
            V::metadata_for_arity::<N>()
        };
        let layout = entry_layout::<KP, V>(metadata);
        unsafe {
            TableReader::new(
                layout,
                self.entries_per_slab,
                0,
                u64::BITS - 1,
                0,
                &self.adjusted_slab_bases,
                metadata,
            )
        }
    }
}

/// Appends entries in ascending hash order and finishes into a [`DenseRun`].
pub(crate) struct DenseRunBuilder<KP: PersistedKey, V: AggregationValue + ?Sized> {
    adjusted_slab_bases: Vec<usize>,
    slabs: Vec<Slab>,
    entries_per_slab: usize,
    layout: EntryLayout,
    len: usize,
    /// Address of the next entry to write, stepped by the stride within a
    /// slab and recomputed at slab boundaries.
    write_ptr: *mut u8,
    bucket_starts: Vec<u32>,
    next_bucket: usize,
    metadata: V::StorageMetadata,
    _phantom: PhantomData<(KP, *const V)>,
}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> DenseRunBuilder<KP, V> {
    /// Allocates room for up to `capacity` entries.
    pub(crate) fn with_capacity(
        allocator: &mut SlabAllocator,
        capacity: usize,
        metadata: V::StorageMetadata,
    ) -> Self {
        let layout = entry_layout::<KP, V>(metadata);
        let entries_per_slab = BUFFER_SIZE / layout.stride;
        let slabs = allocator.create_strided_slabs(capacity.max(1), layout.stride, layout.align);
        let adjusted_slab_bases = adjusted_bases(&slabs, entries_per_slab, layout.stride);
        let write_ptr = slabs[0].ptr;
        Self {
            adjusted_slab_bases,
            slabs,
            entries_per_slab,
            layout,
            len: 0,
            write_ptr,
            bucket_starts: vec![0u32; BUCKET_COUNT + 1],
            next_bucket: 0,
            metadata,
            _phantom: PhantomData,
        }
    }

    /// The address the next entry will occupy, tracking slab boundaries.
    #[inline(always)]
    fn claim_slot(&mut self, hash: u64) -> *mut u8 {
        let bucket = (hash >> (64 - super::sorted_run::BUCKET_BITS)) as usize;
        while self.next_bucket <= bucket {
            self.bucket_starts[self.next_bucket] = self.len as u32;
            self.next_bucket += 1;
        }
        let slot = self.write_ptr;
        self.len += 1;
        if self.len.is_multiple_of(self.entries_per_slab) {
            let slab = self.len / self.entries_per_slab;
            if slab < self.slabs.len() {
                self.write_ptr = self.slabs[slab].ptr;
            }
        } else {
            self.write_ptr = unsafe { self.write_ptr.add(self.layout.stride) };
        }
        slot
    }

    /// Copies one whole entry (hash, key and value) from `src`.
    ///
    /// # Safety
    ///
    /// `src` must point to an entry laid out with this builder's layout.
    #[inline(always)]
    pub(crate) unsafe fn append_raw(&mut self, hash: u64, src: *const u8) {
        let dst = self.claim_slot(hash);
        unsafe { std::ptr::copy_nonoverlapping(src, dst, self.layout.stride) };
    }

    /// Writes a merged entry: the hash, the key, and the value via `write`.
    ///
    /// The value storage handed to `write` is fresh slab memory (zeroed), so
    /// `write` must fully initialise it, e.g. with
    /// [`AggregationValue::copy_from`].
    #[inline(always)]
    pub(crate) fn append_merged(&mut self, hash: u64, key: KP, write: impl FnOnce(&mut V)) {
        let dst = self.claim_slot(hash);
        unsafe {
            *(dst.add(self.layout.hash_offset) as *mut u64) = hash;
            (dst.add(self.layout.key_offset) as *mut KP).write(key);
            write(V::from_entry_mut(
                dst.add(self.layout.value_offset),
                self.metadata,
            ));
        }
    }

    pub(crate) fn finish(mut self) -> DenseRun<KP, V> {
        while self.next_bucket <= BUCKET_COUNT {
            self.bucket_starts[self.next_bucket] = self.len as u32;
            self.next_bucket += 1;
        }
        DenseRun {
            adjusted_slab_bases: self.adjusted_slab_bases,
            _slabs: self.slabs,
            entries_per_slab: self.entries_per_slab,
            len: self.len,
            bucket_starts: self.bucket_starts,
            metadata: self.metadata,
            _phantom: PhantomData,
        }
    }
}
