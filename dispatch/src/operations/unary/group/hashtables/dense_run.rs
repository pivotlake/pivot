//! A worker's consolidated, bucket-grouped array of group entries.
//!
//! At flush, every worker copies the entries of its whole table stack into
//! one of these: entries packed back to back, grouped by the top
//! [`BUCKET_BITS`](super::BUCKET_BITS) of their hash. The merge then reads
//! one run per worker, so a partition job visits worker-count sources instead
//! of one per stacked table, and each source slice is a dense, sequential
//! byte range.
//!
//! Consolidation is a pure copy: entries are not deduplicated (the partition
//! fold combines repeats anyway), so it is two branchless passes at memory
//! bandwidth. Entries reuse the table entry layout, and never straddle a slab
//! boundary, so a [`TableReader`] decodes a dense run the same way it decodes
//! a table.

use crate::memory::{BUFFER_SIZE, Slab, SlabAllocator};
use crate::operations::unary::group::hashtables::hash_table::{adjusted_bases, entry_layout};
use crate::operations::unary::group::hashtables::sorted_run::BUCKET_COUNT;
use crate::operations::unary::group::hashtables::{
    AggregationValue, MultiSlabTable, PersistedKey, TableReader,
};
use crate::operations::unary::group::hll::Hll;
use std::marker::PhantomData;

/// One worker's consolidated entries, grouped by hash bucket.
pub struct DenseRun<KP: PersistedKey, V: AggregationValue + ?Sized> {
    /// Slab addresses adjusted by their first global entry offset.
    adjusted_slab_bases: Vec<usize>,
    /// Owns the run's memory; only ever read through `adjusted_slab_bases`.
    _slabs: Vec<Slab>,
    entries_per_slab: usize,
    len: usize,
    /// `bucket_starts[b]` is the first entry of bucket `b`; the extra final
    /// element is `len`.
    bucket_starts: Vec<u32>,
    metadata: V::StorageMetadata,
    _phantom: PhantomData<(KP, *const V)>,
}

unsafe impl<KP: PersistedKey, V: AggregationValue + ?Sized> Send for DenseRun<KP, V> {}
unsafe impl<KP: PersistedKey, V: AggregationValue + ?Sized> Sync for DenseRun<KP, V> {}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> DenseRun<KP, V> {
    /// Copies every entry of `tables` into one bucket-grouped run, folding
    /// each hash into the worker's distinct-count sketch on the way.
    ///
    /// Two branchless slot scans: one to count each bucket, one to place the
    /// entries.
    pub fn consolidate(
        tables: &[MultiSlabTable<KP, V>],
        allocator: &mut SlabAllocator,
        metadata: V::StorageMetadata,
        hll: &mut Hll,
    ) -> Self {
        let layout = entry_layout::<KP, V>(metadata);
        let entries_per_slab = BUFFER_SIZE / layout.stride;
        let len: usize = tables.iter().map(|table| table.len()).sum();
        let slabs = allocator.create_strided_slabs(len.max(1), layout.stride, layout.align);
        let adjusted_slab_bases = adjusted_bases(&slabs, entries_per_slab, layout.stride);
        let bucket_shift = 64 - super::sorted_run::BUCKET_BITS;

        let mut bucket_starts = vec![0u32; BUCKET_COUNT + 1];
        for table in tables {
            let reader = table.reader::<0>();
            for slot in 0..table.capacity() {
                let hash = reader.hash_of(reader.entry_ptr(slot));
                // Count 0 of the histogram absorbs the empty slots (hash 0 is
                // the empty sentinel); real buckets are offset by one so the
                // occupied prefix sums stay exact, with no branch per slot.
                let bucket = (hash >> bucket_shift) as usize;
                bucket_starts[(bucket + 1) * (hash != 0) as usize] += 1;
            }
        }
        bucket_starts[0] = 0;
        for b in 0..BUCKET_COUNT {
            bucket_starts[b + 1] += bucket_starts[b];
        }

        let entry_at = |index: usize| -> *mut u8 {
            let slab = index / entries_per_slab;
            let base = adjusted_slab_bases[slab];
            base.wrapping_add(index * layout.stride) as *mut u8
        };
        let mut cursors = bucket_starts.clone();
        for table in tables {
            let reader = table.reader::<0>();
            for slot in 0..table.capacity() {
                let src = reader.entry_ptr(slot);
                let hash = reader.hash_of(src);
                if hash == 0 {
                    continue;
                }
                hll.add(hash);
                let bucket = (hash >> bucket_shift) as usize;
                let dst = entry_at(cursors[bucket] as usize);
                unsafe { std::ptr::copy_nonoverlapping(src, dst, layout.stride) };
                cursors[bucket] += 1;
            }
        }

        Self {
            adjusted_slab_bases,
            _slabs: slabs,
            entries_per_slab,
            len,
            bucket_starts,
            metadata,
            _phantom: PhantomData,
        }
    }

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
