//! A worker's consolidated routing stubs: one `(hash, entry)` pair per group
//! entry, grouped by hash bucket.
//!
//! At flush, a worker that stacked tables builds one of these over its whole
//! stack. Keys and aggregate values never move: the tables stay alive and
//! the merge reads them in place. What gets packed is only what the fold
//! actually consumes per entry, the hash (to probe its dedup table) and the
//! entry's address (to hand to the output), so the merge walks one dense,
//! sequential stub stream per worker instead of poking a few entries out of
//! each of hundreds of tables.
//!
//! Like the other runs, a stub run records where each hash bucket starts, so
//! a merge partition's slice is two array reads.

use crate::memory::{BUFFER_SIZE, Slab, SlabAllocator};
use crate::operations::unary::group::hashtables::sorted_run::bucket_bits_for;
use crate::operations::unary::group::hashtables::{
    AggregationValue, BUCKET_BITS, MultiSlabTable, PersistedKey,
};
use crate::operations::unary::group::hll::Hll;

/// One entry's routing stub: its stored hash and its address inside the
/// table that owns it.
#[derive(Clone, Copy)]
pub(crate) struct Stub {
    pub(crate) hash: u64,
    pub(crate) entry: *const u8,
}

/// Whole stubs per 2MB slab.
const STUBS_PER_SLAB: usize = BUFFER_SIZE / size_of::<Stub>();

/// One worker's routing stubs, grouped by hash bucket.
pub struct StubRun {
    /// One pointer per slab, to the slab's first stub.
    slab_bases: Vec<*const Stub>,
    /// Owns the stub memory; only ever read through `slab_bases`.
    _slabs: Vec<Slab>,
    len: usize,
    /// `bucket_starts[b]` is the first stub of bucket `b`; the extra final
    /// element is `len`.
    bucket_starts: Vec<u32>,
    /// Hash-prefix bits this run's index resolves.
    bucket_bits: u32,
}

unsafe impl Send for StubRun {}
unsafe impl Sync for StubRun {}

impl StubRun {
    /// Builds the stubs for every entry of `tables`, folding each hash into
    /// the worker's distinct-count sketch on the way.
    ///
    /// Two branchless slot scans: one to count each bucket, one to place the
    /// stubs. The tables are only read; their entries stay where they are.
    pub fn consolidate<KP: PersistedKey, V: AggregationValue + ?Sized>(
        tables: &[MultiSlabTable<KP, V>],
        allocator: &mut SlabAllocator,
        hll: &mut Hll,
    ) -> Self {
        let len: usize = tables.iter().map(|table| table.len()).sum();
        let slabs =
            allocator.create_strided_slabs(len.max(1), size_of::<Stub>(), align_of::<Stub>());
        let slab_bases: Vec<*const Stub> =
            slabs.iter().map(|slab| slab.ptr as *const Stub).collect();
        let bucket_bits = bucket_bits_for(len);
        let bucket_count = 1usize << bucket_bits;
        let bucket_shift = 64 - bucket_bits;

        let mut bucket_starts = vec![0u32; bucket_count + 1];
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
        for b in 0..bucket_count {
            bucket_starts[b + 1] += bucket_starts[b];
        }

        let stub_at = |index: usize| -> *mut Stub {
            let base = slab_bases[index / STUBS_PER_SLAB];
            unsafe { base.add(index % STUBS_PER_SLAB) as *mut Stub }
        };
        let mut cursors = bucket_starts.clone();
        for table in tables {
            let reader = table.reader::<0>();
            for slot in 0..table.capacity() {
                let entry = reader.entry_ptr(slot) as *const u8;
                let hash = reader.hash_of(entry);
                if hash == 0 {
                    continue;
                }
                hll.add(hash);
                let bucket = (hash >> bucket_shift) as usize;
                unsafe { stub_at(cursors[bucket] as usize).write(Stub { hash, entry }) };
                cursors[bucket] += 1;
            }
        }

        Self {
            slab_bases,
            _slabs: slabs,
            len,
            bucket_starts,
            bucket_bits,
        }
    }

    /// Number of stubs in the run.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Hash-prefix bits this run's index resolves.
    #[inline(always)]
    pub(crate) fn bucket_bits(&self) -> u32 {
        self.bucket_bits
    }

    /// This run's stub range within the global bucket range `[lo, hi)`.
    ///
    /// `lo` and `hi` are indices at [`BUCKET_BITS`] resolution and must be
    /// aligned to this run's coarser resolution, which the merge guarantees
    /// by never splitting finer than any source resolves.
    #[inline(always)]
    pub(crate) fn bucket_range(&self, lo: usize, hi: usize) -> (usize, usize) {
        let shift = BUCKET_BITS - self.bucket_bits;
        (
            self.bucket_starts[lo >> shift] as usize,
            self.bucket_starts[hi >> shift] as usize,
        )
    }

    /// Hands the caller each contiguous stub slice covering `[from, to)`.
    ///
    /// Stubs never straddle a slab boundary, so a range is a handful of
    /// contiguous slices the caller can walk sequentially.
    #[inline(always)]
    pub(crate) fn slices(&self, from: usize, to: usize, mut take: impl FnMut(&[Stub])) {
        let mut index = from;
        while index < to {
            let slab = index / STUBS_PER_SLAB;
            let offset = index % STUBS_PER_SLAB;
            let end = ((slab + 1) * STUBS_PER_SLAB).min(to);
            let base = self.slab_bases[slab];
            let slice = unsafe { std::slice::from_raw_parts(base.add(offset), end - index) };
            take(slice);
            index = end;
        }
    }
}
