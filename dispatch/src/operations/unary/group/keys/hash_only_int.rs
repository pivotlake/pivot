//! Keys-only integer GROUP BY for exact global `COUNT(DISTINCT int)`.
//!
//! For a global distinct count we never read the keys back — we only need the
//! number of distinct values. So instead of storing the integer key alongside
//! its hash (a 16-byte `Entry` for an `i64`), this extractor stores **only** a
//! *bijective* 64-bit hash of the key and uses `()` as the persisted key, giving
//! an 8-byte `Entry { hash: u64, key: (), value: ZST }`. Dedup is by hash
//! equality, which — because the mix below is a bijection on `u64` — is exactly
//! key equality. Half the bytes per entry means roughly twice the entries per
//! cache line, which is the dominant cost for this latency-bound, high-cardinality
//! build (the probe is bound by cache-miss latency, so denser entries win).
//!
//! This is paired with the count-only group output, so the no-op
//! [`KeyColumns`] below is never materialised (the group emits per-partition
//! counts, not keys). It must therefore only be used via `group_by_distinct_count`.

use super::int_pair::IntBits;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::PersistedKey;
use crate::operations::unary::group::keys::{KeyColumns, KeyExtractor};
use ahash::RandomState;
use arrow_array::cast::AsArray;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{ArrayRef, PrimitiveArray, RecordBatch};
use arrow_schema::Field;
use std::marker::PhantomData;
use std::sync::Arc;

/// `()` is a valid persisted key: zero-sized, so an entry that stores only the
/// (bijective) hash carries no separate key bytes.
impl PersistedKey for () {}

/// SplitMix64 finalizer — a *bijection* on `u64` (each step is invertible: an
/// xor-shift-right and a multiply by an odd constant). Distinct inputs therefore
/// map to distinct outputs, so deduping by this value is exact, and its strong
/// avalanche gives uniform top bits for slot/partition placement.
///
/// Every step maps 0 to 0, so the unique preimage of the hash-table's
/// empty-slot sentinel (hash 0) is the key whose bit pattern is 0 — that one
/// key is counted out of band rather than stored (see `DEDUP_BY_HASH`).
#[inline(always)]
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A keys-only [`KeyExtractor`] over a single integer column: stores a bijective
/// hash and a `()` key (8-byte entries). For exact global `COUNT(DISTINCT col)`.
pub struct HashOnlyIntKeyExtractor<T: ArrowPrimitiveType>(PhantomData<T>)
where
    T::Native: IntBits;

unsafe impl<T: ArrowPrimitiveType> Send for HashOnlyIntKeyExtractor<T> where T::Native: IntBits {}

impl<T: ArrowPrimitiveType + Send + 'static> KeyExtractor for HashOnlyIntKeyExtractor<T>
where
    T::Native: IntBits,
{
    // Dedup is purely by the bijective hash (the key is `()`), so the single
    // 0-hash key is counted out of band instead of remapped — both the in-place
    // insert and the post-switch radix scatter skip it (see `AggregatedTable`).
    const DEDUP_BY_HASH: bool = true;

    // Radix-eligible: a scattered row is `(hash, (), ZST)` — just the 8-byte
    // bijective hash — and the merge's insert-by-stored-hash dedups `()` keys by
    // hash equality, which the bijection makes exactly key equality. So a
    // high-cardinality distinct build scatters into cache-resident partitions
    // instead of probing one giant per-worker table that misses cache on every
    // row, and loses no exactness doing it.
    const SUPPORTS_RADIX: bool = true;

    // Entries here are 8 bytes — a quarter of the ~32-byte entries the default
    // switch threshold is tuned for — so the in-place table stays L2-resident to
    // 4x the slot count. Defer the switch by the same factor: same byte budget,
    // and a medium-cardinality distinct (tens of thousands of keys) keeps the
    // cheap in-place path instead of scattering into a 4096-way merge.
    const RADIX_SWITCH_SCALE: usize = 4;

    type Config = ();
    type Persisted = ();
    type LiveKey<'a, 'b> = ();
    type PersistedLiveKey<'a> = ();
    type Reader<'b> = &'b PrimitiveArray<T>;
    type Columns = NoKeyColumns;

    fn make_reader<'b>(
        batch: &'b RecordBatch,
        key_cols: &[usize],
        _config: &(),
    ) -> Self::Reader<'b> {
        batch.column(key_cols[0]).as_primitive::<T>()
    }

    #[inline(always)]
    fn hash(reader: &Self::Reader<'_>, idx: usize, _state: &RandomState) -> u64 {
        // Fixed bijective mix (not the keyed RandomState): consistency across
        // workers and exact dedup both require the same 1:1 function everywhere.
        mix64(unsafe { reader.value_unchecked(idx) }.to_u64())
    }

    #[inline(always)]
    fn live_key<'a, 'b>(
        _reader: &Self::Reader<'b>,
        _idx: usize,
        _arena: &'a mut WorkerArena,
    ) -> Self::LiveKey<'a, 'b> {
    }

    fn resolve_persisted(_arena: &SharedArena, _persisted: ()) {}
}

/// No-op key columns: a keys-only group is only ever used count-only, so this is
/// never materialised.
pub struct NoKeyColumns;

impl KeyColumns for NoKeyColumns {
    type Key = ();
    type Config = ();

    fn with_capacity(_allocator: &mut SlabAllocator, _rows: usize, _config: &()) -> Self {
        NoKeyColumns
    }

    #[inline(always)]
    fn push(&mut self, _key: &()) {}

    fn finish(
        self,
        _arena: &Arc<SharedArena>,
        _allocator: &mut SlabAllocator,
    ) -> (Vec<Field>, Vec<ArrayRef>) {
        (Vec::new(), Vec::new())
    }
}
