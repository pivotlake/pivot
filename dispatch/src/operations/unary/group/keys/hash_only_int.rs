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
//! [`KeyColumnBuilder`] below is never materialised (the group emits per-partition
//! counts, not keys). It must therefore only be used via `group_by_distinct_count`.

use crate::arrays::IntBits;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::PersistedKey;
use crate::operations::unary::group::keys::{InlineKey, KeyColumnBuilder, KeyExtractor};
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
    // Dedup is purely by the bijective hash (the key is `()`), so the table
    // counts the single 0-hash key out of band instead of remapping it. Stays
    // in-place (no radix) — the out-of-band count lives on the in-place table.
    const DEDUP_BY_HASH: bool = true;

    type Config = ();
    type Persisted = ();
    type LiveKey<'a, 'b> = ();
    type Stored = InlineKey<()>;
    type Reader<'b> = &'b PrimitiveArray<T>;
    type ColumnBuilder = NoKeyColumnBuilder;
    type Scratch = ();

    fn make_reader<'b>(
        batch: &'b RecordBatch,
        key_cols: &[usize],
        _config: &(),
        _scratch: &'b mut (),
    ) -> Self::Reader<'b> {
        batch.column(key_cols[0]).as_primitive::<T>()
    }

    #[inline(always)]
    fn prepare_and_hash(reader: &mut Self::Reader<'_>, _state: &RandomState, hashes: &mut [u64]) {
        for (i, h) in hashes.iter_mut().enumerate() {
            // Fixed bijective mix (not the keyed RandomState): consistency across
            // workers and exact dedup both require the same 1:1 function everywhere.
            *h = mix64(unsafe { reader.value_unchecked(i) }.to_u64());
        }
    }

    #[inline(always)]
    fn live_key<'a, 'r>(
        _reader: &'r Self::Reader<'_>,
        _idx: usize,
        _arena: &'a mut WorkerArena,
    ) -> Self::LiveKey<'a, 'r> {
    }
}

/// No-op key columns: a keys-only group is only ever used count-only, so this is
/// never materialised.
pub struct NoKeyColumnBuilder;

impl KeyColumnBuilder for NoKeyColumnBuilder {
    type Key = ();
    type Config = ();

    fn with_capacity(_allocator: &mut SlabAllocator, _rows: usize, _config: &()) -> Self {
        NoKeyColumnBuilder
    }

    #[inline(always)]
    fn push(&mut self, _key: &()) {}

    fn finish(
        self,
        _arena: &Arc<SharedArena>,
        _output_buffers: &Arc<[arrow_buffer::Buffer]>,
        _allocator: &mut SlabAllocator,
    ) -> (Vec<Field>, Vec<ArrayRef>) {
        (Vec::new(), Vec::new())
    }
}
