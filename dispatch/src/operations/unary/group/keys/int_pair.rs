//! Two-integer-key GROUP BY.
//!
//! Keys: a pair of integer columns packed into a single `u128` (first key's
//! bits in the high 64, second's in the low 64),
//! which is `Copy`/`Hash`/`Eq` and needs no arena. The aggregate value is the
//! separate concern of a
//! [`AggregationValue`](crate::operations::unary::group::values::AggregationValue)
//! (typically [`Dynamic`](crate::operations::unary::group::values::Dynamic)).

use crate::arrays::{ArrayBuilder, IntBits, PrimitiveBuilder};
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::keys::{InlineKey, KeyColumnBuilder, KeyExtractor};
use ahash::RandomState;
use arrow_array::cast::AsArray;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{ArrayRef, PrimitiveArray, RecordBatch};
use arrow_schema::Field;
use std::marker::PhantomData;
use std::sync::Arc;

/// Pack a key pair into a `u128`: first key's bits in the high 64, second's low.
#[inline(always)]
fn pack<A: IntBits, B: IntBits>(a: A, b: B) -> u128 {
    ((a.to_u64() as u128) << 64) | (b.to_u64() as u128)
}

/// Per-batch reader for the pair extractor: the two key columns.
pub struct PairReader<'b, A: ArrowPrimitiveType, B: ArrowPrimitiveType> {
    a: &'b PrimitiveArray<A>,
    b: &'b PrimitiveArray<B>,
}

impl<A: ArrowPrimitiveType, B: ArrowPrimitiveType> PairReader<'_, A, B>
where
    A::Native: IntBits,
    B::Native: IntBits,
{
    #[inline(always)]
    fn packed(&self, idx: usize) -> u128 {
        let a = unsafe { self.a.value_unchecked(idx) };
        let b = unsafe { self.b.value_unchecked(idx) };
        pack(a, b)
    }
}

/// `GROUP BY (A, B)` over two integer key columns.
pub struct IntPairKeyExtractor<A: ArrowPrimitiveType, B: ArrowPrimitiveType>(PhantomData<(A, B)>);

unsafe impl<A: ArrowPrimitiveType, B: ArrowPrimitiveType> Send for IntPairKeyExtractor<A, B> {}

impl<A, B> KeyExtractor for IntPairKeyExtractor<A, B>
where
    A: ArrowPrimitiveType + Send + 'static,
    B: ArrowPrimitiveType + Send + 'static,
    A::Native: IntBits,
    B::Native: IntBits,
{
    // Keep deduplicating in the bounded table after the radix transition so
    // repeated keys fold in place instead of scattering one raw row each.
    const RADIX_DEDUP_BEFORE_SCATTER: bool = true;

    type Config = ();
    type Persisted = u128;
    type LiveKey<'a, 'b> = u128;
    type Stored = InlineKey<u128>;
    type Reader<'b> = PairReader<'b, A, B>;
    type ColumnBuilder = IntPairKeyColumnBuilder<A, B>;
    type Scratch = ();

    fn make_reader<'b>(
        batch: &'b RecordBatch,
        key_cols: &[usize],
        _config: &(),
        _scratch: &'b mut (),
    ) -> Self::Reader<'b> {
        let a = batch.column(key_cols[0]).as_primitive::<A>();
        let b = batch.column(key_cols[1]).as_primitive::<B>();
        PairReader { a, b }
    }

    #[inline(always)]
    fn prepare_and_hash(reader: &mut Self::Reader<'_>, state: &RandomState, hashes: &mut [u64]) {
        for (i, h) in hashes.iter_mut().enumerate() {
            *h = state.hash_one(reader.packed(i));
        }
    }

    #[inline(always)]
    fn live_key<'a, 'r>(
        reader: &'r Self::Reader<'_>,
        idx: usize,
        _arena: &'a mut WorkerArena,
    ) -> Self::LiveKey<'a, 'r> {
        reader.packed(idx)
    }
}

/// Emits the two unpacked key columns (`k0`, `k1`).
pub struct IntPairKeyColumnBuilder<A: ArrowPrimitiveType, B: ArrowPrimitiveType>
where
    A::Native: IntBits,
    B::Native: IntBits,
{
    a: PrimitiveBuilder<A>,
    b: PrimitiveBuilder<B>,
}

impl<A: ArrowPrimitiveType, B: ArrowPrimitiveType> KeyColumnBuilder
    for IntPairKeyColumnBuilder<A, B>
where
    A::Native: IntBits,
    B::Native: IntBits,
{
    type Key = u128;
    type Config = ();

    fn with_capacity(allocator: &mut SlabAllocator, rows: usize, _config: &()) -> Self {
        Self {
            a: PrimitiveBuilder::<A>::with_capacity(allocator, rows),
            b: PrimitiveBuilder::<B>::with_capacity(allocator, rows),
        }
    }

    #[inline(always)]
    fn push(&mut self, key: &u128) {
        let packed = *key;
        self.a.push(&A::Native::from_u64((packed >> 64) as u64), 1);
        self.b.push(&B::Native::from_u64(packed as u64), 1);
    }

    fn finish(
        self,
        _arena: &Arc<SharedArena>,
        _output_buffers: &Arc<[arrow_buffer::Buffer]>,
        _allocator: &mut SlabAllocator,
    ) -> (Vec<Field>, Vec<ArrayRef>) {
        let fields = vec![
            Field::new("k0", A::DATA_TYPE, false),
            Field::new("k1", B::DATA_TYPE, false),
        ];
        (
            fields,
            vec![self.a.into_array(None), self.b.into_array(None)],
        )
    }
}
