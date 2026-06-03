//! Two-integer-key GROUP BY.
//!
//! Keys: a pair of integer columns (e.g. `WatchID`, `ClientIP`) packed into a
//! single `u128` (first key's bits in the high 64, second's in the low 64),
//! which is `Copy`/`Hash`/`Eq` and needs no arena. The aggregate value is the
//! separate concern of a
//! [`ValueExtractor`](crate::operations::unary::group::value_extractions::ValueExtractor)
//! (typically [`AggRowValueExtractor`](crate::operations::unary::group::value_extractions::AggRowValueExtractor)).

use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::key_extractions::{KeyColumns, KeyExtractor};
use ahash::RandomState;
use arrow_array::builder::PrimitiveBuilder;
use arrow_array::cast::AsArray;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{ArrayRef, PrimitiveArray, RecordBatch};
use arrow_schema::Field;
use std::marker::PhantomData;
use std::sync::Arc;

/// Reversible conversion between an integer's bit pattern and `u64`, so a key
/// pair packs losslessly into a `u128` and unpacks back to the original value.
pub trait IntBits: Copy {
    fn to_u64(self) -> u64;
    fn from_u64(bits: u64) -> Self;
}

macro_rules! impl_int_bits {
    ($($t:ty => $u:ty),*) => {
        $(impl IntBits for $t {
            #[inline(always)]
            fn to_u64(self) -> u64 { self as $u as u64 }
            #[inline(always)]
            fn from_u64(bits: u64) -> Self { bits as $u as $t }
        })*
    }
}
impl_int_bits!(i8 => u8, i16 => u16, i32 => u32, i64 => u64, u8 => u8, u16 => u16, u32 => u32, u64 => u64);

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
    type Persisted = u128;
    type LiveKey<'a, 'b> = u128;
    type PersistedLiveKey<'a> = u128;
    type Reader<'b> = PairReader<'b, A, B>;
    type Columns = IntPairKeyColumns<A, B>;

    fn make_reader<'b>(batch: &'b RecordBatch, key_cols: &[usize]) -> Self::Reader<'b> {
        let a = batch.column(key_cols[0]).as_primitive::<A>();
        let b = batch.column(key_cols[1]).as_primitive::<B>();
        PairReader { a, b }
    }

    #[inline(always)]
    fn rows(reader: &Self::Reader<'_>) -> usize {
        reader.a.len()
    }

    #[inline(always)]
    fn hash(reader: &Self::Reader<'_>, idx: usize, state: &RandomState) -> u64 {
        state.hash_one(reader.packed(idx))
    }

    #[inline(always)]
    fn live_key<'a, 'b>(
        reader: &Self::Reader<'b>,
        idx: usize,
        _arena: &'a mut WorkerArena,
    ) -> Self::LiveKey<'a, 'b> {
        reader.packed(idx)
    }

    fn resolve_persisted(_arena: &SharedArena, persisted: u128) -> u128 {
        persisted
    }
}

/// Emits the two unpacked key columns (`k0`, `k1`).
pub struct IntPairKeyColumns<A: ArrowPrimitiveType, B: ArrowPrimitiveType>
where
    A::Native: IntBits,
    B::Native: IntBits,
{
    a: PrimitiveBuilder<A>,
    b: PrimitiveBuilder<B>,
}

impl<A: ArrowPrimitiveType, B: ArrowPrimitiveType> KeyColumns for IntPairKeyColumns<A, B>
where
    A::Native: IntBits,
    B::Native: IntBits,
{
    type Key = u128;

    fn with_capacity(rows: usize) -> Self {
        Self {
            a: PrimitiveBuilder::<A>::with_capacity(rows),
            b: PrimitiveBuilder::<B>::with_capacity(rows),
        }
    }

    #[inline(always)]
    fn push(&mut self, key: &u128) {
        let packed = *key;
        self.a
            .append_value(A::Native::from_u64((packed >> 64) as u64));
        self.b.append_value(B::Native::from_u64(packed as u64));
    }

    fn finish(mut self, _arena: &Arc<SharedArena>) -> (Vec<Field>, Vec<ArrayRef>) {
        let fields = vec![
            Field::new("k0", A::DATA_TYPE, false),
            Field::new("k1", B::DATA_TYPE, false),
        ];
        let columns: Vec<ArrayRef> = vec![Arc::new(self.a.finish()), Arc::new(self.b.finish())];
        (fields, columns)
    }
}
