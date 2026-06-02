//! Two-integer-key GROUP BY with multiple sum/count value slots.
//!
//! Keys: a pair of integer columns (e.g. `WatchID`, `ClientIP`) packed into a
//! single `u128` (first key's bits in the high 64, second's in the low 64),
//! which is `Copy`/`Hash`/`Eq` and needs no arena.
//!
//! Value: an [`AggRow<N>`] of `N` `i64` slots, one per `COUNT(*)` / `SUM` /
//! `COUNT(col)` in the query. `N` is monomorphised per arity so each hash-table
//! entry is exactly as wide as the query needs.

use crate::operations::KeyExtractor;
use crate::operations::unary::group::aggregations::{AggRow, GroupAggKind, GroupAggSlot};
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::{Table, TableStorage};
use ahash::RandomState;
use arrow_array::builder::{Int64Builder, PrimitiveBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{ArrowPrimitiveType, Int16Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, PrimitiveArray, RecordBatch};
use arrow_schema::{ArrowError, DataType, Field, Schema};
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

/// Per-slot reader: how to produce slot `s`'s contribution for a given row.
enum SlotReader<'b> {
    /// `COUNT(*)` or `COUNT(non-null col)` — contributes 1.
    One,
    SumI16(&'b PrimitiveArray<Int16Type>),
    SumI32(&'b PrimitiveArray<Int32Type>),
    SumI64(&'b PrimitiveArray<Int64Type>),
}

impl SlotReader<'_> {
    #[inline(always)]
    fn at(&self, idx: usize) -> i64 {
        match self {
            SlotReader::One => 1,
            SlotReader::SumI16(a) => unsafe { a.value_unchecked(idx) as i64 },
            SlotReader::SumI32(a) => unsafe { a.value_unchecked(idx) as i64 },
            SlotReader::SumI64(a) => unsafe { a.value_unchecked(idx) },
        }
    }
}

/// Per-batch reader for the pair extractor.
pub struct PairReader<'b, A: ArrowPrimitiveType, B: ArrowPrimitiveType> {
    a: &'b PrimitiveArray<A>,
    b: &'b PrimitiveArray<B>,
    slots: Vec<SlotReader<'b>>,
}

impl<A: ArrowPrimitiveType, B: ArrowPrimitiveType> PairReader<'_, A, B>
where
    A::Native: IntBits,
    B::Native: IntBits,
{
    #[inline(always)]
    fn pack(&self, idx: usize) -> u128 {
        let a = unsafe { self.a.value_unchecked(idx) }.to_u64();
        let b = unsafe { self.b.value_unchecked(idx) }.to_u64();
        ((a as u128) << 64) | (b as u128)
    }
}

/// `GROUP BY (A, B)` with `N` sum/count value slots.
pub struct IntPairAggExtractor<A: ArrowPrimitiveType, B: ArrowPrimitiveType, const N: usize>(
    PhantomData<(A, B)>,
);

unsafe impl<A: ArrowPrimitiveType, B: ArrowPrimitiveType, const N: usize> Send
    for IntPairAggExtractor<A, B, N>
{
}

impl<A, B, const N: usize> KeyExtractor for IntPairAggExtractor<A, B, N>
where
    A: ArrowPrimitiveType + Send + 'static,
    B: ArrowPrimitiveType + Send + 'static,
    A::Native: IntBits,
    B::Native: IntBits,
{
    type Persisted = u128;
    type LiveKey<'a, 'b> = u128;
    type PersistedLiveKey<'a> = u128;
    type Value = AggRow<N>;
    type Reader<'b> = PairReader<'b, A, B>;

    fn make_reader<'b>(
        batch: &'b RecordBatch,
        key_cols: &[usize],
        value_slots: &[GroupAggSlot],
    ) -> Self::Reader<'b> {
        assert_eq!(value_slots.len(), N, "slot count must match N");
        let a = batch.column(key_cols[0]).as_primitive::<A>();
        let b = batch.column(key_cols[1]).as_primitive::<B>();
        let slots = value_slots
            .iter()
            .map(|slot| match slot.kind {
                GroupAggKind::CountStar | GroupAggKind::Count => SlotReader::One,
                GroupAggKind::Sum => {
                    let col = batch.column(slot.column);
                    match col.data_type() {
                        DataType::Int16 => SlotReader::SumI16(col.as_primitive::<Int16Type>()),
                        DataType::Int32 => SlotReader::SumI32(col.as_primitive::<Int32Type>()),
                        DataType::Int64 => SlotReader::SumI64(col.as_primitive::<Int64Type>()),
                        other => panic!("grouped SUM: unsupported column type {other:?}"),
                    }
                }
            })
            .collect();
        PairReader { a, b, slots }
    }

    #[inline(always)]
    fn rows(reader: &Self::Reader<'_>) -> usize {
        reader.a.len()
    }

    #[inline(always)]
    fn hash(reader: &Self::Reader<'_>, idx: usize, state: &RandomState) -> u64 {
        state.hash_one(reader.pack(idx))
    }

    #[inline(always)]
    fn live_key<'a, 'b>(
        reader: &Self::Reader<'b>,
        idx: usize,
        _arena: &'a mut WorkerArena,
    ) -> Self::LiveKey<'a, 'b> {
        reader.pack(idx)
    }

    // (resolve_persisted / value / create_record_batch below)

    #[inline(always)]
    fn value(reader: &Self::Reader<'_>, idx: usize) -> AggRow<N> {
        let mut out = [0i64; N];
        for (s, slot) in reader.slots.iter().enumerate() {
            out[s] = slot.at(idx);
        }
        AggRow(out)
    }

    fn resolve_persisted(_arena: &SharedArena, persisted: u128) -> u128 {
        persisted
    }

    fn create_record_batch<S: TableStorage<Self>>(
        table: Table<Self, S>,
        _arena: &Arc<SharedArena>,
    ) -> Result<RecordBatch, ArrowError> {
        let len = table.len();
        let mut a_b = PrimitiveBuilder::<A>::with_capacity(len);
        let mut b_b = PrimitiveBuilder::<B>::with_capacity(len);
        let mut val_bs: Vec<Int64Builder> =
            (0..N).map(|_| Int64Builder::with_capacity(len)).collect();

        for entry in table.iter(0) {
            let packed = *entry.key();
            a_b.append_value(A::Native::from_u64((packed >> 64) as u64));
            b_b.append_value(B::Native::from_u64(packed as u64));
            let row = entry.value().0;
            for (s, vb) in val_bs.iter_mut().enumerate() {
                vb.append_value(row[s]);
            }
        }

        let mut fields = vec![
            Field::new("k0", A::DATA_TYPE, false),
            Field::new("k1", B::DATA_TYPE, false),
        ];
        let mut columns: Vec<ArrayRef> = vec![Arc::new(a_b.finish()), Arc::new(b_b.finish())];
        for (s, mut vb) in val_bs.into_iter().enumerate() {
            fields.push(Field::new(format!("v{s}"), DataType::Int64, false));
            columns.push(Arc::new(vb.finish()));
        }

        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
    }
}
