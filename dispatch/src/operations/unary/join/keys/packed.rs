//! The join key over a tuple of integer-natured columns.

use super::{JoinKey, combined_key_validity};
use crate::arrays::IntBits;
use ahash::RandomState;
use arrow_array::cast::AsArray;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{PrimitiveArray, RecordBatch};
use arrow_buffer::NullBuffer;
use std::marker::PhantomData;

/// The per-batch reader of a [`PackedKey`]: one downcast primitive array per
/// key column, plus the columns' combined validity.
pub struct PackedReader<Arrays> {
    arrays: Arrays,
    validity: Option<NullBuffer>,
}

/// A join keyed on a tuple of integer-natured columns (integers, dates,
/// timestamps, `Decimal64`: anything whose native value is [`IntBits`]), each
/// value widened to one `u64` lane of the stored key. Lane equality is value
/// equality because both sides of a condition arrive as one type, so no
/// verification is needed.
///
/// The impls below are stamped for every supported arity, so a new key count
/// never needs its own implementation, only a dispatch arm naming the
/// concrete tuple.
pub struct PackedKey<T>(PhantomData<fn() -> T>);

macro_rules! impl_packed_join_key {
    ($lanes:literal; $($T:ident => $idx:tt),+) => {
        impl<$($T,)+> JoinKey for PackedKey<($($T,)+)>
        where
            $($T: ArrowPrimitiveType<Native: IntBits>,)+
        {
            type Stored = [u64; $lanes];
            type Reader<'a> = PackedReader<($(&'a PrimitiveArray<$T>,)+)>;
            type Verifier<'a> = ();

            fn make_reader<'a>(
                batch: &'a RecordBatch,
                key_columns: &[usize],
                _state: &RandomState,
            ) -> Self::Reader<'a> {
                assert_eq!(key_columns.len(), $lanes, "one key column per packed lane");
                PackedReader {
                    arrays: ($(batch.column(key_columns[$idx]).as_primitive::<$T>(),)+),
                    validity: combined_key_validity(batch, key_columns),
                }
            }

            #[inline(always)]
            fn read_stored(reader: &Self::Reader<'_>, idx: usize) -> [u64; $lanes] {
                [$(unsafe { reader.arrays.$idx.value_unchecked(idx) }.to_u64(),)+]
            }

            #[inline(always)]
            fn hash_row(reader: &Self::Reader<'_>, idx: usize, state: &RandomState) -> u64 {
                state.hash_one(Self::read_stored(reader, idx))
            }

            #[inline(always)]
            fn is_null(reader: &Self::Reader<'_>, idx: usize) -> bool {
                match &reader.validity {
                    Some(validity) => !validity.is_valid(idx),
                    None => false,
                }
            }

            fn make_verifier(_build_row_batches: &[RecordBatch], _build_key_columns: &[usize]) {}

            #[inline(always)]
            fn verify(
                _reader: &Self::Reader<'_>,
                _verifier: &(),
                _probe_idx: usize,
                _build_row_id: u32,
            ) -> bool {
                true
            }
        }
    };
}

impl_packed_join_key!(2; A => 0, B => 1);
impl_packed_join_key!(3; A => 0, B => 1, C => 2);
impl_packed_join_key!(4; A => 0, B => 1, C => 2, D => 3);
