//! The join key over one primitive column.

use super::JoinKey;
use ahash::RandomState;
use arrow_array::cast::AsArray;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{Array, PrimitiveArray, RecordBatch};
use std::hash::Hash;
use std::marker::PhantomData;

/// A join keyed on one primitive column of type `T`: the stored value is the
/// native value itself, so no verification is needed.
///
/// A wrapper rather than an impl on `T` directly because a blanket impl over
/// every `ArrowPrimitiveType` would forbid, by coherence, every other
/// [`JoinKey`] impl.
pub struct SingleColumnKey<T>(PhantomData<fn() -> T>);

impl<T: ArrowPrimitiveType<Native: Hash + Eq>> JoinKey for SingleColumnKey<T> {
    type Stored = T::Native;
    type Reader<'a> = &'a PrimitiveArray<T>;
    type Verifier<'a> = ();

    fn make_reader<'a>(
        batch: &'a RecordBatch,
        key_columns: &[usize],
        _state: &RandomState,
    ) -> Self::Reader<'a> {
        let [key_column] = key_columns else {
            panic!("a single-column join key reads exactly one column");
        };
        batch.column(*key_column).as_primitive::<T>()
    }

    #[inline(always)]
    fn read_stored(reader: &Self::Reader<'_>, idx: usize) -> T::Native {
        unsafe { reader.value_unchecked(idx) }
    }

    #[inline(always)]
    fn hash_row(reader: &Self::Reader<'_>, idx: usize, state: &RandomState) -> u64 {
        state.hash_one(Self::read_stored(reader, idx))
    }

    #[inline(always)]
    fn is_null(reader: &Self::Reader<'_>, idx: usize) -> bool {
        reader.is_null(idx)
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
