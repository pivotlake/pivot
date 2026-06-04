use crate::arrays::{ArrayBuilder, PrimitiveBuilder};
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::PersistedKey;
use crate::operations::unary::group::key_extractions::{KeyColumns, KeyExtractor};
use ahash::RandomState;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{Array, ArrayRef, PrimitiveArray, RecordBatch};
use arrow_schema::Field;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;

/// Implement [`PersistedKey`] for all standard integer types so they can
/// be used directly as hash table keys without indirection.
macro_rules! impl_persisted_key {
    ($($t:ty),*) => {
        $(impl PersistedKey for $t {})*
    }
}

impl_persisted_key!(i8, i16, i32, i64, u8, u16, u32, u64, u128);

/// A [`KeyExtractor`] for a single Arrow primitive (integer) column.
///
/// Native integer types are `Copy`, so live and persisted key forms are
/// identical — no arena allocation is needed.
pub struct IntKeyExtractor<T: ArrowPrimitiveType>(PhantomData<T>)
where
    T::Native: PersistedKey + Hash + Eq;

unsafe impl<T: ArrowPrimitiveType> Send for IntKeyExtractor<T> where
    T::Native: PersistedKey + Hash + Eq
{
}

impl<T: ArrowPrimitiveType + Send + 'static> KeyExtractor for IntKeyExtractor<T>
where
    T::Native: PersistedKey + Hash + Eq,
{
    type Persisted = T::Native;
    type LiveKey<'a, 'b> = T::Native;
    type PersistedLiveKey<'a> = T::Native;
    type Reader<'b> = &'b PrimitiveArray<T>;
    type Columns = IntKeyColumns<T>;

    fn make_reader<'b>(batch: &'b RecordBatch, key_cols: &[usize]) -> Self::Reader<'b> {
        batch
            .column(key_cols[0])
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .expect("int key column type mismatch")
    }

    #[inline(always)]
    fn rows(reader: &Self::Reader<'_>) -> usize {
        reader.len()
    }

    #[inline(always)]
    fn hash(reader: &Self::Reader<'_>, idx: usize, state: &RandomState) -> u64 {
        state.hash_one(unsafe { reader.value_unchecked(idx) })
    }

    #[inline(always)]
    fn live_key<'a, 'b>(
        reader: &Self::Reader<'b>,
        idx: usize,
        _arena: &'a mut WorkerArena,
    ) -> Self::LiveKey<'a, 'b> {
        unsafe { reader.value_unchecked(idx) }
    }

    fn resolve_persisted(_arena: &SharedArena, persisted: T::Native) -> T::Native {
        persisted
    }
}

/// Emits the single primitive key column.
pub struct IntKeyColumns<T: ArrowPrimitiveType>(PrimitiveBuilder<T>);

impl<T: ArrowPrimitiveType> KeyColumns for IntKeyColumns<T> {
    type Key = T::Native;

    fn with_capacity(allocator: &mut SlabAllocator, rows: usize) -> Self {
        Self(PrimitiveBuilder::<T>::with_capacity(allocator, rows))
    }

    #[inline(always)]
    fn push(&mut self, key: &T::Native) {
        self.0.push(key, 1);
    }

    fn finish(self, _arena: &Arc<SharedArena>) -> (Vec<Field>, Vec<ArrayRef>) {
        let fields = vec![Field::new("key", T::DATA_TYPE, false)];
        (fields, vec![self.0.into_array(None)])
    }
}
