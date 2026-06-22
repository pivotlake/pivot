use crate::arrays::{ArrayBuilder, PrimitiveBuilder};
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::PersistedKey;
use crate::operations::unary::group::keys::{KeyColumns, KeyExtractor};
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
    const SUPPORTS_RADIX: bool = true;
    // A single fixed-width int (≤ 8 bytes) at entry offset 8 → copy-and-patch
    // consume's 8-byte key compare is valid (the high pad bytes stay zeroed).
    const CAP_KEY: bool = std::mem::size_of::<T::Native>() <= 8;
    type Config = ();
    type Persisted = T::Native;
    type LiveKey<'a, 'b> = T::Native;
    type PersistedLiveKey<'a> = T::Native;
    type Reader<'b> = &'b PrimitiveArray<T>;
    type Columns = IntKeyColumn<T>;
    type Scratch = ();

    fn make_reader<'b>(
        batch: &'b RecordBatch,
        key_cols: &[usize],
        _config: &(),
        _scratch: &'b mut (),
    ) -> Self::Reader<'b> {
        batch
            .column(key_cols[0])
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .expect("int key column type mismatch")
    }

    #[inline(always)]
    fn prepare_and_hash(reader: &mut Self::Reader<'_>, state: &RandomState, hashes: &mut [u64]) {
        for (i, h) in hashes.iter_mut().enumerate() {
            *h = state.hash_one(unsafe { reader.value_unchecked(i) });
        }
    }

    #[inline(always)]
    fn live_key<'a, 'r>(
        reader: &'r Self::Reader<'_>,
        idx: usize,
        _arena: &'a mut WorkerArena,
    ) -> Self::LiveKey<'a, 'r> {
        unsafe { reader.value_unchecked(idx) }
    }

    #[inline(always)]
    fn cap_key_u64(reader: &Self::Reader<'_>, idx: usize) -> u64 {
        // Zero-extend the native key's bytes into a u64 (low `size_of` bytes),
        // matching the entry's stored key region (little-endian; pad stays zero).
        let v = unsafe { reader.value_unchecked(idx) };
        let mut out = 0u64;
        unsafe {
            std::ptr::copy_nonoverlapping(
                &v as *const T::Native as *const u8,
                &mut out as *mut u64 as *mut u8,
                std::mem::size_of::<T::Native>(),
            );
        }
        out
    }

    fn resolve_persisted(_arena: &SharedArena, persisted: T::Native) -> T::Native {
        persisted
    }
}

/// Emits the single primitive key column.
pub struct IntKeyColumn<T: ArrowPrimitiveType>(PrimitiveBuilder<T>);

impl<T: ArrowPrimitiveType> KeyColumns for IntKeyColumn<T> {
    type Key = T::Native;
    type Config = ();

    fn with_capacity(allocator: &mut SlabAllocator, rows: usize, _config: &()) -> Self {
        Self(PrimitiveBuilder::<T>::with_capacity(allocator, rows))
    }

    #[inline(always)]
    fn push(&mut self, key: &T::Native) {
        self.0.push(key, 1);
    }

    fn finish(
        self,
        _arena: &Arc<SharedArena>,
        _allocator: &mut SlabAllocator,
    ) -> (Vec<Field>, Vec<ArrayRef>) {
        let fields = vec![Field::new("key", T::DATA_TYPE, false)];
        (fields, vec![self.0.into_array(None)])
    }
}
