use crate::operations::KeyExtractor;
use crate::operations::unary::group::aggregations::Count;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::PersistedKey;
use crate::operations::unary::group::hashtables::{Table, TableStorage};
use arrow_array::builder::{PrimitiveBuilder, UInt64Builder};
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{Array, ArrayRef, PrimitiveArray, RecordBatch};
use arrow_schema::{ArrowError, DataType, Field, Schema};
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

impl_persisted_key!(i8, i16, i32, i64, u8, u16, u32, u64);

/// A `KeyExtractor` for Arrow primitive (integer) columns. This allows creating group bys with ints
/// as keys.
///
/// Since native integer types are `Copy`, the live and persisted key forms
/// are identical — no arena allocation is needed. This makes integer GROUP BY
/// essentially zero-copy on the key side.
///
/// Parameterized over `T: ArrowPrimitiveType` so a single implementation
/// covers `Int32Array`, `UInt64Array`, etc.
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
    type ArrayRef<'a> = &'a PrimitiveArray<T>;
    type Persisted = T::Native;
    type LiveKey<'a, 'b> = T::Native;
    type PersistedLiveKey<'a> = T::Native;
    type Value = Count;

    fn live_key<'a, 'b>(
        column: &Self::ArrayRef<'b>,
        idx: usize,
        _arena: &'a mut WorkerArena,
    ) -> Self::LiveKey<'a, 'b> {
        unsafe { column.value_unchecked(idx) }
    }

    fn resolve_persisted(_arena: &SharedArena, persisted: T::Native) -> T::Native {
        persisted
    }

    fn downcast_column(column: &dyn Array) -> Option<&PrimitiveArray<T>> {
        column.as_any().downcast_ref::<PrimitiveArray<T>>()
    }

    fn create_record_batch<S: TableStorage<Self>>(
        table: Table<Self, S>,
        _arena: &Arc<SharedArena>,
    ) -> Result<RecordBatch, ArrowError> {
        let mut key_b = PrimitiveBuilder::<T>::with_capacity(table.len());
        let mut val_b = UInt64Builder::with_capacity(table.len());

        for entry in table.iter(0) {
            key_b.append_value(*entry.key());
            val_b.append_value(entry.value().value as u64);
        }

        let keys: ArrayRef = Arc::new(key_b.finish());
        let vals: ArrayRef = Arc::new(val_b.finish());

        let schema = Arc::new(Schema::new(vec![
            Field::new("key", T::DATA_TYPE, false),
            Field::new("value", DataType::UInt64, false),
        ]));

        RecordBatch::try_new(schema, vec![keys, vals])
    }
}
