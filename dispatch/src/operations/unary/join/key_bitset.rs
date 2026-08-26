//! Exact key bitset over a hash join's build keys.
//!
//! When an integer-keyed build seals, its key set is complete, and a scan
//! feeding the probe side can drop rows whose key the build side does not
//! hold before they touch any operator: a bitset test is a shift and a load
//! into a cache-resident array. The filter is exact (one bit per value of the
//! build key domain, offset by the minimum), so it never rejects a row that
//! would have matched and correctness never depends on it — a consumer that
//! finds its [`KeyBitsetSlot`] unarmed simply filters nothing.

use std::sync::{Arc, RwLock};

use arrow_array::cast::AsArray;
use arrow_array::types::{
    ArrowPrimitiveType, Date32Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type,
    UInt16Type, UInt32Type,
};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;

/// Widest key domain (`max - min + 1`) the filter covers: 64M values is 8MB
/// of bits, small enough to stay cache-resident under probing.
const MAX_DOMAIN: u64 = 1 << 26;

/// Build-key density (`build rows / domain width`) at or above which the
/// filter is not built. A set that dense passes nearly every row the build
/// key range admits, so the per-row test is pure overhead. Row count bounds
/// distinct values from above, so the gate needs no pass over the keys.
const MAX_DENSITY: f64 = 0.9;

/// Most build rows the filter is built from. The set pass runs serially at
/// the build's final gather arrival while every other worker waits, so a
/// huge build side would stall the whole flow for longer than the filter
/// could ever save.
const MAX_BUILD_ROWS: usize = 1 << 24;

/// One bit per value of `[min, min + domain)`; a set bit means some build row
/// carries that key.
#[derive(Debug)]
pub struct KeyBitset {
    min: i64,
    domain: u64,
    words: Vec<u64>,
}

impl KeyBitset {
    /// Build the filter from the build side's key arrays, or `None` when the
    /// key type is not a supported integer or the domain, row count, or
    /// density gates say a filter is not worth its cost. `arrays` must hold
    /// every build key column array and `bounds` the column's non-null
    /// (min, max); a partial set would reject matching probe rows.
    ///
    /// Every gate reads only `bounds` and `total_rows`, so a build side that
    /// seals no filter pays nothing beyond them; the single pass over the
    /// keys happens only for a build the gates admit.
    pub(crate) fn try_build<'arrays>(
        arrays: impl Iterator<Item = &'arrays ArrayRef>,
        bounds: Option<(i64, i64)>,
        total_rows: usize,
    ) -> Option<KeyBitset> {
        let (min, max) = bounds?;
        let domain = max.abs_diff(min) + 1;
        if domain > MAX_DOMAIN || total_rows > MAX_BUILD_ROWS {
            return None;
        }
        if total_rows as f64 >= MAX_DENSITY * domain as f64 {
            return None;
        }

        let mut arrays = arrays.peekable();
        let data_type = arrays.peek()?.data_type().clone();
        let mut words = vec![0u64; domain.div_ceil(64) as usize];
        for array in arrays {
            for value in integer_values(array, &data_type)?.flatten() {
                let bit = value.abs_diff(min);
                words[(bit / 64) as usize] |= 1 << (bit % 64);
            }
        }
        Some(KeyBitset { min, domain, words })
    }

    /// Append the indices of `keys` rows whose value is a present build key
    /// to `survivors`. Nulls and out-of-domain values can never match, so
    /// they are never appended.
    ///
    /// The test runs per row of every scanned batch, so it is monomorphized
    /// per key type over the raw value slice: no per-row dynamic dispatch.
    pub fn select(&self, keys: &ArrayRef, survivors: &mut Vec<u32>) {
        fn select_typed<T: ArrowPrimitiveType>(
            bitset: &KeyBitset,
            keys: &ArrayRef,
            survivors: &mut Vec<u32>,
        ) where
            T::Native: Into<i64>,
        {
            let keys = keys.as_primitive::<T>();
            match keys.nulls().filter(|nulls| nulls.null_count() > 0) {
                None => {
                    for (row, &value) in keys.values().iter().enumerate() {
                        if bitset.contains_value(value.into()) {
                            survivors.push(row as u32);
                        }
                    }
                }
                Some(nulls) => {
                    for (row, &value) in keys.values().iter().enumerate() {
                        if nulls.is_valid(row) && bitset.contains_value(value.into()) {
                            survivors.push(row as u32);
                        }
                    }
                }
            }
        }
        match keys.data_type() {
            DataType::Int8 => select_typed::<Int8Type>(self, keys, survivors),
            DataType::Int16 => select_typed::<Int16Type>(self, keys, survivors),
            DataType::Int32 => select_typed::<Int32Type>(self, keys, survivors),
            DataType::Int64 => select_typed::<Int64Type>(self, keys, survivors),
            DataType::UInt8 => select_typed::<UInt8Type>(self, keys, survivors),
            DataType::UInt16 => select_typed::<UInt16Type>(self, keys, survivors),
            DataType::UInt32 => select_typed::<UInt32Type>(self, keys, survivors),
            DataType::Date32 => select_typed::<Date32Type>(self, keys, survivors),
            other => unreachable!("no key bitset is built over a {other} key"),
        }
    }

    #[inline(always)]
    fn contains_value(&self, value: i64) -> bool {
        // A branchless in-domain test: values below `min` wrap to a huge
        // offset and fail the width comparison.
        let bit = (value as u64).wrapping_sub(self.min as u64);
        bit < self.domain && self.words[(bit / 64) as usize] & (1 << (bit % 64)) != 0
    }
}

/// The single value of a one-element array widened to `i64`, or `None` when
/// the type is not a supported integer or the value is null.
pub(crate) fn integer_scalar(array: &ArrayRef) -> Option<i64> {
    integer_values(array, array.data_type())?.next().flatten()
}

/// The array's values widened to `i64`, or `None` for a non-integer key type.
fn integer_values<'array>(
    array: &'array ArrayRef,
    data_type: &DataType,
) -> Option<Box<dyn Iterator<Item = Option<i64>> + 'array>> {
    fn widen<'array, T: ArrowPrimitiveType>(
        array: &'array ArrayRef,
    ) -> Box<dyn Iterator<Item = Option<i64>> + 'array>
    where
        T::Native: Into<i64>,
    {
        Box::new(
            array
                .as_primitive::<T>()
                .iter()
                .map(|value| value.map(Into::into)),
        )
    }
    Some(match data_type {
        DataType::Int8 => widen::<Int8Type>(array),
        DataType::Int16 => widen::<Int16Type>(array),
        DataType::Int32 => widen::<Int32Type>(array),
        DataType::Int64 => widen::<Int64Type>(array),
        DataType::UInt8 => widen::<UInt8Type>(array),
        DataType::UInt16 => widen::<UInt16Type>(array),
        DataType::UInt32 => widen::<UInt32Type>(array),
        DataType::Date32 => widen::<Date32Type>(array),
        _ => return None,
    })
}

/// The shared cell pairing a join build (producer) with the probe-side scan
/// filters that consume its [`KeyBitset`]. Unarmed until the build seals;
/// consumers that read `None` filter nothing, so arming is an optimization
/// and never a correctness event.
#[derive(Debug, Default)]
pub struct KeyBitsetSlot {
    filter: RwLock<Option<Arc<KeyBitset>>>,
}

impl KeyBitsetSlot {
    pub fn new() -> Self {
        Self::default()
    }

    /// Arm the slot with the sealed build's filter; `None` (the build's key
    /// shape or distribution made a filter not worth having) leaves it
    /// unarmed.
    pub fn publish(&self, filter: Option<KeyBitset>) {
        *self.filter.write().expect("key bitset slot poisoned") = filter.map(Arc::new);
    }

    /// The armed filter, or `None` while the build has not sealed one.
    pub fn get(&self) -> Option<Arc<KeyBitset>> {
        self.filter
            .read()
            .expect("key bitset slot poisoned")
            .clone()
    }
}
