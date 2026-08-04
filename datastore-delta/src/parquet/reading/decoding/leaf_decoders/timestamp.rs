//! Leaf decoder for a timestamp column stored finer than a second.
//!
//! A pivot timestamp is an epoch second, so a file that stores milli-, micro-
//! or nanoseconds has each decoded value divided by its unit's divisor. The
//! division floors, so a value before the epoch lands on the second that
//! contains it rather than the one after.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{ArrayRef, PrimitiveArray, Scalar};
use dispatch::memory::SlabAllocator;

use crate::parquet::reading::decoding::leaf_decoders::{LeafDecoder, PrimitiveLeafDecoder, Result};
use crate::parquet::types::page::DecompressedPage;

/// Reads a sub-second timestamp leaf and converts every value to epoch
/// seconds. `T` is the carrier the reconciled column decodes to: `Int64` for a
/// column the table declares TIMESTAMP, `Timestamp(Second)` otherwise.
pub struct SecondsFromSubsecondDecoder<T: ArrowPrimitiveType<Native = i64>> {
    inner: PrimitiveLeafDecoder<T>,
    /// Stored units per second (see `timestamp_to_seconds_divisor`).
    divisor: i64,
}

impl<T: ArrowPrimitiveType<Native = i64>> SecondsFromSubsecondDecoder<T> {
    pub fn new(max_def_level: i16, divisor: i64) -> Self {
        Self {
            inner: PrimitiveLeafDecoder::<T>::new(max_def_level),
            divisor,
        }
    }
}

impl<T: ArrowPrimitiveType<Native = i64>> LeafDecoder for SecondsFromSubsecondDecoder<T> {
    fn available(&self) -> usize {
        self.inner.available()
    }

    fn insert_page(&mut self, page: DecompressedPage, allocator: &mut SlabAllocator) {
        self.inner.insert_page(page, allocator);
    }

    fn read(&mut self, allocator: &mut SlabAllocator, size: usize) -> Result<ArrayRef> {
        let stored = self.inner.read(allocator, size)?;
        let divisor = self.divisor;
        let seconds: PrimitiveArray<T> = stored
            .as_primitive::<T>()
            .unary(|value| value.div_euclid(divisor));
        Ok(Arc::new(seconds))
    }

    /// A pushed-down constant is an epoch second while this leaf's pages hold
    /// the file's own units, so there is nothing to compare them against: the
    /// pushdown is skipped and the query's `Filter` applies the condition.
    fn set_eq_constant(&mut self, _value: &Scalar<ArrayRef>) {}
}
