//! Column decoder for parquet DECIMAL leaves read as `Float64`.
//!
//! A DECIMAL column stores scaled integers (`value * 10^scale`) over an INT32
//! or INT64 physical type. Pivot computes decimals as `Float64` (see
//! `Type::Decimal`), so this decoder wraps the plain integer
//! [`PrimitiveColumnDecoder`] and divides the scale out of each decoded batch.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{ArrowPrimitiveType, Float64Type};
use arrow_array::{ArrayRef, PrimitiveArray};
use dispatch::memory::SlabAllocator;

use crate::parquet::reading::decoding::column_decoders::primitive::ReadLeBytes;
use crate::parquet::reading::decoding::column_decoders::{
    ColumnDecoder, PrimitiveColumnDecoder, Result,
};
use crate::parquet::types::page::DecompressedPage;

/// Decodes an INT32/INT64-backed DECIMAL leaf into `Float64` values.
pub struct DecimalFloatDecoder<T: ArrowPrimitiveType>
where
    T::Native: ReadLeBytes,
{
    inner: PrimitiveColumnDecoder<T>,
    /// `10^-scale`: multiplying the stored integer by it yields the value.
    factor: f64,
}

impl<T: ArrowPrimitiveType> DecimalFloatDecoder<T>
where
    T::Native: ReadLeBytes,
{
    pub fn new(max_def_level: i16, factor: f64) -> Self {
        Self {
            inner: PrimitiveColumnDecoder::<T>::new(max_def_level),
            factor,
        }
    }
}

impl<T: ArrowPrimitiveType> ColumnDecoder for DecimalFloatDecoder<T>
where
    T::Native: ReadLeBytes + Into<i64>,
{
    fn available(&self) -> usize {
        self.inner.available()
    }

    fn insert_page(&mut self, page: DecompressedPage, allocator: &mut SlabAllocator) {
        self.inner.insert_page(page, allocator);
    }

    fn read(&mut self, allocator: &mut SlabAllocator, size: usize) -> Result<ArrayRef> {
        let scaled = self.inner.read(allocator, size)?;
        let scaled = scaled.as_primitive::<T>();
        let values: PrimitiveArray<Float64Type> = scaled.unary(|v| {
            let v: i64 = v.into();
            v as f64 * self.factor
        });
        Ok(Arc::new(values))
    }
}
