//! The per-batch column readers a [`BoundSlot`](super::BoundSlot) binds — one
//! reader type per *value width*, each downcasting its column family once and
//! yielding one row's value:
//!
//! - [`I64Reader`] — an integer column (Int16/Int32/Int64), widened to `i64`.
//! - [`F64Reader`] — a float column (Float32/Float64), widened to `f64`.
//! - [`U128Reader`] — a `Decimal128` column, read as a full `i128`.
//!
//! Splitting by width keeps each reader single-purpose: a `read` returns exactly one
//! type, so the fold that consumes it needs no `is_float`/`as`-cast branching. The
//! op the reader feeds is chosen by the [`BoundSlot`](super::BoundSlot) variant, not
//! by the reader.

use crate::operations::unary::group::values::read::{IntRead, Read};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal128Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type,
};
use arrow_array::{PrimitiveArray, RecordBatch};
use arrow_schema::DataType;

/// An integer column bound at one of the supported widths, read as `i64`. The width
/// is a payload, not a cross-product with the op: adding a width is one more variant
/// here, shared by every integer op.
pub enum I64Reader<'b> {
    I16(&'b PrimitiveArray<Int16Type>),
    I32(&'b PrimitiveArray<Int32Type>),
    I64(&'b PrimitiveArray<Int64Type>),
}

impl<'b> I64Reader<'b> {
    pub(crate) fn bind(batch: &'b RecordBatch, column: usize) -> Self {
        let col = batch.column(column);
        match col.data_type() {
            DataType::Int16 => I64Reader::I16(col.as_primitive::<Int16Type>()),
            DataType::Int32 => I64Reader::I32(col.as_primitive::<Int32Type>()),
            DataType::Int64 => I64Reader::I64(col.as_primitive::<Int64Type>()),
            other => panic!("integer aggregate over unsupported column type {other:?}"),
        }
    }
    #[inline(always)]
    pub(crate) fn read(&self, idx: usize) -> i64 {
        match self {
            I64Reader::I16(a) => IntRead::<Int16Type>::read(a, idx),
            I64Reader::I32(a) => IntRead::<Int32Type>::read(a, idx),
            I64Reader::I64(a) => IntRead::<Int64Type>::read(a, idx),
        }
    }
}

/// A float column bound at one of the supported widths, read as `f64`.
pub enum F64Reader<'b> {
    F32(&'b PrimitiveArray<Float32Type>),
    F64(&'b PrimitiveArray<Float64Type>),
}

impl<'b> F64Reader<'b> {
    pub(crate) fn bind(batch: &'b RecordBatch, column: usize) -> Self {
        let col = batch.column(column);
        match col.data_type() {
            DataType::Float32 => F64Reader::F32(col.as_primitive::<Float32Type>()),
            DataType::Float64 => F64Reader::F64(col.as_primitive::<Float64Type>()),
            other => panic!("float aggregate over unsupported column type {other:?}"),
        }
    }
    #[inline(always)]
    pub(crate) fn read(&self, idx: usize) -> f64 {
        match self {
            F64Reader::F32(a) => (unsafe { a.value_unchecked(idx) }) as f64,
            F64Reader::F64(a) => unsafe { a.value_unchecked(idx) },
        }
    }
}

/// A `Decimal128(38, 0)` column, read as a full `i128`.
///
/// `Decimal128` is the Arrow type a *wide* (`i128`) cell emits for its
/// `COUNT`/`SUM`/`MIN`/`MAX` slots, so this reader binds whenever such a column is
/// aggregated — an aggregate re-reading partials a prior level already widened.
/// Reading the full `i128` (rather than truncating to `i64`) keeps a partial `SUM`
/// that overflows `i64` exact.
pub struct U128Reader<'b>(&'b PrimitiveArray<Decimal128Type>);

impl<'b> U128Reader<'b> {
    pub(crate) fn bind(batch: &'b RecordBatch, column: usize) -> Self {
        let col = batch.column(column);
        match col.data_type() {
            DataType::Decimal128(_, _) => U128Reader(col.as_primitive::<Decimal128Type>()),
            other => panic!("wide aggregate over unsupported column type {other:?}"),
        }
    }
    #[inline(always)]
    pub(crate) fn read(&self, idx: usize) -> i128 {
        unsafe { self.0.value_unchecked(idx) }
    }
}
