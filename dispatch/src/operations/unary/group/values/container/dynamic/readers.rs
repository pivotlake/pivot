//! Numeric column readers used by the dynamic container.
//!
//! Each reader downcasts its Arrow column once when binding the batch. Reading
//! a row then returns one stable Rust type without inspecting the data type.

use crate::operations::unary::group::values::read::{IntRead, Read};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal128Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type,
};
use arrow_array::{PrimitiveArray, RecordBatch};
use arrow_schema::DataType;

/// An Int16, Int32, or Int64 column read as `i64`.
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

/// A Decimal128 column read as `i128`, preserving wide partial aggregates.
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
