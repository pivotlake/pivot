//! Batch-bound aggregation operations for [`Dynamic`](super::Dynamic).
//!
//! [`Operation`] combines an aggregation kind with its input column. The column
//! reader is selected once per batch, so row processing dispatches on one enum
//! instead of repeatedly inspecting Arrow data types.

use super::super::super::cell::{F64Cell, IntCell, StringCell, WideCell};
use super::super::super::fold::Fold;
use super::super::super::read::{Read, StrRead};
use super::super::super::{
    AggregationKind, AggregationSlot, Count, F64Max, F64Min, F64Sum, Max, Min, StrMax, StrMin, Sum,
    U128Max, U128Min, U128Sum,
};
use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{ArrayRef, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field};
use std::array;
use std::sync::Arc;

use super::readers::{F64Reader, I64Reader, U128Reader};

/// An aggregation operation bound to its input column for one batch.
///
/// Prefixes identify the value returned by the reader: `I64`, `F64`, or
/// `U128` for the engine's `i128` wide values.
pub enum Operation<'b> {
    Count,
    I64Sum(I64Reader<'b>),
    I64Min(I64Reader<'b>),
    I64Max(I64Reader<'b>),
    F64Sum(F64Reader<'b>),
    F64Min(F64Reader<'b>),
    F64Max(F64Reader<'b>),
    U128Sum(U128Reader<'b>),
    U128Min(U128Reader<'b>),
    U128Max(U128Reader<'b>),
    StrMin(&'b StringViewArray),
    StrMax(&'b StringViewArray),
}

impl<'b> Operation<'b> {
    pub(in super::super) fn bind(batch: &'b RecordBatch, descriptor: &AggregationSlot) -> Self {
        use AggregationKind::*;
        match descriptor.kind {
            CountStar | Count => Operation::Count,
            Sum => Self::bind_numeric(
                batch,
                descriptor.column,
                Operation::I64Sum,
                Operation::F64Sum,
                Operation::U128Sum,
            ),
            Min => Self::bind_extreme(
                batch,
                descriptor.column,
                Operation::StrMin,
                Operation::I64Min,
                Operation::F64Min,
                Operation::U128Min,
            ),
            Max => Self::bind_extreme(
                batch,
                descriptor.column,
                Operation::StrMax,
                Operation::I64Max,
                Operation::F64Max,
                Operation::U128Max,
            ),
        }
    }

    /// Binds a numeric column to the reader for its value family.
    fn bind_numeric(
        batch: &'b RecordBatch,
        column: usize,
        on_i64: fn(I64Reader<'b>) -> Self,
        on_f64: fn(F64Reader<'b>) -> Self,
        on_u128: fn(U128Reader<'b>) -> Self,
    ) -> Self {
        match batch.column(column).data_type() {
            DataType::Int16 | DataType::Int32 | DataType::Int64 => {
                on_i64(I64Reader::bind(batch, column))
            }
            DataType::Float32 | DataType::Float64 => on_f64(F64Reader::bind(batch, column)),
            DataType::Decimal128(_, _) => on_u128(U128Reader::bind(batch, column)),
            other => panic!("numeric aggregate over unsupported column type {other:?}"),
        }
    }

    /// Binds a string or numeric column for `MIN` and `MAX`.
    fn bind_extreme(
        batch: &'b RecordBatch,
        column: usize,
        on_str: fn(&'b StringViewArray) -> Self,
        on_i64: fn(I64Reader<'b>) -> Self,
        on_f64: fn(F64Reader<'b>) -> Self,
        on_u128: fn(U128Reader<'b>) -> Self,
    ) -> Self {
        if let DataType::Utf8View = batch.column(column).data_type() {
            on_str(StrRead::bind(batch, column))
        } else {
            Self::bind_numeric(batch, column, on_i64, on_f64, on_u128)
        }
    }

    /// Seeds one cell from the first row in a group.
    ///
    /// With `ONLY_ADDITIVE`, non-additive variants are unreachable. The planner
    /// only enables that specialization for COUNT and integer SUM slots.
    #[inline(always)]
    pub(in super::super) fn seed<
        A: IntCell + StringCell + F64Cell + WideCell,
        const ONLY_ADDITIVE: bool,
    >(
        &self,
        row_index: usize,
        worker_context: &mut WorkerArena,
    ) -> A {
        match self {
            Operation::Count => Count::<A>::seed(()),
            Operation::I64Sum(reader) => Sum::<A>::seed(reader.read(row_index)),
            Operation::U128Sum(reader) => U128Sum::<A>::seed(reader.read(row_index)),
            Operation::I64Min(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    Min::<A>::seed(reader.read(row_index))
                }
            }
            Operation::I64Max(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    Max::<A>::seed(reader.read(row_index))
                }
            }
            Operation::F64Sum(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    F64Sum::<A>::seed(reader.read(row_index))
                }
            }
            Operation::F64Min(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    F64Min::<A>::seed(reader.read(row_index))
                }
            }
            Operation::F64Max(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    F64Max::<A>::seed(reader.read(row_index))
                }
            }
            Operation::U128Min(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    U128Min::<A>::seed(reader.read(row_index))
                }
            }
            Operation::U128Max(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    U128Max::<A>::seed(reader.read(row_index))
                }
            }
            Operation::StrMin(array) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    StrMin::<A>::seed(StrRead::read(array, row_index), worker_context)
                }
            }
            Operation::StrMax(array) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    StrMax::<A>::seed(StrRead::read(array, row_index), worker_context)
                }
            }
        }
    }

    /// Folds one row into an existing cell.
    #[inline(always)]
    pub(in super::super) fn update<
        A: IntCell + StringCell + F64Cell + WideCell,
        const ONLY_ADDITIVE: bool,
    >(
        &self,
        cell: A,
        row_index: usize,
        worker_context: &mut WorkerArena,
        shared: &Arc<SharedArena>,
    ) -> A {
        match self {
            Operation::Count => Count::<A>::update(cell, ()),
            Operation::I64Sum(reader) => Sum::<A>::update(cell, reader.read(row_index)),
            Operation::U128Sum(reader) => U128Sum::<A>::update(cell, reader.read(row_index)),
            Operation::I64Min(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    Min::<A>::update(cell, reader.read(row_index))
                }
            }
            Operation::I64Max(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    Max::<A>::update(cell, reader.read(row_index))
                }
            }
            Operation::F64Sum(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    F64Sum::<A>::update(cell, reader.read(row_index))
                }
            }
            Operation::F64Min(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    F64Min::<A>::update(cell, reader.read(row_index))
                }
            }
            Operation::F64Max(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    F64Max::<A>::update(cell, reader.read(row_index))
                }
            }
            Operation::U128Min(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    U128Min::<A>::update(cell, reader.read(row_index))
                }
            }
            Operation::U128Max(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    U128Max::<A>::update(cell, reader.read(row_index))
                }
            }
            Operation::StrMin(array) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    StrMin::<A>::update(
                        cell,
                        StrRead::read(array, row_index),
                        worker_context,
                        shared,
                    )
                }
            }
            Operation::StrMax(array) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    StrMax::<A>::update(
                        cell,
                        StrRead::read(array, row_index),
                        worker_context,
                        shared,
                    )
                }
            }
        }
    }
}

/// Aggregation operations bound to one record batch.
///
/// Up to eight operations stay inline. This lets the compiler keep their column
/// pointers in registers across the probe loop. Larger signatures use a boxed
/// slice to keep the enum reasonably sized.
pub enum Operations<'b> {
    Inline {
        len: usize,
        operations: [Operation<'b>; 8],
    },
    Boxed(Box<[Operation<'b>]>),
}

impl<'b> Operations<'b> {
    pub(in super::super) fn bind(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self {
        if slots.len() <= 8 {
            Operations::Inline {
                len: slots.len(),
                // `as_slice` hides these placeholder entries.
                operations: array::from_fn(|index| {
                    if index < slots.len() {
                        Operation::bind(batch, &slots[index])
                    } else {
                        Operation::Count
                    }
                }),
            }
        } else {
            Operations::Boxed(
                slots
                    .iter()
                    .map(|descriptor| Operation::bind(batch, descriptor))
                    .collect(),
            )
        }
    }

    #[inline(always)]
    pub(in super::super) fn as_slice(&self) -> &[Operation<'b>] {
        match self {
            Operations::Inline { len, operations } => &operations[..*len],
            Operations::Boxed(operations) => operations,
        }
    }
}

/// Merges two partial cells according to an aggregation descriptor.
#[inline(always)]
pub(in super::super) fn merge_operation<A: IntCell + StringCell + F64Cell + WideCell>(
    a: A,
    b: A,
    descriptor: &AggregationSlot,
    shared: &Arc<SharedArena>,
) -> A {
    let ty = &descriptor.output_type;
    match descriptor.kind {
        AggregationKind::CountStar | AggregationKind::Count => Count::<A>::merge(a, b),
        AggregationKind::Sum if ty.is_floating() => F64Sum::<A>::merge(a, b),
        AggregationKind::Sum => Sum::<A>::merge(a, b),
        AggregationKind::Min if *ty == DataType::Utf8View => StrMin::<A>::merge(a, b, shared),
        AggregationKind::Min if ty.is_floating() => F64Min::<A>::merge(a, b),
        AggregationKind::Min => Min::<A>::merge(a, b),
        AggregationKind::Max if *ty == DataType::Utf8View => StrMax::<A>::merge(a, b, shared),
        AggregationKind::Max if ty.is_floating() => F64Max::<A>::merge(a, b),
        AggregationKind::Max => Max::<A>::merge(a, b),
    }
}

/// Finishes one operation's output column using its declared Arrow type.
pub(in super::super) fn finish_operation<A: IntCell + StringCell + F64Cell + WideCell>(
    name: &str,
    descriptor: &AggregationSlot,
    col: SlabColumn<A>,
    arena: &Arc<SharedArena>,
) -> (Field, ArrayRef) {
    let ty = &descriptor.output_type;
    match descriptor.kind {
        AggregationKind::CountStar | AggregationKind::Count => Count::<A>::finish(name, col),
        AggregationKind::Sum if ty.is_floating() => F64Sum::<A>::finish(name, col),
        AggregationKind::Sum => Sum::<A>::finish(name, col),
        AggregationKind::Min if *ty == DataType::Utf8View => StrMin::<A>::finish(name, col, arena),
        AggregationKind::Min if ty.is_floating() => F64Min::<A>::finish(name, col),
        AggregationKind::Min => Min::<A>::finish(name, col),
        AggregationKind::Max if *ty == DataType::Utf8View => StrMax::<A>::finish(name, col, arena),
        AggregationKind::Max if ty.is_floating() => F64Max::<A>::finish(name, col),
        AggregationKind::Max => Max::<A>::finish(name, col),
    }
}
