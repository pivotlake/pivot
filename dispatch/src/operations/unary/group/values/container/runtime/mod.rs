//! Per-slot operations for [`RuntimeAggregation`].
//!
//! A [`BoundSlot`] downcasts one input column once per batch and pairs it with
//! the aggregate operation to apply. The row loop then dispatches on that bound
//! variant instead of repeatedly inspecting Arrow types.
//!
//! All operations consume and return the common cell type `A`. Cell traits
//! encode and decode floats, strings, and wide integers within that cell.
//! Signatures needing any of those representations use `i128`; narrow integer
//! signatures use `i64`.

mod readers;
mod value;

use super::super::cell::{F64Cell, IntCell, StringCell, WideCell};
use super::super::fold::Fold;
use super::super::read::{Read, StrRead};
use super::super::{
    AggregationKind, AggregationSlot, Count, F64Max, F64Min, F64Sum, Max, Min, StrMax, StrMin, Sum,
    U128Max, U128Min, U128Sum,
};
use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{ArrayRef, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field};
use std::sync::Arc;

pub use readers::{F64Reader, I64Reader, U128Reader};
pub use value::{RuntimeAggregation, RuntimeAggregationContext};

/// One aggregate slot bound to a typed reader for the current batch.
///
/// The prefixes describe what the reader returns: `i64`, `f64`, or `i128`.
pub enum BoundSlot<'b> {
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

impl<'b> BoundSlot<'b> {
    pub(in super::super) fn bind(batch: &'b RecordBatch, slot: &AggregationSlot) -> Self {
        use AggregationKind::*;
        match slot.kind {
            CountStar | Count => BoundSlot::Count,
            Sum => Self::bind_numeric(
                batch,
                slot.column,
                BoundSlot::I64Sum,
                BoundSlot::F64Sum,
                BoundSlot::U128Sum,
            ),
            Min => Self::bind_extreme(
                batch,
                slot.column,
                BoundSlot::StrMin,
                BoundSlot::I64Min,
                BoundSlot::F64Min,
                BoundSlot::U128Min,
            ),
            Max => Self::bind_extreme(
                batch,
                slot.column,
                BoundSlot::StrMax,
                BoundSlot::I64Max,
                BoundSlot::F64Max,
                BoundSlot::U128Max,
            ),
        }
    }

    /// Bind a numeric column and wrap it in the matching operation variant.
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

    /// Bind string `MIN`/`MAX` directly; delegate numeric inputs to
    /// [`bind_numeric`](BoundSlot::bind_numeric).
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
}

/// Initialize one slot from row `idx`.
///
/// In the `ALL_ADDITIVE` instantiation, non-additive arms are unreachable and
/// constant-fold away. Keeping the const check inside each arm preserves the
/// compact dispatch generated for `COUNT` and integer `SUM`.
#[inline(always)]
pub(in super::super) fn seed_slot<
    A: IntCell + StringCell + F64Cell + WideCell,
    const ALL_ADDITIVE: bool,
>(
    slot: &BoundSlot<'_>,
    idx: usize,
    wc: &mut WorkerArena,
) -> A {
    match slot {
        BoundSlot::Count => Count::<A>::seed(()),
        BoundSlot::I64Sum(r) => Sum::<A>::seed(r.read(idx)),
        BoundSlot::U128Sum(r) => U128Sum::<A>::seed(r.read(idx)),
        BoundSlot::I64Min(r) => {
            if ALL_ADDITIVE {
                unreachable!()
            } else {
                Min::<A>::seed(r.read(idx))
            }
        }
        BoundSlot::I64Max(r) => {
            if ALL_ADDITIVE {
                unreachable!()
            } else {
                Max::<A>::seed(r.read(idx))
            }
        }
        BoundSlot::F64Sum(r) => {
            if ALL_ADDITIVE {
                unreachable!()
            } else {
                F64Sum::<A>::seed(r.read(idx))
            }
        }
        BoundSlot::F64Min(r) => {
            if ALL_ADDITIVE {
                unreachable!()
            } else {
                F64Min::<A>::seed(r.read(idx))
            }
        }
        BoundSlot::F64Max(r) => {
            if ALL_ADDITIVE {
                unreachable!()
            } else {
                F64Max::<A>::seed(r.read(idx))
            }
        }
        BoundSlot::U128Min(r) => {
            if ALL_ADDITIVE {
                unreachable!()
            } else {
                U128Min::<A>::seed(r.read(idx))
            }
        }
        BoundSlot::U128Max(r) => {
            if ALL_ADDITIVE {
                unreachable!()
            } else {
                U128Max::<A>::seed(r.read(idx))
            }
        }
        BoundSlot::StrMin(a) => {
            if ALL_ADDITIVE {
                unreachable!()
            } else {
                StrMin::<A>::seed(StrRead::read(a, idx), wc)
            }
        }
        BoundSlot::StrMax(a) => {
            if ALL_ADDITIVE {
                unreachable!()
            } else {
                StrMax::<A>::seed(StrRead::read(a, idx), wc)
            }
        }
    }
}

/// Fold row `idx` into an existing cell.
///
/// Numeric operations need only the cell and reader. String operations use the
/// worker arena to persist a new winner and the shared arena to compare with
/// the current winner. `ALL_ADDITIVE` is specialized as in [`seed_slot`].
#[inline(always)]
pub(in super::super) fn update_slot<
    A: IntCell + StringCell + F64Cell + WideCell,
    const ALL_ADDITIVE: bool,
>(
    cell: A,
    slot: &BoundSlot<'_>,
    idx: usize,
    wc: &mut WorkerArena,
    arena: &Arc<SharedArena>,
) -> A {
    match slot {
        BoundSlot::Count => Count::<A>::update(cell, ()),
        BoundSlot::I64Sum(r) => Sum::<A>::update(cell, r.read(idx)),
        BoundSlot::U128Sum(r) => U128Sum::<A>::update(cell, r.read(idx)),
        BoundSlot::I64Min(r) => {
            if ALL_ADDITIVE {
                unreachable!()
            } else {
                Min::<A>::update(cell, r.read(idx))
            }
        }
        BoundSlot::I64Max(r) => {
            if ALL_ADDITIVE {
                unreachable!()
            } else {
                Max::<A>::update(cell, r.read(idx))
            }
        }
        BoundSlot::F64Sum(r) => {
            if ALL_ADDITIVE {
                unreachable!()
            } else {
                F64Sum::<A>::update(cell, r.read(idx))
            }
        }
        BoundSlot::F64Min(r) => {
            if ALL_ADDITIVE {
                unreachable!()
            } else {
                F64Min::<A>::update(cell, r.read(idx))
            }
        }
        BoundSlot::F64Max(r) => {
            if ALL_ADDITIVE {
                unreachable!()
            } else {
                F64Max::<A>::update(cell, r.read(idx))
            }
        }
        BoundSlot::U128Min(r) => {
            if ALL_ADDITIVE {
                unreachable!()
            } else {
                U128Min::<A>::update(cell, r.read(idx))
            }
        }
        BoundSlot::U128Max(r) => {
            if ALL_ADDITIVE {
                unreachable!()
            } else {
                U128Max::<A>::update(cell, r.read(idx))
            }
        }
        BoundSlot::StrMin(a) => {
            if ALL_ADDITIVE {
                unreachable!()
            } else {
                StrMin::<A>::update(cell, StrRead::read(a, idx), wc, arena)
            }
        }
        BoundSlot::StrMax(a) => {
            if ALL_ADDITIVE {
                unreachable!()
            } else {
                StrMax::<A>::update(cell, StrRead::read(a, idx), wc, arena)
            }
        }
    }
}

/// Merge two partial cells for one slot.
///
/// No input reader exists during merge, so `output_type` distinguishes string
/// and floating-point operations from integer operations. The caller bypasses
/// this dispatch entirely for an all-additive signature.
#[inline(always)]
pub(in super::super) fn merge_slot<A: IntCell + StringCell + F64Cell + WideCell>(
    a: A,
    b: A,
    slot: &AggregationSlot,
    arena: &Arc<SharedArena>,
) -> A {
    let ty = &slot.output_type;
    match slot.kind {
        AggregationKind::CountStar | AggregationKind::Count => Count::<A>::merge(a, b),
        AggregationKind::Sum if ty.is_floating() => F64Sum::<A>::merge(a, b),
        AggregationKind::Sum => Sum::<A>::merge(a, b),
        AggregationKind::Min if *ty == DataType::Utf8View => StrMin::<A>::merge(a, b, arena),
        AggregationKind::Min if ty.is_floating() => F64Min::<A>::merge(a, b),
        AggregationKind::Min => Min::<A>::merge(a, b),
        AggregationKind::Max if *ty == DataType::Utf8View => StrMax::<A>::merge(a, b, arena),
        AggregationKind::Max if ty.is_floating() => F64Max::<A>::merge(a, b),
        AggregationKind::Max => Max::<A>::merge(a, b),
    }
}

/// Render one completed cell column using the slot's operation and result type.
pub(in super::super) fn finish_slot<A: IntCell + StringCell + F64Cell + WideCell>(
    name: &str,
    slot: &AggregationSlot,
    col: SlabColumn<A>,
    arena: &Arc<SharedArena>,
) -> (Field, ArrayRef) {
    let ty = &slot.output_type;
    match slot.kind {
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
