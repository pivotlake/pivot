//! Axis 1 — **READ**: what a slot pulls out of an input row.
//!
//! A read turns row `idx` of a batch into a cell value `A`. There are only three:
//! - [`One`] — contributes `1`, reads no column (`COUNT`);
//! - [`Col<T>`](Col) — the row's (widened) integer column value (`SUM`/`MIN`/`MAX`
//!   all read identically — they differ only in the [`Fold`](super::fold));
//! - [`Str`] — persists the row's string into the value arena, yielding its
//!   [`ArenaKey`](crate::operations::unary::group::ArenaKey) bits (`MIN`/`MAX` over
//!   a string column).
//!
//! [`Op<R, F>`](super::op::Op) pairs a `Read` with a [`Fold`](super::fold) into a
//! compiled atom. [`Mono`](super::container) / [`Dynamic`](super::container) — whose
//! slots share a fold but differ in read — instead carry a per-slot [`SlotReader`]
//! enum, the runtime form of the same three reads.

mod column;
mod one;
mod string;

pub use column::Col;
pub use one::One;
pub use string::Str;

use super::cell::NumericCell;
use super::{AggregationKind, AggregationSlot};
use crate::operations::unary::group::arena::WorkerArena;
use crate::operations::unary::group::keys::ArenaKey;
use arrow_array::cast::AsArray;
use arrow_array::types::{Int16Type, Int32Type, Int64Type};
use arrow_array::{PrimitiveArray, RecordBatch, StringViewArray};
use arrow_schema::DataType;

/// One slot's read, as a typed op. Implemented by [`One`], [`Col<T>`](Col) and
/// [`Str`]; composed with a [`Fold`](super::fold) by [`Op`](super::op::Op).
pub trait Read<A> {
    /// Per-batch downcast input (the column, or `()` for [`One`]).
    type Reader<'b>;
    fn make_reader(batch: &RecordBatch, column: usize) -> Self::Reader<'_>;
    /// Read row `idx` as a cell. `arena` is where [`Str`] persists; the numeric
    /// reads ignore it.
    fn read(reader: &Self::Reader<'_>, idx: usize, arena: &mut WorkerArena) -> A;
}

/// The runtime form of a slot's read, for the containers whose slots share one
/// fold but read different columns ([`Mono`]/[`Dynamic`]). Built from the slot's
/// kind and column type; read as numeric ([`read_num`](Self::read_num)) or string
/// ([`read_str`](Self::read_str)) depending on the fold family — a slot is only
/// ever read the one way its container's fold dictates.
///
/// [`Mono`]: super::container::Mono
/// [`Dynamic`]: super::container::Dynamic
pub enum SlotReader<'b> {
    /// `COUNT` — no column.
    One,
    Col16(&'b PrimitiveArray<Int16Type>),
    Col32(&'b PrimitiveArray<Int32Type>),
    Col64(&'b PrimitiveArray<Int64Type>),
    Str(&'b StringViewArray),
}

impl<'b> SlotReader<'b> {
    /// Bind a slot's input column. A `COUNT` reads nothing; every other kind
    /// reads its column, dispatched by the column's type (an integer width, or a
    /// string view for a string `MIN`/`MAX`).
    pub fn new(batch: &'b RecordBatch, slot: &AggregationSlot) -> Self {
        match slot.kind {
            AggregationKind::CountStar | AggregationKind::Count => SlotReader::One,
            AggregationKind::Sum | AggregationKind::Min | AggregationKind::Max => {
                let col = batch.column(slot.column);
                match col.data_type() {
                    DataType::Int16 => SlotReader::Col16(col.as_primitive::<Int16Type>()),
                    DataType::Int32 => SlotReader::Col32(col.as_primitive::<Int32Type>()),
                    DataType::Int64 => SlotReader::Col64(col.as_primitive::<Int64Type>()),
                    DataType::Utf8View => SlotReader::Str(col.as_string_view()),
                    other => panic!("grouped aggregate: unsupported column type {other:?}"),
                }
            }
        }
    }

    /// Read this slot as a numeric contribution (the additive/extreme families).
    #[inline(always)]
    pub fn read_num<A: NumericCell>(&self, idx: usize) -> A {
        match self {
            SlotReader::One => A::from_i64(1),
            SlotReader::Col16(a) => A::from_i64(unsafe { a.value_unchecked(idx) } as i64),
            SlotReader::Col32(a) => A::from_i64(unsafe { a.value_unchecked(idx) } as i64),
            SlotReader::Col64(a) => A::from_i64(unsafe { a.value_unchecked(idx) }),
            SlotReader::Str(_) => unreachable!("numeric fold over a string slot"),
        }
    }

    /// Read this slot as a string contribution (the string extreme family),
    /// persisting it into `arena` and returning its [`ArenaKey`] bits. Used to
    /// *seed* a new group, where the string is kept unconditionally.
    #[inline(always)]
    pub fn read_str(&self, idx: usize, arena: &mut WorkerArena) -> u128 {
        arena.push(self.str_at(idx)).as_u128()
    }

    /// The raw string at the row, *without* persisting. A string extreme's
    /// `update` compares this against the current cell first, so a losing row
    /// never touches the arena.
    #[inline(always)]
    pub fn str_at(&self, idx: usize) -> &str {
        match self {
            SlotReader::Str(a) => a.value(idx),
            _ => unreachable!("string fold over a numeric slot"),
        }
    }
}

/// Resolve a slot's stored string-cell bits back to its bytes (for a string
/// fold's compare / finish). The inverse of [`SlotReader::read_str`].
#[inline(always)]
pub fn arena_key(bits: u128) -> ArenaKey {
    ArenaKey::from_raw(bits)
}
