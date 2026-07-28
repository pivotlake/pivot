//! What a slot's value is *stored* as.
//!
//! A slot stores a bare [`Cell`] — a plain `Copy` value, in practice an `i64`
//! (narrow) or `i128` (wide). That raw storage is all there is: [`IntCell`],
//! [`F64Cell`], [`StringCell`], and [`WideCell`] add no storage of their own, they
//! are just different ways to *read and write those same bits* — as an integer
//! accumulator, an `f64`, a string `ArenaKey`, or a passthrough `i128`. *How* a value
//! combines is the [`Fold`](super::fold)'s job (plain `Ord::min` / `+` for the numeric
//! folds, an arena compare for the string ones) — never the cell's. [`IntCell`]
//! additionally owns the Arrow rendering, since that depends on the width
//! (`i64 → Int64`, `i128 → Decimal128`).

use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::keys::ArenaKey;
use arrow_array::types::{ArrowPrimitiveType, Decimal128Type, Int64Type};
use arrow_array::{ArrayRef, Decimal128Array, Float64Array, Int64Array, StringViewArray};
use arrow_buffer::{NullBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field};
use std::sync::Arc;

/// The bare requirement to live in an aggregate slot: a plain `Copy` value. Held
/// by every cell type (`i64`, `i128`, `ArenaKey`, the `u128` union), so it's a
/// blanket marker — the per-aggregate behaviour is the [`Read`](super::read) /
/// [`Fold`](super::fold), never the cell.
pub trait Cell: Copy + Default + Send + Sync + 'static {}
impl<T: Copy + Default + Send + Sync + 'static> Cell for T {}

/// The raw `i64` (narrow) / `i128` (wide) storage read as an integer. Bundles the
/// std arithmetic the numeric folds combine with (`+`, `Ord`, `From<i64>`,
/// `Into<i128>`) and the width-dependent Arrow rendering (`i64 → Int64`,
/// `i128 → Decimal128`), so a fold can just say `A: IntCell`. The combine logic lives
/// in the [`Fold`](super::fold), not here.
pub trait IntCell: Cell + Ord + std::ops::Add<Output = Self> + From<i64> + Into<i128> {
    /// The Arrow primitive whose `Native` is this width.
    type Arrow: ArrowPrimitiveType<Native = Self>;
    /// The width's largest value: a `MIN`'s empty cell, absorbed by any fold.
    const MAX_VALUE: Self;
    /// The width's smallest value: a `MAX`'s empty cell.
    const MIN_VALUE: Self;
    /// Render a finished grouped column, handing the engine slab to Arrow
    /// zero-copy. `nulls` marks the groups whose slot never saw a non-NULL
    /// value; those render as SQL NULL.
    fn finish(name: &str, col: SlabColumn<Self>, nulls: Option<NullBuffer>) -> (Field, ArrayRef);
    /// This width's Arrow data type.
    fn data_type() -> DataType;
    /// A single-row array — the global (no-GROUP-BY) output. `None` is a SQL NULL
    /// (an aggregate over zero rows), so the column is nullable.
    fn scalar_array(value: Option<Self>) -> ArrayRef;
}
impl IntCell for i64 {
    type Arrow = Int64Type;
    const MAX_VALUE: Self = i64::MAX;
    const MIN_VALUE: Self = i64::MIN;
    fn finish(name: &str, col: SlabColumn<Self>, nulls: Option<NullBuffer>) -> (Field, ArrayRef) {
        let len = col.len();
        let values = ScalarBuffer::<i64>::new(col.into_buffer(), 0, len);
        (
            Field::new(name, DataType::Int64, nulls.is_some()),
            Arc::new(Int64Array::new(values, nulls)),
        )
    }
    fn data_type() -> DataType {
        DataType::Int64
    }
    fn scalar_array(value: Option<Self>) -> ArrayRef {
        Arc::new(Int64Array::from(vec![value]))
    }
}

impl IntCell for i128 {
    type Arrow = Decimal128Type;
    const MAX_VALUE: Self = i128::MAX;
    const MIN_VALUE: Self = i128::MIN;
    fn finish(name: &str, col: SlabColumn<Self>, nulls: Option<NullBuffer>) -> (Field, ArrayRef) {
        let len = col.len();
        let values = ScalarBuffer::<i128>::new(col.into_buffer(), 0, len);
        let nullable = nulls.is_some();
        let arr = Decimal128Array::new(values, nulls)
            .with_precision_and_scale(38, 0)
            .expect("(38, 0) is a valid decimal128 precision/scale");
        (
            Field::new(name, DataType::Decimal128(38, 0), nullable),
            Arc::new(arr),
        )
    }
    fn data_type() -> DataType {
        DataType::Decimal128(38, 0)
    }
    fn scalar_array(value: Option<Self>) -> ArrayRef {
        Arc::new(
            Decimal128Array::from(vec![value])
                .with_precision_and_scale(38, 0)
                .expect("(38, 0) is a valid decimal128 precision/scale"),
        )
    }
}
/// Storing an `f64` accumulator in a numeric value cell.
///
/// A grouped float `SUM`/`MIN`/`MAX` keeps its running `f64` in the very cell an
/// integer aggregate would use, bit-punned through [`f64::to_bits`]: the cell holds
/// the float's raw bits, reinterpreted back on read. The float folds own that
/// reinterpretation (the cell is just storage), exactly as a string extreme views
/// its cell as an [`ArenaKey`] through [`StringCell`]. Both widths hold the 64-bit
/// pattern (`i128` in its low half), so a float slot rides whichever width the rest
/// of the signature already forces.
pub trait F64Cell: Cell {
    /// Pack an `f64`'s bits into the cell.
    fn from_f64(v: f64) -> Self;
    /// Read the cell's bits back as the `f64` a float slot stored.
    fn into_f64(self) -> f64;
    /// Render a finished column of float cells as a zero-or-near-zero-copy `Float64`
    /// array. The output phase narrows this to `Float32` for a `REAL` slot via the
    /// slot's declared `output_type` cast. `nulls` marks the groups whose slot
    /// never saw a non-NULL value.
    fn finish_float(
        name: &str,
        col: SlabColumn<Self>,
        nulls: Option<NullBuffer>,
    ) -> (Field, ArrayRef);
}

impl F64Cell for i64 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v.to_bits() as i64
    }
    #[inline(always)]
    fn into_f64(self) -> f64 {
        f64::from_bits(self as u64)
    }
    fn finish_float(
        name: &str,
        col: SlabColumn<Self>,
        nulls: Option<NullBuffer>,
    ) -> (Field, ArrayRef) {
        let len = col.len();
        // Each `i64` cell holds an `f64`'s raw bits, so the same 8-byte slab
        // reinterprets as an `f64` buffer with no copy.
        let values = ScalarBuffer::<f64>::new(col.into_buffer(), 0, len);
        (
            Field::new(name, DataType::Float64, nulls.is_some()),
            Arc::new(Float64Array::new(values, nulls)),
        )
    }
}

impl F64Cell for i128 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        // Zero-extend the 64 bits into the low half of the wide cell.
        v.to_bits() as i128
    }
    #[inline(always)]
    fn into_f64(self) -> f64 {
        f64::from_bits(self as u64)
    }
    fn finish_float(
        name: &str,
        col: SlabColumn<Self>,
        nulls: Option<NullBuffer>,
    ) -> (Field, ArrayRef) {
        let len = col.len();
        // The f64 bits sit in each cell's low 64; the 16-byte stride can't
        // reinterpret in place, so gather them back into a fresh Float64 buffer.
        let cells = ScalarBuffer::<i128>::new(col.into_buffer(), 0, len);
        let arr = Float64Array::new(
            cells.iter().map(|&c| f64::from_bits(c as u64)).collect(),
            nulls.clone(),
        );
        (
            Field::new(name, DataType::Float64, nulls.is_some()),
            Arc::new(arr),
        )
    }
}

/// Storing a re-read wide (`i128`) partial in a numeric value cell.
///
/// When an aggregate's input column is `Decimal128` — the Arrow type a wide
/// (`i128`) partial is emitted as, so this arises whenever an aggregate re-reads
/// partials a prior level already widened — the cell must hold that full `i128`
/// losslessly. Only the 128-bit cell can; the planner always widens such a
/// signature, so the `i64` arms are the fail-out (a bug if reached), exactly like
/// [`StringCell`]. Reading the `Decimal128` as `i64` instead would truncate a
/// partial `SUM` that overflows `i64`.
pub trait WideCell: Cell {
    /// Store a full `i128` partial in the cell.
    fn from_i128(v: i128) -> Self;
    /// Read the cell back as the `i128` partial it holds.
    fn into_i128(self) -> i128;
}

/// The fail-out: a re-read wide partial requires 128-bit storage, so the planner
/// must widen. Any `i64` arm being hit is a planning bug.
const NARROW_WIDE_CELL: &str = "wide (i128) partial re-read requires 128-bit (i128) storage; the planner must widen, so reaching i64 is a bug";

impl WideCell for i128 {
    #[inline(always)]
    fn from_i128(v: i128) -> Self {
        v
    }
    #[inline(always)]
    fn into_i128(self) -> i128 {
        self
    }
}
impl WideCell for i64 {
    fn from_i128(_: i128) -> Self {
        panic!("{NARROW_WIDE_CELL}")
    }
    fn into_i128(self) -> i128 {
        panic!("{NARROW_WIDE_CELL}")
    }
}

/// Storing a string extreme's [`ArenaKey`] in a numeric value cell.
///
/// A grouped string `MIN`/`MAX` keeps its winning `ArenaKey` — a 128-bit Arrow
/// `StringView` header — in the very slot a numeric aggregate would use, so a
/// `Dynamic` value can mix a string extreme with integer ones without a second
/// storage path. Only the 128-bit cell (`i128`) can hold the key; `i64` is the
/// fail-out, since the planner always widens a signature containing a string
/// extreme to `i128`. The `i64` methods therefore panic: reaching them means the
/// planner handed a string slot to a narrow cell, which is a bug, not a runtime
/// condition.
pub trait StringCell: Cell {
    /// Pack a winning `ArenaKey` into the cell.
    fn from_key(key: ArenaKey) -> Self;
    /// Read the cell back as the `ArenaKey` a string slot stored in it.
    fn into_key(self) -> ArenaKey;
    /// Render a finished column of string-extreme cells as a zero-copy
    /// `Utf8View` array over the value arena's ring buffers. `nulls` marks the
    /// groups whose slot never saw a non-NULL value; their cells are the default
    /// (an empty inline view), never resolved through the arena.
    fn finish(
        name: &str,
        col: SlabColumn<Self>,
        arena: &Arc<SharedArena>,
        nulls: Option<NullBuffer>,
    ) -> (Field, ArrayRef);
}

/// The fail-out: a string extreme requires 128-bit storage, so the planner must
/// widen its signature to `i128`. Any `i64` arm being hit is a planning bug.
const NARROW_STRING_CELL: &str = "string aggregate requires 128-bit (i128) storage; the planner must widen — reaching i64 is a bug";

impl StringCell for i64 {
    fn from_key(_: ArenaKey) -> Self {
        panic!("{NARROW_STRING_CELL}")
    }
    fn into_key(self) -> ArenaKey {
        panic!("{NARROW_STRING_CELL}")
    }
    fn finish(
        _: &str,
        _: SlabColumn<Self>,
        _: &Arc<SharedArena>,
        _: Option<NullBuffer>,
    ) -> (Field, ArrayRef) {
        panic!("{NARROW_STRING_CELL}")
    }
}

impl StringCell for i128 {
    #[inline(always)]
    fn from_key(key: ArenaKey) -> Self {
        // `ArenaKey` is a transparent `u128` (a StringView header); the cell holds
        // its raw bits, reinterpreted back on read. The numeric reading is never
        // applied to a string slot, so this bit-punning never crosses families.
        key.as_u128() as i128
    }
    #[inline(always)]
    fn into_key(self) -> ArenaKey {
        ArenaKey::from_raw(self as u128)
    }
    fn finish(
        name: &str,
        col: SlabColumn<Self>,
        arena: &Arc<SharedArena>,
        nulls: Option<NullBuffer>,
    ) -> (Field, ArrayRef) {
        let len = col.len();
        // The cells are valid `ArenaKey`s (StringView headers); reinterpret the
        // slab as `u128` views (zero-copy) over the arena's ring buffers. A NULL
        // group's cell is 0, a valid empty inline view, masked by `nulls`.
        let views = ScalarBuffer::<u128>::new(col.into_buffer(), 0, len);
        let buffers = arena.to_arrow_buffers();
        let nullable = nulls.is_some();
        // Safety: the views are valid ArenaKeys and the arena (Arc-held in each
        // Buffer) outlives the array — the same contract as string keys.
        let arr: ArrayRef =
            Arc::new(unsafe { StringViewArray::new_unchecked(views, buffers, nulls) });
        (Field::new(name, DataType::Utf8View, nullable), arr)
    }
}
