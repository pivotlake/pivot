//! What a slot's value is *stored* as.
//!
//! [`Cell`] is a bare marker — any `Copy` value can sit in a slot, so it's a
//! blanket bound with nothing to implement. *How* cells combine is the
//! [`Fold`](super::fold)'s job (plain `Ord::min` / `+` for the numeric folds, an
//! arena compare for the string ones) — never the cell's. The only thing a
//! numeric width owns is how it renders to Arrow (`NumericArrow`), since that
//! genuinely depends on the width (`i64 → Int64`, `i128 → Decimal128`). The
//! [`Numeric`] bound alias bundles that with the std arithmetic the numeric folds
//! lean on, so a fold can just say `A: Numeric`.

use crate::arrays::SlabColumn;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::keys::ArenaKey;
use arrow_array::types::{ArrowPrimitiveType, Decimal128Type, Int64Type};
use arrow_array::{ArrayRef, Decimal128Array, Float64Array, Int64Array, StringViewArray};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field};
use std::sync::Arc;

/// The bare requirement to live in an aggregate slot: a plain `Copy` value. Held
/// by every cell type (`i64`, `i128`, `ArenaKey`, the `u128` union), so it's a
/// blanket marker — the per-aggregate behaviour is the [`Read`](super::read) /
/// [`Fold`](super::fold), never the cell.
pub trait Cell: Copy + Default + Send + Sync + 'static {}
impl<T: Copy + Default + Send + Sync + 'static> Cell for T {}

/// A numeric aggregate cell (`i64` narrow / `i128` wide): a [`Cell`] that adds
/// (`+`), orders (`Ord`), builds from a per-row `i64` and widens to `i128`, and
/// renders to Arrow (`NumericArrow`). A bound alias — no methods of its own, so
/// the numeric folds combine with std ops, not cell methods.
pub trait Numeric: Cell + Ord + std::ops::Add<Output = Self> + From<i64> + Into<i128> {
    /// The Arrow primitive whose `Native` is this width.
    type Arrow: ArrowPrimitiveType<Native = Self>;
    /// Render a finished grouped column, handing the engine slab to Arrow zero-copy.
    fn finish(name: &str, col: SlabColumn<Self>) -> (Field, ArrayRef);
    /// This width's Arrow data type.
    fn data_type() -> DataType;
    /// A single-row array — the global (no-GROUP-BY) output. `None` is a SQL NULL
    /// (an aggregate over zero rows), so the column is nullable.
    fn scalar_array(value: Option<Self>) -> ArrayRef;
}
impl Numeric for i64 {
    type Arrow = Int64Type;
    fn finish(name: &str, col: SlabColumn<Self>) -> (Field, ArrayRef) {
        let len = col.len();
        let values = ScalarBuffer::<i64>::new(col.into_buffer(), 0, len);
        (
            Field::new(name, DataType::Int64, false),
            Arc::new(Int64Array::new(values, None)),
        )
    }
    fn data_type() -> DataType {
        DataType::Int64
    }
    fn scalar_array(value: Option<Self>) -> ArrayRef {
        Arc::new(Int64Array::from(vec![value]))
    }
}

impl Numeric for i128 {
    type Arrow = Decimal128Type;
    fn finish(name: &str, col: SlabColumn<Self>) -> (Field, ArrayRef) {
        let len = col.len();
        let values = ScalarBuffer::<i128>::new(col.into_buffer(), 0, len);
        let arr = Decimal128Array::new(values, None)
            .with_precision_and_scale(38, 0)
            .expect("(38, 0) is a valid decimal128 precision/scale");
        (
            Field::new(name, DataType::Decimal128(38, 0), false),
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
    /// `Utf8View` array over the value arena's ring buffers.
    fn finish(name: &str, col: SlabColumn<Self>, arena: &Arc<SharedArena>) -> (Field, ArrayRef);
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
    fn finish(_: &str, _: SlabColumn<Self>, _: &Arc<SharedArena>) -> (Field, ArrayRef) {
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
    fn finish(name: &str, col: SlabColumn<Self>, arena: &Arc<SharedArena>) -> (Field, ArrayRef) {
        let len = col.len();
        // The cells are valid `ArenaKey`s (StringView headers); reinterpret the
        // slab as `u128` views (zero-copy) over the arena's ring buffers.
        let views = ScalarBuffer::<u128>::new(col.into_buffer(), 0, len);
        let buffers = arena.to_arrow_buffers();
        // Safety: the views are valid ArenaKeys and the arena (Arc-held in each
        // Buffer) outlives the array — the same contract as string keys.
        let arr: ArrayRef =
            Arc::new(unsafe { StringViewArray::new_unchecked(views, buffers, None) });
        (Field::new(name, DataType::Utf8View, false), arr)
    }
}

/// Storing an `f64` aggregate value in a numeric value cell: the float
/// counterpart to [`StringCell`].
///
/// A `SUM`/`MIN`/`MAX` over a `Float64` column folds in `f64`, but the value
/// containers ([`Dynamic`](super::container::Dynamic)) store every slot in one
/// uniform numeric cell (`i64` narrow / `i128` wide). So a float slot keeps the
/// raw *bit pattern* of its running `f64` in that integer cell, reinterpreted
/// back to `f64` inside the [`FloatSum`](super::fold::FloatSum)/[`FloatMin`](super::fold::FloatMin)/[`FloatMax`](super::fold::FloatMax)
/// fold, the same "pack a foreign value into the numeric cell" trick
/// [`StringCell`] uses for an `ArenaKey`. The cell is never read through the
/// integer ([`Numeric`]) path, so the bit-punning never crosses families.
///
/// Both widths can hold an `f64` (its 64 bits fit either), so unlike a string
/// extreme a float never forces the planner to widen; it rides whatever width
/// the rest of the signature picks.
pub trait FloatCell: Cell {
    /// Pack an `f64`'s bits into the cell.
    fn from_f64(value: f64) -> Self;
    /// Read the cell's bits back as the `f64` a float slot stored in it.
    fn into_f64(self) -> f64;
    /// Render a finished column of float cells as a `Float64` array.
    fn finish_float(name: &str, col: SlabColumn<Self>) -> (Field, ArrayRef);
}

impl FloatCell for i64 {
    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        value.to_bits() as i64
    }
    #[inline(always)]
    fn into_f64(self) -> f64 {
        f64::from_bits(self as u64)
    }
    fn finish_float(name: &str, col: SlabColumn<Self>) -> (Field, ArrayRef) {
        let len = col.len();
        // Each cell holds an `f64`'s bits in its 8 bytes, so the slab's bytes are
        // already the `Float64` array's bytes, so reinterpret zero-copy.
        let values = ScalarBuffer::<f64>::new(col.into_buffer(), 0, len);
        (
            Field::new(name, DataType::Float64, false),
            Arc::new(Float64Array::new(values, None)),
        )
    }
}

impl FloatCell for i128 {
    #[inline(always)]
    fn from_f64(value: f64) -> Self {
        // Zero-extend the 64 `f64` bits into the wide cell; `into_f64` truncates
        // back to the low 64.
        value.to_bits() as i128
    }
    #[inline(always)]
    fn into_f64(self) -> f64 {
        f64::from_bits(self as u64)
    }
    fn finish_float(name: &str, col: SlabColumn<Self>) -> (Field, ArrayRef) {
        let len = col.len();
        // Wide cells are 16 bytes each (a float only occupies the low 64), so the
        // slab can't be reinterpreted directly; unpack each cell to its `f64`. This
        // is the rare path (a float beside a wide sum or a string extreme) over the
        // low-cardinality group output.
        let cells = ScalarBuffer::<i128>::new(col.into_buffer(), 0, len);
        let arr = Float64Array::from_iter_values(cells.iter().map(|&c| Self::into_f64(c)));
        (Field::new(name, DataType::Float64, false), Arc::new(arr))
    }
}
