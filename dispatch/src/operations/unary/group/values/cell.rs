//! What a slot's value is *stored* as.
//!
//! [`Cell`] is the bare storage marker — any `Copy` value can sit in a slot, so
//! it's a blanket bound, nothing to implement. The width-specific numeric
//! behaviour (how an `i64` vs a `u128` adds, takes an extreme, and renders to
//! Arrow) lives in [`NumericCell`], which the numeric [`Fold`](super::fold)s use.
//! A string slot stores an [`ArenaKey`](super::super::keys::ArenaKey) (a `u128`)
//! and never touches [`NumericCell`] — its behaviour is the string fold's.

use crate::arrays::SlabColumn;
use arrow_array::{ArrayRef, Decimal128Array, Int64Array};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field};
use std::sync::Arc;

/// The bare requirement to live in an aggregate slot: a plain `Copy` value. Held
/// by every cell type (`i64`, `u128`, `ArenaKey`), so it's a blanket marker — the
/// per-aggregate behaviour is the [`Read`](super::read::Read)/[`Fold`](super::fold::Fold),
/// never the cell.
pub trait Cell: Copy + Default + Send + Sync + 'static {}
impl<T: Copy + Default + Send + Sync + 'static> Cell for T {}

/// A numeric aggregate cell: `i64` (narrow) or `u128` (wide, carrying an `i128`).
///
/// The width is chosen by the planner from the summed column's type — `i64` is
/// enough for counts and 16/32-bit sums, `u128` keeps a 64-bit sum from
/// overflowing — and it determines the Arrow output: `Int64` or `Decimal128(38, 0)`
/// (matching DuckDB's `HUGEINT`). All arithmetic is *signed*; `u128` reinterprets
/// as `i128`, so a wide sum of negatives stays correct (`u128`'s own `Ord`/`+`
/// would not).
pub trait NumericCell: Cell {
    /// This width's signed bounds — the seeds the global aggregate's running
    /// `MIN`/`MAX` start from (a running `MIN` starts at `MAX`, and vice versa).
    /// The grouped path never needs them: a new group stores its first row's value
    /// directly. They are width facts, not aggregate facts — a string extreme uses
    /// neither (it stores `ArenaKey`s and the string fold compares bytes).
    const MIN: Self;
    const MAX: Self;

    /// Build from a per-row `i64` contribution (a `COUNT`'s `1`, or a widened
    /// column value).
    fn from_i64(v: i64) -> Self;
    /// This cell as a signed 128-bit value — the universal sort key, and the
    /// route a `COUNT` narrows back to `i64` through.
    fn to_i128(self) -> i128;
    /// Signed add — the additive fold.
    fn add(self, other: Self) -> Self;
    /// Signed min/max — the extreme folds.
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    /// Render a finished column of these cells into the Arrow array + field,
    /// handing the engine slab to Arrow zero-copy (the grouped output path).
    fn finish(name: &str, col: SlabColumn<Self>) -> (Field, ArrayRef);

    /// This width's Arrow output type (`Int64` / `Decimal128(38, 0)`).
    fn data_type() -> DataType;
    /// A single-row array of this width — the global (no-GROUP-BY) output.
    fn scalar_array(value: Self) -> ArrayRef;
}

impl NumericCell for i64 {
    const MIN: Self = i64::MIN;
    const MAX: Self = i64::MAX;
    #[inline(always)]
    fn from_i64(v: i64) -> Self {
        v
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self as i128
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        self + other
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Ord::min(self, other)
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Ord::max(self, other)
    }
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
    fn scalar_array(value: Self) -> ArrayRef {
        Arc::new(Int64Array::from(vec![value]))
    }
}

impl NumericCell for u128 {
    const MIN: Self = i128::MIN as u128;
    const MAX: Self = i128::MAX as u128;
    #[inline(always)]
    fn from_i64(v: i64) -> Self {
        v as i128 as u128
    }
    #[inline(always)]
    fn to_i128(self) -> i128 {
        self as i128
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        (self as i128).wrapping_add(other as i128) as u128
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Ord::min(self as i128, other as i128) as u128
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Ord::max(self as i128, other as i128) as u128
    }
    fn finish(name: &str, col: SlabColumn<Self>) -> (Field, ArrayRef) {
        // The cells carry `i128` bit patterns (Decimal128's native); the `u128`
        // and `i128` buffers are byte-identical, so reinterpret zero-copy.
        // Precision/scale (38, 0) matches DuckDB's HUGEINT (default scale is 10).
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
    fn scalar_array(value: Self) -> ArrayRef {
        Arc::new(
            Decimal128Array::from(vec![value as i128])
                .with_precision_and_scale(38, 0)
                .expect("(38, 0) is a valid decimal128 precision/scale"),
        )
    }
}
