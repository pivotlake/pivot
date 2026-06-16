//! What a slot's value is *stored* as.
//!
//! [`Cell`] is a bare marker — any `Copy` value can sit in a slot, so it's a
//! blanket bound with nothing to implement. *How* cells combine is the
//! [`Fold`](super::fold)'s job (plain `Ord::min` / `+` for the numeric folds, an
//! arena compare for the string ones) — never the cell's. The only thing a
//! numeric width owns is how it renders to Arrow ([`NumericArrow`]), since that
//! genuinely depends on the width (`i64 → Int64`, `i128 → Decimal128`). The
//! [`Numeric`] bound alias bundles that with the std arithmetic the numeric folds
//! lean on, so a fold can just say `A: Numeric`.

use crate::arrays::SlabColumn;
use arrow_array::types::{ArrowPrimitiveType, Decimal128Type, Int64Type};
use arrow_array::{ArrayRef, Decimal128Array, Int64Array};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{DataType, Field};
use std::sync::Arc;

/// The bare requirement to live in an aggregate slot: a plain `Copy` value. Held
/// by every cell type (`i64`, `i128`, `ArenaKey`, the `u128` union), so it's a
/// blanket marker — the per-aggregate behaviour is the [`Read`](super::read) /
/// [`Fold`](super::fold), never the cell.
pub trait Cell: Copy + Default + Send + Sync + 'static {}
impl<T: Copy + Default + Send + Sync + 'static> Cell for T {}

/// How a numeric width renders its finished column to Arrow. `i64 → Int64`,
/// `i128 → Decimal128(38, 0)` (matching DuckDB's `HUGEINT`). This is the one
/// irreducible width fact — there's no std reverse-map from a native type to its
/// Arrow `PrimitiveType` — so it lives here; the *arithmetic* does not.
pub trait NumericArrow: Cell {
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

impl NumericArrow for i64 {
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

impl NumericArrow for i128 {
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

/// A numeric aggregate cell (`i64` narrow / `i128` wide): a [`Cell`] that adds
/// (`+`), orders (`Ord`), builds from a per-row `i64` and widens to `i128`, and
/// renders to Arrow ([`NumericArrow`]). A bound alias — no methods of its own, so
/// the numeric folds combine with std ops, not cell methods.
pub trait Numeric:
    Cell + Ord + std::ops::Add<Output = Self> + From<i64> + Into<i128> + NumericArrow
{
}
impl<T> Numeric for T where
    T: Cell + Ord + std::ops::Add<Output = T> + From<i64> + Into<i128> + NumericArrow
{
}
