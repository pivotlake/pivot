//! The accumulator width for aggregate slots — the single knob shared by the
//! global ([`aggregate`](crate::operations::unary::aggregate)) and grouped
//! ([`group`](crate::operations::unary::group)) paths.
//!
//! Every current aggregate (`Sum`/`Count`/`CountStar`) combines by plain
//! addition, so a slot is just a running integer. The only question is its
//! width, decided the same way in both paths from the summed column's type:
//! `i64` is enough for counts and for sums over 16/32-bit columns (a whole-table
//! scan can't overflow it), but a sum over a 64-bit column (e.g. `SUM(UserID)` =
//! ~2.5e26) needs `i128`. `i64` is the default because it keeps a grouped hash
//! table entry half as wide; `i128` is opt-in for the wide-sum case.

use crate::arrays::{ArrayBuilder, PrimitiveBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{ArrowPrimitiveType, Decimal128Type, Int64Type};
use arrow_array::{ArrayRef, Decimal128Array, Int64Array};
use arrow_schema::DataType;
use std::sync::Arc;

/// An aggregate slot's accumulator integer: `i64` (narrow) or `i128` (wide).
///
/// Combining is plain addition (the same for sum and count, which is why a
/// grouped [`AggregationRow`](super::AggregationRowValueExtractor) merges blindly
/// elementwise); a width only differs in storage and in the Arrow type a `SUM`
/// emits (`Int64` vs `Decimal128(38, 0)`, matching DuckDB's `HUGEINT`).
pub trait Accumulator:
    Copy + Default + Send + Sync + 'static + std::ops::AddAssign + std::ops::Add<Output = Self>
{
    /// Arrow primitive whose `Native` is `Self` — the slab builder element type.
    type Arrow: ArrowPrimitiveType<Native = Self>;

    /// Widen a per-row `i64` contribution (from the [`Aggregate`](super::Aggregate)
    /// ops) into this accumulator.
    fn from_i64(v: i64) -> Self;
    /// Narrow back to `i64` for sort keys / count outputs (both always fit).
    fn as_i64(self) -> i64;

    /// The Arrow type a `SUM` accumulated at this width emits.
    fn sum_datatype() -> DataType;
    /// Finish a slab-backed column of `SUM` values into its output array
    /// (applies the decimal precision/scale for the `i128` case).
    fn finish_sum(builder: PrimitiveBuilder<Self::Arrow>) -> ArrayRef;
    /// Build a single-row `SUM` output array (the global aggregate's result).
    fn one_sum_array(value: Self) -> ArrayRef;
}

impl Accumulator for i64 {
    type Arrow = Int64Type;
    #[inline(always)]
    fn from_i64(v: i64) -> i64 {
        v
    }
    #[inline(always)]
    fn as_i64(self) -> i64 {
        self
    }
    fn sum_datatype() -> DataType {
        DataType::Int64
    }
    fn finish_sum(builder: PrimitiveBuilder<Int64Type>) -> ArrayRef {
        builder.into_array(None)
    }
    fn one_sum_array(value: i64) -> ArrayRef {
        Arc::new(Int64Array::from(vec![value]))
    }
}

impl Accumulator for i128 {
    type Arrow = Decimal128Type;
    #[inline(always)]
    fn from_i64(v: i64) -> i128 {
        v as i128
    }
    #[inline(always)]
    fn as_i64(self) -> i64 {
        self as i64
    }
    fn sum_datatype() -> DataType {
        // Scale 0 = a plain integer; precision 38 is Decimal128's max and covers
        // any i128 a real scan produces.
        DataType::Decimal128(38, 0)
    }
    fn finish_sum(builder: PrimitiveBuilder<Decimal128Type>) -> ArrayRef {
        let arr = builder.into_array(None);
        Arc::new(
            arr.as_primitive::<Decimal128Type>()
                .clone()
                .with_precision_and_scale(38, 0)
                .unwrap(),
        )
    }
    fn one_sum_array(value: i128) -> ArrayRef {
        Arc::new(
            Decimal128Array::from(vec![value])
                .with_precision_and_scale(38, 0)
                .unwrap(),
        )
    }
}
