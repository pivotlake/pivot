//! The accumulator width for aggregate slots — the single knob shared by the
//! global ([`aggregate`](crate::operations::unary::aggregate)) and grouped
//! ([`group`](crate::operations::unary::group)) paths.
//!
//! Every current aggregate (`Sum`/`Count`/`CountStar`) combines by plain
//! addition, so a slot is just a running integer. The only question is its
//! width, decided the same way in both paths from the summed column's type:
//! `i64` is enough for counts and for sums over 16/32-bit columns (a whole-table
//! scan can't overflow it), but a sum over a 64-bit column whose total can far
//! exceed `i64::MAX` needs `i128`. `i64` is the default because it keeps a grouped hash
//! table entry half as wide; `i128` is opt-in for the wide-sum case.

use arrow_array::ArrayRef;
use arrow_array::cast::AsArray;
use arrow_array::types::{ArrowPrimitiveType, Decimal128Type, Int64Type};
use std::sync::Arc;

/// An aggregate slot's accumulator integer: `i64` (narrow) or `i128` (wide).
///
/// A width is just an integer that combines by addition ([`core::ops::AddAssign`]), is
/// built from a per-row `i64` contribution ([`From<i64>`]), widens losslessly to
/// `i128` ([`Into<i128>`], for narrowing `COUNT` columns through a checked
/// `i64::try_from`), is comparable ([`Ord`], for top-k sort keys), and maps to
/// one Arrow output column type ([`Arrow`](Accumulator::Arrow)).
pub trait Accumulator:
    Copy + Default + Send + Sync + 'static + std::ops::AddAssign + Ord + From<i64> + Into<i128>
{
    /// The Arrow primitive backing this width's output column (`Int64Type` /
    /// `Decimal128Type`); its `Native` is the accumulator itself.
    type Arrow: ArrowPrimitiveType<Native = Self>;

    /// Finish a freshly built `Self::Arrow` column into its output array:
    /// identity for `Int64`, sets precision/scale `(38, 0)` for `Decimal128`
    /// (whose default scale is 10) so it matches DuckDB's `HUGEINT`.
    fn finalize(array: ArrayRef) -> ArrayRef;
}

impl Accumulator for i64 {
    type Arrow = Int64Type;
    #[inline(always)]
    fn finalize(array: ArrayRef) -> ArrayRef {
        array
    }
}

impl Accumulator for i128 {
    type Arrow = Decimal128Type;
    fn finalize(array: ArrayRef) -> ArrayRef {
        Arc::new(
            array
                .as_primitive::<Decimal128Type>()
                .clone()
                .with_precision_and_scale(38, 0)
                .unwrap(),
        )
    }
}
