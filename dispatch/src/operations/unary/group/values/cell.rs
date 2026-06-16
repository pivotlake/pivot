//! The numeric cell type for aggregate slots — the `i64`/`i128` a slot's value is
//! stored and emitted as. The single width knob shared by the global
//! ([`aggregate`](crate::operations::unary::aggregate)) and grouped
//! ([`group`](crate::operations::unary::group)) paths.
//!
//! The width is decided the same way in both paths from the summed column's type:
//! `i64` is enough for counts and for sums over 16/32-bit columns (a whole-table
//! scan can't overflow it), but a sum over a 64-bit column whose total can far
//! exceed `i64::MAX` needs `i128`. `i64` is the default because it keeps a grouped
//! hash table entry half as wide; `i128` is opt-in for the wide-sum case.
//!
//! Not every aggregation cell is a [`Cell`]: a string `MIN`/`MAX` stores an
//! [`ArenaKey`](crate::operations::unary::group::ArenaKey) instead. This is only
//! the *numeric* width.

use arrow_array::ArrayRef;
use arrow_array::cast::AsArray;
use arrow_array::types::{ArrowPrimitiveType, Decimal128Type, Int64Type};
use std::sync::Arc;

/// A numeric aggregation cell: `i64` (narrow) or `i128` (wide).
///
/// It bundles the small set of capabilities a numeric slot needs — combine by
/// addition ([`AddAssign`]), build from a per-row `i64` contribution
/// ([`From<i64>`]), widen losslessly to `i128` ([`Into<i128>`], for narrowing
/// `COUNT` through a checked `i64::try_from`), compare ([`Ord`], for top-k sort
/// keys and the extremes), the extreme seeds ([`MIN`](Cell::MIN)/[`MAX`](Cell::MAX)),
/// and the one Arrow column it emits as ([`Arrow`](Cell::Arrow)). *How* cells fold
/// is not here — that's the slot's [`CellFold`](super::CellFold) / kind.
pub trait Cell:
    Copy + Default + Send + Sync + 'static + std::ops::AddAssign + Ord + From<i64> + Into<i128>
{
    /// This width's extremes — the identity seeds for the order statistics: a
    /// running `MIN` starts at `MAX` (every value is `≤` it) and a running `MAX`
    /// at `MIN`. (`SUM`/`COUNT` use the additive identity, [`Default`]'s zero.)
    const MIN: Self;
    const MAX: Self;

    /// The Arrow primitive backing this width's output column (`Int64Type` /
    /// `Decimal128Type`); its `Native` is the cell itself.
    type Arrow: ArrowPrimitiveType<Native = Self>;

    /// Finish a freshly built `Self::Arrow` column into its output array:
    /// identity for `Int64`, sets precision/scale `(38, 0)` for `Decimal128`
    /// (whose default scale is 10) so it matches DuckDB's `HUGEINT`.
    fn finalize(array: ArrayRef) -> ArrayRef;
}

impl Cell for i64 {
    const MIN: Self = i64::MIN;
    const MAX: Self = i64::MAX;
    type Arrow = Int64Type;
    #[inline(always)]
    fn finalize(array: ArrayRef) -> ArrayRef {
        array
    }
}

impl Cell for i128 {
    const MIN: Self = i128::MIN;
    const MAX: Self = i128::MAX;
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
