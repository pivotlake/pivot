//! [`FloatSum`] / [`FloatMin`] / [`FloatMax`]: the `SUM`/`MIN`/`MAX` ops over a
//! `Float64` column, folding the `f64` their [`Read`](super::super::read) yields.
//!
//! They mirror the integer [`Sum`](super::Sum)/[`Min`](super::Min)/[`Max`](super::Max),
//! but fold in `f64`: the running value is kept as its bit pattern in the numeric
//! cell ([`FloatCell`]), reinterpreted to `f64` for each combine and packed back.
//! An `f64` fits either width, so a float never forces the wide cell; it rides
//! whatever width `A` the rest of the signature picks.

use super::{Fold, FoldAcc};
use crate::arrays::SlabColumn;
use crate::operations::unary::group::values::cell::FloatCell;
use arrow_array::ArrayRef;
use arrow_schema::Field;
use std::marker::PhantomData;

/// `SUM` over a `Float64` column, accumulating in width `A` (its cell holds the
/// running `f64`'s bits).
pub struct FloatSum<A = i64>(PhantomData<A>);
/// `MIN` over a `Float64` column.
pub struct FloatMin<A = i64>(PhantomData<A>);
/// `MAX` over a `Float64` column.
pub struct FloatMax<A = i64>(PhantomData<A>);

impl<A: FloatCell> FoldAcc for FloatSum<A> {
    type Acc = A;
    type SharedContext = ();
    type WorkerContext = ();

    #[inline(always)]
    fn merge(a: A, b: A, _ctx: &()) -> A {
        A::from_f64(a.into_f64() + b.into_f64())
    }
    fn finish(name: &str, col: SlabColumn<A>, _ctx: &()) -> (Field, ArrayRef) {
        A::finish_float(name, col)
    }
}

impl<A: FloatCell> Fold<f64> for FloatSum<A> {
    #[inline(always)]
    fn seed(v: f64, _wc: &mut ()) -> A {
        A::from_f64(v)
    }
    #[inline(always)]
    fn update(acc: A, v: f64, _wc: &mut (), _ctx: &()) -> A {
        A::from_f64(acc.into_f64() + v)
    }
}

/// The float extremes order via [`keep_min`]/[`keep_max`], which rank every
/// `NaN` (either sign bit) above all finite values, matching DuckDB's
/// `MIN`/`MAX`.
macro_rules! float_extreme {
    ($Op:ident, $keep:ident) => {
        impl<A: FloatCell> FoldAcc for $Op<A> {
            type Acc = A;
            type SharedContext = ();
            type WorkerContext = ();

            #[inline(always)]
            fn merge(a: A, b: A, _ctx: &()) -> A {
                A::from_f64($keep(a.into_f64(), b.into_f64()))
            }
            fn finish(name: &str, col: SlabColumn<A>, _ctx: &()) -> (Field, ArrayRef) {
                A::finish_float(name, col)
            }
        }

        impl<A: FloatCell> Fold<f64> for $Op<A> {
            #[inline(always)]
            fn seed(v: f64, _wc: &mut ()) -> A {
                A::from_f64(v)
            }
            #[inline(always)]
            fn update(acc: A, v: f64, _wc: &mut (), _ctx: &()) -> A {
                A::from_f64($keep(acc.into_f64(), v))
            }
        }
    };
}

/// `MIN` keep: the smaller of `a`/`b`, with `NaN` ranked greatest (DuckDB
/// semantics), so a `NaN` only wins when both are `NaN`. Plain `<` handles the
/// finite case and treats `-0.0`/`+0.0` as equal (SQL does not distinguish them);
/// the explicit `NaN` arms fix `<`'s "always false against `NaN`" so a `NaN`
/// never displaces a finite minimum, regardless of its sign bit (which
/// `total_cmp` would mishandle for a negative `NaN`). Shared with the global
/// aggregate path so both fold the same ordering.
#[inline(always)]
pub(crate) fn keep_min(a: f64, b: f64) -> f64 {
    if b.is_nan() {
        a
    } else if a.is_nan() || b < a {
        b
    } else {
        a
    }
}
/// `MAX` keep: the larger of `a`/`b`, with `NaN` ranked greatest, so a `NaN`
/// wins whenever either side is `NaN`. See [`keep_min`].
#[inline(always)]
pub(crate) fn keep_max(a: f64, b: f64) -> f64 {
    if b.is_nan() {
        b
    } else if a.is_nan() || a > b {
        a
    } else {
        b
    }
}

float_extreme!(FloatMin, keep_min);
float_extreme!(FloatMax, keep_max);

#[cfg(test)]
mod tests {
    use super::{keep_max, keep_min};

    // A negative-signbit NaN, which `f64::total_cmp` would wrongly rank below all
    // finite values; DuckDB ranks every NaN (either sign) as the greatest.
    const NEG_NAN: f64 = f64::from_bits(0xFFF8_0000_0000_0000);

    #[test]
    fn extremes_over_finite_values() {
        assert_eq!(keep_min(1.0, 2.0), 1.0);
        assert_eq!(keep_min(2.0, 1.0), 1.0);
        assert_eq!(keep_max(1.0, 2.0), 2.0);
        assert_eq!(keep_max(2.0, 1.0), 2.0);
        assert_eq!(keep_min(-3.5, 10.0), -3.5);
        assert_eq!(keep_max(-3.5, 10.0), 10.0);
    }

    #[test]
    fn nan_ranks_greatest_in_either_position_and_sign() {
        // MIN: a NaN (positive or negative sign) never displaces a finite minimum.
        assert_eq!(keep_min(1.0, f64::NAN), 1.0);
        assert_eq!(keep_min(f64::NAN, 1.0), 1.0);
        assert_eq!(keep_min(1.0, NEG_NAN), 1.0);
        assert_eq!(keep_min(NEG_NAN, 1.0), 1.0);
        // MAX: a NaN always wins.
        assert!(keep_max(1.0, f64::NAN).is_nan());
        assert!(keep_max(f64::NAN, 1.0).is_nan());
        assert!(keep_max(1.0, NEG_NAN).is_nan());
        assert!(keep_max(NEG_NAN, 1.0).is_nan());
    }

    #[test]
    fn both_nan_folds_to_nan() {
        assert!(keep_min(f64::NAN, NEG_NAN).is_nan());
        assert!(keep_max(f64::NAN, NEG_NAN).is_nan());
    }
}
