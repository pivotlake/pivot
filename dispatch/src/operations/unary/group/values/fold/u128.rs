//! [`U128Sum`]/[`U128Min`]/[`U128Max`] — re-fold a wide (`i128`) partial read from a
//! `Decimal128` column (see `U128Reader`): an aggregate re-reading partials a prior
//! level already widened.
//!
//! These helpers preserve the full `i128` read from a prior aggregate's
//! `Decimal128` output. Merge and rendering are identical to the ordinary
//! integer operations once both values are in an `i128` cell.

use super::super::cell::WideCell;
use std::marker::PhantomData;

/// `SUM` re-folding an `i128` partial into cell width `A`.
pub struct U128Sum<A>(PhantomData<A>);
/// `MIN` re-folding an `i128` partial.
pub struct U128Min<A>(PhantomData<A>);
/// `MAX` re-folding an `i128` partial.
pub struct U128Max<A>(PhantomData<A>);

impl<A: WideCell> U128Sum<A> {
    #[inline(always)]
    pub fn seed(v: i128) -> A {
        A::from_i128(v)
    }
    #[inline(always)]
    pub fn update(acc: A, v: i128) -> A {
        A::from_i128(acc.into_i128() + v)
    }
}

macro_rules! wide_extreme {
    ($Op:ident, $keep:ident) => {
        impl<A: WideCell> $Op<A> {
            #[inline(always)]
            pub fn seed(v: i128) -> A {
                A::from_i128(v)
            }
            #[inline(always)]
            pub fn update(acc: A, v: i128) -> A {
                A::from_i128(Ord::$keep(acc.into_i128(), v))
            }
        }
    };
}

wide_extreme!(U128Min, min);
wide_extreme!(U128Max, max);
