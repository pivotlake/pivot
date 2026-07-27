//! [`U128Sum`]/[`U128Min`]/[`U128Max`] — re-fold a wide (`i128`) partial read from a
//! `Decimal128` column (see `U128Reader`): an aggregate re-reading partials a prior
//! level already widened.
//!
//! Like the [`F64`](super::F64Sum) folds these are inherent helpers, not
//! [`Fold`](super::Fold)s: the value is a full `i128` held in the cell width `A` via
//! [`WideCell`]. Only `seed` and `update` are defined — a wide cell `merge`s and
//! renders identically whether it was seeded from an `i64` or an `i128` input (both
//! accumulate in `A = i128`), so the [`Variable`](super::super::container::Variable)
//! container drives `merge` and `finish` through the integer
//! [`Sum`](super::Sum)/[`Min`](super::Min)/[`Max`](super::Max) arms. Reading the
//! `Decimal128` as `i128` here (not `i64`) keeps a partial that overflows `i64`
//! exact.

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
