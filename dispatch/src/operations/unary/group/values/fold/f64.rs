//! [`F64Sum`]/[`F64Min`]/[`F64Max`] — the float counterparts of the integer
//! [`Sum`](super::Sum)/[`Min`](super::Min)/[`Max`](super::Max), folding the `f64` an
//! `F64Reader` yields.
//!
//! Like the string extremes these are *not* [`Fold`](super::Fold)s: a float
//! accumulator lives bit-punned in the cell width `A` via [`F64Cell`], so the op
//! owns the reinterpret and exposes plain inherent methods, called from the runtime
//! [`Dynamic`](super::super::container::Dynamic) container's float arms (there is no
//! `Compiled` float slot). `MIN`/`MAX` order with [`f64::total_cmp`] so the extreme
//! is deterministic regardless of row order (NaN sorts greatest, matching DuckDB).

use crate::arrays::SlabColumn;
use crate::operations::unary::group::values::cell::F64Cell;
use arrow_array::ArrayRef;
use arrow_schema::Field;
use std::marker::PhantomData;

/// `SUM` over a float column, accumulating in cell width `A`.
pub struct F64Sum<A>(PhantomData<A>);
/// `MIN` over a float column.
pub struct F64Min<A>(PhantomData<A>);
/// `MAX` over a float column.
pub struct F64Max<A>(PhantomData<A>);

impl<A: F64Cell> F64Sum<A> {
    /// Materialise a new group's cell from a row's value.
    #[inline(always)]
    pub fn seed(v: f64) -> A {
        A::from_f64(v)
    }
    /// Fold a value into an existing cell.
    #[inline(always)]
    pub fn update(acc: A, v: f64) -> A {
        A::from_f64(acc.into_f64() + v)
    }
    /// Combine two finished partials.
    #[inline(always)]
    pub fn merge(a: A, b: A) -> A {
        A::from_f64(a.into_f64() + b.into_f64())
    }
    /// Render a finished column of cells as a `Float64` array.
    pub fn finish(name: &str, col: SlabColumn<A>) -> (Field, ArrayRef) {
        A::finish_float(name, col)
    }
}

macro_rules! float_extreme {
    ($Op:ident, $keep:ident) => {
        impl<A: F64Cell> $Op<A> {
            #[inline(always)]
            pub fn seed(v: f64) -> A {
                A::from_f64(v)
            }
            #[inline(always)]
            pub fn update(acc: A, v: f64) -> A {
                A::from_f64($keep(acc.into_f64(), v))
            }
            #[inline(always)]
            pub fn merge(a: A, b: A) -> A {
                A::from_f64($keep(a.into_f64(), b.into_f64()))
            }
            pub fn finish(name: &str, col: SlabColumn<A>) -> (Field, ArrayRef) {
                A::finish_float(name, col)
            }
        }
    };
}

/// Total-order keep of the smaller value (NaN sorts greatest, so a present NaN never
/// wins a `MIN` unless every value is NaN).
#[inline(always)]
fn keep_min(a: f64, b: f64) -> f64 {
    if b.total_cmp(&a).is_lt() { b } else { a }
}
/// Total-order keep of the larger value.
#[inline(always)]
fn keep_max(a: f64, b: f64) -> f64 {
    if b.total_cmp(&a).is_gt() { b } else { a }
}

float_extreme!(F64Min, keep_min);
float_extreme!(F64Max, keep_max);
