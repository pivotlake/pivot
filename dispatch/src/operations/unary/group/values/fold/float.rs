//! [`SumF<A>`](SumF) / [`MinF<A>`](MinF) / [`MaxF<A>`](MaxF) - `SUM`/`MIN`/`MAX`
//! over the `f64` their [`Read`](super::super::read::FloatRead) yields.
//!
//! Like the string extremes these are *not* [`Fold`](super::Fold)s, though for a
//! different reason than impossibility: their cell is the integer width `A` and
//! their value is `f64`, which would satisfy `Fold` fine. They are inherent
//! methods because a float aggregate only ever runs in the runtime
//! [`Dynamic`](super::super::container::Dynamic) container's float arms (there is
//! no `Compiled` float slot, so a `Fold` impl would have no call site), and
//! because, like a string extreme, the op reinterprets the cell rather than
//! folding a numeric value directly.
//!
//! The accumulator is the plain cell width `A`: an `f64` is just 64 bits, so a
//! float aggregate rides the same `A` cell an integer slot uses, no separate cell
//! type and no container reinterpret. Viewing those bits as an `f64` is *this op's*
//! business, via [`FloatCell`]. A `MIN`/`MAX` keep uses `f64::min`/`f64::max`,
//! which favour the non-NaN operand. Note this differs from a database that orders
//! NaN as the largest float (where `MAX` of a column containing NaN is NaN); the
//! folds here drop NaN instead, an edge case left for a follow-up.

use crate::arrays::SlabColumn;
use crate::operations::unary::group::values::cell::FloatCell;
use arrow_array::ArrayRef;
use arrow_schema::Field;
use std::marker::PhantomData;

/// `SUM` over a float column, accumulating in cell width `A`.
pub struct SumF<A = i64>(PhantomData<A>);
/// `MIN` over a float column.
pub struct MinF<A = i64>(PhantomData<A>);
/// `MAX` over a float column.
pub struct MaxF<A = i64>(PhantomData<A>);

impl<A: FloatCell> SumF<A> {
    /// Materialise a new group's cell from a row's value.
    #[inline(always)]
    pub fn seed(v: f64) -> A {
        A::from_float(v)
    }
    /// Fold a value into an existing cell.
    #[inline(always)]
    pub fn update(acc: A, v: f64) -> A {
        A::from_float(acc.into_float() + v)
    }
    /// Combine two finished partials.
    #[inline(always)]
    pub fn merge(a: A, b: A) -> A {
        A::from_float(a.into_float() + b.into_float())
    }
    /// Render a finished column of cells into the Arrow `Float64` array + field.
    pub fn finish(name: &str, col: SlabColumn<A>) -> (Field, ArrayRef) {
        A::finish(name, col)
    }
}

macro_rules! float_extreme {
    ($Op:ident, $keep:ident) => {
        impl<A: FloatCell> $Op<A> {
            #[inline(always)]
            pub fn seed(v: f64) -> A {
                A::from_float(v)
            }
            #[inline(always)]
            pub fn update(acc: A, v: f64) -> A {
                A::from_float(f64::$keep(acc.into_float(), v))
            }
            #[inline(always)]
            pub fn merge(a: A, b: A) -> A {
                A::from_float(f64::$keep(a.into_float(), b.into_float()))
            }
            pub fn finish(name: &str, col: SlabColumn<A>) -> (Field, ArrayRef) {
                A::finish(name, col)
            }
        }
    };
}

float_extreme!(MinF, min);
float_extreme!(MaxF, max);
