//! Per-cell merge functions — the `F` of a homogeneous [`Mono`](super::Mono)
//! value, where every cell folds the same way.
//!
//! These are the "if they're all the same, use a specific one" case: a query
//! whose slots are all `SUM`/`COUNT` folds with [`Add`], all `MIN` with [`Min`],
//! all `MAX` with [`Max`] — branch-free, no per-slot dispatch. Heterogeneous
//! signatures fall back to [`DynamicMixed`](super::DynamicMixed) instead.

use super::accumulator::Accumulator;

/// How a single accumulator cell combines with another.
pub trait CellFold<A>: Send + Sync + 'static {
    fn fold(a: A, b: A) -> A;
}

/// `SUM` / `COUNT` — add.
pub struct Add;
impl<A: Accumulator> CellFold<A> for Add {
    #[inline(always)]
    fn fold(a: A, b: A) -> A {
        let mut x = a;
        x += b;
        x
    }
}

/// `MIN` — keep the smaller.
pub struct Min;
impl<A: Accumulator> CellFold<A> for Min {
    #[inline(always)]
    fn fold(a: A, b: A) -> A {
        a.min(b)
    }
}

/// `MAX` — keep the larger.
pub struct Max;
impl<A: Accumulator> CellFold<A> for Max {
    #[inline(always)]
    fn fold(a: A, b: A) -> A {
        a.max(b)
    }
}
