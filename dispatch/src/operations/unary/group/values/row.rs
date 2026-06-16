//! The raw aggregation cell array — the storage every aggregation value wraps.

use super::cell::Cell;

/// `N` accumulator cells of width `A` — the in-table payload of one group.
///
/// This is just storage. The aggregation *value* types ([`Mono`](super::Mono),
/// [`DynamicMixed`](super::DynamicMixed), [`CompiledMixed`](super::CompiledMixed))
/// wrap a row and add the merge behaviour; the row itself knows nothing about how
/// its cells combine. It sits inline in every hash-table entry, so its width — `N`
/// (query arity) and `A` (`i64`, or `u128` once string extremes share a cell) — is
/// monomorphised to exactly what the query needs.
#[derive(Clone, Copy)]
pub struct AggregationRow<const N: usize, A: Cell = i64>(pub [A; N]);

impl<const N: usize, A: Cell> Default for AggregationRow<N, A> {
    fn default() -> Self {
        Self([A::default(); N])
    }
}

impl<const N: usize, A: Cell> AggregationRow<N, A> {
    /// Combine cell-wise with `other` under `f`.
    #[inline(always)]
    pub fn zip(self, other: Self, f: impl Fn(A, A) -> A) -> Self {
        Self(std::array::from_fn(|s| f(self.0[s], other.0[s])))
    }
}
