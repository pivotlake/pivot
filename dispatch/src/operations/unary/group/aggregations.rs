use crate::operations::unary::group::hashtables::Value;

/// A simple counting aggregation, that can be inline in a `HashTable` and holds the current count
/// for a key
#[derive(Default, Copy, Clone)]
pub struct Count {
    pub value: usize,
}

impl Count {
    /// Create a count with a specific initial value.
    pub fn new(size: usize) -> Self {
        Self { value: size }
    }
}

impl Value for Count {
    #[inline]
    fn single() -> Self {
        Self { value: 1 }
    }

    fn merge(mut self, v: Self) -> Self {
        self.value += v.value;
        self
    }
}

/// Which per-group aggregate a value slot accumulates during the consume phase.
///
/// `Avg` is not represented here: DuckDB lowers `AVG(c)` to `sum(c)` + `count(c)`
/// with a divide projection, so a grouped average arrives as a `Sum` slot plus a
/// `Count` slot and the division happens in the downstream projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupAggKind {
    /// `COUNT(*)` — +1 per row, ignores the column.
    CountStar,
    /// `COUNT(col)` — +1 per non-null row.
    Count,
    /// `SUM(col)` — += the (widened) column value.
    Sum,
}

/// One aggregate output slot: which aggregate, over which input column.
#[derive(Clone, Copy, Debug)]
pub struct GroupAggSlot {
    pub kind: GroupAggKind,
    pub column: usize,
}

impl GroupAggSlot {
    pub fn new(kind: GroupAggKind, column: usize) -> Self {
        Self { kind, column }
    }
}

/// A group-by aggregate value: `N` `i64` accumulator slots merged elementwise.
///
/// Every slot (count or sum) accumulates into an `i64`, so the merge and the
/// output emission are uniform — the per-slot [`GroupAggKind`] only matters
/// during the consume phase (whether a row contributes `1` or a column value).
/// `N` is monomorphised per query arity so the value is exactly as wide as the
/// query needs (it sits inline in every hash-table entry, and grouping keys
/// like `WatchID` can produce ~100M entries, so width matters).
#[derive(Copy, Clone)]
pub struct AggRow<const N: usize>(pub [i64; N]);

impl<const N: usize> Default for AggRow<N> {
    fn default() -> Self {
        Self([0; N])
    }
}

impl<const N: usize> Value for AggRow<N> {
    #[inline]
    fn single() -> Self {
        // Not used for multi-aggregate grouping (values are built per row from
        // the slot configuration); provided only to satisfy the trait.
        Self([0; N])
    }

    #[inline]
    fn merge(mut self, v: Self) -> Self {
        for i in 0..N {
            self.0[i] += v.0[i];
        }
        self
    }
}
