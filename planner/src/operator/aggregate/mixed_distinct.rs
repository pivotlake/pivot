//! A grouped aggregate mixing one `COUNT(DISTINCT x)` with non-distinct
//! aggregates (e.g. ClickBench Q9:
//! `RegionID, SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth),
//! COUNT(DISTINCT UserID) GROUP BY RegionID`).
//!
//! Lowered to a two-level GROUP BY by exploiting decomposability: the
//! non-distinct aggregates (SUM/COUNT/COUNT(*)) are sums of per-subgroup
//! partials, and `COUNT(DISTINCT x)` is the number of distinct `(group, x)`
//! subgroups. So:
//! * **Inner** GROUP BY `(group, x)` computes each non-distinct aggregate's
//!   partial → `[group, x, p0, p1, …]`.
//! * **Outer** GROUP BY `group` re-sums each partial (SUM over the inner column)
//!   and uses `COUNT(*)` of the inner rows for the distinct count.
//!
//! The outer slots are emitted in the original expression order — distinct expr
//! → `COUNT(*)`, each non-distinct expr → `SUM` of its inner partial — so the
//! output column layout matches DuckDB's aggregate output and the downstream
//! projection (e.g. the AVG divide) lines up. AVG is already split by DuckDB into
//! `sum`+`count` exprs, handled generically here. Single integer group key only.

use super::Aggregate;
use crate::compile::Error;
use crate::expression::{AggregateFunc, Expression, Ref};
use crate::types::Type;
use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
use dispatch::{AggregationKind, AggregationSlot, Dynamic, IntKeyExtractor, IntPairKeyExtractor};

/// The two-level COUNT(DISTINCT) aggregation is all additive (dedup counts and
/// re-summed partials), so both levels use the `ONLY_ADDITIVE` `Dynamic`: the
/// const prunes the per-slot Min/Max/string arms, collapsing the fold to the
/// same branch-free additive codegen as a hand-written `Mono`.
type AddVal<const N: usize, const EAGER: bool> = Dynamic<N, i64, true, EAGER>;

impl Aggregate {
    pub(super) fn compile_grouped_mixed_distinct(
        &self,
        input: dispatch::RecordBatchOperatorSpec,
    ) -> Result<dispatch::RecordBatchOperatorSpec, Error> {
        if self.groups.len() != 1 {
            return Err(Error::UnsupportedAggregateGroupAmount(self.groups.len()));
        }
        let g = match &self.groups[0] {
            Expression::Ref(r) => r,
            e => return Err(Error::UnexpectedAggExpression(e.clone())),
        };
        let g_col = g.column_idx;

        let Plan {
            distinct_col: x,
            inner_slots,
            outer_slots,
        } = self.plan_two_level()?;
        let x_col = x.column_idx;

        // MEASUREMENT baseline: route both levels' `Dynamic` to the eager (original
        // in-place) consume instead of the deferred path.
        let use_eager = std::env::var_os("PIVOT_GB_OLDDYNAMIC").is_some();

        // group_by_aggregate monomorphised over key type + slot arity, and over the
        // `EAGER` consume-strategy const (chosen once here so each build is straight
        // line).
        macro_rules! grouped_eager {
            ($spec:expr, $K:ty, $cols:expr, $slots:expr, $eager:literal) => {{
                let slots = $slots;
                match slots.len() {
                    1 => $spec.group_by_aggregate::<$K, AddVal<1, $eager>>($cols, slots, None),
                    2 => $spec.group_by_aggregate::<$K, AddVal<2, $eager>>($cols, slots, None),
                    3 => $spec.group_by_aggregate::<$K, AddVal<3, $eager>>($cols, slots, None),
                    4 => $spec.group_by_aggregate::<$K, AddVal<4, $eager>>($cols, slots, None),
                    5 => $spec.group_by_aggregate::<$K, AddVal<5, $eager>>($cols, slots, None),
                    6 => $spec.group_by_aggregate::<$K, AddVal<6, $eager>>($cols, slots, None),
                    n => return Err(Error::UnsupportedAggregateExpressionAmount(n)),
                }
            }};
        }
        macro_rules! grouped {
            ($spec:expr, $K:ty, $cols:expr, $slots:expr) => {{
                if use_eager {
                    grouped_eager!($spec, $K, $cols, $slots, true)
                } else {
                    grouped_eager!($spec, $K, $cols, $slots, false)
                }
            }};
        }
        macro_rules! build {
            ($G:ty, $X:ty) => {{
                let inner = grouped!(
                    input,
                    IntPairKeyExtractor<$G, $X>,
                    vec![g_col, x_col],
                    inner_slots
                );
                Ok(grouped!(inner, IntKeyExtractor<$G>, vec![0], outer_slots))
            }};
        }
        macro_rules! by_x {
            ($G:ty) => {
                match &x.return_type {
                    Type::Int8 => build!($G, Int8Type),
                    Type::Int16 => build!($G, Int16Type),
                    Type::Int32 => build!($G, Int32Type),
                    Type::Int64 => build!($G, Int64Type),
                    dt => Err(Error::DataTypeNotSupportedForGroupBy(dt.clone())),
                }
            };
        }
        match &g.return_type {
            Type::Int8 => by_x!(Int8Type),
            Type::Int16 => by_x!(Int16Type),
            Type::Int32 => by_x!(Int32Type),
            Type::Int64 => by_x!(Int64Type),
            dt => Err(Error::DataTypeNotSupportedForGroupBy(dt.clone())),
        }
    }

    /// Walk the aggregate expressions once, building the inner (non-distinct
    /// partials over `(g, x)`) and outer (re-sum partials + COUNT(*) for the
    /// distinct) slot lists. The outer slots stay in expression order so the
    /// output columns match DuckDB's aggregate layout.
    fn plan_two_level(&self) -> Result<Plan<'_>, Error> {
        // The inner emits `[g, x, partial0, partial1, …]`, so partial `k` is at
        // column `2 + k`. Coalesce inner partials that compute the *same* value,
        // so each is scattered/merged once: all COUNT/COUNT(*) slots are
        // identical (pivot's Count contributes +1 per row regardless of
        // column/null, == COUNT(*)), and SUMs of the same column are identical.
        // Fewer inner slots ⇒ a narrower hash-table entry, the dominant cost of
        // the high-cardinality inner.
        let mut distinct_col: Option<&Ref> = None;
        let mut inner = InnerSlots::default();
        let mut outer_slots: Vec<AggregationSlot> = Vec::new();

        for e in &self.expressions {
            let Expression::AggregateFunc(func) = e else {
                return Err(Error::UnsupportedAggregateExpression(e.clone()));
            };
            match func {
                AggregateFunc::CountDistinct(a) => {
                    if distinct_col.is_some() {
                        return Err(Error::UnsupportedAggregateExpression(e.clone()));
                    }
                    distinct_col = Some(&a.column);
                    // distinct count = number of inner (distinct-pair) rows for g.
                    outer_slots.push(AggregationSlot::new(AggregationKind::CountStar, 0));
                }
                AggregateFunc::CountStar(_) => {
                    let col = inner.intern(Partial::Count, AggregationKind::CountStar, 0);
                    outer_slots.push(AggregationSlot::new(AggregationKind::Sum, col));
                }
                // Same value as COUNT(*) in pivot's Count semantics → coalesce.
                AggregateFunc::Count(a) => {
                    let col =
                        inner.intern(Partial::Count, AggregationKind::Count, a.column.column_idx);
                    outer_slots.push(AggregationSlot::new(AggregationKind::Sum, col));
                }
                AggregateFunc::Sum(a) => {
                    let key = Partial::Sum(a.column.column_idx);
                    let col = inner.intern(key, AggregationKind::Sum, a.column.column_idx);
                    outer_slots.push(AggregationSlot::new(AggregationKind::Sum, col));
                }
                _ => return Err(Error::UnsupportedAggregateExpression(e.clone())),
            }
        }

        // Guaranteed by the caller (exactly one CountDistinct), but stay total.
        let distinct_col = distinct_col.ok_or(Error::UnsupportedAggregateExpressionAmount(0))?;
        Ok(Plan {
            distinct_col,
            inner_slots: inner.slots,
            outer_slots,
        })
    }
}

/// The two-level lowering: the distinct argument plus the inner/outer slot lists.
struct Plan<'a> {
    distinct_col: &'a Ref,
    inner_slots: Vec<AggregationSlot>,
    outer_slots: Vec<AggregationSlot>,
}

/// Identifies an inner partial for coalescing: a count, or a sum of a column.
#[derive(PartialEq)]
enum Partial {
    Count,
    Sum(usize),
}

/// Accumulates the inner partial slots, deduplicating equal ones and tracking
/// which inner output column each landed in.
#[derive(Default)]
struct InnerSlots {
    slots: Vec<AggregationSlot>,
    interned: Vec<(Partial, usize)>,
}

impl InnerSlots {
    /// Return the inner output column for `key`, appending a new slot (built from
    /// `kind`/`column`) the first time the key is seen. The inner emits
    /// `[g, x, partial0, …]`, so a new partial lands at column `2 + slots.len()`.
    fn intern(&mut self, key: Partial, kind: AggregationKind, column: usize) -> usize {
        if let Some((_, col)) = self.interned.iter().find(|(k, _)| *k == key) {
            return *col;
        }
        let col = 2 + self.slots.len();
        self.slots.push(AggregationSlot::new(kind, column));
        self.interned.push((key, col));
        col
    }
}
