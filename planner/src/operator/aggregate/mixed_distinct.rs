//! A grouped aggregate mixing one `COUNT(DISTINCT x)` with non-distinct
//! aggregates (e.g. `g, SUM(a), COUNT(*), AVG(b), COUNT(DISTINCT user)
//! GROUP BY g`, or `phrase, MIN(url), MIN(title), COUNT(*),
//! COUNT(DISTINCT user) GROUP BY phrase`).
//!
//! Lowered to a two-level GROUP BY by exploiting decomposability: every
//! non-distinct aggregate is foldable from per-subgroup partials (SUM/COUNT are
//! sums of partial sums/counts; MIN/MAX are the extreme of partial extremes),
//! and `COUNT(DISTINCT x)` is the number of distinct `(group…, x)` subgroups. So:
//! * **Inner** GROUP BY `(group…, x)` computes each non-distinct aggregate's
//!   partial, emitting `[group…, x, p0, p1, …]`.
//! * **Outer** GROUP BY `group…` re-folds each partial (re-summing a SUM/COUNT,
//!   re-extremising a MIN/MAX) and uses `COUNT(*)` of the inner rows for the
//!   distinct count.
//!
//! The outer slots are emitted in the original expression order (distinct expr to
//! `COUNT(*)`, each non-distinct expr to the re-fold of its inner partial), so the
//! output column layout matches DuckDB's aggregate output and the downstream
//! projection (e.g. the AVG divide) lines up. AVG is already split by DuckDB into
//! `sum`+`count` exprs, handled generically here.
//!
//! Both levels lower through [`lower_grouped`], so they support every group-key
//! and value shape the general grouped path does: an integer or string group
//! key (or several), and `MIN`/`MAX` over integers or strings. A `MIN`/`MAX` over
//! a string makes its level *wide* (the `i128` cell holding the `ArenaKey`), so
//! the level's `COUNT`/`SUM` partials emit as `Decimal128` and the outer re-reads
//! them through that width (see `NumReader` in `dispatch`).

use super::grouped::lower_grouped;
use super::{Aggregate, extreme_kind};
use crate::compile::Error;
use crate::expression::{AggregateFunc, Expression, NumericAggregate};
use crate::types::Type;
use dispatch::{AggregationKind, AggregationSlot};

impl Aggregate {
    pub(super) fn compile_grouped_mixed_distinct(
        &self,
        input: dispatch::RecordBatchOperatorSpec,
    ) -> Result<dispatch::RecordBatchOperatorSpec, Error> {
        // Materialise any computed group key into a leading column, so from here
        // every group key is a plain column; the distinct argument and the
        // aggregate value columns shift right by the number of materialised keys.
        let (input, groups, key_shift) = self.materialize_group_keys(input)?;

        let Plan {
            distinct_col,
            distinct_type,
            inner_slots,
            outer_slots,
        } = self.plan_two_level(&groups, key_shift)?;

        // Inner keys are the group columns followed by the distinct argument; the
        // inner dedups `(group…, x)`. Its row encoding emits the key columns in
        // this order, so the outer's group columns are the leading `0..n_groups`.
        let mut inner_keys = groups.clone();
        inner_keys.push((distinct_col, distinct_type));
        let outer_keys: Vec<(usize, Type)> = groups
            .iter()
            .enumerate()
            .map(|(col, (_, ty))| (col, ty.clone()))
            .collect();

        // A string extreme forces its level to the wide `i128` cell. (A SUM over a
        // 64-bit column does too in the single-level path; the partials this path
        // re-sums are counts and narrow sums that fit `i64`, so they don't.)
        let inner_wide = inner_slots.iter().any(|s| s.kind.is_string_extreme());
        let outer_wide = outer_slots.iter().any(|s| s.kind.is_string_extreme());

        // Empty `sig`: the synthetic slots match no `Compiled` specialisation, so
        // both levels fold per-slot in `Dynamic`. Only the final (outer) level
        // carries the pushed-down LIMIT.
        let inner = lower_grouped(input, &inner_keys, inner_slots, &[], inner_wide, None)?;
        lower_grouped(
            inner,
            &outer_keys,
            outer_slots,
            &[],
            outer_wide,
            self.output_limit,
        )
    }

    /// Walk the aggregate expressions once, building the inner (non-distinct
    /// partials over `(group…, x)`) and outer (re-folded partials + `COUNT(*)` for
    /// the distinct) slot lists. The outer slots stay in expression order so the
    /// output columns match DuckDB's aggregate layout. `key_shift` offsets each
    /// value column past any materialised group key.
    fn plan_two_level(&self, groups: &[(usize, Type)], key_shift: usize) -> Result<Plan, Error> {
        // The inner emits `[group…, x, partial0, partial1, …]`: `n_groups + 1` key
        // columns then the partials, so partial `k` is at column
        // `n_groups + 1 + k`. Coalesce inner partials that compute the *same*
        // value, so each is scattered/merged once: all COUNT/COUNT(*) slots are
        // identical (pivot's Count contributes +1 per row regardless of
        // column/null, == COUNT(*)), SUMs of the same column are identical, and a
        // MIN/MAX of the same column and direction is identical. Fewer inner slots
        // ⇒ a narrower hash-table entry, the dominant cost of the high-cardinality
        // inner.
        let key_count = groups.len() + 1;
        let mut distinct: Option<(usize, Type)> = None;
        let mut inner = InnerSlots::new(key_count);
        let mut outer_slots: Vec<AggregationSlot> = Vec::new();

        for e in &self.expressions {
            let Expression::AggregateFunc(func) = e else {
                return Err(Error::UnsupportedAggregateExpression(e.clone()));
            };
            match func {
                AggregateFunc::CountDistinct(a) => {
                    if distinct.is_some() {
                        return Err(Error::UnsupportedAggregateExpression(e.clone()));
                    }
                    distinct = Some((
                        a.column.column_idx + key_shift,
                        a.column.return_type.clone(),
                    ));
                    // distinct count = number of inner (distinct-pair) rows for the
                    // group; the outer's CountStar ignores its column.
                    outer_slots.push(AggregationSlot::new(AggregationKind::CountStar, 0));
                }
                AggregateFunc::CountStar(_) => {
                    let col = inner.intern(Partial::Count, AggregationKind::CountStar, 0);
                    outer_slots.push(AggregationSlot::new(AggregationKind::Sum, col));
                }
                // Same value as COUNT(*) in pivot's Count semantics → coalesce.
                AggregateFunc::Count(a) => {
                    let col = inner.intern(
                        Partial::Count,
                        AggregationKind::Count,
                        a.column.column_idx + key_shift,
                    );
                    outer_slots.push(AggregationSlot::new(AggregationKind::Sum, col));
                }
                AggregateFunc::Sum(a) => {
                    let column = a.column.column_idx + key_shift;
                    let col = inner.intern(Partial::Sum(column), AggregationKind::Sum, column);
                    outer_slots.push(AggregationSlot::new(AggregationKind::Sum, col));
                }
                // MIN/MAX fold the same way at both levels: the inner extreme over
                // each subgroup, the outer extreme over those. The string extreme
                // emits a `Utf8View` partial the outer re-extremises; the numeric
                // one an integer (or `Decimal128`, when the level is wide).
                AggregateFunc::Min(a) => {
                    outer_slots.push(inner.intern_extreme(a, key_shift, false, e)?);
                }
                AggregateFunc::Max(a) => {
                    outer_slots.push(inner.intern_extreme(a, key_shift, true, e)?);
                }
                _ => return Err(Error::UnsupportedAggregateExpression(e.clone())),
            }
        }

        // Guaranteed by the caller (exactly one CountDistinct), but stay total.
        let (distinct_col, distinct_type) =
            distinct.ok_or(Error::UnsupportedAggregateExpressionAmount(0))?;
        Ok(Plan {
            distinct_col,
            distinct_type,
            inner_slots: inner.slots,
            outer_slots,
        })
    }
}

/// The two-level lowering: the distinct argument's column/type plus the
/// inner/outer slot lists.
struct Plan {
    distinct_col: usize,
    distinct_type: Type,
    inner_slots: Vec<AggregationSlot>,
    outer_slots: Vec<AggregationSlot>,
}

/// Identifies an inner partial for coalescing: a count, a sum of a column, or a
/// MIN/MAX (`is_max`) of a column. Two expressions with the same key share one
/// inner slot.
#[derive(PartialEq)]
enum Partial {
    Count,
    Sum(usize),
    Extreme { is_max: bool, column: usize },
}

/// Accumulates the inner partial slots, deduplicating equal ones and tracking
/// which inner output column each landed in.
struct InnerSlots {
    /// Number of leading inner key columns (`group… , x`); partials follow.
    key_count: usize,
    slots: Vec<AggregationSlot>,
    interned: Vec<(Partial, usize)>,
}

impl InnerSlots {
    fn new(key_count: usize) -> Self {
        Self {
            key_count,
            slots: Vec::new(),
            interned: Vec::new(),
        }
    }

    /// Return the inner output column for `key`, appending a new slot (built from
    /// `kind`/`column`) the first time the key is seen. The inner emits
    /// `[group…, x, partial0, …]`, so a new partial lands at column
    /// `key_count + slots.len()`.
    fn intern(&mut self, key: Partial, kind: AggregationKind, column: usize) -> usize {
        if let Some((_, col)) = self.interned.iter().find(|(k, _)| *k == key) {
            return *col;
        }
        let col = self.key_count + self.slots.len();
        self.slots.push(AggregationSlot::new(kind, column));
        self.interned.push((key, col));
        col
    }

    /// Intern a MIN/MAX over `a`'s column and return the matching *outer* slot,
    /// which re-extremises the inner partial with the same kind. Both levels use
    /// `StrMin`/`StrMax` for a string column and `Min`/`Max` for an integer one.
    fn intern_extreme(
        &mut self,
        a: &NumericAggregate,
        key_shift: usize,
        is_max: bool,
        expr: &Expression,
    ) -> Result<AggregationSlot, Error> {
        let column = a.column.column_idx + key_shift;
        let (string, numeric) = if is_max {
            (AggregationKind::StrMax, AggregationKind::Max)
        } else {
            (AggregationKind::StrMin, AggregationKind::Min)
        };
        let kind = extreme_kind(&a.column.return_type, string, numeric)
            .ok_or_else(|| Error::UnsupportedAggregateExpression(expr.clone()))?;
        let inner_col = self.intern(Partial::Extreme { is_max, column }, kind, column);
        Ok(AggregationSlot::new(kind, inner_col))
    }
}
