//! Grouped `COUNT(DISTINCT x)` (alone, or mixed with non-distinct aggregates),
//! lowered to a two-level GROUP BY by rewriting it into two stacked *regular*
//! grouped aggregates. Every aggregate is defined once (in `aggregation_slots`,
//! reached through [`build_group_by_operator`]); this module only decides which synthetic
//! expression each level gets and how the outer references the inner's partials.
//!
//! Decomposability makes the rewrite correct: every non-distinct aggregate folds
//! from per-subgroup partials (SUM/COUNT are sums of partial sums/counts; MIN/MAX
//! are the extreme of partial extremes), and `COUNT(DISTINCT x)` is the number of
//! distinct `(group…, x)` subgroups. So:
//! * **Inner** GROUP BY `(group…, x)` is a regular aggregate computing each
//!   non-distinct aggregate's partial, emitting `[group…, x, p0, p1, …]`. With no
//!   non-distinct aggregates it degenerates to a keys-only dedup
//!   ([`build_dedup_operator`]).
//! * **Outer** GROUP BY `group…` is another regular aggregate that re-folds each
//!   partial (re-summing a SUM/COUNT, re-extremising a MIN/MAX) and uses
//!   `COUNT(*)` of the inner rows for the distinct count.
//!
//! The outer expressions stay in the original expression order, so the output
//! column layout matches DuckDB's aggregate output and the downstream projection
//! (e.g. the AVG divide) lines up. AVG is already split by DuckDB into
//! `sum`+`count` exprs, handled generically here.
//!
//! Both levels lower through [`build_group_by_operator`], so they support every group-key and
//! value shape the general grouped path does.
//! A string `MIN`/`MAX` makes its level *wide* (the `i128` cell holding the
//! `ArenaKey`); its `COUNT`/`SUM` partials then emit as `Decimal128` and the
//! outer re-reads them through that width (see `NumReader` in `dispatch`).

use super::Aggregate;
use super::grouped::{
    build_dedup_operator, build_group_by_operator, column_nullable, derive_outer_keys,
};
use crate::compile::Error;
use crate::expression::{AggregateFunc, CountStar, Expression, NumericAggregate, Ref};
use crate::types::Type;
use dispatch::RecordBatchOperatorSpec;

impl Aggregate {
    /// Lower `g…, <non-distinct aggs>, COUNT(DISTINCT x) GROUP BY g…` into two
    /// stacked regular GROUP BYs. Worked example:
    ///
    /// ```text
    ///   SELECT g, SUM(a), COUNT(DISTINCT x) GROUP BY g
    ///
    ///     input rows (g, a, x)
    ///         |
    ///         v   INNER:  GROUP BY (g, x)            collapses duplicate (g, x) pairs;
    ///         |     partial SUM(a) per (g, x)        emits one row per distinct (g, x):
    ///         |                                         [ g , x , SUM_a ]
    ///         |                                          c0  c1   c2
    ///         v   OUTER:  GROUP BY g  (inner column 0)
    ///         |     COUNT(*)  ->  COUNT(DISTINCT x)  (# inner rows in the g group)
    ///         |     SUM(c2)   ->  SUM(a)             (re-sum the partial sums)
    ///         v
    ///       result  [ g , SUM(a) , COUNT(DISTINCT x) ]   in SELECT order
    /// ```
    ///
    /// So `COUNT(DISTINCT x)` becomes "dedup the `(g, x)` pairs, then count the
    /// survivors per `g`", and every non-distinct aggregate rides along as a partial
    /// the outer re-folds (decomposability; see the module docs).
    pub(super) fn compile_grouped_distinct(
        &self,
        input: RecordBatchOperatorSpec,
        nullability: &[bool],
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // Step 1: every group key is a plain column after `materialize_inputs`
        // (called once at the top of compilation), as is every aggregate
        // argument including the distinct one, so resolve the keys directly.
        let groups = self.resolved_keys();

        // Step 2: split each SELECT aggregate into an inner partial expression and an
        // outer re-fold expression (and pick out the single distinct argument `x`).
        let TwoLevel {
            distinct_col,
            distinct_type,
            inner_exprs,
            outer_exprs,
        } = self.plan_two_level(&groups, nullability)?;

        // Step 3: build the INNER level over keys `(g…, x)`. The key encoding emits
        // the key columns in this order, so the inner output is
        // `[g…, x, partial0, …]` and the outer's group columns are the leading
        // `0..n_groups`. With non-distinct aggregates the inner is a regular grouped
        // aggregate (computing the partials); with none it is a pure keys-only dedup.
        let mut inner_keys = groups.clone();
        inner_keys.push((distinct_col, distinct_type));
        let inner = if inner_exprs.is_empty() {
            build_dedup_operator(input, &inner_keys, nullability)?
        } else {
            build_group_by_operator(input, &inner_keys, &inner_exprs, None, nullability)?
        };

        // Step 4: build the OUTER level grouping by the inner's leading group columns
        // (`derive_outer_keys` remaps `g…` onto columns `0..n_groups`). It counts the
        // inner rows per group (the distinct count) and re-folds each partial. Only
        // this final level carries any pushed-down LIMIT. The inner output's
        // nullability mirrors its columns: the keys keep their input nullability,
        // and a partial is NULL only when its subgroup saw no non-NULL value,
        // which requires the aggregated input column itself to be nullable (a
        // count partial is a number, never NULL). Blanket-marking the partials
        // nullable would put the whole outer level on the mask-tracking path
        // for every NULL-free COUNT(DISTINCT) query.
        let outer_nullability: Vec<bool> = groups
            .iter()
            .map(|(col, _)| col)
            .chain(std::iter::once(&distinct_col))
            .map(|&col| column_nullable(nullability, col))
            .chain(inner_exprs.iter().map(|e| match e {
                Expression::AggregateFunc(
                    AggregateFunc::Sum(a) | AggregateFunc::Min(a) | AggregateFunc::Max(a),
                ) => column_nullable(nullability, a.column().column_idx),
                _ => false,
            }))
            .collect();
        build_group_by_operator(
            inner,
            &derive_outer_keys(&groups),
            &outer_exprs,
            self.output_limit,
            &outer_nullability,
        )
    }

    /// Walk the SELECT aggregates once, splitting each into an inner partial and an
    /// outer re-fold (both ordinary aggregate expression lists, so each level lowers
    /// through the regular [`build_group_by_operator`]). The inner emits
    /// `[g… , x , p0 , p1 , …]` (`n_groups + 1` key columns, then one partial per
    /// non-distinct aggregate), so partial `k` sits at column `partial_base + k`,
    /// which the matching outer re-fold reads.
    ///
    /// ```text
    ///   SELECT aggregate     inner partial        outer re-fold (reads partial col)
    ///   ------------------   ------------------   ---------------------------------
    ///   COUNT(DISTINCT x)    (none; x is a key)   COUNT(*)         count the inner rows
    ///   COUNT(*) / COUNT(c)  COUNT(*) / COUNT(c)  SUM(partial)     re-sum the counts
    ///   SUM(c)               SUM(c)               SUM(partial)     re-sum the sums
    ///   MIN(c) / MAX(c)      MIN(c) / MAX(c)      MIN/MAX(partial) re-take the extreme
    /// ```
    ///
    /// The outer list stays in SELECT order so the output columns match DuckDB's
    /// aggregate layout. Every aggregate argument is already a plain column (the
    /// distinct one included), materialised before lowering.
    fn plan_two_level(
        &self,
        groups: &[(usize, Type)],
        nullability: &[bool],
    ) -> Result<TwoLevel, Error> {
        // The inner emits `[group…, x, partial0, partial1, …]`: `n_groups + 1` key
        // columns then the partials, so partial `k` is at column `n_groups + 1 + k`.
        let partial_base = groups.len() + 1;
        let mut distinct: Option<(usize, Type)> = None;
        let mut inner_exprs: Vec<Expression> = Vec::new();
        let mut outer_exprs: Vec<Expression> = Vec::new();

        for e in &self.expressions {
            let Expression::AggregateFunc(func) = e else {
                return Err(Error::UnsupportedAggregateExpression(e.clone()));
            };
            match func {
                AggregateFunc::CountDistinct(a) => {
                    if distinct.is_some() {
                        return Err(Error::UnsupportedAggregateExpression(e.clone()));
                    }
                    let column = a.column().column_idx;
                    distinct = Some((column, a.column().return_type.clone()));
                    // distinct count = number of inner `(group…, x)` rows for the
                    // group. Over a nullable `x` the inner emits the NULLs as one
                    // extra key row, which SQL does not count as a distinct value,
                    // so the outer counts only the non-NULL `x` keys (inner key
                    // column `groups.len()`); a NULL-free `x` counts every inner
                    // row with the cheaper `COUNT(*)`.
                    if column_nullable(nullability, column) {
                        outer_exprs.push(count_column_expr(
                            groups.len(),
                            a.column().return_type.clone(),
                        ));
                    } else {
                        outer_exprs.push(count_star_expr());
                    }
                }
                // The non-distinct aggregates: the inner computes the original
                // aggregate over each `(group…, x)` subgroup, the outer re-folds
                // that partial. `build_group_by_operator` coalesces duplicate inner
                // partials and re-expands its output, so each expression still has
                // its own partial column `partial_base + k`.
                AggregateFunc::CountStar(_)
                | AggregateFunc::Count(_)
                | AggregateFunc::Sum(_)
                | AggregateFunc::Min(_)
                | AggregateFunc::Max(_) => {
                    let read = Ref {
                        column_idx: partial_base + inner_exprs.len(),
                        return_type: refold_partial_type(func),
                        // A synthesised reference into the inner output; no source name.
                        name: None,
                    };
                    outer_exprs.push(refold_expr(func, read));
                    inner_exprs.push(e.clone());
                }
                _ => return Err(Error::UnsupportedAggregateExpression(e.clone())),
            }
        }

        // Guaranteed by the caller (exactly one CountDistinct), but stay total.
        let (distinct_col, distinct_type) =
            distinct.ok_or(Error::UnsupportedAggregateExpressionAmount(0))?;
        Ok(TwoLevel {
            distinct_col,
            distinct_type,
            inner_exprs,
            outer_exprs,
        })
    }
}

/// The two-level rewrite: the distinct argument's column/type plus the inner and
/// outer aggregate expression lists.
struct TwoLevel {
    distinct_col: usize,
    distinct_type: Type,
    inner_exprs: Vec<Expression>,
    outer_exprs: Vec<Expression>,
}

/// The element type the outer re-fold reads from `func`'s inner partial column.
/// SUM keeps the summed column's type (so an `Int64` sum widens the outer
/// accumulator, exactly as the single-level path does); MIN/MAX keep theirs (so
/// the outer picks the same string-vs-numeric extreme); COUNT/COUNT(*) partials
/// are `i64` counts.
fn refold_partial_type(func: &AggregateFunc) -> Type {
    match func {
        // A SUM partial is emitted at the slot's declared HUGEINT type
        // (`Decimal128`) whatever the input width, so the re-fold must read a
        // 128-bit column: declaring the input width here would leave the
        // outer level narrow and panic on the wide re-read.
        AggregateFunc::Sum(_) => Type::Int128,
        AggregateFunc::Min(a) | AggregateFunc::Max(a) => a.column().return_type.clone(),
        _ => Type::Int64,
    }
}

/// The outer aggregate that re-folds `func`'s inner partial read through
/// `partial`: SUM/COUNT/COUNT(*) re-sum the per-subgroup partials, MIN/MAX
/// re-extremise them with the same direction. `func` is one of those five kinds
/// (the caller's match guarantees it).
fn refold_expr(func: &AggregateFunc, partial: Ref) -> Expression {
    // The re-fold's result column keeps the original aggregate's declared type
    // (a re-summed COUNT is still a BIGINT, a re-summed SUM still a HUGEINT, a
    // re-extremised MIN/MAX still the input type).
    let agg = NumericAggregate {
        argument: Box::new(Expression::Ref(partial)),
        return_type: func.return_type().clone(),
    };
    Expression::AggregateFunc(match func {
        AggregateFunc::Min(_) => AggregateFunc::Min(agg),
        AggregateFunc::Max(_) => AggregateFunc::Max(agg),
        _ => AggregateFunc::Sum(agg),
    })
}

/// A synthetic `COUNT(*)` expression (a `BIGINT` row count).
fn count_star_expr() -> Expression {
    Expression::AggregateFunc(AggregateFunc::CountStar(CountStar {
        params: Vec::new(),
        return_type: Type::Int64,
    }))
}

/// `COUNT(<column>)`: counts only the rows where `column` is non-NULL.
fn count_column_expr(column: usize, column_type: Type) -> Expression {
    Expression::AggregateFunc(AggregateFunc::Count(NumericAggregate {
        argument: Box::new(Expression::Ref(Ref {
            column_idx: column,
            return_type: column_type,
            name: None,
        })),
        return_type: Type::Int64,
    }))
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn grouped_count_distinct_excludes_nulls(mut testing_planner: TestingPlanner) {
        let mut rows = run(
            &mut testing_planner,
            "SELECT CASE WHEN a = 2 OR a = 4 THEN 0 ELSE 1 END AS g, COUNT(DISTINCT s) AS d \
             FROM nullable_table GROUP BY g",
        );

        rows.sort_by_key(|r| r["g"].as_i64().unwrap());
        // g = 0 (a = 2, 4): s is NULL, NULL. g = 1 (a = 1, 3, 5, 6): x, y, x, z.
        assert_eq!(rows[0]["d"].as_i64(), Some(0));
        assert_eq!(rows[1]["d"].as_i64(), Some(3));
    }

    #[rstest]
    fn grouped_count_distinct_of_all_null_group_is_zero(mut testing_planner: TestingPlanner) {
        let mut rows = run(
            &mut testing_planner,
            "SELECT a, COUNT(DISTINCT b) AS d, SUM(b) AS sb FROM nullable_table GROUP BY a",
        );

        rows.sort_by_key(|r| r["a"].as_i64().unwrap());
        assert_eq!(rows[0]["d"].as_i64(), Some(1));
        assert_eq!(rows[1]["d"].as_i64(), Some(0));
        assert!(rows[1]["sb"].is_null());
    }
}
