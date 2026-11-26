//! Grouped `COUNT(DISTINCT …)` (one or several, alone or mixed with
//! non-distinct aggregates), lowered to a two-level GROUP BY by rewriting it into
//! two stacked *regular* grouped aggregates. Every aggregate is defined once (in `aggregation_slots`,
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
//! Several distinct arguments `x1, x2, …` first expand every input row into one
//! copy per argument ([`Aggregate::expand_per_distinct_argument`]): copy `i`
//! keeps `xi` and NULLs the other distinct arguments. The inner then groups by
//! `(group…, x1, x2, …)`, so each argument's values dedup within its own copies,
//! and the outer counts each argument's non-NULL keys. Only the first copy keeps
//! the non-distinct aggregates' arguments (the others NULL them, and `COUNT(*)`
//! counts a column only the first copy sets), so every partial still sees each
//! input row once.
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
use arrow_array::{ArrayRef, BooleanArray, RecordBatch, new_null_array};
use arrow_buffer::BooleanBuffer;
use arrow_schema::DataType;
use dispatch::RecordBatchOperatorSpec;
use std::sync::Arc;

impl Aggregate {
    /// Lower a grouped aggregate holding one or more `COUNT(DISTINCT)`s: with
    /// several distinct arguments, expand the rows into one copy per argument
    /// first (see the module docs), then lower through
    /// [`compile_two_level`](Self::compile_two_level).
    pub(super) fn compile_grouped_distinct(
        &self,
        input: RecordBatchOperatorSpec,
        nullability: &[bool],
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let distinct_columns = self.distinct_argument_columns();
        if distinct_columns.len() > 1 {
            let (expanded_input, expanded_nullability, expanded) =
                self.expand_per_distinct_argument(input, nullability, &distinct_columns)?;
            return expanded.compile_two_level(expanded_input, &expanded_nullability);
        }
        self.compile_two_level(input, nullability)
    }

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
    /// the outer re-folds (decomposability; see the module docs). Several distinct
    /// arguments, already expanded, make the inner keys `(g, x1, x2, …)`.
    fn compile_two_level(
        &self,
        input: RecordBatchOperatorSpec,
        nullability: &[bool],
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // Step 1: every group key is a plain column after `materialize_inputs`
        // (called once at the top of compilation), as is every aggregate
        // argument including the distinct one, so resolve the keys directly.
        let groups = self.resolved_keys();

        // Step 2: split each SELECT aggregate into an inner partial expression and an
        // outer re-fold expression (and pick out the distinct arguments `x…`).
        let TwoLevel {
            distinct_columns,
            inner_exprs,
            outer_exprs,
        } = self.plan_two_level(&groups, nullability)?;

        // Step 3: build the INNER level over keys `(g…, x…)`. The key encoding emits
        // the key columns in this order, so the inner output is
        // `[g…, x…, partial0, …]` and the outer's group columns are the leading
        // `0..n_groups`. With non-distinct aggregates the inner is a regular grouped
        // aggregate (computing the partials); with none it is a pure keys-only dedup.
        let mut inner_keys = groups.clone();
        inner_keys.extend(distinct_columns.iter().cloned());
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
            .chain(&distinct_columns)
            .map(|(col, _)| column_nullable(nullability, *col))
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
    /// `[g… , x… , p0 , p1 , …]` (one key column per group and per distinct
    /// argument, then one partial per non-distinct aggregate), so partial `k` sits
    /// at column `partial_base + k`, which the matching outer re-fold reads.
    ///
    /// ```text
    ///   SELECT aggregate     inner partial        outer re-fold (reads partial col)
    ///   ------------------   ------------------   ---------------------------------
    ///   COUNT(DISTINCT x)    (none; x is a key)   COUNT(*)/COUNT(x) count the inner rows
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
        // Guaranteed by the caller (at least one CountDistinct), but stay total.
        let distinct_columns = self.distinct_argument_columns();
        if distinct_columns.is_empty() {
            return Err(Error::UnsupportedAggregateExpressionAmount(0));
        }
        // The inner emits `[group…, x…, partial0, partial1, …]`: one key column per
        // group and per distinct argument, then the partials, so partial `k` is at
        // column `partial_base + k`.
        let partial_base = groups.len() + distinct_columns.len();
        let mut inner_exprs: Vec<Expression> = Vec::new();
        let mut outer_exprs: Vec<Expression> = Vec::new();

        for e in &self.expressions {
            let Expression::AggregateFunc(func) = e else {
                return Err(Error::UnsupportedAggregateExpression(e.clone()));
            };
            match func {
                AggregateFunc::CountDistinct(a) => {
                    let column = a.column().column_idx;
                    let key_column = groups.len()
                        + distinct_columns
                            .iter()
                            .position(|(distinct_column, _)| *distinct_column == column)
                            .expect("every distinct argument is listed");
                    // distinct count = number of inner `(group…, x)` rows for the
                    // group. Over a nullable `x` the inner emits the NULLs as one
                    // extra key row, which SQL does not count as a distinct value,
                    // so the outer counts only the non-NULL `x` keys; a NULL-free
                    // `x` counts every inner row with the cheaper `COUNT(*)`. An
                    // expanded argument is NULL in the other arguments' copies.
                    if column_nullable(nullability, column) {
                        outer_exprs.push(count_column_expr(
                            key_column,
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

        Ok(TwoLevel {
            distinct_columns,
            inner_exprs,
            outer_exprs,
        })
    }

    /// The column and type each `COUNT(DISTINCT)` reads, each column once, in
    /// SELECT order.
    fn distinct_argument_columns(&self) -> Vec<(usize, Type)> {
        let mut columns: Vec<(usize, Type)> = Vec::new();
        for e in &self.expressions {
            if let Expression::AggregateFunc(AggregateFunc::CountDistinct(a)) = e
                && !columns
                    .iter()
                    .any(|(column, _)| *column == a.column().column_idx)
            {
                columns.push((a.column().column_idx, a.column().return_type.clone()));
            }
        }
        columns
    }

    /// Repeat every input row once per distinct argument (see the module docs),
    /// appending after the input's columns one column per distinct argument (set
    /// in its own copy, NULL in the others), one per non-distinct aggregate
    /// argument (set in the first copy only) and, for a `COUNT(*)`, a marker set
    /// in the first copy only. Returns the expanded input, its nullability, and
    /// this aggregate rewritten to read the appended columns, with each `COUNT(*)`
    /// counting the marker.
    fn expand_per_distinct_argument(
        &self,
        input: RecordBatchOperatorSpec,
        nullability: &[bool],
        distinct_columns: &[(usize, Type)],
    ) -> Result<(RecordBatchOperatorSpec, Vec<bool>, Aggregate), Error> {
        let distinct_sources: Arc<[usize]> =
            distinct_columns.iter().map(|(column, _)| *column).collect();
        let mut value_sources: Vec<usize> = Vec::new();
        let mut has_count_star = false;
        for e in &self.expressions {
            match e {
                Expression::AggregateFunc(AggregateFunc::CountStar(_)) => has_count_star = true,
                Expression::AggregateFunc(AggregateFunc::CountDistinct(_)) => {}
                Expression::AggregateFunc(
                    AggregateFunc::Count(a)
                    | AggregateFunc::Sum(a)
                    | AggregateFunc::Min(a)
                    | AggregateFunc::Max(a),
                ) => {
                    if !value_sources.contains(&a.column().column_idx) {
                        value_sources.push(a.column().column_idx);
                    }
                }
                _ => return Err(Error::UnsupportedAggregateExpression(e.clone())),
            }
        }

        // The appended columns follow the input's: the distinct arguments, then
        // the values, then the marker.
        let first_distinct_column = nullability.len();
        let first_value_column = first_distinct_column + distinct_sources.len();
        let marker_column = first_value_column + value_sources.len();
        let expressions = self
            .expressions
            .iter()
            .map(|e| {
                let Expression::AggregateFunc(func) = e else {
                    unreachable!("every expression was checked to be an aggregate above")
                };
                let (first_column, sources) = match func {
                    AggregateFunc::CountStar(count) => {
                        return Expression::AggregateFunc(AggregateFunc::Count(NumericAggregate {
                            argument: Box::new(Expression::Ref(Ref {
                                column_idx: marker_column,
                                return_type: Type::Boolean,
                                name: None,
                            })),
                            return_type: count.return_type.clone(),
                        }));
                    }
                    AggregateFunc::CountDistinct(_) => {
                        (first_distinct_column, &distinct_sources[..])
                    }
                    _ => (first_value_column, &value_sources[..]),
                };
                let mut rewritten = func.clone();
                for argument in rewritten.arguments_mut() {
                    let Expression::Ref(column) = argument else {
                        unreachable!("aggregate arguments are materialised into columns")
                    };
                    column.column_idx = first_column
                        + sources
                            .iter()
                            .position(|source| *source == column.column_idx)
                            .expect("every aggregate argument is listed");
                }
                Expression::AggregateFunc(rewritten)
            })
            .collect();

        let appended_count =
            distinct_sources.len() + value_sources.len() + usize::from(has_count_star);
        let expanded_nullability = nullability
            .iter()
            .copied()
            .chain(std::iter::repeat_n(true, appended_count))
            .collect();
        let value_sources: Arc<[usize]> = value_sources.into();
        let expanded_input = input.unnest(distinct_sources.len(), move || {
            let distinct_sources = distinct_sources.clone();
            let value_sources = value_sources.clone();
            move |batch: &RecordBatch, copy: usize| {
                let rows = batch.num_rows();
                let set_or_null = |column: usize, set: bool| -> ArrayRef {
                    if set {
                        batch.column(column).clone()
                    } else {
                        new_null_array(batch.column(column).data_type(), rows)
                    }
                };
                let mut columns: Vec<ArrayRef> = distinct_sources
                    .iter()
                    .enumerate()
                    .map(|(argument, &column)| set_or_null(column, argument == copy))
                    .collect();
                columns.extend(
                    value_sources
                        .iter()
                        .map(|&column| set_or_null(column, copy == 0)),
                );
                if has_count_star {
                    columns.push(if copy == 0 {
                        Arc::new(BooleanArray::new(BooleanBuffer::new_set(rows), None))
                    } else {
                        new_null_array(&DataType::Boolean, rows)
                    });
                }
                Ok(columns)
            }
        });
        let expanded = Aggregate {
            groups: self.groups.clone(),
            expressions,
            output_limit: self.output_limit,
        };
        Ok((expanded_input, expanded_nullability, expanded))
    }
}

/// The two-level rewrite: the distinct arguments' columns/types plus the inner
/// and outer aggregate expression lists.
struct TwoLevel {
    distinct_columns: Vec<(usize, Type)>,
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
