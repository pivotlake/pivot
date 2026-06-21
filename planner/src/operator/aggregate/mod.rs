//! [`Aggregate`] — GROUP BY + aggregate functions.
//!
//! [`Aggregate::compile`] is a router: it inspects the groups and aggregate
//! expressions and dispatches to one of a handful of lowering strategies, each
//! in its own submodule:
//!
//! | query shape | strategy |
//! |---|---|
//! | `COUNT(DISTINCT x)` alone | [`count_distinct`] |
//! | grouped, one `COUNT(DISTINCT)` + other aggregates | [`mixed_distinct`] |
//! | no GROUP BY | [`global`] |
//! | any other GROUP BY (incl. a computed key) | [`grouped`] |
//!
//! The shared expression→slot lowering and accumulator-width rule live here,
//! since the global and grouped paths both use them.

mod count_distinct;
mod global;
mod grouped;
mod mixed_distinct;

use crate::compile::Error;
use crate::expression::{AggregateFunc, Expression};
use crate::types::Type;
use dispatch::{AggregationKind, AggregationSlot, RecordBatchOperatorSpec};
use duckdb_planner::operator as duckdb_operator;
use std::fmt;

/// GROUP BY + aggregate functions.
#[derive(Debug)]
pub struct Aggregate {
    pub groups: Vec<Expression>,
    pub expressions: Vec<Expression>,
    /// Set by the `group → TopN` detection pass to `Some((value_slot, limit))`
    /// when this grouped aggregate feeds an `ORDER BY <slot> DESC LIMIT limit`,
    /// so the group operator emits only each partition's top-`limit` rows.
    pub top_k: Option<(usize, usize)>,
}

impl TryFrom<duckdb_operator::Aggregate> for Aggregate {
    type Error = super::Error;
    fn try_from(a: duckdb_operator::Aggregate) -> Result<Self, Self::Error> {
        Ok(Aggregate {
            top_k: None,
            groups: a
                .groups
                .into_iter()
                .map(Expression::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            expressions: a
                .expressions
                .into_iter()
                .map(Expression::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl fmt::Display for Aggregate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let groups = render(&self.groups);
        let exprs = render(&self.expressions);
        write!(f, "Aggregate(groups: [{groups}], exprs: [{exprs}])")
    }
}

fn render(exprs: &[Expression]) -> String {
    exprs
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

impl Aggregate {
    pub fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // `COUNT(DISTINCT x)` as the sole aggregate lowers to a two-level GROUP
        // BY (both the global and single-key forms route here).
        if let [Expression::AggregateFunc(AggregateFunc::CountDistinct(a))] =
            self.expressions.as_slice()
        {
            return self.compile_count_distinct(input, a);
        }

        // A grouped aggregate mixing one `COUNT(DISTINCT)` with non-distinct
        // aggregates (e.g. ClickBench Q9) is also a two-level GROUP BY.
        let n_distinct = self
            .expressions
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    Expression::AggregateFunc(AggregateFunc::CountDistinct(_))
                )
            })
            .count();
        if !self.groups.is_empty() && n_distinct == 1 {
            return self.compile_grouped_mixed_distinct(input);
        }

        if self.groups.is_empty() {
            return self.compile_global(input);
        }

        // Every GROUP BY (plain column keys, two-int-key, the row fallback, or a
        // single computed key) lowers through the general grouped path.
        self.compile_grouped(input)
    }

    fn is_lone_count_star(&self) -> bool {
        matches!(
            self.expressions.as_slice(),
            [Expression::AggregateFunc(AggregateFunc::CountStar(_))]
        )
    }
}

/// Lower each aggregate expression to its [`AggregationSlot`]. Shared by the
/// global and grouped multi-aggregate paths, which build slots identically. A
/// `MIN`/`MAX` over a `Utf8` column folds the byte extreme (`StrMin`/`StrMax`);
/// over an integer column the numeric one. `COUNT(*)` ignores its column, so the
/// index is a placeholder.
fn aggregation_slots(exprs: &[Expression]) -> Result<Vec<AggregationSlot>, Error> {
    exprs
        .iter()
        .map(|e| {
            let Expression::AggregateFunc(func) = e else {
                return Err(Error::UnsupportedAggregateExpression(e.clone()));
            };
            Ok(match func {
                AggregateFunc::CountStar(_) => AggregationSlot::new(AggregationKind::CountStar, 0),
                AggregateFunc::Count(a) => {
                    AggregationSlot::new(AggregationKind::Count, a.column.column_idx)
                }
                AggregateFunc::Sum(a) => {
                    AggregationSlot::new(AggregationKind::Sum, a.column.column_idx)
                }
                AggregateFunc::Min(a) => extreme_kind(
                    &a.column.return_type,
                    AggregationKind::StrMin,
                    AggregationKind::Min,
                )
                .map(|kind| AggregationSlot::new(kind, a.column.column_idx))
                .ok_or_else(|| Error::UnsupportedAggregateExpression(e.clone()))?,
                AggregateFunc::Max(a) => extreme_kind(
                    &a.column.return_type,
                    AggregationKind::StrMax,
                    AggregationKind::Max,
                )
                .map(|kind| AggregationSlot::new(kind, a.column.column_idx))
                .ok_or_else(|| Error::UnsupportedAggregateExpression(e.clone()))?,
                _ => return Err(Error::UnsupportedAggregateExpression(e.clone())),
            })
        })
        .collect()
}

/// The MIN/MAX kind for a column of type `ty`: the byte-wise `string` extreme for
/// a `Utf8` column (folded through the value container's arena path), the
/// `numeric` extreme for the integer widths the executor can read
/// (`Int16`/`Int32`/`Int64`). `None` for any other type, so the caller reports a
/// clean `UnsupportedAggregateExpression` rather than a worker panic in the reader.
fn extreme_kind(
    ty: &Type,
    string: AggregationKind,
    numeric: AggregationKind,
) -> Option<AggregationKind> {
    match ty {
        Type::Utf8 => Some(string),
        Type::Int16 | Type::Int32 | Type::Int64 => Some(numeric),
        _ => None,
    }
}

/// The accumulator-width rule shared by the global and grouped paths: `i128`
/// only when a `SUM` reads a 64-bit column (whose total can overflow `i64`),
/// else `i64`.
fn sum_reads_wide_column(exprs: &[Expression]) -> bool {
    exprs.iter().any(|e| {
        matches!(
            e,
            Expression::AggregateFunc(AggregateFunc::Sum(a)) if a.column.return_type == Type::Int64
        )
    })
}
