//! Aggregate function calls — [`AggregateFunc`] and its payloads
//! [`CountStar`] / [`NumericAggregate`]. Unlike the scalar expressions,
//! aggregates have no per-row `compile`; they are lowered by the Aggregate
//! operator (see [`crate::compile`]).

use super::{Error, Expression, Ref};
use duckdb_planner::expression as duckdb_expression;
use std::fmt::{self, Display};

#[derive(Debug, Clone)]
pub struct CountStar {
    pub params: Vec<Expression>,
}

impl TryFrom<duckdb_expression::AggregateFunc> for CountStar {
    type Error = Error;
    fn try_from(a: duckdb_expression::AggregateFunc) -> Result<Self, Self::Error> {
        if a.aggregate_function != "count_star" {
            return Err(Error::UnsupportedAggregateFunction(a.aggregate_function));
        }
        Ok(CountStar {
            params: a
                .params
                .into_iter()
                .map(Expression::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

/// A single-column numeric aggregate (`SUM(col)` / `AVG(col)`). Carries the
/// bound column reference being aggregated.
#[derive(Debug, Clone)]
pub struct NumericAggregate {
    pub column: Ref,
}

impl TryFrom<duckdb_expression::AggregateFunc> for NumericAggregate {
    type Error = Error;
    fn try_from(a: duckdb_expression::AggregateFunc) -> Result<Self, Self::Error> {
        if a.params.len() != 1 {
            return Err(Error::InvalidParameterCount {
                function: a.aggregate_function,
                expected: 1,
                actual: a.params.len(),
            });
        }
        let column = match Expression::try_from(a.params.into_iter().next().unwrap())? {
            Expression::Ref(r) => r,
            other => {
                return Err(Error::UnsupportedAggregateFunction(format!(
                    "non-column argument: {other}"
                )));
            }
        };
        Ok(NumericAggregate { column })
    }
}

/// An aggregate function call (e.g. `SUM`, `COUNT`).
#[derive(Debug, Clone)]
pub enum AggregateFunc {
    CountStar(CountStar),
    Sum(NumericAggregate),
    Avg(NumericAggregate),
    /// `MIN(col)` / `MAX(col)` over an integer column — keep the running extreme.
    Min(NumericAggregate),
    Max(NumericAggregate),
    /// `COUNT(col)` — counts non-null values. DuckDB lowers `AVG(col)` to
    /// `sum(col) / count(col)`, so this shows up in average plans.
    Count(NumericAggregate),
    /// `COUNT(DISTINCT col)` — counts the distinct non-null values of `col`.
    /// Lowered in compilation to a two-level GROUP BY (dedup on the group keys
    /// plus `col`, then count rows per group); see [`crate::compile`].
    CountDistinct(NumericAggregate),
}

impl TryFrom<duckdb_expression::AggregateFunc> for AggregateFunc {
    type Error = Error;
    fn try_from(a: duckdb_expression::AggregateFunc) -> Result<Self, Self::Error> {
        // DISTINCT is only supported for `COUNT` so far; reject `SUM(DISTINCT)`
        // etc. rather than silently computing the non-distinct aggregate.
        if a.distinct && a.aggregate_function != "count" {
            return Err(Error::UnsupportedAggregateFunction(format!(
                "DISTINCT {}",
                a.aggregate_function
            )));
        }
        match a.aggregate_function.as_str() {
            "count_star" => Ok(AggregateFunc::CountStar(a.try_into()?)),
            "sum" => Ok(AggregateFunc::Sum(a.try_into()?)),
            "avg" => Ok(AggregateFunc::Avg(a.try_into()?)),
            "min" => Ok(AggregateFunc::Min(a.try_into()?)),
            "max" => Ok(AggregateFunc::Max(a.try_into()?)),
            "count" if a.distinct => Ok(AggregateFunc::CountDistinct(a.try_into()?)),
            "count" => Ok(AggregateFunc::Count(a.try_into()?)),
            _ => Err(Error::UnsupportedAggregateFunction(a.aggregate_function)),
        }
    }
}

impl Display for AggregateFunc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AggregateFunc::CountStar(_) => f.write_str("count_star()"),
            AggregateFunc::Sum(a) => write!(f, "sum(#{})", a.column.column_idx),
            AggregateFunc::Avg(a) => write!(f, "avg(#{})", a.column.column_idx),
            AggregateFunc::Min(a) => write!(f, "min(#{})", a.column.column_idx),
            AggregateFunc::Max(a) => write!(f, "max(#{})", a.column.column_idx),
            AggregateFunc::Count(a) => write!(f, "count(#{})", a.column.column_idx),
            AggregateFunc::CountDistinct(a) => {
                write!(f, "count(distinct #{})", a.column.column_idx)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn group_by_count(mut testing_planner: TestingPlanner) {
        // Grouped output is positional: the key is `key`, the aggregate `v0`.
        let rows = run(
            &mut testing_planner,
            "SELECT name, COUNT(*) FROM example_table GROUP BY name",
        );

        let alice = rows.iter().find(|r| r["key"] == "alice").unwrap();
        assert_eq!(alice["v0"], 2); // alice appears twice
    }
}
