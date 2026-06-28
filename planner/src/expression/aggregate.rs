//! Aggregate function calls — [`AggregateFunc`] and its payloads
//! [`CountStar`] / [`NumericAggregate`]. Unlike the scalar expressions,
//! aggregates have no per-row `compile`; they are lowered by the Aggregate
//! operator (see [`crate::compile`]).

use super::{Error, Expression, Ref};
use crate::types::Type;
use duckdb_planner::expression as duckdb_expression;
use std::fmt::{self, Display};

#[derive(Debug, Clone)]
pub struct CountStar {
    pub params: Vec<Expression>,
    /// DuckDB's declared result type for the call (`BIGINT`).
    pub return_type: Type,
}

impl TryFrom<duckdb_expression::AggregateFunc> for CountStar {
    type Error = Error;
    fn try_from(a: duckdb_expression::AggregateFunc) -> Result<Self, Self::Error> {
        if a.aggregate_function != "count_star" {
            return Err(Error::UnsupportedAggregateFunction(a.aggregate_function));
        }
        let return_type = crate::types::type_from_logical(a.return_type)?;
        Ok(CountStar {
            params: a
                .params
                .into_iter()
                .map(Expression::try_from)
                .collect::<Result<Vec<_>, _>>()?,
            return_type,
        })
    }
}

/// A single-column numeric aggregate (`SUM(col)` / `AVG(col)`). Carries the
/// bound column reference being aggregated and DuckDB's declared result type for
/// the call (e.g. `HUGEINT` for an integer `SUM`, `BIGINT` for `COUNT`, the input
/// type for `MIN`/`MAX`).
#[derive(Debug, Clone)]
pub struct NumericAggregate {
    pub column: Ref,
    pub return_type: Type,
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
        let return_type = crate::types::type_from_logical(a.return_type)?;
        let column = match Expression::try_from(a.params.into_iter().next().unwrap())? {
            Expression::Ref(r) => r,
            other => {
                return Err(Error::UnsupportedAggregateFunction(format!(
                    "non-column argument: {other}"
                )));
            }
        };
        Ok(NumericAggregate {
            column,
            return_type,
        })
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

impl AggregateFunc {
    /// DuckDB's declared result type for the call.
    pub fn return_type(&self) -> &Type {
        match self {
            AggregateFunc::CountStar(c) => &c.return_type,
            AggregateFunc::Sum(a)
            | AggregateFunc::Avg(a)
            | AggregateFunc::Min(a)
            | AggregateFunc::Max(a)
            | AggregateFunc::Count(a)
            | AggregateFunc::CountDistinct(a) => &a.return_type,
        }
    }
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
            AggregateFunc::Sum(a) => write!(f, "sum({})", a.column.name_or_index()),
            AggregateFunc::Avg(a) => write!(f, "avg({})", a.column.name_or_index()),
            AggregateFunc::Min(a) => write!(f, "min({})", a.column.name_or_index()),
            AggregateFunc::Max(a) => write!(f, "max({})", a.column.name_or_index()),
            AggregateFunc::Count(a) => write!(f, "count({})", a.column.name_or_index()),
            AggregateFunc::CountDistinct(a) => {
                write!(f, "count(distinct {})", a.column.name_or_index())
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
        // Output columns keep DuckDB's resolved names (the group key `name` and
        // the aggregate `count_star()`), not internal positional placeholders.
        let rows = run(
            &mut testing_planner,
            "SELECT name, COUNT(*) FROM example_table GROUP BY name",
        );

        let alice = rows.iter().find(|r| r["name"] == "alice").unwrap();
        assert_eq!(alice["count_star()"], 2); // alice appears twice
    }
}
