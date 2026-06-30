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

/// A single-argument numeric aggregate (`SUM(col)` / `SUM(a * b)`). Carries the
/// argument expression being aggregated and DuckDB's declared result type for the
/// call (e.g. `HUGEINT` for an integer `SUM`, `BIGINT` for `COUNT`, the input type
/// for `MIN`/`MAX`). The argument may be any expression (e.g. `a * b`); a
/// projection that materialises every computed argument into a column is inserted
/// before the aggregate
/// ([`Aggregate::materialize_inputs`](crate::operator::Aggregate)), so by
/// compilation [`column`](Self::column) is always a plain column reference.
#[derive(Debug, Clone)]
pub struct NumericAggregate {
    /// Boxed to break the `Expression` → `AggregateFunc` → `NumericAggregate`
    /// type cycle.
    pub argument: Box<Expression>,
    pub return_type: Type,
}

impl NumericAggregate {
    /// The input column this aggregate reads. Valid only after the Aggregate
    /// operator has materialised every computed argument into a column, which it
    /// always does before lowering, so every slot-building site reads a plain
    /// reference here.
    pub fn column(&self) -> &Ref {
        match self.argument.as_ref() {
            Expression::Ref(r) => r,
            other => unreachable!("aggregate argument not materialised to a column: {other}"),
        }
    }
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
        let argument = Expression::try_from(a.params.into_iter().next().unwrap())?;
        Ok(NumericAggregate {
            argument: Box::new(argument),
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

    /// The expression(s) this aggregate is computed over (`SUM(x)`'s `x`). Empty
    /// for `COUNT(*)`, which reads no column. An iterator so it stays correct once
    /// a function takes more than one argument.
    pub fn arguments(&self) -> impl Iterator<Item = &Expression> {
        match self {
            AggregateFunc::CountStar(_) => None,
            AggregateFunc::Sum(a)
            | AggregateFunc::Avg(a)
            | AggregateFunc::Min(a)
            | AggregateFunc::Max(a)
            | AggregateFunc::Count(a)
            | AggregateFunc::CountDistinct(a) => Some(a.argument.as_ref()),
        }
        .into_iter()
    }

    /// Mutable view of [`arguments`](Self::arguments), for rewriting each argument
    /// in place (e.g. pointing it at a materialised column).
    pub fn arguments_mut(&mut self) -> impl Iterator<Item = &mut Expression> {
        match self {
            AggregateFunc::CountStar(_) => None,
            AggregateFunc::Sum(a)
            | AggregateFunc::Avg(a)
            | AggregateFunc::Min(a)
            | AggregateFunc::Max(a)
            | AggregateFunc::Count(a)
            | AggregateFunc::CountDistinct(a) => Some(a.argument.as_mut()),
        }
        .into_iter()
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
            AggregateFunc::Sum(a) => write!(f, "sum({})", a.argument),
            AggregateFunc::Avg(a) => write!(f, "avg({})", a.argument),
            AggregateFunc::Min(a) => write!(f, "min({})", a.argument),
            AggregateFunc::Max(a) => write!(f, "max({})", a.argument),
            AggregateFunc::Count(a) => write!(f, "count({})", a.argument),
            AggregateFunc::CountDistinct(a) => write!(f, "count(distinct {})", a.argument),
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
