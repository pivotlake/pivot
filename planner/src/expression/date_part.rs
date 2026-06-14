//! [`DatePart`] — SQL `extract(<part> FROM ts)` and its [`DatePartKind`].

use super::shared::{SECS_PER_DAY, SECS_PER_HOUR, civil_from_days, days_from_civil, iso_week};
use super::{Error, Expression};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::DataType;
use duckdb_planner::expression as duckdb_expression;
use std::fmt::{self, Display};
use std::sync::Arc;

/// Which field of a timestamp a [`DatePart`] extracts. DuckDB lowers
/// `extract(<part> FROM ts)` to a scalar function named after the part (e.g.
/// `minute`, `year`); this enumerates the parts we evaluate from `EventTime`'s
/// Int64 epoch-seconds representation. See [`DatePart`]'s compile impl for the
/// per-part arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatePartKind {
    /// Whole seconds since the Unix epoch (the stored value, unchanged).
    Epoch,
    /// Second of minute, 0–59.
    Second,
    /// Millisecond of minute, 0–59000 (whole seconds only, so `second * 1000`).
    Millisecond,
    /// Microsecond of minute, `second * 1_000_000`.
    Microsecond,
    /// Minute of hour, 0–59.
    Minute,
    /// Hour of day, 0–23.
    Hour,
    /// Day of month, 1–31.
    Day,
    /// Month of year, 1–12.
    Month,
    /// Quarter of year, 1–4.
    Quarter,
    /// Full year (e.g. 2024).
    Year,
    /// Decade — `year / 10` (e.g. 202 for 2024).
    Decade,
    /// Century — e.g. 21 for years 2001–2100.
    Century,
    /// Millennium — e.g. 3 for years 2001–3000.
    Millennium,
    /// Day of week, 0 (Sunday)–6 (Saturday).
    DayOfWeek,
    /// ISO day of week, 1 (Monday)–7 (Sunday).
    IsoDayOfWeek,
    /// Day of year, 1–366.
    DayOfYear,
    /// ISO 8601 week of year, 1–53.
    Week,
}

impl DatePartKind {
    /// Resolve a DuckDB scalar-function name (as `extract` lowers it) to a part.
    /// Includes the common DuckDB aliases (`dow`, `doy`, `weekofyear`).
    pub fn from_function_name(name: &str) -> Option<Self> {
        Some(match name {
            "epoch" => DatePartKind::Epoch,
            "second" => DatePartKind::Second,
            "millisecond" => DatePartKind::Millisecond,
            "microsecond" => DatePartKind::Microsecond,
            "minute" => DatePartKind::Minute,
            "hour" => DatePartKind::Hour,
            "day" => DatePartKind::Day,
            "month" => DatePartKind::Month,
            "quarter" => DatePartKind::Quarter,
            "year" => DatePartKind::Year,
            "decade" => DatePartKind::Decade,
            "century" => DatePartKind::Century,
            "millennium" => DatePartKind::Millennium,
            "dayofweek" | "dow" => DatePartKind::DayOfWeek,
            "isodow" => DatePartKind::IsoDayOfWeek,
            "dayofyear" | "doy" => DatePartKind::DayOfYear,
            "week" | "weekofyear" => DatePartKind::Week,
            _ => return None,
        })
    }

    /// The canonical DuckDB function name for this part, used for plan display.
    pub fn name(&self) -> &'static str {
        match self {
            DatePartKind::Epoch => "epoch",
            DatePartKind::Second => "second",
            DatePartKind::Millisecond => "millisecond",
            DatePartKind::Microsecond => "microsecond",
            DatePartKind::Minute => "minute",
            DatePartKind::Hour => "hour",
            DatePartKind::Day => "day",
            DatePartKind::Month => "month",
            DatePartKind::Quarter => "quarter",
            DatePartKind::Year => "year",
            DatePartKind::Decade => "decade",
            DatePartKind::Century => "century",
            DatePartKind::Millennium => "millennium",
            DatePartKind::DayOfWeek => "dayofweek",
            DatePartKind::IsoDayOfWeek => "isodow",
            DatePartKind::DayOfYear => "dayofyear",
            DatePartKind::Week => "week",
        }
    }
}

/// SQL `extract(<part> FROM source)` — a timestamp field accessor. DuckDB
/// lowers each part to a scalar function (`minute`, `year`, …); `EventTime` is
/// stored as Int64 epoch *seconds* (see [`Type::Timestamp`]), so every part is
/// a pure integer computation. See its compile impl.
///
/// [`Type::Timestamp`]: crate::types::Type::Timestamp
#[derive(Debug, Clone)]
pub struct DatePart {
    pub kind: DatePartKind,
    pub source: Box<Expression>,
}

impl DatePart {
    /// Build from a DuckDB function call once its name has been recognised as a
    /// date part. Validates the single-argument arity.
    pub(super) fn from_function(
        kind: DatePartKind,
        mut f: duckdb_expression::Function,
    ) -> Result<Self, Error> {
        if f.params.len() != 1 {
            let actual = f.params.len();
            return Err(Error::InvalidParameterCount {
                function: f.function,
                expected: 1,
                actual,
            });
        }
        let source = Box::new(Expression::try_from(f.params.remove(0))?);
        Ok(DatePart { kind, source })
    }
}

impl Display for DatePart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.kind.name(), self.source)
    }
}

impl DatePart {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        // EventTime is stored as Int64 epoch *seconds* (UTC), so every part is a
        // pure integer computation. Euclidean div/rem keep the time-of-day and
        // calendar fields well-defined for pre-epoch (negative) timestamps,
        // matching DuckDB's `extract(<part> FROM ...)`.
        let kind = self.kind;
        let source_builder = self.source.compile()?;
        Ok(Box::new(move || {
            let mut source_expr = source_builder();
            Box::new(move |batch: &RecordBatch| {
                let src = source_expr(batch);
                let (arr, _) = src.as_datum().get();
                let i64arr = arrow::compute::cast(arr, &DataType::Int64).unwrap();
                let vals = i64arr.as_primitive::<Int64Type>();
                // Dispatch on the part ONCE per batch, then run a single
                // monomorphic, branch-free row loop per arm — so e.g. `minute`
                // compiles to exactly its two-op loop with no per-row `kind`
                // test (no reliance on the optimizer hoisting a loop-invariant
                // branch). The civil-date parts share `civil_from_days`.
                use crate::expression::DatePartKind::*;
                macro_rules! map_part {
                    ($f:expr) => {{
                        let out: Int64Array = vals.iter().map(|v| v.map($f)).collect();
                        out
                    }};
                }
                let day = |t: i64| t.div_euclid(SECS_PER_DAY);
                let out = match kind {
                    Epoch => map_part!(|t: i64| t),
                    Second => map_part!(|t: i64| t.rem_euclid(60)),
                    Millisecond => map_part!(|t: i64| t.rem_euclid(60) * 1_000),
                    Microsecond => map_part!(|t: i64| t.rem_euclid(60) * 1_000_000),
                    Minute => map_part!(|t: i64| t.div_euclid(60).rem_euclid(60)),
                    Hour => map_part!(|t: i64| t.div_euclid(SECS_PER_HOUR).rem_euclid(24)),
                    // 0 = Sunday … 6 = Saturday. The epoch day (1970-01-01) was a
                    // Thursday (4), so `(days + 4) mod 7` rebases to Sunday = 0.
                    DayOfWeek => map_part!(|t: i64| (day(t) + 4).rem_euclid(7)),
                    // 1 = Monday … 7 = Sunday.
                    IsoDayOfWeek => map_part!(|t: i64| (day(t) + 3).rem_euclid(7) + 1),
                    Day => map_part!(|t: i64| civil_from_days(day(t)).2),
                    Month => map_part!(|t: i64| civil_from_days(day(t)).1),
                    Quarter => map_part!(|t: i64| (civil_from_days(day(t)).1 - 1) / 3 + 1),
                    Year => map_part!(|t: i64| civil_from_days(day(t)).0),
                    Decade => map_part!(|t: i64| civil_from_days(day(t)).0.div_euclid(10)),
                    Century => {
                        map_part!(|t: i64| (civil_from_days(day(t)).0 - 1).div_euclid(100) + 1)
                    }
                    Millennium => {
                        map_part!(|t: i64| (civil_from_days(day(t)).0 - 1).div_euclid(1000) + 1)
                    }
                    DayOfYear => map_part!(|t: i64| {
                        let d = day(t);
                        d - days_from_civil(civil_from_days(d).0, 1, 1) + 1
                    }),
                    Week => map_part!(|t: i64| iso_week(day(t))),
                };
                ExprResult::Array(Arc::new(out) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use arrow_array::{ArrayRef, Int64Array};
    use crate::types::Type;
    use rstest::rstest;
    use std::sync::Arc;

    #[rstest]
    fn extracts_year(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "ts",
            &[(
                "EventTime",
                Type::Timestamp,
                // 1704067200 = 2024-01-01 00:00:00 UTC.
                Arc::new(Int64Array::from(vec![1_704_067_200i64])) as ArrayRef,
            )],
        );

        let rows = run(&mut testing_planner, "SELECT extract(year FROM EventTime) FROM ts");

        assert_eq!(rows[0]["col0"], 2024);
    }
}
