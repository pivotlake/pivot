//! [`DateTrunc`] — SQL `date_trunc(unit, source)`.

use super::{Expression, Function};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::Type;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, RecordBatch, TimestampMicrosecondArray};
use arrow_schema::DataType;
use std::fmt::{self, Display};
use std::sync::Arc;

/// Microseconds in a second, the unit a timestamp counts in.
const MICROS_PER_SEC: i64 = 1_000_000;

/// SQL `date_trunc(unit, source)` — truncate a timestamp down to `unit`
/// (e.g. `date_trunc('minute', ts)`). DuckDB passes the unit as a string
/// constant in the first argument and the timestamp expression second.
#[derive(Debug, Clone)]
pub struct DateTrunc {
    pub unit: String,
    pub source: Box<Expression>,
    /// The bound result type ([`Type::Timestamp`] or [`Type::TimestampTz`],
    /// tracking the source), so truncation never changes zone-ness.
    pub result: Type,
}

impl Display for DateTrunc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "date_trunc('{}', {})", self.unit, self.source)
    }
}

impl DateTrunc {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        // A timestamp is stored as Int64 epoch *microseconds*, so truncating to
        // a unit is flooring to that many microseconds. `M = (t / step) * step`.
        let step: i64 = MICROS_PER_SEC
            * match self.unit.as_str() {
                "second" => 1,
                "minute" => 60,
                "hour" => 3600,
                "day" => 86400,
                other => {
                    return Err(compile::Error::UnsupportedExpression(Expression::Function(
                        Function::DateTrunc(DateTrunc {
                            unit: other.to_string(),
                            source: self.source.clone(),
                            result: self.result.clone(),
                        }),
                    )));
                }
            };
        // The output timestamp keeps the source's zone-ness; flooring epoch
        // microseconds is zone-independent under the UTC session.
        let zone = match crate::types::physical_arrow_type(&self.result) {
            DataType::Timestamp(_, zone) => zone,
            other => unreachable!("date_trunc result is a timestamp, not {other}"),
        };
        let source_builder = self.source.compile()?;
        Ok(Box::new(move || {
            let mut source_expr = source_builder();
            let zone = zone.clone();
            Box::new(move |batch: &RecordBatch| {
                let src = source_expr(batch);
                let (arr, _) = src.as_datum().get();
                let i64arr = arrow::compute::cast(arr, &DataType::Int64).unwrap();
                let vals = i64arr.as_primitive::<Int64Type>();
                // Floor to `step` and emit a real timestamp (epoch
                // microseconds), the type date_trunc returns, rather than a
                // bare int.
                let truncated: TimestampMicrosecondArray = vals
                    .iter()
                    .map(|v| v.map(|x| x.div_euclid(step) * step))
                    .collect();
                ExprResult::Array(Arc::new(truncated.with_timezone_opt(zone.clone())) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::cast::AsArray;
    use arrow_array::types::TimestampMicrosecondType;
    use arrow_array::{ArrayRef, Int64Array};
    use arrow_schema::{DataType, TimeUnit};
    use rstest::rstest;
    use std::sync::Arc;

    #[rstest]
    fn truncates_to_minute(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "events",
            &[(
                "EventTime",
                Type::Timestamp,
                Arc::new(Int64Array::from(vec![
                    0i64,
                    90_000_000,
                    150_500_000,
                    3_690_000_000,
                ])) as ArrayRef,
            )],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT date_trunc('minute', EventTime) FROM events",
        );

        // date_trunc yields a real TIMESTAMP (epoch microseconds floored to the
        // minute), so the sub-second part of a source value goes with it.
        let col = batches[0].column(0);
        assert_eq!(
            col.data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        let mut micros: Vec<i64> = col
            .as_primitive::<TimestampMicrosecondType>()
            .values()
            .to_vec();
        micros.sort();
        assert_eq!(micros, vec![0, 60_000_000, 120_000_000, 3_660_000_000]);
    }

    #[rstest]
    fn truncating_a_timestamptz_keeps_its_zone(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "events",
            &[(
                "EventTime",
                Type::TimestampTz,
                Arc::new(Int64Array::from(vec![90_500_000i64, 3_690_000_000])) as ArrayRef,
            )],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT date_trunc('hour', EventTime) FROM events",
        );

        // Truncation floors the UTC instant and keeps the zone marker, so the
        // result is still a TIMESTAMP WITH TIME ZONE.
        let col = batches[0].column(0);
        assert_eq!(
            col.data_type(),
            &DataType::Timestamp(
                TimeUnit::Microsecond,
                Some(crate::types::UTC_TIMEZONE.into()),
            )
        );
        let mut micros: Vec<i64> = col
            .as_primitive::<TimestampMicrosecondType>()
            .values()
            .to_vec();
        micros.sort();
        assert_eq!(micros, vec![0, 3_600_000_000]);
    }
}
