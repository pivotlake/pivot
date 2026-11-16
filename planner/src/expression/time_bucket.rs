//! [`TimeBucket`] — SQL `time_bucket(bucket_width, source[, offset | origin])`.

use super::Expression;
use super::date_part::{civil_from_days, days_from_civil};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::{Type, physical_arrow_type};
use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Int64Type, TimestampMicrosecondType};
use arrow_array::{ArrayRef, Date32Array, RecordBatch, TimestampMicrosecondArray};
use arrow_schema::DataType;
use std::fmt::{self, Display};
use std::sync::Arc;

/// Microseconds in a day, the step between a `DATE`'s days and a timestamp's
/// microseconds.
pub const MICROS_PER_DAY: i64 = 86_400 * 1_000_000;

/// Count the whole months from January 1970 to the month that holds a day,
/// itself counted in days since the epoch.
pub fn count_epoch_months(days: i64) -> i64 {
    let (year, month, _) = civil_from_days(days);
    (year - 1970) * 12 + month - 1
}

/// The buckets a [`TimeBucket`] floors onto: every instant `origin + k * width`.
/// The origin is kept reduced into `[0, width)`, which names the same buckets
/// as any other origin a whole number of widths away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketGrid {
    /// Buckets of a fixed length, both fields counted in microseconds since
    /// the epoch.
    Micros { width: i64, origin: i64 },
    /// Buckets of whole calendar months, both fields counted in months since
    /// January 1970. Every bucket starts at midnight on the first of a month.
    Months { width: i64, origin: i64 },
}

impl BucketGrid {
    /// Build the grid of fixed-length buckets that has a bucket starting at
    /// `origin`. `width` must be positive.
    pub fn from_micros(width: i64, origin: i64) -> BucketGrid {
        BucketGrid::Micros {
            width,
            origin: origin.rem_euclid(width),
        }
    }

    /// Build the grid of calendar-month buckets that has a bucket starting at
    /// the month `origin`. `width` must be positive.
    pub fn from_months(width: i64, origin: i64) -> BucketGrid {
        BucketGrid::Months {
            width,
            origin: origin.rem_euclid(width),
        }
    }

    /// Floor an instant, in microseconds since the epoch, to the start of the
    /// bucket that holds it.
    fn floor_micros(self, micros: i64) -> i64 {
        match self {
            // The distance past the bucket start is computed from the two
            // remainders, which both sit in `[0, width)`, so no instant can
            // overflow on the way.
            BucketGrid::Micros { width, origin } => {
                micros.wrapping_sub((micros.rem_euclid(width) - origin).rem_euclid(width))
            }
            BucketGrid::Months { width, origin } => {
                let months = count_epoch_months(micros.div_euclid(MICROS_PER_DAY));
                let bucket_start = months - (months - origin).rem_euclid(width);
                let days = days_from_civil(
                    1970 + bucket_start.div_euclid(12),
                    bucket_start.rem_euclid(12) + 1,
                    1,
                );
                days * MICROS_PER_DAY
            }
        }
    }
}

impl Display for BucketGrid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BucketGrid::Micros { width, origin } => {
                write!(f, "{width} microseconds from {origin}")
            }
            BucketGrid::Months { width, origin } => write!(f, "{width} months from {origin}"),
        }
    }
}

/// SQL `time_bucket(bucket_width, source[, offset | origin])` — floor a date or
/// timestamp to the start of its bucket. The width and the optional third
/// argument are constants, folded into the [`BucketGrid`] when the call is
/// built.
#[derive(Debug, Clone)]
pub struct TimeBucket {
    pub grid: BucketGrid,
    pub source: Box<Expression>,
    /// The bound result type, which is the source's own ([`Type::Date`],
    /// [`Type::Timestamp`] or [`Type::TimestampTz`]).
    pub result: Type,
}

impl Display for TimeBucket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "time_bucket({}, {})", self.grid, self.source)
    }
}

impl TimeBucket {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let grid = self.grid;
        let result_arrow = physical_arrow_type(&self.result);
        let source_builder = self.source.compile()?;
        Ok(Box::new(move || {
            let mut source_expr = source_builder();
            let result_arrow = result_arrow.clone();
            Box::new(move |batch: &RecordBatch| {
                let source = source_expr(batch);
                let (array, _) = source.as_datum().get();
                let bucketed: ArrayRef = match &result_arrow {
                    DataType::Date32 => Arc::new(floor_dates(grid, array.as_primitive())),
                    DataType::Timestamp(_, zone) => {
                        // Bucketing the UTC instant is zone-independent under
                        // the UTC session, so a zoned source keeps its marker.
                        let micros = arrow::compute::cast(array, &DataType::Int64).unwrap();
                        Arc::new(
                            floor_timestamps(grid, micros.as_primitive())
                                .with_timezone_opt(zone.clone()),
                        )
                    }
                    other => unreachable!("time_bucket result is a date or timestamp, not {other}"),
                };
                ExprResult::Array(bucketed)
            }) as ExprEvalFn
        }))
    }
}

/// Floor each timestamp to its bucket start. The two infinite timestamps have
/// no bucket and pass through unchanged.
fn floor_timestamps(
    grid: BucketGrid,
    micros: &arrow_array::PrimitiveArray<Int64Type>,
) -> TimestampMicrosecondArray {
    micros.unary::<_, TimestampMicrosecondType>(|instant| {
        if instant == i64::MAX || instant == -i64::MAX {
            return instant;
        }
        grid.floor_micros(instant)
    })
}

/// Floor each date to the day its bucket starts on, reading the date as its
/// midnight. The two infinite dates have no bucket and pass through unchanged.
fn floor_dates(grid: BucketGrid, days: &arrow_array::PrimitiveArray<Date32Type>) -> Date32Array {
    days.unary(|day| {
        if day == i32::MAX || day == -i32::MAX {
            return day;
        }
        grid.floor_micros(i64::from(day) * MICROS_PER_DAY)
            .div_euclid(MICROS_PER_DAY) as i32
    })
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::{ArrayRef, Int32Array, Int64Array};
    use rstest::rstest;
    use std::sync::Arc;

    /// 2024-03-05 10:22:33.5, a Tuesday, in microseconds since the epoch.
    const TUESDAY_MORNING: i64 = 1_709_634_153_500_000;

    /// Register `events` with one row whose `ts` column of `column_type`
    /// holds `micros`.
    fn add_event(testing_planner: &TestingPlanner, column_type: Type, micros: i64) {
        let column = Arc::new(Int64Array::from(vec![micros])) as ArrayRef;
        testing_planner.add_table("events", &[("ts", column_type, column)]);
    }

    #[rstest]
    #[case::minutes("INTERVAL '15 minutes', ts", "2024-03-05T10:15:00")]
    #[case::weeks_start_on_monday("INTERVAL '7 days', ts", "2024-03-04T00:00:00")]
    #[case::months("INTERVAL '3 months', ts", "2024-01-01T00:00:00")]
    #[case::years("INTERVAL '5 years', ts", "2020-01-01T00:00:00")]
    #[case::offset("INTERVAL '1 hour', ts, INTERVAL '10 minutes'", "2024-03-05T10:10:00")]
    #[case::month_offset("INTERVAL '3 months', ts, INTERVAL '1 month'", "2024-02-01T00:00:00")]
    #[case::origin(
        "INTERVAL '1 hour', ts, TIMESTAMP '2024-01-01 00:20:00'",
        "2024-03-05T10:20:00"
    )]
    fn floors_a_timestamp_to_its_bucket(
        mut testing_planner: TestingPlanner,
        #[case] arguments: &str,
        #[case] expected: &str,
    ) {
        add_event(&testing_planner, Type::Timestamp, TUESDAY_MORNING);

        let rows = run(
            &mut testing_planner,
            &format!("SELECT time_bucket({arguments}) FROM events"),
        );

        assert_eq!(only_column(&rows[0]), expected);
    }

    #[rstest]
    fn floors_a_timestamp_before_the_epoch(mut testing_planner: TestingPlanner) {
        let half_an_hour_before_the_epoch = -1_800_000_000;
        add_event(
            &testing_planner,
            Type::Timestamp,
            half_an_hour_before_the_epoch,
        );

        let rows = run(
            &mut testing_planner,
            "SELECT time_bucket(INTERVAL '1 day', ts) FROM events",
        );

        assert_eq!(only_column(&rows[0]), "1969-12-31T00:00:00");
    }

    #[rstest]
    fn bucketing_a_timestamptz_keeps_its_zone(mut testing_planner: TestingPlanner) {
        add_event(&testing_planner, Type::TimestampTz, TUESDAY_MORNING);

        let rows = run(
            &mut testing_planner,
            "SELECT time_bucket(INTERVAL '1 hour', ts) FROM events",
        );

        assert_eq!(only_column(&rows[0]), "2024-03-05T10:00:00Z");
    }

    #[rstest]
    fn bucketing_a_date_yields_a_date(mut testing_planner: TestingPlanner) {
        let tuesday = Arc::new(Int32Array::from(vec![19_787])) as ArrayRef;
        testing_planner.add_table("events", &[("day", Type::Date, tuesday)]);

        let rows = run(
            &mut testing_planner,
            "SELECT time_bucket(INTERVAL '7 days', day) FROM events",
        );

        assert_eq!(only_column(&rows[0]), "2024-03-04");
    }

    #[rstest]
    fn rejects_a_width_mixing_months_and_days(mut testing_planner: TestingPlanner) {
        add_event(&testing_planner, Type::Timestamp, TUESDAY_MORNING);

        let error = testing_planner
            .plan("SELECT time_bucket(INTERVAL '1 month 1 day', ts) FROM events")
            .err()
            .unwrap();

        assert!(
            error
                .to_string()
                .contains("cannot have a day or time component")
        );
    }
}
