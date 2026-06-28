//! [`IntervalArithmetic`]: `date`/`timestamp` ± `INTERVAL`.
//!
//! An interval added to or subtracted from a temporal value is lowered to a
//! constant integer offset in the value's physical unit (days for `DATE`,
//! seconds for `TIMESTAMP`) and applied with the same wrapping add/sub kernel as
//! ordinary integer arithmetic. The temporal operand is reinterpreted to its int
//! for the add and back to `Date32`/`Timestamp` after, so the expression owns its
//! output type like every other temporal expression. Month and year intervals
//! are calendar-variable, so they need real calendar math and are rejected here.

use super::arithmetic::ArithmeticOp;
use super::{Error, Expression};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::{self, Type, physical_arrow_type};
use arrow::compute::kernels::numeric::{add_wrapping, sub_wrapping};
use arrow_array::{ArrayRef, Datum, Int32Array, Int64Array, RecordBatch};
use arrow_schema::DataType;
use duckdb_planner::expression as duckdb_expression;
use std::fmt::{self, Display};

const SECS_PER_DAY: i64 = 86_400;
const MICROS_PER_SEC: i64 = 1_000_000;

/// A DuckDB `INTERVAL`'s three independent components. The bridge serializes them
/// into the constant's `raw_value` as `"<months> <days> <micros>"` (months are
/// calendar-variable, so they stay separate from the fixed day/microsecond parts).
struct IntervalParts {
    months: i32,
    days: i32,
    micros: i64,
}

/// Parse the bridge's `"<months> <days> <micros>"` interval encoding.
fn parse_interval(raw: &str) -> Result<IntervalParts, Error> {
    let malformed = || Error::UnsupportedInterval(format!("malformed interval constant '{raw}'"));
    let [months, days, micros] = raw.split_whitespace().collect::<Vec<_>>()[..] else {
        return Err(malformed());
    };
    Ok(IntervalParts {
        months: months.parse().map_err(|_| malformed())?,
        days: days.parse().map_err(|_| malformed())?,
        micros: micros.parse().map_err(|_| malformed())?,
    })
}

/// `date`/`timestamp` ± `INTERVAL`, reduced to a constant integer offset.
#[derive(Debug, Clone)]
pub struct IntervalArithmetic {
    pub op: ArithmeticOp,
    pub operand: Box<Expression>,
    /// The interval as an offset in the result's unit (days for `Date`, seconds
    /// for `Timestamp`).
    pub offset: i64,
    /// The temporal result type (`Date` or `Timestamp`).
    pub result: Type,
}

impl IntervalArithmetic {
    /// Build from a DuckDB `+`/`-` whose `params[interval_idx]` is an `INTERVAL`
    /// constant; the other operand is the temporal value. The caller (in
    /// [`Function::try_from`](super::Function)) has already located the interval.
    pub(super) fn try_build(
        f: duckdb_expression::Function,
        interval_idx: usize,
    ) -> Result<Self, Error> {
        if f.params.len() != 2 {
            return Err(Error::InvalidParameterCount {
                function: f.function,
                expected: 2,
                actual: f.params.len(),
            });
        }
        let op = match f.function.as_str() {
            "+" => ArithmeticOp::Add,
            "-" => ArithmeticOp::Sub,
            other => return Err(Error::UnsupportedScalarFunction(other.to_string())),
        };
        // DuckDB types the `+`/`-` itself, so its return type is the temporal
        // result (DATE for whole-day intervals, TIMESTAMP otherwise).
        let result = types::type_from_logical(f.return_type.clone())?;
        let interval = match &f.params[interval_idx] {
            duckdb_expression::Expression::Constant(scalar) => parse_interval(&scalar.raw_value)?,
            _ => unreachable!("caller selected an interval constant"),
        };
        let offset = interval_offset(&interval, &result)?;
        let mut params = f.params;
        let operand = Box::new(Expression::try_from(params.swap_remove(1 - interval_idx))?);
        Ok(IntervalArithmetic {
            op,
            operand,
            offset,
            result,
        })
    }

    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let result_arrow = physical_arrow_type(&self.result);
        // The physical int the temporal result reinterprets through.
        let int_arrow = match self.result {
            Type::Date => DataType::Int32,
            Type::Timestamp => DataType::Int64,
            _ => unreachable!("interval result is Date or Timestamp"),
        };
        let op = self.op;
        let offset = self.offset;
        let operand = self.operand.compile()?;
        Ok(Box::new(move || {
            let mut eval = operand();
            let result_arrow = result_arrow.clone();
            let int_arrow = int_arrow.clone();
            Box::new(move |batch: &RecordBatch| {
                let value = eval(batch);
                let (arr, _) = value.as_datum().get();
                // Convert the operand to the result's temporal type first: DuckDB
                // promotes `date + interval` to a TIMESTAMP, so a DATE operand (days)
                // must become a timestamp (seconds at midnight) before the offset,
                // which is in the result's unit, is applied.
                let as_result = arrow::compute::cast(arr, &result_arrow).unwrap();
                let ints = arrow::compute::cast(&as_result, &int_arrow).unwrap();
                let shifted = apply(&ints, offset, op, &int_arrow);
                ExprResult::Array(arrow::compute::cast(&shifted, &result_arrow).unwrap())
            }) as ExprEvalFn
        }))
    }
}

/// Apply the constant `offset` to the reinterpreted int column with the wrapping
/// add/sub kernel, building the offset scalar at the column's width.
fn apply(ints: &dyn Datum, offset: i64, op: ArithmeticOp, int_arrow: &DataType) -> ArrayRef {
    let kernel = match op {
        ArithmeticOp::Add => add_wrapping,
        ArithmeticOp::Sub => sub_wrapping,
        ArithmeticOp::Mul => unreachable!("interval arithmetic is only + or -"),
    };
    match int_arrow {
        DataType::Int32 => kernel(ints, &Int32Array::new_scalar(offset as i32)).unwrap(),
        _ => kernel(ints, &Int64Array::new_scalar(offset)).unwrap(),
    }
}

/// Reduce an interval to a constant offset in the result's unit. Months/years are
/// calendar-variable (rejected); a `DATE` only takes whole-day intervals, a
/// `TIMESTAMP` takes days plus a sub-day part truncated to whole seconds (pivot
/// stores timestamps at second granularity, so any sub-second part is dropped).
fn interval_offset(interval: &IntervalParts, result: &Type) -> Result<i64, Error> {
    if interval.months != 0 {
        return Err(Error::UnsupportedInterval(
            "month and year intervals require calendar arithmetic".to_string(),
        ));
    }
    match result {
        Type::Date => {
            if interval.micros != 0 {
                return Err(Error::UnsupportedInterval(
                    "sub-day interval applied to a DATE".to_string(),
                ));
            }
            Ok(interval.days as i64)
        }
        Type::Timestamp => {
            Ok(interval.days as i64 * SECS_PER_DAY + interval.micros / MICROS_PER_SEC)
        }
        other => Err(Error::UnsupportedInterval(format!(
            "interval arithmetic on a non-temporal {other}"
        ))),
    }
}

impl Display for IntervalArithmetic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({} {} interval[{} {}])",
            self.operand, self.op, self.offset, self.result
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::cast::AsArray;
    use arrow_array::types::TimestampSecondType;
    use arrow_array::{ArrayRef, Int32Array};
    use arrow_schema::{DataType, TimeUnit};
    use rstest::rstest;
    use std::sync::Arc;

    #[rstest]
    fn date_plus_interval_yields_a_timestamp(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "events",
            &[(
                "d",
                Type::Date,
                Arc::new(Int32Array::from(vec![0])) as ArrayRef,
            )],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT d + interval '7 days' FROM events",
        );

        // DuckDB promotes `date + interval` to a TIMESTAMP: 1970-01-01 + 7 days =
        // 1970-01-08 00:00:00 = 604800 epoch seconds.
        let col = batches[0].column(0);
        assert_eq!(
            col.data_type(),
            &DataType::Timestamp(TimeUnit::Second, None)
        );
        assert_eq!(
            col.as_primitive::<TimestampSecondType>().value(0),
            7 * 86_400
        );
    }

    #[rstest]
    fn now_minus_interval_subtracts_seconds(mut testing_planner: TestingPlanner) {
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let batches = run_batches(&mut testing_planner, "SELECT now() - interval '5 days'");

        let after = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let col = batches[0].column(0);
        assert_eq!(
            col.data_type(),
            &DataType::Timestamp(TimeUnit::Second, None)
        );
        let secs = col.as_primitive::<TimestampSecondType>().value(0);
        let five_days = 5 * 86_400;
        assert!(
            (before - five_days..=after - five_days).contains(&secs),
            "{secs} not in [{}, {}]",
            before - five_days,
            after - five_days
        );
    }
}
