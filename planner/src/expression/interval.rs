//! [`IntervalArithmetic`]: `date`/`timestamp` ± `INTERVAL`.
//!
//! An interval added to or subtracted from a temporal value is lowered to a
//! constant integer offset in the value's physical unit (days for `DATE`,
//! seconds for `TIMESTAMP`) and applied with the same wrapping add/sub kernel as
//! ordinary integer arithmetic. The temporal operand is reinterpreted to its int
//! for the add and back to `Date32`/`Timestamp` after, so the expression owns its
//! output type like every other temporal expression. Month and year intervals
//! are calendar-variable, so they need real calendar math and are rejected here.

use super::Expression;
use super::arithmetic::ArithmeticOp;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::{Type, physical_arrow_type};
use arrow::compute::kernels::numeric::{add_wrapping, sub_wrapping};
use arrow_array::{ArrayRef, Datum, Int32Array, Int64Array, RecordBatch};
use arrow_schema::DataType;
use std::fmt::{self, Display};

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
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let result_arrow = physical_arrow_type(&self.result);
        // The physical int the temporal result reinterprets through.
        let int_arrow = match self.result {
            Type::Date => DataType::Int32,
            Type::Timestamp(_) => DataType::Int64,
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
    use arrow_array::types::TimestampMicrosecondType;
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

        // DuckDB promotes `date + interval` to its microsecond TIMESTAMP:
        // 1970-01-01 + 7 days = 1970-01-08 00:00:00 = 604800 epoch seconds.
        let col = batches[0].column(0);
        assert_eq!(
            col.data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        assert_eq!(
            col.as_primitive::<TimestampMicrosecondType>().value(0),
            7 * 86_400 * 1_000_000
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
            &DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        let secs = col.as_primitive::<TimestampMicrosecondType>().value(0) / 1_000_000;
        let five_days = 5 * 86_400;
        assert!(
            (before - five_days..=after - five_days).contains(&secs),
            "{secs} not in [{}, {}]",
            before - five_days,
            after - five_days
        );
    }
}
