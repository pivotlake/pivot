//! [`DateTrunc`] — SQL `date_trunc(unit, source)`.

use super::{Expression, Function};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::TimestampUnit;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::DataType;
use std::fmt::{self, Display};

/// SQL `date_trunc(unit, source)` — truncate a timestamp down to `unit`
/// (e.g. `date_trunc('minute', ts)`). DuckDB passes the unit as a string
/// constant in the first argument and the timestamp expression second.
#[derive(Debug, Clone)]
pub struct DateTrunc {
    pub unit: String,
    /// The resolution `source` counts in, which the truncated result keeps: a
    /// value is floored in its own unit rather than converted to another.
    pub source_unit: TimestampUnit,
    pub source: Box<Expression>,
}

impl Display for DateTrunc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "date_trunc('{}', {})", self.unit, self.source)
    }
}

impl DateTrunc {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        // A timestamp is stored as an Int64 epoch count in its own unit, so
        // truncating to a unit is flooring to that many of them:
        // `M = (t / step) * step`, where `step` is the unit's seconds scaled
        // into the column's resolution.
        let secs: i64 = match self.unit.as_str() {
            "second" => 1,
            "minute" => 60,
            "hour" => 3600,
            "day" => 86400,
            other => {
                return Err(compile::Error::UnsupportedExpression(Expression::Function(
                    Function::DateTrunc(DateTrunc {
                        unit: other.to_string(),
                        source_unit: self.source_unit,
                        source: self.source.clone(),
                    }),
                )));
            }
        };
        let step = secs * self.source_unit.per_second();
        let source_unit = self.source_unit;
        let source_builder = self.source.compile()?;
        Ok(Box::new(move || {
            let result_type = DataType::Timestamp(source_unit.arrow_unit(), None);
            let mut source_expr = source_builder();
            Box::new(move |batch: &RecordBatch| {
                let src = source_expr(batch);
                let (arr, _) = src.as_datum().get();
                let i64arr = arrow::compute::cast(arr, &DataType::Int64).unwrap();
                let vals = i64arr.as_primitive::<Int64Type>();
                // Floor to `step` and emit a real TIMESTAMP of the source's own
                // unit, the type date_trunc returns, rather than a bare int. The
                // cast only re-labels the counts; it does not rescale them.
                let truncated: Int64Array = vals
                    .iter()
                    .map(|v| v.map(|x| x.div_euclid(step) * step))
                    .collect();
                let typed = arrow::compute::cast(&truncated, &result_type)
                    .expect("an Int64 count re-labels as a timestamp of any unit");
                ExprResult::Array(typed as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::TimestampUnit;
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
                Type::Timestamp(TimestampUnit::Second),
                Arc::new(Int64Array::from(vec![0i64, 90, 150, 3690])) as ArrayRef,
            )],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT date_trunc('minute', EventTime) FROM events",
        );

        // The column counts seconds, but DuckDB types `date_trunc` as its
        // microsecond TIMESTAMP, so the floored value arrives in microseconds.
        let col = batches[0].column(0);
        assert_eq!(
            col.data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        let mut secs: Vec<i64> = col
            .as_primitive::<TimestampMicrosecondType>()
            .values()
            .iter()
            .map(|micros| micros / 1_000_000)
            .collect();
        secs.sort();
        assert_eq!(secs, vec![0, 60, 120, 3660]);
    }
}
