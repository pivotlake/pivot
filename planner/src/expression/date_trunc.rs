//! [`DateTrunc`] — SQL `date_trunc(unit, source)`.

use super::{Error, Expression, Function, constant_string};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::DataType;
use duckdb_planner::expression as duckdb_expression;
use std::fmt::{self, Display};
use std::sync::Arc;

/// SQL `date_trunc(unit, source)` — truncate a timestamp down to `unit`
/// (e.g. `date_trunc('minute', EventTime)`). DuckDB passes the unit as a string
/// constant in the first argument and the timestamp expression second.
#[derive(Debug, Clone)]
pub struct DateTrunc {
    pub unit: String,
    pub source: Box<Expression>,
}

impl TryFrom<duckdb_expression::Function> for DateTrunc {
    type Error = Error;
    fn try_from(mut f: duckdb_expression::Function) -> Result<Self, Self::Error> {
        if f.params.len() != 2 {
            let actual = f.params.len();
            return Err(Error::InvalidParameterCount {
                function: f.function,
                expected: 2,
                actual,
            });
        }
        let source = Box::new(Expression::try_from(f.params.remove(1))?);
        let unit = constant_string(
            Expression::try_from(f.params.remove(0))?,
            "date_trunc: unit",
        )?
        .to_ascii_lowercase();
        Ok(DateTrunc { unit, source })
    }
}

impl Display for DateTrunc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "date_trunc('{}', {})", self.unit, self.source)
    }
}

impl DateTrunc {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        // EventTime is stored as Int64 epoch *seconds*, so truncating to a unit
        // is flooring to that many seconds. `M = (t / secs) * secs`.
        let secs: i64 = match self.unit.as_str() {
            "second" => 1,
            "minute" => 60,
            "hour" => 3600,
            "day" => 86400,
            other => {
                return Err(compile::Error::UnsupportedExpression(Expression::Function(
                    Function::DateTrunc(DateTrunc {
                        unit: other.to_string(),
                        source: self.source.clone(),
                    }),
                )));
            }
        };
        let source_builder = self.source.compile()?;
        Ok(Box::new(move || {
            let mut source_expr = source_builder();
            Box::new(move |batch: &RecordBatch| {
                let src = source_expr(batch);
                let (arr, _) = src.as_datum().get();
                let i64arr = arrow::compute::cast(arr, &DataType::Int64).unwrap();
                let vals = i64arr.as_primitive::<Int64Type>();
                let truncated: Int64Array = vals
                    .iter()
                    .map(|v| v.map(|x| x.div_euclid(secs) * secs))
                    .collect();
                ExprResult::Array(Arc::new(truncated) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::{ArrayRef, Int64Array};
    use rstest::rstest;
    use std::sync::Arc;

    #[rstest]
    fn truncates_to_minute(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "events",
            &[(
                "EventTime",
                Type::Timestamp,
                Arc::new(Int64Array::from(vec![0i64, 90, 150, 3690])) as ArrayRef,
            )],
        );

        let mut rows = run(
            &mut testing_planner,
            "SELECT date_trunc('minute', EventTime) FROM events",
        );

        rows.sort_by_key(|r| r["col0"].as_i64().unwrap());

        assert_eq!(
            rows.iter()
                .map(|r| r["col0"].as_i64().unwrap())
                .collect::<Vec<_>>(),
            vec![0, 60, 120, 3660]
        );
    }
}
