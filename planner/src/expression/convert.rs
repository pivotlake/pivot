//! [`TemporalConvert`] — the integer→temporal conversion functions that let a
//! query read an integer column as a date/time (both DuckDB built-ins):
//!
//! - `make_date(days)` → `Date32`
//! - `make_timestamp(microseconds)` → `Timestamp(Microsecond)`
//!
//! pivot stores a `DATE` as `Int32` days and a `TIMESTAMP` as `Int64`
//! microseconds, the same counts DuckDB reads these functions as, so each
//! conversion is just the matching arrow cast over the source column:
//! microseconds→`Timestamp(Microsecond)` is a zero-copy reinterpret (identical
//! i64 bits), and days→`Date32` is the same for an `Int32` source (a cheap
//! widening cast for a narrower integer source).

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::Type;
use arrow_array::RecordBatch;
use arrow_schema::DataType;
use std::fmt::{self, Display};

/// A unary integer→temporal conversion. `result` is the pivot type the call
/// yields (`Date` or `Timestamp`); `target` is the arrow type its source casts
/// into.
#[derive(Debug, Clone)]
pub struct TemporalConvert {
    /// Display name of the originating function (`make_date` / `make_timestamp`).
    pub(crate) name: &'static str,
    /// The pivot type this call produces, for `result_type`.
    pub result: Type,
    /// The arrow type the integer source is cast into.
    pub(crate) target: DataType,
    /// `target`'s physical integer type (`Int32` for `Date32`, `Int64` for
    /// `Timestamp`). A source already of this type reinterprets straight to
    /// `target`; a narrower one (e.g. `UInt16`) widens through it first, since
    /// arrow has no direct `UInt16`→`Date32` cast.
    pub(crate) via: DataType,
    /// The integer source expression.
    pub(crate) source: Box<Expression>,
}

impl TemporalConvert {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let target = self.target.clone();
        let via = self.via.clone();
        let source_builder = self.source.compile()?;
        Ok(Box::new(move || {
            let target = target.clone();
            let via = via.clone();
            let mut source_expr = source_builder();
            Box::new(move |batch: &RecordBatch| {
                let src = source_expr(batch);
                let (arr, _) = src.as_datum().get();
                // A source already of the physical integer type reinterprets
                // straight to `target`; a narrower one widens through it first.
                let out = if arr.data_type() == &via {
                    arrow::compute::cast(arr, &target)
                } else {
                    let widened = arrow::compute::cast(arr, &via)
                        .expect("integer source widens to its temporal physical type");
                    arrow::compute::cast(&widened, &target)
                }
                .expect("integer source casts to its temporal type");
                ExprResult::Array(out)
            }) as ExprEvalFn
        }))
    }
}

impl Display for TemporalConvert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.name, self.source)
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

    /// `make_timestamp` reads its integer as the microseconds DuckDB defines it
    /// to be, so a column counting something coarser is scaled by the query
    /// itself and the result still lands on a real TIMESTAMP.
    #[rstest]
    fn make_timestamp_reads_microseconds(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "events",
            &[(
                "seconds",
                Type::Int64,
                Arc::new(Int64Array::from(vec![1_704_067_200i64])) as ArrayRef,
            )],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT make_timestamp(seconds * 1000000) FROM events",
        );

        let col = batches[0].column(0);
        assert_eq!(
            col.data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        assert_eq!(
            col.as_primitive::<TimestampMicrosecondType>().value(0),
            1_704_067_200_000_000
        );
    }
}
