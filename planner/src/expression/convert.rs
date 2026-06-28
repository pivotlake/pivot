//! [`TemporalConvert`] — the integer→temporal conversion functions that let a
//! query read an integer column as a date/time (both DuckDB built-ins):
//!
//! - `make_date(days)` → `Date32`
//! - `make_timestamp(seconds)` → `Timestamp(Second)`
//!
//! pivot stores a `DATE` as `Int32` days and a `TIMESTAMP` as `Int64` seconds,
//! so each conversion is just the matching arrow cast over the source column:
//! seconds→`Timestamp(Second)` is a zero-copy reinterpret (identical i64 bits),
//! and days→`Date32` is the same for an `Int32` source (a cheap widening cast
//! for a narrower integer like ClickBench's `UInt16` `EventDate`).

use super::{Error, Expression};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::Type;
use arrow_array::RecordBatch;
use arrow_schema::{DataType, TimeUnit};
use duckdb_planner::expression as duckdb_expression;
use std::fmt::{self, Display};

/// A unary integer→temporal conversion. `result` is the pivot type the call
/// yields (`Date` or `Timestamp`); `target` is the arrow type its source casts
/// into.
#[derive(Debug, Clone)]
pub struct TemporalConvert {
    /// Display name of the originating function (`make_date` / `make_timestamp`).
    name: &'static str,
    /// The pivot type this call produces, for `result_type`.
    pub result: Type,
    /// The arrow type the integer source is cast into.
    target: DataType,
    /// `target`'s physical integer type (`Int32` for `Date32`, `Int64` for
    /// `Timestamp`). A source already of this type reinterprets straight to
    /// `target`; a narrower one (e.g. `UInt16`) widens through it first, since
    /// arrow has no direct `UInt16`→`Date32` cast.
    via: DataType,
    /// The integer source expression.
    source: Box<Expression>,
}

impl TemporalConvert {
    /// `make_date(days)` → `Date32`.
    pub(super) fn make_date(f: duckdb_expression::Function) -> Result<Self, Error> {
        Self::build(
            "make_date",
            Type::Date,
            DataType::Date32,
            DataType::Int32,
            f,
        )
    }

    /// `make_timestamp(seconds)` → `Timestamp(Second)`.
    pub(super) fn make_timestamp(f: duckdb_expression::Function) -> Result<Self, Error> {
        Self::build(
            "make_timestamp",
            Type::Timestamp,
            DataType::Timestamp(TimeUnit::Second, None),
            DataType::Int64,
            f,
        )
    }

    fn build(
        name: &'static str,
        result: Type,
        target: DataType,
        via: DataType,
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
        Ok(TemporalConvert {
            name,
            result,
            target,
            via,
            source,
        })
    }

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
