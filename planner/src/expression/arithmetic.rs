//! [`Arithmetic`] — binary integer `+`/`-`/`*` and its [`ArithmeticOp`].

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::{MAX_DECIMAL64_PRECISION, MICROS_PER_DAY, NANOS_PER_MICRO, Type};
use arrow::compute::kernels::numeric::{add_wrapping, mul_wrapping, sub_wrapping};
use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal64Type, Decimal128Type, DecimalType, DurationMicrosecondType};
use arrow_array::{ArrayRef, Datum, IntervalMonthDayNanoArray, RecordBatch};
use arrow_buffer::IntervalMonthDayNano;
use arrow_schema::{ArrowError, DataType};
use std::fmt::{self, Display};
use std::sync::Arc;

/// Signature shared by arrow's wrapping arithmetic kernels.
type ArithKernel = fn(&dyn Datum, &dyn Datum) -> std::result::Result<ArrayRef, ArrowError>;

/// The operator of a binary integer [`Arithmetic`] expression.
#[derive(Debug, Clone, Copy)]
pub enum ArithmeticOp {
    Add,
    Sub,
    Mul,
}

impl Display for ArithmeticOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArithmeticOp::Add => f.write_str("+"),
            ArithmeticOp::Sub => f.write_str("-"),
            ArithmeticOp::Mul => f.write_str("*"),
        }
    }
}

/// Binary integer arithmetic (`lhs + rhs`, `lhs - rhs`, `lhs * rhs`). DuckDB
/// lowers these as a `BOUND_FUNCTION` whose function name is the operator
/// symbol itself (`"+"`, `"-"`, `"*"`). Division is separate
/// ([`Divide`](super::Divide)) because SQL `/` yields a non-integer quotient.
#[derive(Debug, Clone)]
pub struct Arithmetic {
    pub op: ArithmeticOp,
    pub left: Box<Expression>,
    pub right: Box<Expression>,
    pub return_type: Type,
}

impl Display for Arithmetic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} {} {})", self.left, self.op, self.right)
    }
}

impl Arithmetic {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        // Wrapping kernels match DuckDB's behaviour for in-range values; we
        // accept silent wraparound (rather than an error) on overflow.
        let kernel: ArithKernel = match self.op {
            ArithmeticOp::Add => add_wrapping,
            ArithmeticOp::Sub => sub_wrapping,
            ArithmeticOp::Mul => mul_wrapping,
        };
        let bound_decimal = match &self.return_type {
            Type::Decimal { precision, scale } => Some((*precision, *scale)),
            _ => None,
        };
        // `timestamp - timestamp` is bound to an INTERVAL, but arrow's kernel
        // answers a `Duration` of microseconds, so the difference is rebuilt
        // into interval fields below.
        let yields_interval = self.return_type == Type::Interval;
        let left_builder = self.left.compile()?;
        let right_builder = self.right.compile()?;
        Ok(Box::new(move || {
            let mut left_expr = left_builder();
            let mut right_expr = right_builder();
            Box::new(move |batch: &RecordBatch| {
                let left = left_expr(batch);
                let right = right_expr(batch);
                let out = kernel(left.as_datum(), right.as_datum())
                    .unwrap_or_else(|error| panic!("arithmetic kernel failed: {error}"));
                let out = match bound_decimal {
                    Some((precision, scale)) => restamp_decimal(out, precision, scale),
                    None => out,
                };
                let out = if yields_interval {
                    interval_from_duration(&out)
                } else {
                    out
                };
                ExprResult::Array(out)
            }) as ExprEvalFn
        }))
    }
}

/// Rebuild a microsecond `Duration`, what arrow's kernel answers for
/// `timestamp - timestamp`, into interval fields: whole days split out, the
/// remainder left as sub-day time.
///
/// Months stay zero, as in DuckDB's `Interval::FromMicro` and Postgres's
/// `timestamp_mi`. How many months a span covers depends on which months they
/// were, so naming one breaks `a + (b - a) = b`: 2020-01-01 to 2020-04-10 is
/// 100 days, and `3 mons 10 days` added back lands a day late.
///
/// Splitting the days off also keeps the nanoseconds under a day, well inside
/// the ~292 years the field holds.
fn interval_from_duration(duration: &ArrayRef) -> ArrayRef {
    let micros = duration.as_primitive::<DurationMicrosecondType>();
    let spans: IntervalMonthDayNanoArray = micros.unary(|total| IntervalMonthDayNano {
        months: 0,
        days: (total / MICROS_PER_DAY) as i32,
        nanoseconds: total % MICROS_PER_DAY * NANOS_PER_MICRO,
    });
    Arc::new(spans)
}

/// Restamp a decimal kernel result to the plan's bound result type.
///
/// Arrow's decimal arithmetic derives the same result *scale* as DuckDB's
/// binder (`max(s1, s2)` for add/sub, `s1 + s2` for multiply), so the stored
/// unscaled integers already match the plan. Only the declared *precision*
/// can disagree, in two known ways. A multiply is declared `p1 + p2 + 1`
/// digits by Arrow (saturating at the carrier's own cap) but `p1 + p2` by
/// DuckDB. An add/sub whose operands fit in 18 digits is declared 19 digits
/// by Arrow, while DuckDB keeps it declared at 18: in DuckDB's own engine
/// crossing 18 digits switches the physical storage from int64 to int128, so
/// its binder pins the width and relies on a runtime overflow check instead.
/// Precision is only an annotation on a decimal array, so swapping it touches
/// no values.
///
/// The kernel result's carrier always matches the bound type's: DuckDB casts
/// both operands of an operation whose result crosses 18 digits to the wide
/// type, so a `Decimal64` kernel result only ever pairs with a
/// `Decimal64`-carried bound type, and likewise for `Decimal128`.
fn restamp_decimal(array: ArrayRef, precision: u8, scale: i8) -> ArrayRef {
    fn restamp<T: DecimalType>(array: &ArrayRef, precision: u8, scale: i8) -> ArrayRef {
        let decimal = array.as_primitive::<T>();
        assert_eq!(
            decimal.scale(),
            scale,
            "arrow and the planner derived different scales for a decimal result"
        );
        Arc::new(
            decimal
                .clone()
                .with_precision_and_scale(precision, scale)
                .expect("the plan's decimal shape was validated at build"),
        )
    }
    match array.data_type() {
        DataType::Decimal64(_, _) => {
            assert!(
                precision <= MAX_DECIMAL64_PRECISION,
                "a Decimal64 kernel result cannot restamp to a Decimal128-carried bound type"
            );
            restamp::<Decimal64Type>(&array, precision, scale)
        }
        DataType::Decimal128(_, _) => {
            assert!(
                precision > MAX_DECIMAL64_PRECISION,
                "a Decimal128 kernel result cannot restamp to a Decimal64-carried bound type"
            );
            restamp::<Decimal128Type>(&array, precision, scale)
        }
        other => unreachable!("a decimal arithmetic kernel returned {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::cast::AsArray;
    use arrow_array::types::IntervalMonthDayNanoType;
    use arrow_array::{ArrayRef, Int64Array};
    use arrow_buffer::IntervalMonthDayNano;
    use arrow_schema::{DataType, IntervalUnit};
    use rstest::rstest;
    use std::sync::Arc;

    /// 1970-01-01 00:00:00 and 2 days 03:04:05.5 later, in microseconds.
    const EPOCH: i64 = 0;
    const LATER: i64 = 183_845_500_000;

    /// The same 2 days 03:04:05.5, as the interval fields it lands on.
    fn later_as_interval() -> IntervalMonthDayNano {
        IntervalMonthDayNano {
            months: 0,
            days: 2,
            nanoseconds: 11_045_500_000_000,
        }
    }

    fn events_table(planner: &TestingPlanner) {
        planner.add_table(
            "events",
            &[(
                "ts",
                Type::Timestamp,
                Arc::new(Int64Array::from(vec![EPOCH, LATER])) as ArrayRef,
            )],
        );
    }

    #[rstest]
    fn subtracting_two_timestamps_yields_an_interval(mut testing_planner: TestingPlanner) {
        events_table(&testing_planner);

        let batches = run_batches(&mut testing_planner, "SELECT max(ts) - min(ts) FROM events");

        // DuckDB binds `timestamp - timestamp` to an INTERVAL, and splits the
        // difference's whole days out of its sub-day remainder.
        let col = batches[0].column(0);
        assert_eq!(
            col.data_type(),
            &DataType::Interval(IntervalUnit::MonthDayNano)
        );
        assert_eq!(
            col.as_primitive::<IntervalMonthDayNanoType>().value(0),
            later_as_interval()
        );
    }

    #[rstest]
    fn now_minus_a_timestamp_measures_the_age_of_a_row(mut testing_planner: TestingPlanner) {
        events_table(&testing_planner);
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;

        let batches = run_batches(&mut testing_planner, "SELECT now() - max(ts) FROM events");

        let age = batches[0]
            .column(0)
            .as_primitive::<IntervalMonthDayNanoType>()
            .value(0);
        let elapsed_days = ((before - LATER) / 86_400_000_000) as i32;
        assert!(
            age.days >= elapsed_days,
            "{age:?} is before the query started"
        );
    }

    /// Sorting by an interval is rejected cleanly rather than sorted wrongly.
    /// DuckDB keys the sort on `normalized_interval(age)`, and an ORDER BY key
    /// has to be a plain column reference here, so the carried value has
    /// nowhere to live. Supporting it means computing the key into an appended
    /// column, the way a join's computed keys already are
    /// (`append_computed_keys`), and projecting it back off after the sort.
    /// Sorting the uncarried column instead would be wrong: `1 mon` and
    /// `100 days` rank one way as stored and the other way once carried.
    #[rstest]
    fn ordering_by_an_interval_is_not_supported_yet(mut testing_planner: TestingPlanner) {
        events_table(&testing_planner);

        let error = compile_expecting_error(
            &mut testing_planner,
            "SELECT ts - TIMESTAMP '1970-01-01' AS age FROM events ORDER BY age DESC",
        );

        assert!(
            error.contains("Unsupported order by expression"),
            "unexpected error: {error}"
        );
    }

    #[rstest]
    fn an_interval_compares_against_an_interval_constant(mut testing_planner: TestingPlanner) {
        events_table(&testing_planner);

        // The 2 days 03:04:05.5 span sits between the two constants, and the
        // month is DuckDB's fixed 30 days.
        let rows = run(
            &mut testing_planner,
            "SELECT max(ts) - min(ts) > INTERVAL '2 days' AS over_two_days,
                    max(ts) - min(ts) < INTERVAL '1 month' AS under_a_month
             FROM events",
        );

        assert_eq!(rows[0]["over_two_days"], true);
        assert_eq!(rows[0]["under_a_month"], true);
    }

    #[rstest]
    fn nested_add_and_mul_in_projection(mut testing_planner: TestingPlanner) {
        // (a + b) * 2 over the (a, b) pairs (1,10)…(5,50).
        let mut rows = run(
            &mut testing_planner,
            "SELECT (a + b) * 2 FROM example_table",
        );

        rows.sort_by_key(|r| only_column(r).as_i64().unwrap());

        assert_eq!(
            rows.iter()
                .map(|r| only_column(r).as_i64().unwrap())
                .collect::<Vec<_>>(),
            vec![22, 44, 66, 88, 110]
        );
    }
}
