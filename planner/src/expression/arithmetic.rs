//! [`Arithmetic`] — binary integer `+`/`-`/`*` and its [`ArithmeticOp`].

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::{MAX_DECIMAL64_PRECISION, Type};
use arrow::compute::kernels::numeric::{add_wrapping, mul_wrapping, sub_wrapping};
use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal64Type, Decimal128Type, DecimalType};
use arrow_array::{ArrayRef, Datum, RecordBatch};
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
                ExprResult::Array(out)
            }) as ExprEvalFn
        }))
    }
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
            "arrow and DuckDB derived different scales for a decimal result"
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
    use rstest::rstest;

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
