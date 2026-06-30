//! [`Arithmetic`] — binary integer `+`/`-`/`*` and its [`ArithmeticOp`].

use super::{Error, Expression};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::{self, Type};
use arrow::compute::kernels::numeric::{add_wrapping, mul_wrapping, sub_wrapping};
use arrow_array::{ArrayRef, Datum, RecordBatch};
use arrow_schema::ArrowError;
use duckdb_planner::expression as duckdb_expression;
use std::fmt::{self, Display};

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

impl TryFrom<duckdb_expression::Function> for Arithmetic {
    type Error = Error;
    fn try_from(mut f: duckdb_expression::Function) -> Result<Self, Self::Error> {
        let op = match f.function.as_str() {
            "+" => ArithmeticOp::Add,
            "-" => ArithmeticOp::Sub,
            "*" => ArithmeticOp::Mul,
            _ => return Err(Error::UnsupportedScalarFunction(f.function)),
        };
        // Unary forms (e.g. `-x`) bind to the same function names with one
        // parameter; only the binary forms are supported.
        if f.params.len() != 2 {
            let actual = f.params.len();
            return Err(Error::InvalidParameterCount {
                function: f.function,
                expected: 2,
                actual,
            });
        }
        let return_type = types::type_from_logical(f.return_type)?;
        let right = Box::new(Expression::try_from(f.params.remove(1))?);
        let left = Box::new(Expression::try_from(f.params.remove(0))?);
        Ok(Arithmetic {
            op,
            left,
            right,
            return_type,
        })
    }
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
        let left_builder = self.left.compile()?;
        let right_builder = self.right.compile()?;
        Ok(Box::new(move || {
            let mut left_expr = left_builder();
            let mut right_expr = right_builder();
            Box::new(move |batch: &RecordBatch| {
                let left = left_expr(batch);
                let right = right_expr(batch);
                let out = kernel(left.as_datum(), right.as_datum())
                    .expect("arithmetic operands share a type");
                ExprResult::Array(out)
            }) as ExprEvalFn
        }))
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
