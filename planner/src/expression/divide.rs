//! [`Divide`] — SQL `lhs / rhs` (float division).

use super::{Error, Expression};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow_array::RecordBatch;
use arrow_schema::DataType;
use duckdb_planner::expression as duckdb_expression;
use std::fmt::{self, Display};

/// SQL `lhs / rhs`. Used by `AVG`, which DuckDB lowers to `sum(x) / count(x)`.
#[derive(Debug, Clone)]
pub struct Divide {
    pub left: Box<Expression>,
    pub right: Box<Expression>,
}

impl TryFrom<duckdb_expression::Function> for Divide {
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
        let right = Box::new(Expression::try_from(f.params.remove(1))?);
        let left = Box::new(Expression::try_from(f.params.remove(0))?);
        Ok(Divide { left, right })
    }
}

impl Display for Divide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} / {})", self.left, self.right)
    }
}

impl Divide {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let left_builder = self.left.compile()?;
        let right_builder = self.right.compile()?;
        Ok(Box::new(move || {
            let mut left_expr = left_builder();
            let mut right_expr = right_builder();
            Box::new(move |batch: &RecordBatch| {
                // Float division: cast both operands to Float64 so integer
                // inputs (e.g. sum/count for AVG) divide to a real quotient.
                let left = left_expr(batch);
                let right = right_expr(batch);
                let lf = arrow::compute::cast(left.as_datum().get().0, &DataType::Float64).unwrap();
                let rf =
                    arrow::compute::cast(right.as_datum().get().0, &DataType::Float64).unwrap();
                ExprResult::Array(arrow::compute::kernels::numeric::div(&lf, &rf).unwrap())
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    // `Divide` only appears inside an `AVG` lowering; a bare `SELECT a / 2`
    // projection is rejected earlier because the planner has no Float64 column
    // type. Exercise `Divide::compile` directly instead: col / 2 over [1, 2, 3].
    use super::Divide;
    use crate::expression::{Expression, Ref};
    use crate::types::Type;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Float64Type;
    use arrow_array::{ArrayRef, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn divides_to_float_quotient() {
        // As used by AVG, both operands are full-length columns (sum / count).
        let col = |idx| {
            Box::new(Expression::Ref(Ref {
                column_idx: idx,
                return_type: Type::Int64,
                name: None,
            }))
        };
        let divide = Divide {
            left: col(0),
            right: col(1),
        };

        let mut eval = divide.compile().unwrap()();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("n", DataType::Int64, false),
                Field::new("d", DataType::Int64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1i64, 2, 3])) as ArrayRef,
                Arc::new(Int64Array::from(vec![2i64, 2, 2])) as ArrayRef,
            ],
        )
        .unwrap();

        let result = eval(&batch);
        let (arr, _) = result.as_datum().get();
        let quotients = arr.as_primitive::<Float64Type>();
        assert_eq!(
            (0..quotients.len())
                .map(|i| quotients.value(i))
                .collect::<Vec<_>>(),
            vec![0.5, 1.0, 1.5]
        );
    }
}
