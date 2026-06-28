//! [`Cast`] — `CAST(source AS target)`.
//!
//! DuckDB inserts casts to coerce operands to a common type and for explicit
//! user casts; pivot honors them by casting the source to the target type's
//! arrow [`DataType`] at runtime. A temporal target casts to its real arrow type
//! (`Date32`/`Timestamp`), every other to its physical storage type.

use super::{Error, Expression};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::{self, Type};
use arrow_array::{RecordBatch, Scalar};
use arrow_schema::DataType;
use duckdb_planner::expression as duckdb_expression;
use std::fmt::{self, Display};

/// A `CAST(source AS target)`. `target` is the pivot type (used for the result
/// type when grouping/typing); `target_arrow` is the arrow type the value is
/// cast into.
#[derive(Debug, Clone)]
pub struct Cast {
    pub target: Type,
    target_arrow: DataType,
    source: Box<Expression>,
}

impl TryFrom<duckdb_expression::Cast> for Cast {
    type Error = Error;
    fn try_from(c: duckdb_expression::Cast) -> Result<Self, Self::Error> {
        let target = types::type_from_logical(c.target_type)?;
        let target_arrow = types::physical_arrow_type(&target);
        let source = Box::new(Expression::try_from(*c.child)?);
        Ok(Cast {
            target,
            target_arrow,
            source,
        })
    }
}

impl Display for Cast {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cast({} as {})", self.source, self.target)
    }
}

impl Cast {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let target = self.target_arrow.clone();
        let source_builder = self.source.compile()?;
        Ok(Box::new(move || {
            let target = target.clone();
            let mut source_expr = source_builder();
            Box::new(move |batch: &RecordBatch| {
                let src = source_expr(batch);
                let (arr, is_scalar) = src.as_datum().get();
                let out = arrow::compute::cast(arr, &target).expect("cast source to target type");
                if is_scalar {
                    ExprResult::Scalar(Scalar::new(out))
                } else {
                    ExprResult::Array(out)
                }
            }) as ExprEvalFn
        }))
    }
}
