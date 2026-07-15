//! [`IsNull`] — `expr IS [NOT] NULL`.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow::compute::{is_not_null, is_null};
use arrow_array::{ArrayRef, RecordBatch};
use std::fmt::{self, Display};
use std::sync::Arc;

/// `expr IS NULL` (or, with `negated`, `expr IS NOT NULL`). DuckDB lowers both
/// as a `BoundOperatorExpression` with a single child.
#[derive(Debug, Clone)]
pub struct IsNull {
    pub input: Box<Expression>,
    pub negated: bool,
}

impl Display for IsNull {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = if self.negated {
            "IS NOT NULL"
        } else {
            "IS NULL"
        };
        write!(f, "({} {op})", self.input)
    }
}

impl IsNull {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let input_builder = self.input.compile()?;
        let negated = self.negated;
        Ok(Box::new(move || {
            let mut input_expr = input_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (arr, _) = input.as_datum().get();
                let mask = if negated {
                    is_not_null(arr)
                } else {
                    is_null(arr)
                }
                .expect("null-test kernel accepts any array");
                ExprResult::Array(Arc::new(mask) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}
