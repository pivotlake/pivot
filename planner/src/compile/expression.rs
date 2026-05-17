//! Per-expression compile impls.
//!
//! Each variant of [`Expression`] compiles into an
//! [`ExprFn`] — a builder closure the filter/project operators
//! in `dispatch` can call.
use crate::compile::{Error, ExprEvalFn, ExprFn, ExprResult, stateless_expr};
use crate::expression::{Compare, CompareType, Contains, Expression, Function, Ref};
use crate::types::Type;
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, Datum, RecordBatch};
use arrow_ord::cmp::{eq, neq};
use dispatch::Contains as DispatchContains;
use std::sync::Arc;

impl Ref {
    pub fn compile(&self) -> Result<ExprFn, Error> {
        let column_idx = self.column_idx;
        Ok(stateless_expr(move |batch: &RecordBatch| {
            ExprResult::Array(batch.column(column_idx).clone())
        }))
    }
}

impl Compare {
    pub fn compile(&self) -> Result<ExprFn, Error> {
        let kernel: fn(&dyn Datum, &dyn Datum) -> _ = match self.compare_type {
            CompareType::Equal => eq,
            CompareType::NotEqual => neq,
        };
        let left_builder = self.left.compile()?;
        let right_builder = self.right.compile()?;
        Ok(Box::new(move || {
            let mut left_expr = left_builder();
            let mut right_expr = right_builder();
            Box::new(move |batch: &RecordBatch| {
                let left = left_expr(batch);
                let right = right_expr(batch);
                ExprResult::Array(
                    Arc::new(kernel(left.as_datum(), right.as_datum()).unwrap()) as ArrayRef
                )
            }) as ExprEvalFn
        }))
    }
}

impl Contains {
    pub fn compile(&self) -> Result<ExprFn, Error> {
        match self.haystack.as_ref() {
            Expression::Ref(r) if r.return_type == Type::Utf8 => {}
            expr => {
                return Err(Error::UnsupportedExpressionForContainsHaystack(
                    expr.clone(),
                ));
            }
        }

        let haystack_builder = self.haystack.compile()?;

        // Extract the needle string from the constant expression
        let needle_str: String = match self.needle.as_ref() {
            Expression::Constant(scalar) => {
                let (arr, _) = scalar.get();
                arr.as_string_view_opt()
                    .ok_or_else(|| Error::FailedToDowncastScalarIntoString(scalar.clone()))?
                    .value(0)
                    .to_string()
            }
            expr => return Err(Error::UnsupportedExpressionForContainsNeedle(expr.clone())),
        };
        Ok(Box::new(move || {
            let mut haystack_expr = haystack_builder();
            let mut contains = DispatchContains::new(&needle_str);
            Box::new(move |batch: &RecordBatch| {
                let haystack = haystack_expr(batch);
                let (arr, _) = haystack.as_datum().get();
                let col = arr
                    .as_any()
                    .downcast_ref::<arrow_array::StringViewArray>()
                    .unwrap();
                ExprResult::Array(Arc::new(contains.run(col)) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

impl Function {
    pub fn compile(&self) -> Result<ExprFn, Error> {
        match self {
            Function::Contains(c) => c.compile(),
        }
    }
}

impl Expression {
    pub fn compile(&self) -> Result<ExprFn, Error> {
        match self {
            Expression::Ref(r) => r.compile(),
            Expression::Compare(c) => c.compile(),
            Expression::Constant(c) => {
                let scalar = c.clone();
                Ok(stateless_expr(move |_batch: &RecordBatch| {
                    ExprResult::Scalar(scalar.clone())
                }))
            }
            Expression::Function(f) => f.compile(),
            _ => Err(Error::UnsupportedExpression(self.clone())),
        }
    }
}
