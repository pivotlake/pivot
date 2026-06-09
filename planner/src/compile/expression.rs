//! Per-expression compile impls.
//!
//! Each variant of [`Expression`] compiles into an
//! [`ExprFn`] — a builder closure the filter/project operators
//! in `dispatch` can call.
use crate::compile::{Error, ExprEvalFn, ExprFn, ExprResult, stateless_expr};
use crate::expression::{
    Between, Case, Compare, CompareType, Contains, DateTrunc, Divide, Expression, Function, Ref,
};
use crate::types::Type;
use arrow::compute::kernels::boolean::and;
use arrow::compute::kernels::zip::zip;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, BooleanArray, Datum, Int64Array, RecordBatch, Scalar};
use arrow_ord::cmp::{eq, gt, gt_eq, lt, lt_eq, neq};
use arrow_schema::{ArrowError, DataType};
use dispatch::Contains as DispatchContains;
use std::sync::Arc;

/// Signature shared by arrow's scalar comparison kernels.
type CmpKernel = fn(&dyn Datum, &dyn Datum) -> std::result::Result<BooleanArray, ArrowError>;

/// Run a comparison kernel, coercing both operands to Int64 when their data
/// types differ. The declared logical type (e.g. DATE) and the physical parquet
/// array (e.g. UInt16 day counts) can diverge, and arrow's kernels require
/// matching types; Int64 is a safe common type for every integer/date/timestamp
/// column we compare.
fn compare_coerced(left: &dyn Datum, right: &dyn Datum, kernel: CmpKernel) -> BooleanArray {
    let (la, l_scalar) = left.get();
    let (ra, r_scalar) = right.get();
    if la.data_type() == ra.data_type() {
        kernel(left, right).unwrap()
    } else {
        let lc = arrow::compute::cast(la, &DataType::Int64).unwrap();
        let rc = arrow::compute::cast(ra, &DataType::Int64).unwrap();
        let ld: Box<dyn Datum> = if l_scalar {
            Box::new(Scalar::new(lc))
        } else {
            Box::new(lc)
        };
        let rd: Box<dyn Datum> = if r_scalar {
            Box::new(Scalar::new(rc))
        } else {
            Box::new(rc)
        };
        kernel(ld.as_ref(), rd.as_ref()).unwrap()
    }
}

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
        let kernel: CmpKernel = match self.compare_type {
            CompareType::Equal => eq,
            CompareType::NotEqual => neq,
            CompareType::Less => lt,
            CompareType::Greater => gt,
            CompareType::LessEqual => lt_eq,
            CompareType::GreaterEqual => gt_eq,
        };
        let left_builder = self.left.compile()?;
        let right_builder = self.right.compile()?;
        Ok(Box::new(move || {
            let mut left_expr = left_builder();
            let mut right_expr = right_builder();
            Box::new(move |batch: &RecordBatch| {
                let left = left_expr(batch);
                let right = right_expr(batch);
                ExprResult::Array(Arc::new(compare_coerced(
                    left.as_datum(),
                    right.as_datum(),
                    kernel,
                )) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

impl Between {
    pub fn compile(&self) -> Result<ExprFn, Error> {
        // `input BETWEEN lower AND upper` == `input >= lower AND input <= upper`
        // (the inclusive/exclusive flags swap >=/> and <=/<).
        let lower_kernel: CmpKernel = if self.lower_inclusive { gt_eq } else { gt };
        let upper_kernel: CmpKernel = if self.upper_inclusive { lt_eq } else { lt };
        let input_builder = self.input.compile()?;
        let lower_builder = self.lower.compile()?;
        let upper_builder = self.upper.compile()?;
        Ok(Box::new(move || {
            let mut input_expr = input_builder();
            let mut lower_expr = lower_builder();
            let mut upper_expr = upper_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let lower = lower_expr(batch);
                let upper = upper_expr(batch);
                let ge = compare_coerced(input.as_datum(), lower.as_datum(), lower_kernel);
                let le = compare_coerced(input.as_datum(), upper.as_datum(), upper_kernel);
                ExprResult::Array(Arc::new(and(&ge, &le).unwrap()) as ArrayRef)
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

impl Case {
    pub fn compile(&self) -> Result<ExprFn, Error> {
        // Compile the ELSE branch and each (WHEN, THEN) arm once. Per batch we
        // start from the ELSE value and fold the arms back-to-front with arrow's
        // `zip` (selecting `then` where the WHEN mask is true, else the running
        // result). Applying the *first* arm last makes it win on overlap, giving
        // SQL's first-match-wins CASE semantics. A NULL WHEN reads as false.
        //
        // `zip` requires `then` and the running result share a data type; DuckDB
        // unifies all branch types when binding the CASE, so they always do.
        let else_builder = self.else_expr.compile()?;
        let arm_builders = self
            .checks
            .iter()
            .map(|c| Ok((c.when.compile()?, c.then.compile()?)))
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Box::new(move || {
            let mut else_eval = else_builder();
            let mut arm_evals: Vec<(ExprEvalFn, ExprEvalFn)> = arm_builders
                .iter()
                .map(|(when, then)| (when(), then()))
                .collect();
            Box::new(move |batch: &RecordBatch| {
                let mut result = else_eval(batch);
                for (when_eval, then_eval) in arm_evals.iter_mut().rev() {
                    let when = when_eval(batch);
                    let (when_arr, _) = when.as_datum().get();
                    let mask = when_arr.as_any().downcast_ref::<BooleanArray>().unwrap();
                    let then = then_eval(batch);
                    let zipped = zip(mask, then.as_datum(), result.as_datum()).unwrap();
                    result = ExprResult::Array(zipped);
                }
                result
            }) as ExprEvalFn
        }))
    }
}

impl Divide {
    pub fn compile(&self) -> Result<ExprFn, Error> {
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

impl DateTrunc {
    pub fn compile(&self) -> Result<ExprFn, Error> {
        // EventTime is stored as Int64 epoch *seconds*, so truncating to a unit
        // is flooring to that many seconds. `M = (t / secs) * secs`.
        let secs: i64 = match self.unit.as_str() {
            "second" => 1,
            "minute" => 60,
            "hour" => 3600,
            "day" => 86400,
            other => {
                return Err(Error::UnsupportedExpression(Expression::Function(
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

impl Function {
    pub fn compile(&self) -> Result<ExprFn, Error> {
        match self {
            Function::Contains(c) => c.compile(),
            Function::Divide(d) => d.compile(),
            Function::DateTrunc(dt) => dt.compile(),
            // `drop_cache()` evicts pivot's file cache as a side effect, then
            // returns the regions dropped. Evaluated over the single `DummyScan`
            // row on a worker thread (where `memory_ctx` is valid), so the
            // eviction happens exactly once; the returned array matches the
            // (one-row) batch.
            Function::DropCache => Ok(stateless_expr(|batch: &RecordBatch| {
                let evicted = dispatch::memory_ctx().file_cache().clear() as i64;
                ExprResult::Array(Arc::new(Int64Array::from(vec![evicted; batch.num_rows()])))
            })),
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
            Expression::Between(b) => b.compile(),
            Expression::Case(c) => c.compile(),
            _ => Err(Error::UnsupportedExpression(self.clone())),
        }
    }
}
