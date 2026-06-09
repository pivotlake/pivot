//! Per-expression compile impls.
//!
//! Each variant of [`Expression`] compiles into an
//! [`ExprFn`] — a builder closure the filter/project operators
//! in `dispatch` can call.
use crate::compile::{Error, ExprEvalFn, ExprFn, ExprResult, stateless_expr};
use crate::expression::{
    Between, Case, Compare, CompareType, Conjunction, ConjunctionOp, Contains, DatePart, DateTrunc,
    Divide, Expression, Function, InList, Ref,
};
use crate::types::Type;
use arrow::compute::kernels::boolean::{and, or};
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

impl InList {
    pub fn compile(&self) -> Result<ExprFn, Error> {
        // `x IN (a, b, …)` is the disjunction `x = a OR x = b OR …`. We compile
        // the tested expression and every list value once, then per batch
        // OR-reduce the equality masks. `compare_coerced` aligns differing
        // physical types (e.g. an Int16 column against Int32 literals) the same
        // way `Compare` does, so an IN over any integer/string column works.
        let input_builder = self.input.compile()?;
        let value_builders = self
            .values
            .iter()
            .map(|v| v.compile())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Box::new(move || {
            let mut input_expr = input_builder();
            let mut value_exprs: Vec<ExprEvalFn> = value_builders.iter().map(|b| b()).collect();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let mask = value_exprs
                    .iter_mut()
                    .map(|v| {
                        let value = v(batch);
                        compare_coerced(input.as_datum(), value.as_datum(), eq)
                    })
                    .reduce(|left, right| or(&left, &right).unwrap())
                    // An empty list (`IN ()`) matches nothing; DuckDB folds this
                    // away before planning, so this is only a defensive fallback.
                    .unwrap_or_else(|| BooleanArray::from(vec![false; batch.num_rows()]));
                ExprResult::Array(Arc::new(mask) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

impl Conjunction {
    pub fn compile(&self) -> Result<ExprFn, Error> {
        // Compile each child predicate once, then per batch reduce their boolean
        // masks with the conjunction's kernel (`AND`/`OR`).
        type BoolKernel = fn(&BooleanArray, &BooleanArray) -> Result<BooleanArray, ArrowError>;
        let kernel: BoolKernel = match self.op {
            ConjunctionOp::And => and,
            ConjunctionOp::Or => or,
        };
        let child_builders = self
            .children
            .iter()
            .map(|c| c.compile())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Box::new(move || {
            let mut child_exprs: Vec<ExprEvalFn> = child_builders.iter().map(|b| b()).collect();
            Box::new(move |batch: &RecordBatch| {
                let mask = child_exprs
                    .iter_mut()
                    .map(|c| {
                        let result = c(batch);
                        let (arr, _) = result.as_datum().get();
                        arr.as_any().downcast_ref::<BooleanArray>().unwrap().clone()
                    })
                    .reduce(|left, right| kernel(&left, &right).unwrap())
                    .expect("conjunction always has at least two children");
                ExprResult::Array(Arc::new(mask) as ArrayRef)
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

/// Seconds in a day / hour, for the time-of-day parts.
const SECS_PER_DAY: i64 = 86_400;
const SECS_PER_HOUR: i64 = 3_600;

/// Convert a day count relative to the Unix epoch (1970-01-01) into a
/// `(year, month, day)` civil date. Howard Hinnant's `civil_from_days`
/// (<http://howardhinnant.github.io/date_algorithms.html>); valid for the full
/// proleptic Gregorian range, including negative (pre-epoch) day counts.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // day of era, [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (y + i64::from(m <= 2), m, d)
}

/// Inverse of [`civil_from_days`]: the epoch-relative day count for a civil
/// date. Used to derive day-of-year and ISO week.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Number of ISO 8601 weeks (52 or 53) in the given year. A year has 53 weeks
/// iff it starts on a Thursday, or it is a leap year starting on a Wednesday.
fn iso_weeks_in_year(y: i64) -> i64 {
    let p = |y: i64| (y + y.div_euclid(4) - y.div_euclid(100) + y.div_euclid(400)).rem_euclid(7);
    if p(y) == 4 || p(y - 1) == 3 { 53 } else { 52 }
}

/// ISO 8601 week-of-year (1–53) for an epoch-relative day count. Week 1 is the
/// week containing the year's first Thursday; days before it belong to the
/// prior year's last week, and the year's tail can roll into week 1.
fn iso_week(days: i64) -> i64 {
    let (y, ..) = civil_from_days(days);
    let ordinal = days - days_from_civil(y, 1, 1) + 1;
    let iso_dow = (days + 3).rem_euclid(7) + 1;
    let week = (ordinal - iso_dow + 10).div_euclid(7);
    if week < 1 {
        iso_weeks_in_year(y - 1)
    } else if week > iso_weeks_in_year(y) {
        1
    } else {
        week
    }
}

impl DatePart {
    pub fn compile(&self) -> Result<ExprFn, Error> {
        // EventTime is stored as Int64 epoch *seconds* (UTC), so every part is a
        // pure integer computation. Euclidean div/rem keep the time-of-day and
        // calendar fields well-defined for pre-epoch (negative) timestamps,
        // matching DuckDB's `extract(<part> FROM ...)`.
        let kind = self.kind;
        let source_builder = self.source.compile()?;
        Ok(Box::new(move || {
            let mut source_expr = source_builder();
            Box::new(move |batch: &RecordBatch| {
                let src = source_expr(batch);
                let (arr, _) = src.as_datum().get();
                let i64arr = arrow::compute::cast(arr, &DataType::Int64).unwrap();
                let vals = i64arr.as_primitive::<Int64Type>();
                // Dispatch on the part ONCE per batch, then run a single
                // monomorphic, branch-free row loop per arm — so e.g. `minute`
                // compiles to exactly its two-op loop with no per-row `kind`
                // test (no reliance on the optimizer hoisting a loop-invariant
                // branch). The civil-date parts share `civil_from_days`.
                use crate::expression::DatePartKind::*;
                macro_rules! map_part {
                    ($f:expr) => {{
                        let out: Int64Array = vals.iter().map(|v| v.map($f)).collect();
                        out
                    }};
                }
                let day = |t: i64| t.div_euclid(SECS_PER_DAY);
                let out = match kind {
                    Epoch => map_part!(|t: i64| t),
                    Second => map_part!(|t: i64| t.rem_euclid(60)),
                    Millisecond => map_part!(|t: i64| t.rem_euclid(60) * 1_000),
                    Microsecond => map_part!(|t: i64| t.rem_euclid(60) * 1_000_000),
                    Minute => map_part!(|t: i64| t.div_euclid(60).rem_euclid(60)),
                    Hour => map_part!(|t: i64| t.div_euclid(SECS_PER_HOUR).rem_euclid(24)),
                    // 0 = Sunday … 6 = Saturday. The epoch day (1970-01-01) was a
                    // Thursday (4), so `(days + 4) mod 7` rebases to Sunday = 0.
                    DayOfWeek => map_part!(|t: i64| (day(t) + 4).rem_euclid(7)),
                    // 1 = Monday … 7 = Sunday.
                    IsoDayOfWeek => map_part!(|t: i64| (day(t) + 3).rem_euclid(7) + 1),
                    Day => map_part!(|t: i64| civil_from_days(day(t)).2),
                    Month => map_part!(|t: i64| civil_from_days(day(t)).1),
                    Quarter => map_part!(|t: i64| (civil_from_days(day(t)).1 - 1) / 3 + 1),
                    Year => map_part!(|t: i64| civil_from_days(day(t)).0),
                    Decade => map_part!(|t: i64| civil_from_days(day(t)).0.div_euclid(10)),
                    Century => {
                        map_part!(|t: i64| (civil_from_days(day(t)).0 - 1).div_euclid(100) + 1)
                    }
                    Millennium => {
                        map_part!(|t: i64| (civil_from_days(day(t)).0 - 1).div_euclid(1000) + 1)
                    }
                    DayOfYear => map_part!(|t: i64| {
                        let d = day(t);
                        d - days_from_civil(civil_from_days(d).0, 1, 1) + 1
                    }),
                    Week => map_part!(|t: i64| iso_week(day(t))),
                };
                ExprResult::Array(Arc::new(out) as ArrayRef)
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
            Function::DatePart(d) => d.compile(),
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
            Expression::InList(i) => i.compile(),
            Expression::Conjunction(c) => c.compile(),
            Expression::Case(c) => c.compile(),
            _ => Err(Error::UnsupportedExpression(self.clone())),
        }
    }
}
