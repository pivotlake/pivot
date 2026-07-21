//! Constructs Pivot [`Expression`]s by walking DuckDB's borrowed bound-expression
//! handles. This is the expression half of the plan build (the operator half is
//! in [`operator`](super::operator)): every expression kind's `from_handle` is
//! implemented here, so the `crate::expression` modules hold only Pivot IR (the
//! AST types, their `Display`, and their `compile`).
//!
//! [`Expression::from_handle`] is the single recursive entry point; it dispatches
//! each DuckDB expression variant to its kind's `from_handle`. Scalar-function
//! calls go through [`Function::from_handle`], which routes on the function name.

use arrow_array::Datum;
use arrow_array::cast::AsArray;
use arrow_schema::{DataType, TimeUnit};
use duckdb_planner::duckdb_bridge::duckdb_types::ExpressionType;
use duckdb_planner::handle::{
    AggregateFunc as AggregateFuncHandle, Between as BetweenHandle, Case as CaseHandle,
    Cast as CastHandle, Compare as CompareHandle, Conjunction as ConjunctionHandle,
    Expression as DuckExpression, Function as FunctionHandle, InList as InListHandle,
    Not as NotHandle, Parameter as ParameterHandle, Ref as RefHandle,
};
use duckdb_planner::{Expr, LogicalTypeId, ScalarValue};

use crate::expression::{
    AggregateFunc, Arithmetic, ArithmeticOp, Between, Case, CaseCheck, Cast, Compare, Conjunction,
    ConjunctionOp, Contains, CountStar, DatePart, DatePartKind, DateTrunc, Divide, Error,
    Expression, Function, InList, IntervalArithmetic, Length, Not, NumericAggregate, ParameterRef,
    Prefix, Ref, RegexpJitReplace, RegexpReplace, TemporalConvert, VariantGet,
};
use crate::types::{Type, build_scalar_value, physical_arrow_type, type_from_logical};

impl Expression {
    /// Build a Pivot [`Expression`] (and its whole subtree) from a borrowed DuckDB
    /// bound expression handle. The single recursive entry point the operator walk
    /// and the per-kind builders share; each DuckDB expression variant dispatches
    /// to its kind's `from_handle`.
    pub(crate) fn from_handle(e: Expr<'_>) -> Result<Expression, Error> {
        Ok(match e.expression() {
            DuckExpression::Ref(r) => Expression::Ref(Ref::from_handle(r)?),
            DuckExpression::Compare(c) => Expression::Compare(Compare::from_handle(c)?),
            DuckExpression::Between(b) => Expression::Between(Between::from_handle(b)?),
            DuckExpression::Constant(c) => Expression::Constant(build_scalar_value(c.value())?),
            DuckExpression::AggregateFunc(a) => {
                Expression::AggregateFunc(AggregateFunc::from_handle(a)?)
            }
            DuckExpression::Function(f) => Expression::Function(Function::from_handle(f)?),
            DuckExpression::InList(i) => Expression::InList(InList::from_handle(i)?),
            DuckExpression::Conjunction(c) => Expression::Conjunction(Conjunction::from_handle(c)?),
            DuckExpression::Case(c) => Expression::Case(Case::from_handle(c)?),
            DuckExpression::Not(n) => Expression::Not(Not::from_handle(n)?),
            DuckExpression::Cast(c) => Cast::from_handle(c)?,
            DuckExpression::Parameter(p) => Expression::Parameter(ParameterRef::from_handle(p)?),
            DuckExpression::Unsupported(t) => return Err(Error::UnsupportedExpressionType(t)),
        })
    }
}

impl Ref {
    pub(crate) fn from_handle(view: RefHandle<'_>) -> Result<Ref, Error> {
        Ok(Ref {
            column_idx: view.column_index(),
            return_type: type_from_logical(view.return_type())?,
            // Drop DuckDB's positional aliases (e.g. "0" for an unnamed computed
            // group key): an all-digit name carries no more than the index does
            // and reads as a constant in a plan dump, so fall back to `#idx`.
            name: view
                .alias()
                .filter(|n| !n.is_empty() && !n.bytes().all(|b| b.is_ascii_digit())),
        })
    }
}

impl ParameterRef {
    pub(crate) fn from_handle(view: ParameterHandle<'_>) -> Result<ParameterRef, Error> {
        Ok(ParameterRef {
            index: view
                .index()
                .map_err(|e| Error::UnsupportedParameter(e.to_string()))?,
            // DuckDB resolves a parameter's type where it can (e.g. an INSERT
            // target column), but leaves one used only in a comparison untyped
            // (an unknown logical type that doesn't map). Such a parameter's type
            // comes from the client's Parse declaration at bind time, so leave it
            // `None` here rather than guessing.
            ty: type_from_logical(view.return_type()).ok(),
        })
    }
}

impl Compare {
    pub(crate) fn from_handle(view: CompareHandle<'_>) -> Result<Compare, Error> {
        Ok(Compare {
            left: Box::new(Expression::from_handle(view.left())?),
            right: Box::new(Expression::from_handle(view.right())?),
            compare_type: view.comparison_type().try_into()?,
            return_type: type_from_logical(view.return_type())?,
        })
    }
}

impl Between {
    pub(crate) fn from_handle(view: BetweenHandle<'_>) -> Result<Between, Error> {
        Ok(Between {
            input: Box::new(Expression::from_handle(view.input())?),
            lower: Box::new(Expression::from_handle(view.lower())?),
            upper: Box::new(Expression::from_handle(view.upper())?),
            lower_inclusive: view.lower_inclusive(),
            upper_inclusive: view.upper_inclusive(),
        })
    }
}

impl Not {
    pub(crate) fn from_handle(view: NotHandle<'_>) -> Result<Not, Error> {
        Ok(Not {
            input: Box::new(Expression::from_handle(view.input())?),
        })
    }
}

impl Conjunction {
    pub(crate) fn from_handle(view: ConjunctionHandle<'_>) -> Result<Conjunction, Error> {
        let op = if view.conjunction_type() == ExpressionType::CONJUNCTION_OR {
            ConjunctionOp::Or
        } else {
            ConjunctionOp::And
        };
        Ok(Conjunction {
            op,
            children: view
                .children()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl InList {
    pub(crate) fn from_handle(view: InListHandle<'_>) -> Result<InList, Error> {
        // Child 0 is the tested expression, children 1.. are the list values.
        let children: Vec<Expr<'_>> = view.children().collect();
        Ok(InList {
            input: Box::new(Expression::from_handle(children[0])?),
            values: children[1..]
                .iter()
                .copied()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl Case {
    pub(crate) fn from_handle(view: CaseHandle<'_>) -> Result<Case, Error> {
        let checks = view
            .checks()
            .map(|arm| {
                Ok(CaseCheck {
                    when: Box::new(Expression::from_handle(arm.when)?),
                    then: Box::new(Expression::from_handle(arm.then)?),
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Case {
            checks,
            else_expr: Box::new(Expression::from_handle(view.else_expr())?),
        })
    }
}

impl Cast {
    pub(crate) fn from_handle(view: CastHandle<'_>) -> Result<Expression, Error> {
        // A `BoundCastExpression`'s target is its own result type.
        let target = type_from_logical(view.return_type())?;
        let source = Expression::from_handle(view.child())?;

        // Fold a cast and its `->` chain into one typed read. This lets the
        // kernel read a shredded leaf without building intermediate variants.
        if matches!(source.result_type(), Ok(Type::Variant)) {
            if !VariantGet::supports_cast_to(&target) {
                return Err(Error::UnsupportedScalarFunction(format!(
                    "cannot cast a VARIANT to {target}"
                )));
            }
            let (input, path) = collapse_extractions(source);
            return Ok(Expression::Function(Function::VariantGet(VariantGet {
                input: Box::new(input),
                path,
                as_type: Some(target),
            })));
        }

        Ok(Expression::Cast(Cast {
            target_arrow: physical_arrow_type(&target),
            target,
            source: Box::new(source),
        }))
    }
}

/// Peel a chain of bare `->` extractions, returning the base expression and the
/// concatenated path (empty when `e` isn't an extraction, e.g. a cast directly
/// over a variant column).
fn collapse_extractions(e: Expression) -> (Expression, Vec<String>) {
    match e {
        Expression::Function(Function::VariantGet(extraction)) if extraction.as_type.is_none() => {
            let (input, mut path) = collapse_extractions(*extraction.input);
            path.extend(extraction.path);
            (input, path)
        }
        other => (other, Vec::new()),
    }
}

impl AggregateFunc {
    pub(crate) fn from_handle(view: AggregateFuncHandle<'_>) -> Result<AggregateFunc, Error> {
        let function = view.name();
        let distinct = view.distinct();
        let return_type = type_from_logical(view.return_type())?;
        let params = view
            .children()
            .map(Expression::from_handle)
            .collect::<Result<Vec<_>, _>>()?;

        // DISTINCT is only supported for `COUNT` so far; reject `SUM(DISTINCT)`
        // etc. rather than silently computing the non-distinct aggregate.
        if distinct && function != "count" {
            return Err(Error::UnsupportedAggregateFunction(format!(
                "DISTINCT {function}"
            )));
        }

        match function.as_str() {
            "count_star" => Ok(AggregateFunc::CountStar(CountStar {
                params,
                return_type,
            })),
            "sum" => Ok(AggregateFunc::Sum(numeric_aggregate(
                params,
                return_type,
                function,
            )?)),
            "avg" => Ok(AggregateFunc::Avg(numeric_aggregate(
                params,
                return_type,
                function,
            )?)),
            "min" => Ok(AggregateFunc::Min(numeric_aggregate(
                params,
                return_type,
                function,
            )?)),
            "max" => Ok(AggregateFunc::Max(numeric_aggregate(
                params,
                return_type,
                function,
            )?)),
            "count" if distinct => Ok(AggregateFunc::CountDistinct(numeric_aggregate(
                params,
                return_type,
                function,
            )?)),
            "count" => Ok(AggregateFunc::Count(numeric_aggregate(
                params,
                return_type,
                function,
            )?)),
            _ => Err(Error::UnsupportedAggregateFunction(function)),
        }
    }
}

/// Build a single-argument numeric aggregate's payload, validating the arity.
fn numeric_aggregate(
    params: Vec<Expression>,
    return_type: Type,
    function: String,
) -> Result<NumericAggregate, Error> {
    if params.len() != 1 {
        return Err(Error::InvalidParameterCount {
            function,
            expected: 1,
            actual: params.len(),
        });
    }
    Ok(NumericAggregate {
        argument: Box::new(params.into_iter().next().unwrap()),
        return_type,
    })
}

impl Function {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<Function, Error> {
        let name = func.name();
        match name.as_str() {
            "contains" => Ok(Function::Contains(Contains::from_handle(func)?)),
            // DuckDB's optimizer rewrites `LIKE 'foo%'` into `prefix(col, 'foo')`.
            "prefix" => Ok(Function::Prefix(Prefix::from_handle(func)?)),
            // `date`/`timestamp` ± `INTERVAL` carries an INTERVAL constant operand;
            // plain numeric `+`/`-` does not and stays `Arithmetic`.
            "+" | "-" => match func.children().position(|p| {
                matches!(p.expression(), DuckExpression::Constant(c) if c.return_type() == LogicalTypeId::INTERVAL)
            }) {
                Some(idx) => Ok(Function::IntervalArithmetic(
                    IntervalArithmetic::from_handle(func, idx)?,
                )),
                None => Ok(Function::Arithmetic(Arithmetic::from_handle(func)?)),
            },
            "*" => Ok(Function::Arithmetic(Arithmetic::from_handle(func)?)),
            "length" | "strlen" | "len" => Ok(Function::Length(Length::from_handle(func)?)),
            "regexp_replace" => Ok(Function::RegexpReplace(RegexpReplace::from_handle(func)?)),
            "regexp_jit_replace" => Ok(Function::RegexpJitReplace(RegexpJitReplace::from_handle(
                func,
            )?)),
            "/" => Ok(Function::Divide(Divide::from_handle(func)?)),
            "date_trunc" => Ok(Function::DateTrunc(DateTrunc::from_handle(func)?)),
            "make_date" => Ok(Function::TemporalConvert(TemporalConvert::make_date(func)?)),
            "make_timestamp" => Ok(Function::TemporalConvert(TemporalConvert::make_timestamp(
                func,
            )?)),
            // Variant field access: DuckDB's binder rewrites `d.age` to
            // `variant_extract` (its native VARIANT function) and `d->'age'` to
            // `json_extract` (which resolves against pivot's registry). Both are
            // one extracted field.
            "variant_extract" | "json_extract" => {
                Ok(Function::VariantGet(VariantGet::from_handle(func)?))
            }
            "drop_cache" => {
                function_args(func, 0)?;
                Ok(Function::DropCache)
            }
            "now" => {
                function_args(func, 0)?;
                Ok(Function::Now)
            }
            // `extract(<part> FROM ts)` lowers to a function named after the part
            // (`minute`, `year`, `dayofweek`, …).
            _ => match DatePartKind::from_function_name(&name) {
                Some(kind) => Ok(Function::DatePart(DatePart::from_handle(kind, func)?)),
                None => Err(Error::UnsupportedScalarFunction(name)),
            },
        }
    }
}

impl VariantGet {
    /// A bare `doc->'key'`: one extracted field, yielding the sub-variant. A
    /// cast above it types the read ([`Cast::from_handle`] fuses the chain).
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<VariantGet, Error> {
        let params = function_args(func, 2)?;
        let field = constant_string(Expression::from_handle(params[1])?, "->: field name")?;
        Ok(VariantGet {
            input: Box::new(Expression::from_handle(params[0])?),
            path: vec![field],
            as_type: None,
        })
    }
}

impl Contains {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<Contains, Error> {
        let params = function_args(func, 2)?;
        Ok(Contains {
            needle: Box::new(Expression::from_handle(params[1])?),
            haystack: Box::new(Expression::from_handle(params[0])?),
        })
    }
}

impl Prefix {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<Prefix, Error> {
        let params = function_args(func, 2)?;
        Ok(Prefix {
            haystack: Box::new(Expression::from_handle(params[0])?),
            prefix: Box::new(Expression::from_handle(params[1])?),
        })
    }
}

impl Arithmetic {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<Arithmetic, Error> {
        let op = match func.name().as_str() {
            "+" => ArithmeticOp::Add,
            "-" => ArithmeticOp::Sub,
            "*" => ArithmeticOp::Mul,
            _ => return Err(Error::UnsupportedScalarFunction(func.name())),
        };
        // Unary forms (e.g. `-x`) bind to the same function names with one
        // parameter; only the binary forms are supported.
        let params = function_args(func, 2)?;
        Ok(Arithmetic {
            op,
            left: Box::new(Expression::from_handle(params[0])?),
            right: Box::new(Expression::from_handle(params[1])?),
            return_type: type_from_logical(func.return_type())?,
        })
    }
}

impl Divide {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<Divide, Error> {
        let params = function_args(func, 2)?;
        Ok(Divide {
            left: Box::new(Expression::from_handle(params[0])?),
            right: Box::new(Expression::from_handle(params[1])?),
            return_type: type_from_logical(func.return_type())?,
        })
    }
}

impl Length {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<Length, Error> {
        let params = function_args(func, 1)?;
        Ok(Length {
            return_type: type_from_logical(func.return_type())?,
            input: Box::new(Expression::from_handle(params[0])?),
        })
    }
}

impl DateTrunc {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<DateTrunc, Error> {
        let params = function_args(func, 2)?;
        let unit = constant_string(Expression::from_handle(params[0])?, "date_trunc: unit")?
            .to_ascii_lowercase();
        Ok(DateTrunc {
            unit,
            source: Box::new(Expression::from_handle(params[1])?),
        })
    }
}

impl DatePart {
    pub(crate) fn from_handle(
        kind: DatePartKind,
        func: FunctionHandle<'_>,
    ) -> Result<DatePart, Error> {
        let params = function_args(func, 1)?;
        Ok(DatePart {
            kind,
            source: Box::new(Expression::from_handle(params[0])?),
            return_type: Type::Int64,
        })
    }
}

impl TemporalConvert {
    /// `make_date(days)` → `Date32`.
    pub(crate) fn make_date(func: FunctionHandle<'_>) -> Result<TemporalConvert, Error> {
        Self::build(
            "make_date",
            Type::Date,
            DataType::Date32,
            DataType::Int32,
            func,
        )
    }

    /// `make_timestamp(seconds)` → `Timestamp(Second)`.
    pub(crate) fn make_timestamp(func: FunctionHandle<'_>) -> Result<TemporalConvert, Error> {
        Self::build(
            "make_timestamp",
            Type::Timestamp,
            DataType::Timestamp(TimeUnit::Second, None),
            DataType::Int64,
            func,
        )
    }

    fn build(
        name: &'static str,
        result: Type,
        target: DataType,
        via: DataType,
        func: FunctionHandle<'_>,
    ) -> Result<TemporalConvert, Error> {
        let params = function_args(func, 1)?;
        Ok(TemporalConvert {
            name,
            result,
            target,
            via,
            source: Box::new(Expression::from_handle(params[0])?),
        })
    }
}

impl IntervalArithmetic {
    /// Build a `date`/`timestamp` ± `INTERVAL` from a DuckDB `+`/`-` whose
    /// `params[interval_idx]` is an `INTERVAL` constant; the other operand is the
    /// temporal value. The caller ([`Function::from_handle`]) has already located
    /// the interval operand.
    pub(crate) fn from_handle(
        func: FunctionHandle<'_>,
        interval_idx: usize,
    ) -> Result<IntervalArithmetic, Error> {
        let params = function_args(func, 2)?;
        let op = match func.name().as_str() {
            "+" => ArithmeticOp::Add,
            "-" => ArithmeticOp::Sub,
            other => return Err(Error::UnsupportedScalarFunction(other.to_string())),
        };
        // DuckDB types the `+`/`-` itself, so its return type is the temporal
        // result (DATE for whole-day intervals, TIMESTAMP otherwise).
        let result = type_from_logical(func.return_type())?;
        let interval = match params[interval_idx].expression() {
            DuckExpression::Constant(c) => match c.value() {
                ScalarValue::Interval {
                    months,
                    days,
                    micros,
                } => IntervalParts {
                    months,
                    days,
                    micros,
                },
                _ => unreachable!("caller selected an interval constant"),
            },
            _ => unreachable!("caller selected an interval constant"),
        };
        let offset = interval_offset(&interval, &result)?;
        Ok(IntervalArithmetic {
            op,
            operand: Box::new(Expression::from_handle(params[1 - interval_idx])?),
            offset,
            result,
        })
    }
}

/// A DuckDB `INTERVAL`'s three independent components (months are
/// calendar-variable, so they stay separate from the fixed day/microsecond parts).
struct IntervalParts {
    months: i32,
    days: i32,
    micros: i64,
}

const SECS_PER_DAY: i64 = 86_400;
const MICROS_PER_SEC: i64 = 1_000_000;

/// Reduce an interval to a constant offset in the result's unit. Months/years are
/// calendar-variable (rejected); a `DATE` only takes whole-day intervals, a
/// `TIMESTAMP` takes days plus a sub-day part truncated to whole seconds (pivot
/// stores timestamps at second granularity, so any sub-second part is dropped).
fn interval_offset(interval: &IntervalParts, result: &Type) -> Result<i64, Error> {
    if interval.months != 0 {
        return Err(Error::UnsupportedInterval(
            "month and year intervals require calendar arithmetic".to_string(),
        ));
    }
    match result {
        Type::Date => {
            if interval.micros != 0 {
                return Err(Error::UnsupportedInterval(
                    "sub-day interval applied to a DATE".to_string(),
                ));
            }
            Ok(interval.days as i64)
        }
        Type::Timestamp => {
            Ok(interval.days as i64 * SECS_PER_DAY + interval.micros / MICROS_PER_SEC)
        }
        other => Err(Error::UnsupportedInterval(format!(
            "interval arithmetic on a non-temporal {other}"
        ))),
    }
}

/// Extract the `(input, pattern, replacement)` shared by `regexp_replace` and
/// `regexp_jit_replace`. Only the three-argument first-match form is accepted: a
/// fourth `options` argument (e.g. 'g' for replace-all) would change the
/// semantics. Error contexts are keyed off the function's own name.
fn regex_replace_args(
    func: FunctionHandle<'_>,
) -> Result<(Box<Expression>, String, String), Error> {
    let name = func.name();
    let params = function_args(func, 3)?;
    let replacement = constant_string(
        Expression::from_handle(params[2])?,
        &format!("{name}: replacement"),
    )?;
    let pattern = constant_string(
        Expression::from_handle(params[1])?,
        &format!("{name}: pattern"),
    )?;
    let input = Box::new(Expression::from_handle(params[0])?);
    Ok((input, pattern, replacement))
}

impl RegexpReplace {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<RegexpReplace, Error> {
        let (input, pattern, replacement) = regex_replace_args(func)?;
        Ok(RegexpReplace {
            input,
            pattern,
            replacement,
        })
    }
}

impl RegexpJitReplace {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<RegexpJitReplace, Error> {
        let (input, pattern, replacement) = regex_replace_args(func)?;
        Ok(RegexpJitReplace {
            input,
            pattern,
            replacement,
        })
    }
}

// ---- Shared argument helpers ----

/// Collect a bound function's argument handles, rejecting a wrong argument count
/// the uniform way every scalar-function builder needs. Each builder then indexes
/// the returned `params` positionally.
fn function_args(func: FunctionHandle<'_>, expected: usize) -> Result<Vec<Expr<'_>>, Error> {
    let params: Vec<Expr<'_>> = func.children().collect();
    if params.len() != expected {
        return Err(Error::InvalidParameterCount {
            function: func.name(),
            expected,
            actual: params.len(),
        });
    }
    Ok(params)
}

/// Extract a constant string argument (e.g. a regex pattern or a `date_trunc`
/// unit) from a built expression, erroring with `context` when the argument is
/// not a string constant.
fn constant_string(e: Expression, context: &str) -> Result<String, Error> {
    match e {
        Expression::Constant(scalar) => {
            let (arr, _) = scalar.get();
            Ok(arr
                .as_string_view_opt()
                .ok_or_else(|| {
                    Error::UnsupportedScalarFunction(format!("{context} must be a string"))
                })?
                .value(0)
                .to_string())
        }
        _ => Err(Error::UnsupportedScalarFunction(format!(
            "{context} must be a constant"
        ))),
    }
}
