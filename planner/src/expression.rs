//! Expressions used inside [`Operator`](crate::operator::Operator)s — column
//! references, comparisons, aggregates, scalar functions, and constants.

use crate::types;
use crate::types::{Type, build_scalar_value, type_from_logical};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, Datum, Scalar};
use duckdb_planner::duckdb_bridge::duckdb_types::ExpressionType;
use duckdb_planner::expression as duckdb_expression;
use std::fmt::{self, Display};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    TypeError(#[from] types::Error),
    #[error("Unsupported comparison type: {0:?}")]
    UnsupportedComparisonType(ExpressionType),
    #[error("Unsupported aggregate function: {0}")]
    UnsupportedAggregateFunction(String),
    #[error("Unsupported scalar function: {0}")]
    UnsupportedScalarFunction(String),
    #[error("Invalid parameter count for {function}: expected {expected}, got {actual}")]
    InvalidParameterCount {
        function: String,
        expected: usize,
        actual: usize,
    },
}

/// A bound column reference (points at a column by index in the input).
#[derive(Debug, Clone)]
pub struct Ref {
    pub column_idx: usize,
    pub return_type: Type,
}

impl TryFrom<duckdb_expression::Ref> for Ref {
    type Error = Error;
    fn try_from(r: duckdb_expression::Ref) -> Result<Self, Self::Error> {
        Ok(Ref {
            column_idx: r.column_idx,
            return_type: type_from_logical(r.return_type)?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CompareType {
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
}

impl TryFrom<ExpressionType> for CompareType {
    type Error = Error;
    fn try_from(c: ExpressionType) -> Result<Self, Self::Error> {
        match c {
            ExpressionType::COMPARE_EQUAL => Ok(CompareType::Equal),
            ExpressionType::COMPARE_NOTEQUAL => Ok(CompareType::NotEqual),
            ExpressionType::COMPARE_LESSTHAN => Ok(CompareType::Less),
            ExpressionType::COMPARE_GREATERTHAN => Ok(CompareType::Greater),
            ExpressionType::COMPARE_LESSTHANOREQUALTO => Ok(CompareType::LessEqual),
            ExpressionType::COMPARE_GREATERTHANOREQUALTO => Ok(CompareType::GreaterEqual),
            _ => Err(Error::UnsupportedComparisonType(c)),
        }
    }
}

/// A binary comparison expression (e.g. `<>`, `=`).
#[derive(Debug, Clone)]
pub struct Compare {
    pub left: Box<Expression>,
    pub right: Box<Expression>,
    pub compare_type: CompareType,
    pub return_type: Type,
}

impl TryFrom<duckdb_expression::Compare> for Compare {
    type Error = Error;
    fn try_from(c: duckdb_expression::Compare) -> Result<Self, Self::Error> {
        Ok(Compare {
            left: Box::<Expression>::try_from(c.left)?,
            right: Box::<Expression>::try_from(c.right)?,
            compare_type: c.compare_type.try_into()?,
            return_type: type_from_logical(c.return_type)?,
        })
    }
}

/// A `BETWEEN` range test (`input BETWEEN lower AND upper`). DuckDB folds
/// `x >= a AND x <= b` into this; we compile it back to the conjunction of two
/// comparisons (respecting the inclusive/exclusive flags).
#[derive(Debug, Clone)]
pub struct Between {
    pub input: Box<Expression>,
    pub lower: Box<Expression>,
    pub upper: Box<Expression>,
    pub lower_inclusive: bool,
    pub upper_inclusive: bool,
}

impl TryFrom<duckdb_expression::Between> for Between {
    type Error = Error;
    fn try_from(b: duckdb_expression::Between) -> Result<Self, Self::Error> {
        Ok(Between {
            input: Box::<Expression>::try_from(b.input)?,
            lower: Box::<Expression>::try_from(b.lower)?,
            upper: Box::<Expression>::try_from(b.upper)?,
            lower_inclusive: b.lower_inclusive,
            upper_inclusive: b.upper_inclusive,
        })
    }
}

#[derive(Debug, Clone)]
pub struct CountStar {
    pub params: Vec<Expression>,
}

impl TryFrom<duckdb_expression::AggregateFunc> for CountStar {
    type Error = Error;
    fn try_from(a: duckdb_expression::AggregateFunc) -> Result<Self, Self::Error> {
        if a.aggregate_function != "count_star" {
            return Err(Error::UnsupportedAggregateFunction(a.aggregate_function));
        }
        Ok(CountStar {
            params: a
                .params
                .into_iter()
                .map(Expression::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

/// An aggregate function call (e.g. `SUM`, `COUNT`).
/// A single-column numeric aggregate (`SUM(col)` / `AVG(col)`). Carries the
/// bound column reference being aggregated.
#[derive(Debug, Clone)]
pub struct NumericAggregate {
    pub column: Ref,
}

impl TryFrom<duckdb_expression::AggregateFunc> for NumericAggregate {
    type Error = Error;
    fn try_from(a: duckdb_expression::AggregateFunc) -> Result<Self, Self::Error> {
        if a.params.len() != 1 {
            return Err(Error::InvalidParameterCount {
                function: a.aggregate_function,
                expected: 1,
                actual: a.params.len(),
            });
        }
        let column = match Expression::try_from(a.params.into_iter().next().unwrap())? {
            Expression::Ref(r) => r,
            other => {
                return Err(Error::UnsupportedAggregateFunction(format!(
                    "non-column argument: {other}"
                )));
            }
        };
        Ok(NumericAggregate { column })
    }
}

/// An aggregate function call (e.g. `SUM`, `COUNT`).
#[derive(Debug, Clone)]
pub enum AggregateFunc {
    CountStar(CountStar),
    Sum(NumericAggregate),
    Avg(NumericAggregate),
    /// `COUNT(col)` — counts non-null values. DuckDB lowers `AVG(col)` to
    /// `sum(col) / count(col)`, so this shows up in average plans.
    Count(NumericAggregate),
    /// `COUNT(DISTINCT col)` — counts the distinct non-null values of `col`.
    /// Lowered in compilation to a two-level GROUP BY (dedup on the group keys
    /// plus `col`, then count rows per group); see [`crate::compile`].
    CountDistinct(NumericAggregate),
}

impl TryFrom<duckdb_expression::AggregateFunc> for AggregateFunc {
    type Error = Error;
    fn try_from(a: duckdb_expression::AggregateFunc) -> Result<Self, Self::Error> {
        // DISTINCT is only supported for `COUNT` so far; reject `SUM(DISTINCT)`
        // etc. rather than silently computing the non-distinct aggregate.
        if a.distinct && a.aggregate_function != "count" {
            return Err(Error::UnsupportedAggregateFunction(format!(
                "DISTINCT {}",
                a.aggregate_function
            )));
        }
        match a.aggregate_function.as_str() {
            "count_star" => Ok(AggregateFunc::CountStar(a.try_into()?)),
            "sum" => Ok(AggregateFunc::Sum(a.try_into()?)),
            "avg" => Ok(AggregateFunc::Avg(a.try_into()?)),
            "count" if a.distinct => Ok(AggregateFunc::CountDistinct(a.try_into()?)),
            "count" => Ok(AggregateFunc::Count(a.try_into()?)),
            _ => Err(Error::UnsupportedAggregateFunction(a.aggregate_function)),
        }
    }
}

/// SQL `contains(haystack, needle)`.
#[derive(Debug, Clone)]
pub struct Contains {
    pub needle: Box<Expression>,
    pub haystack: Box<Expression>,
}

impl TryFrom<duckdb_expression::Function> for Contains {
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
        let needle = Box::new(Expression::try_from(f.params.remove(1))?);
        let haystack = Box::new(Expression::try_from(f.params.remove(0))?);
        Ok(Contains { needle, haystack })
    }
}

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

/// SQL `date_trunc(unit, source)` — truncate a timestamp down to `unit`
/// (e.g. `date_trunc('minute', EventTime)`). DuckDB passes the unit as a string
/// constant in the first argument and the timestamp expression second.
#[derive(Debug, Clone)]
pub struct DateTrunc {
    pub unit: String,
    pub source: Box<Expression>,
}

impl TryFrom<duckdb_expression::Function> for DateTrunc {
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
        let source = Box::new(Expression::try_from(f.params.remove(1))?);
        let unit = match Expression::try_from(f.params.remove(0))? {
            Expression::Constant(scalar) => {
                let (arr, _) = scalar.get();
                arr.as_string_view_opt()
                    .ok_or_else(|| {
                        Error::UnsupportedScalarFunction("date_trunc: unit must be a string".into())
                    })?
                    .value(0)
                    .to_ascii_lowercase()
            }
            _ => {
                return Err(Error::UnsupportedScalarFunction(
                    "date_trunc: unit must be a constant".into(),
                ));
            }
        };
        Ok(DateTrunc { unit, source })
    }
}

/// Which field of a timestamp a [`DatePart`] extracts. DuckDB lowers
/// `extract(<part> FROM ts)` to a scalar function named after the part (e.g.
/// `minute`, `year`); this enumerates the parts we evaluate from `EventTime`'s
/// Int64 epoch-seconds representation. See [`DatePart`]'s compile impl for the
/// per-part arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatePartKind {
    /// Whole seconds since the Unix epoch (the stored value, unchanged).
    Epoch,
    /// Second of minute, 0–59.
    Second,
    /// Millisecond of minute, 0–59000 (whole seconds only, so `second * 1000`).
    Millisecond,
    /// Microsecond of minute, `second * 1_000_000`.
    Microsecond,
    /// Minute of hour, 0–59.
    Minute,
    /// Hour of day, 0–23.
    Hour,
    /// Day of month, 1–31.
    Day,
    /// Month of year, 1–12.
    Month,
    /// Quarter of year, 1–4.
    Quarter,
    /// Full year (e.g. 2024).
    Year,
    /// Decade — `year / 10` (e.g. 202 for 2024).
    Decade,
    /// Century — e.g. 21 for years 2001–2100.
    Century,
    /// Millennium — e.g. 3 for years 2001–3000.
    Millennium,
    /// Day of week, 0 (Sunday)–6 (Saturday).
    DayOfWeek,
    /// ISO day of week, 1 (Monday)–7 (Sunday).
    IsoDayOfWeek,
    /// Day of year, 1–366.
    DayOfYear,
    /// ISO 8601 week of year, 1–53.
    Week,
}

impl DatePartKind {
    /// Resolve a DuckDB scalar-function name (as `extract` lowers it) to a part.
    /// Includes the common DuckDB aliases (`dow`, `doy`, `weekofyear`).
    pub fn from_function_name(name: &str) -> Option<Self> {
        Some(match name {
            "epoch" => DatePartKind::Epoch,
            "second" => DatePartKind::Second,
            "millisecond" => DatePartKind::Millisecond,
            "microsecond" => DatePartKind::Microsecond,
            "minute" => DatePartKind::Minute,
            "hour" => DatePartKind::Hour,
            "day" => DatePartKind::Day,
            "month" => DatePartKind::Month,
            "quarter" => DatePartKind::Quarter,
            "year" => DatePartKind::Year,
            "decade" => DatePartKind::Decade,
            "century" => DatePartKind::Century,
            "millennium" => DatePartKind::Millennium,
            "dayofweek" | "dow" => DatePartKind::DayOfWeek,
            "isodow" => DatePartKind::IsoDayOfWeek,
            "dayofyear" | "doy" => DatePartKind::DayOfYear,
            "week" | "weekofyear" => DatePartKind::Week,
            _ => return None,
        })
    }

    /// The canonical DuckDB function name for this part, used for plan display.
    pub fn name(&self) -> &'static str {
        match self {
            DatePartKind::Epoch => "epoch",
            DatePartKind::Second => "second",
            DatePartKind::Millisecond => "millisecond",
            DatePartKind::Microsecond => "microsecond",
            DatePartKind::Minute => "minute",
            DatePartKind::Hour => "hour",
            DatePartKind::Day => "day",
            DatePartKind::Month => "month",
            DatePartKind::Quarter => "quarter",
            DatePartKind::Year => "year",
            DatePartKind::Decade => "decade",
            DatePartKind::Century => "century",
            DatePartKind::Millennium => "millennium",
            DatePartKind::DayOfWeek => "dayofweek",
            DatePartKind::IsoDayOfWeek => "isodow",
            DatePartKind::DayOfYear => "dayofyear",
            DatePartKind::Week => "week",
        }
    }
}

/// SQL `extract(<part> FROM source)` — a timestamp field accessor. DuckDB
/// lowers each part to a scalar function (`minute`, `year`, …); `EventTime` is
/// stored as Int64 epoch *seconds* (see [`Type::Timestamp`]), so every part is
/// a pure integer computation. See its compile impl.
///
/// [`Type::Timestamp`]: crate::types::Type::Timestamp
#[derive(Debug, Clone)]
pub struct DatePart {
    pub kind: DatePartKind,
    pub source: Box<Expression>,
}

impl DatePart {
    /// Build from a DuckDB function call once its name has been recognised as a
    /// date part. Validates the single-argument arity.
    fn from_function(
        kind: DatePartKind,
        mut f: duckdb_expression::Function,
    ) -> Result<Self, Error> {
        if f.params.len() != 1 {
            let actual = f.params.len();
            return Err(Error::InvalidParameterCount {
                function: f.function,
                expected: 1,
                actual,
            });
        }
        let source = Box::new(Expression::try_from(f.params.remove(0))?);
        Ok(DatePart { kind, source })
    }
}

/// A scalar function call (e.g. `year`, `substring`).
#[derive(Debug, Clone)]
pub enum Function {
    Contains(Contains),
    Divide(Divide),
    DateTrunc(DateTrunc),
    DatePart(DatePart),
    /// `drop_cache()` — evict pivot's file cache, returning the regions dropped.
    /// A side-effecting admin function; evaluated once over the [`DummyScan`]
    /// row of a `FROM`-less `SELECT`. See its compile impl.
    ///
    /// [`DummyScan`]: crate::operator::DummyScan
    DropCache,
}

impl TryFrom<duckdb_expression::Function> for Function {
    type Error = Error;
    fn try_from(f: duckdb_expression::Function) -> Result<Self, Self::Error> {
        match f.function.as_str() {
            "contains" => Ok(Function::Contains(f.try_into()?)),
            "/" => Ok(Function::Divide(f.try_into()?)),
            "date_trunc" => Ok(Function::DateTrunc(f.try_into()?)),
            "drop_cache" => {
                if !f.params.is_empty() {
                    return Err(Error::InvalidParameterCount {
                        function: f.function,
                        expected: 0,
                        actual: f.params.len(),
                    });
                }
                Ok(Function::DropCache)
            }
            // `extract(<part> FROM ts)` lowers to a function named after the
            // part (`minute`, `year`, `dayofweek`, …).
            name => match DatePartKind::from_function_name(name) {
                Some(kind) => Ok(Function::DatePart(DatePart::from_function(kind, f)?)),
                None => Err(Error::UnsupportedScalarFunction(f.function)),
            },
        }
    }
}

/// An expression in the logical plan. Discriminated by DuckDB's [`ExpressionType`].
#[derive(Debug, Clone)]
pub enum Expression {
    Ref(Ref),
    Compare(Compare),
    Between(Between),
    Constant(Scalar<ArrayRef>),
    AggregateFunc(AggregateFunc),
    Function(Function),
}

impl Display for CompareType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompareType::Equal => f.write_str("="),
            CompareType::NotEqual => f.write_str("<>"),
            CompareType::Less => f.write_str("<"),
            CompareType::Greater => f.write_str(">"),
            CompareType::LessEqual => f.write_str("<="),
            CompareType::GreaterEqual => f.write_str(">="),
        }
    }
}

/// Format an arrow `Scalar<ArrayRef>` constant as `value:Type` for plan
/// display. Falls back to `?:DataType` for unsupported arrow types.
fn format_constant(s: &Scalar<ArrayRef>) -> String {
    let (arr, _is_scalar) = s.get();
    let formatter = ArrayFormatter::try_new(arr, &FormatOptions::default());
    let value = match formatter {
        Ok(f) => f.value(0).to_string(),
        Err(_) => "?".to_string(),
    };
    format!("{value}:{}", arr.data_type())
}

impl Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Ref(r) => write!(f, "#{}:{}", r.column_idx, r.return_type),
            Expression::Compare(c) => write!(
                f,
                "{} {} {} -> {}",
                c.left, c.compare_type, c.right, c.return_type
            ),
            Expression::Between(b) => {
                write!(f, "{} BETWEEN {} AND {}", b.input, b.lower, b.upper)
            }
            Expression::Constant(c) => f.write_str(&format_constant(c)),
            Expression::AggregateFunc(AggregateFunc::CountStar(_)) => f.write_str("count_star()"),
            Expression::AggregateFunc(AggregateFunc::Sum(a)) => {
                write!(f, "sum(#{})", a.column.column_idx)
            }
            Expression::AggregateFunc(AggregateFunc::Avg(a)) => {
                write!(f, "avg(#{})", a.column.column_idx)
            }
            Expression::AggregateFunc(AggregateFunc::Count(a)) => {
                write!(f, "count(#{})", a.column.column_idx)
            }
            Expression::AggregateFunc(AggregateFunc::CountDistinct(a)) => {
                write!(f, "count(distinct #{})", a.column.column_idx)
            }
            Expression::Function(Function::Contains(c)) => {
                write!(f, "contains({}, {})", c.haystack, c.needle)
            }
            Expression::Function(Function::Divide(d)) => write!(f, "({} / {})", d.left, d.right),
            Expression::Function(Function::DateTrunc(dt)) => {
                write!(f, "date_trunc('{}', {})", dt.unit, dt.source)
            }
            Expression::Function(Function::DatePart(d)) => {
                write!(f, "{}({})", d.kind.name(), d.source)
            }
            Expression::Function(Function::DropCache) => write!(f, "drop_cache()"),
        }
    }
}

impl TryFrom<duckdb_expression::Expression> for Expression {
    type Error = Error;
    fn try_from(e: duckdb_expression::Expression) -> Result<Self, Self::Error> {
        Ok(match e {
            duckdb_expression::Expression::Ref(r) => Expression::Ref(r.try_into()?),
            duckdb_expression::Expression::Compare(c) => Expression::Compare(c.try_into()?),
            duckdb_expression::Expression::Between(b) => Expression::Between(b.try_into()?),
            duckdb_expression::Expression::Constant(c) => {
                Expression::Constant(build_scalar_value(c)?)
            }
            duckdb_expression::Expression::AggregateFunc(a) => {
                Expression::AggregateFunc(a.try_into()?)
            }
            duckdb_expression::Expression::Function(f) => Expression::Function(f.try_into()?),
        })
    }
}

impl TryFrom<Box<duckdb_planner::expression::Expression>> for Box<Expression> {
    type Error = Error;
    fn try_from(e: Box<duckdb_planner::expression::Expression>) -> Result<Self, Self::Error> {
        Ok(Box::new((*e).try_into()?))
    }
}

/// A constant comparison against a single column pushed into a table scan.
#[derive(Debug)]
pub struct ConstantComparison {
    pub column_ref: Box<Expression>,
    pub compare_type: CompareType,
    pub constant: Scalar<ArrayRef>,
}

impl TryFrom<duckdb_expression::ConstantComparison> for ConstantComparison {
    type Error = Error;
    fn try_from(c: duckdb_expression::ConstantComparison) -> Result<Self, Self::Error> {
        Ok(ConstantComparison {
            column_ref: Box::<Expression>::try_from(c.column_ref)?,
            compare_type: c.compare_type.try_into()?,
            constant: build_scalar_value(c.constant)?,
        })
    }
}

/// A filter that was pushed down into a table scan.
#[derive(Debug)]
pub enum TableFilter {
    Expression(Box<Expression>),
    ConstantComparison(ConstantComparison),
}

impl TryFrom<duckdb_expression::TableFilter> for TableFilter {
    type Error = Error;
    fn try_from(f: duckdb_expression::TableFilter) -> Result<Self, Self::Error> {
        Ok(match f {
            duckdb_expression::TableFilter::Expression(e) => TableFilter::Expression(e.try_into()?),
            duckdb_expression::TableFilter::ConstantComparison(c) => {
                TableFilter::ConstantComparison(c.try_into()?)
            }
        })
    }
}

impl Display for TableFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TableFilter::Expression(e) => write!(f, "{e}"),
            TableFilter::ConstantComparison(c) => write!(
                f,
                "{} {} {}",
                c.column_ref,
                c.compare_type,
                format_constant(&c.constant),
            ),
        }
    }
}
