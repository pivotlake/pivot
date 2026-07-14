//! [`Function`] — the scalar-function enum dispatching DuckDB `BOUND_FUNCTION`s
//! to the per-function expression types.

use super::{
    Arithmetic, Contains, DatePart, DateTrunc, Divide, IntervalArithmetic, Length, Prefix,
    RegexpJitReplace, RegexpReplace, TemporalConvert, VariantGet, VariantToJson,
};
use crate::compile::{self, ExprFn, ExprResult, stateless_expr};
use crate::types::Type;
use arrow_array::{Int64Array, RecordBatch, TimestampSecondArray};
use std::fmt::{self, Display};
use std::sync::Arc;

/// The binding signature of a pivot-defined scalar function: argument types,
/// return type, and whether DuckDB must leave the call un-folded. The bridge
/// reads this to register the function for DuckDB's binder, so a pivot scalar
/// function needs no C++ registration.
pub struct ScalarFunctionSignature {
    pub arguments: Vec<Type>,
    pub return_type: Type,
    /// `true` marks the function `VOLATILE` so DuckDB can't constant-fold the
    /// call away before pivot re-plans it (e.g. a no-arg `drop_cache()`).
    pub volatile: bool,
}

/// The binding signature for a pivot-defined scalar function `name`, or `None`
/// for names pivot doesn't define (DuckDB's own built-ins like `+`/`length`,
/// which DuckDB binds itself and pivot only intercepts at compile time). The
/// compile-time mapping of the same names lives in `Function::from_handle`.
pub fn builtin_scalar_function(name: &str) -> Option<ScalarFunctionSignature> {
    match name {
        "drop_cache" => Some(ScalarFunctionSignature {
            arguments: vec![],
            return_type: Type::Int64,
            volatile: true,
        }),
        // `now()`: current wall-clock time. VOLATILE so DuckDB can't fold the
        // call into its own `TIMESTAMP WITH TIME ZONE` constant; pivot evaluates
        // it instead, returning a `TIMESTAMP` (epoch seconds) like the rest of
        // its time path. (The bare `CURRENT_TIMESTAMP` keyword is a separate
        // DuckDB value-function that yields a TZ type pivot doesn't model, so
        // only the `now()` call form is intercepted here.)
        "now" => Some(ScalarFunctionSignature {
            arguments: vec![],
            return_type: Type::Timestamp,
            volatile: true,
        }),
        // Not a DuckDB built-in (unlike `regexp_replace`), so its signature is
        // registered here for DuckDB's binder.
        "regexp_jit_replace" => Some(ScalarFunctionSignature {
            arguments: vec![Type::Utf8, Type::Utf8, Type::Utf8],
            return_type: Type::Utf8,
            volatile: false,
        }),
        // DuckDB binds `doc->'key'` as `json_extract`. The stub is a no-op, so
        // mark it volatile to prevent DuckDB from evaluating or folding it.
        "json_extract" => Some(ScalarFunctionSignature {
            arguments: vec![Type::Variant, Type::Utf8],
            return_type: Type::Variant,
            volatile: true,
        }),
        _ => None,
    }
}

/// A scalar function call (e.g. `year`, `substring`).
#[derive(Debug, Clone)]
pub enum Function {
    Contains(Contains),
    Prefix(Prefix),
    Arithmetic(Arithmetic),
    Length(Length),
    RegexpReplace(RegexpReplace),
    /// `regexp_jit_replace` — like `RegexpReplace` but always PCRE2 JIT-compiled.
    RegexpJitReplace(RegexpJitReplace),
    Divide(Divide),
    DateTrunc(DateTrunc),
    DatePart(DatePart),
    /// `date`/`timestamp` ± `INTERVAL` (e.g. `now() - interval '5 days'`).
    IntervalArithmetic(IntervalArithmetic),
    /// `make_date(days)` / `make_timestamp(seconds)` — read an integer column as a
    /// real `DATE` / `TIMESTAMP`.
    TemporalConvert(TemporalConvert),
    /// `drop_cache()` — evict pivot's compressed cache, returning the regions dropped.
    /// A side-effecting admin function; evaluated once over the [`DummyScan`]
    /// row of a `FROM`-less `SELECT`. See its compile impl.
    ///
    /// [`DummyScan`]: crate::operator::DummyScan
    DropCache,
    /// `now()` yields the wall-clock time captured once when the query
    /// compiles, so every row of the statement sees the same instant. Result is
    /// a `TIMESTAMP` (epoch seconds).
    Now,
    /// A variant (JSON) path read: `doc->'key'` chains, optionally typed by a
    /// fused `CAST`.
    VariantGet(VariantGet),
    /// A variant value rendered as JSON text, wrapped around variant-typed
    /// output columns by plan build.
    VariantToJson(VariantToJson),
}

impl Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Function::Contains(c) => write!(f, "{c}"),
            Function::Prefix(p) => write!(f, "{p}"),
            Function::Arithmetic(a) => write!(f, "{a}"),
            Function::Length(l) => write!(f, "{l}"),
            Function::RegexpReplace(r) => write!(f, "{r}"),
            Function::RegexpJitReplace(r) => write!(f, "{r}"),
            Function::Divide(d) => write!(f, "{d}"),
            Function::DateTrunc(dt) => write!(f, "{dt}"),
            Function::DatePart(d) => write!(f, "{d}"),
            Function::IntervalArithmetic(i) => write!(f, "{i}"),
            Function::TemporalConvert(c) => write!(f, "{c}"),
            Function::DropCache => write!(f, "drop_cache()"),
            Function::Now => write!(f, "now()"),
            Function::VariantGet(v) => write!(f, "{v}"),
            Function::VariantToJson(v) => write!(f, "{v}"),
        }
    }
}

impl Function {
    /// The type a call to this function yields.
    pub fn result_type(&self) -> Type {
        match self {
            // The string matchers yield booleans.
            Function::Contains(_) | Function::Prefix(_) => Type::Boolean,
            // Arithmetic and the extractors carry DuckDB's bound result type.
            Function::Arithmetic(a) => a.return_type.clone(),
            Function::Length(l) => l.return_type.clone(),
            Function::DatePart(d) => d.return_type.clone(),
            // The regex replacers rewrite strings.
            Function::RegexpReplace(_) | Function::RegexpJitReplace(_) => Type::Utf8,
            // `/` always computes a float quotient.
            Function::Divide(_) => Type::Float64,
            // `date_trunc` and `now()` yield a timestamp.
            Function::DateTrunc(_) | Function::Now => Type::Timestamp,
            // `date`/`timestamp` ± interval keeps the temporal operand's type,
            // and `make_date`/`make_timestamp` produce the type they convert to.
            Function::IntervalArithmetic(i) => i.result.clone(),
            Function::TemporalConvert(c) => c.result.clone(),
            // A variant path read yields its cast's type (the sub-variant when
            // bare); a JSON-text render always yields a string.
            Function::VariantGet(v) => v.result_type(),
            Function::VariantToJson(_) => Type::Utf8,
            // `drop_cache()` returns the evicted-entry count.
            Function::DropCache => Type::Int64,
        }
    }

    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        match self {
            Function::Contains(c) => c.compile(),
            Function::Prefix(p) => p.compile(),
            Function::Arithmetic(a) => a.compile(),
            Function::Length(l) => l.compile(),
            Function::RegexpReplace(r) => r.compile(),
            Function::RegexpJitReplace(r) => r.compile(),
            Function::Divide(d) => d.compile(),
            Function::DateTrunc(dt) => dt.compile(),
            Function::DatePart(d) => d.compile(),
            Function::IntervalArithmetic(i) => i.compile(),
            Function::TemporalConvert(c) => c.compile(),
            Function::VariantGet(v) => v.compile(),
            Function::VariantToJson(v) => v.compile(),
            // `drop_cache()` evicts pivot's in-memory compressed cache *and* the on-disk
            // cache (so remote reads go cold to the network) as a side effect, then
            // returns the total entries dropped. Evaluated over the single
            // `DummyScan` row on a worker thread (where `memory_ctx` and the
            // worker's disk-cache handle are valid), so the eviction happens
            // exactly once; the returned array matches the (one-row) batch.
            Function::DropCache => Ok(stateless_expr(|batch: &RecordBatch| {
                let extents = dispatch::memory_ctx().compressed_cache().clear();
                let decompressed = dispatch::memory_ctx().decompressed_cache().clear();
                let objects = dispatch::io::clear_disk_cache();
                let evicted = (extents + decompressed + objects) as i64;
                ExprResult::Array(Arc::new(Int64Array::from(vec![evicted; batch.num_rows()])))
            })),
            // Capture the instant once, here at compile time, so every worker and
            // every row of the statement observes the same `now()`. Emitted as a
            // real `Timestamp` (epoch seconds, pivot's timestamp representation).
            // Negative (pre-epoch) clocks are clamped to 0, which can't happen on a
            // sane host.
            Function::Now => {
                let now_secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|elapsed| elapsed.as_secs() as i64)
                    .unwrap_or(0);
                Ok(stateless_expr(move |batch: &RecordBatch| {
                    ExprResult::Array(Arc::new(TimestampSecondArray::from(vec![
                        now_secs;
                        batch.num_rows()
                    ])))
                }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use arrow_array::cast::AsArray;
    use arrow_array::types::TimestampSecondType;
    use arrow_schema::{DataType, TimeUnit};
    use rstest::rstest;

    #[rstest]
    fn drop_cache_returns_one_row(mut testing_planner: TestingPlanner) {
        // FROM-less SELECT runs over the DummyScan; drop_cache() evicts and
        // returns the (non-negative) region count.
        let rows = run(&mut testing_planner, "SELECT drop_cache()");

        assert_eq!(rows.len(), 1);
        assert!(only_column(&rows[0]).as_i64().unwrap() >= 0);
    }

    #[rstest]
    fn now_returns_current_time_as_a_timestamp(mut testing_planner: TestingPlanner) {
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let batches = run_batches(&mut testing_planner, "SELECT now()");

        let after = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let col = batches[0].column(0);
        // `now()` surfaces as a real TIMESTAMP carrying the captured epoch second.
        assert_eq!(
            col.data_type(),
            &DataType::Timestamp(TimeUnit::Second, None)
        );
        let now = col.as_primitive::<TimestampSecondType>().value(0);
        assert!(
            (before..=after).contains(&now),
            "{now} not in [{before}, {after}]"
        );
    }
}
