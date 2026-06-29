//! [`Function`] — the scalar-function enum dispatching DuckDB `BOUND_FUNCTION`s
//! to the per-function expression types.

use super::{
    Arithmetic, Contains, DatePart, DatePartKind, DateTrunc, Divide, Error, Length, RegexpReplace,
};
use crate::compile::{self, ExprFn, ExprResult, stateless_expr};
use crate::types::Type;
use arrow_array::{Int64Array, RecordBatch};
use duckdb_planner::expression as duckdb_expression;
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
/// compile-time mapping of the same names lives in [`Function::try_from`].
pub fn builtin_scalar_function(name: &str) -> Option<ScalarFunctionSignature> {
    match name {
        "drop_cache" => Some(ScalarFunctionSignature {
            arguments: vec![],
            return_type: Type::Int64,
            volatile: true,
        }),
        _ => None,
    }
}

/// A scalar function call (e.g. `year`, `substring`).
#[derive(Debug, Clone)]
pub enum Function {
    Contains(Contains),
    Arithmetic(Arithmetic),
    Length(Length),
    RegexpReplace(RegexpReplace),
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
            "+" | "-" | "*" => Ok(Function::Arithmetic(f.try_into()?)),
            "length" | "strlen" | "len" => Ok(Function::Length(f.try_into()?)),
            "regexp_replace" => Ok(Function::RegexpReplace(f.try_into()?)),
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

impl Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Function::Contains(c) => write!(f, "{c}"),
            Function::Arithmetic(a) => write!(f, "{a}"),
            Function::Length(l) => write!(f, "{l}"),
            Function::RegexpReplace(r) => write!(f, "{r}"),
            Function::Divide(d) => write!(f, "{d}"),
            Function::DateTrunc(dt) => write!(f, "{dt}"),
            Function::DatePart(d) => write!(f, "{d}"),
            Function::DropCache => write!(f, "drop_cache()"),
        }
    }
}

impl Function {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        match self {
            Function::Contains(c) => c.compile(),
            Function::Arithmetic(a) => a.compile(),
            Function::Length(l) => l.compile(),
            Function::RegexpReplace(r) => r.compile(),
            Function::Divide(d) => d.compile(),
            Function::DateTrunc(dt) => dt.compile(),
            Function::DatePart(d) => d.compile(),
            // `drop_cache()` evicts pivot's in-memory file cache *and* the on-disk
            // cache (so remote reads go cold to the network) as a side effect, then
            // returns the total entries dropped. Evaluated over the single
            // `DummyScan` row on a worker thread (where `memory_ctx` and the
            // worker's disk-cache handle are valid), so the eviction happens
            // exactly once; the returned array matches the (one-row) batch.
            Function::DropCache => Ok(stateless_expr(|batch: &RecordBatch| {
                let extents = dispatch::memory_ctx().file_memory_cache().clear();
                let decompressed = dispatch::memory_ctx().decompressed_cache().clear();
                let objects = dispatch::io::clear_disk_cache();
                let evicted = (extents + decompressed + objects) as i64;
                ExprResult::Array(Arc::new(Int64Array::from(vec![evicted; batch.num_rows()])))
            })),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn drop_cache_returns_one_row(mut testing_planner: TestingPlanner) {
        // FROM-less SELECT runs over the DummyScan; drop_cache() evicts and
        // returns the (non-negative) region count.
        let rows = run(&mut testing_planner, "SELECT drop_cache()");

        assert_eq!(rows.len(), 1);
        assert!(only_column(&rows[0]).as_i64().unwrap() >= 0);
    }
}
