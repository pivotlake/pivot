//! The `compact` table function backing the `COMPACT` statement.
//!
//! The parser rewrites `COMPACT <table> [FINAL]` into `CALL compact(catalog,
//! schema, table, final)`, so DuckDB's binder needs a `compact` function to
//! resolve; this is that function. It only ever *binds*: the server recognises
//! the bound plan (see [`Plan::as_compact`](crate::Plan::as_compact)) and runs
//! the compaction itself, coordinator-side, because a sweep drives dataflows
//! of its own and would deadlock the pool if compiled into one. Reaching
//! `compile` therefore means the call was used as a row source (say, inside a
//! larger SELECT), which is not a supported way to run it.

use super::{TableFunction, TableFunctionSignature, invalid_argument};
use crate::catalog::Column;
use crate::compile::Error;
use crate::types::Type;
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use duckdb_planner::ScalarValue;

pub struct CompactTableFunction;

impl TableFunction for CompactTableFunction {
    fn name(&self) -> &str {
        "compact"
    }

    fn signature(&self) -> TableFunctionSignature {
        TableFunctionSignature {
            arguments: vec![Type::Utf8, Type::Utf8, Type::Utf8, Type::Boolean],
            columns: vec![Column {
                name: "sweeps".to_string(),
                col_type: Type::Int64,
            }],
        }
    }

    fn compile(
        &self,
        _args: &[ScalarValue],
        _dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        Err(invalid_argument(
            "compact",
            "COMPACT runs as its own statement and cannot be part of a query".to_string(),
        ))
    }
}
