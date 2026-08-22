//! [`TableFunctionScan`]: a scan over a table-valued function such as
//! `generate_series(start, stop[, step])`.
//!
//! This module holds the generic machinery: the [`TableFunction`] trait each
//! function implements, the operator that looks one up and runs it, and the
//! column projection shared by all of them. Every function depends only on its
//! arguments, so they all live here ([`series`]).
//!
//! A function produces its *full* output (every column it declares, in order)
//! as a dataflow; the operator then projects that to the columns DuckDB asked
//! for. DuckDB prunes and reorders a table function's output and resolves refs
//! above the scan positionally against that, so the projection is what keeps a
//! `SELECT <subset>` returning the right columns (the table-function twin of
//! [`Input`](super::Input)'s column projection).

mod series;

use crate::compile::Error;
use crate::expression::Expression;
use crate::types::Type;
use arrow_array::{RecordBatch, RecordBatchOptions};
use arrow_schema::Schema;
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use duckdb_planner::ScalarValue;
use std::fmt;
use std::sync::Arc;

/// One table-valued function: given its bound constant arguments, it builds the
/// dataflow that emits its rows. Implementors emit their *full* declared output
/// (every column, in declared order); [`TableFunctionScan`] projects it down to
/// the requested columns.
pub trait TableFunction: Send + Sync {
    /// The SQL name this function is invoked as (e.g. `"generate_series"`).
    fn name(&self) -> &str;

    /// Build the dataflow emitting this function's full output, from nothing but
    /// this function's arguments.
    fn compile(
        &self,
        args: &[ScalarValue],
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec, Error>;
}

/// The built-in table functions, keyed by name. DuckDB binds the call against
/// its own system catalog; this is where the plan finds the implementation that
/// runs it.
fn find_table_function(name: &str) -> Option<Box<dyn TableFunction>> {
    match name {
        "range" => Some(Box::new(series::SeriesTableFunction::range())),
        "generate_series" => Some(Box::new(series::SeriesTableFunction::generate_series())),
        _ => None,
    }
}

/// A scan over a table-valued function. Holds the function name, its bound
/// constant arguments, and the output columns this scan must emit (each a
/// positional [`Ref`](crate::expression::Ref) into the function's full output).
#[derive(Debug)]
pub struct TableFunctionScan {
    function_name: String,
    args: Vec<ScalarValue>,
    columns: Vec<Expression>,
}

impl TableFunctionScan {
    pub(crate) fn new(
        function_name: String,
        args: Vec<ScalarValue>,
        columns: Vec<Expression>,
    ) -> Self {
        Self {
            function_name,
            args,
            columns,
        }
    }

    /// The type of each output column, in order.
    pub(crate) fn output_types(&self) -> Result<Vec<Type>, Error> {
        self.columns.iter().map(Expression::result_type).collect()
    }
}

impl fmt::Display for TableFunctionScan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let args: Vec<String> = self.args.iter().map(|a| a.to_string()).collect();
        write!(
            f,
            "TableFunctionScan({}({}))",
            self.function_name,
            args.join(", ")
        )
    }
}

impl TableFunctionScan {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let function = find_table_function(&self.function_name)
            .ok_or_else(|| Error::UnsupportedTableFunction(self.function_name.clone()))?;
        let full = function.compile(&self.args, dispatcher)?;
        self.project(full)
    }

    /// Project `spec` (the function's complete output, in declared order) down to
    /// the columns this scan was asked for. Mirrors how [`Input`](super::Input)
    /// turns its `columns` into a scan projection.
    fn project(&self, spec: RecordBatchOperatorSpec) -> Result<RecordBatchOperatorSpec, Error> {
        let indices = Arc::new(self.output_column_indices()?);
        Ok(spec.project(move || {
            let indices = indices.clone();
            move |batch: RecordBatch| {
                if indices.is_empty() {
                    // No column referenced (e.g. `COUNT(*)`): drop every column
                    // but keep the row count, which a plain `project(&[])` loses.
                    RecordBatch::try_new_with_options(
                        Arc::new(Schema::empty()),
                        vec![],
                        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
                    )
                    .expect("empty projection preserves the row count")
                } else {
                    batch
                        .project(&indices)
                        .expect("table-function column indices are within the generated schema")
                }
            }
        }))
    }

    /// The requested output columns as positional indices into the generated
    /// batch, skipping DuckDB's `usize::MAX` "no column needed" sentinel.
    fn output_column_indices(&self) -> Result<Vec<usize>, Error> {
        self.columns
            .iter()
            .filter_map(|expr| match expr {
                Expression::Ref(r) if r.column_idx == usize::MAX => None,
                Expression::Ref(r) => Some(Ok(r.column_idx)),
                _ => Some(Err(Error::UnexpectedInputExpression(expr.clone()))),
            })
            .collect()
    }
}

/// Build an `InvalidTableFunctionArgument` error for `function`.
pub(crate) fn invalid_argument(function: &str, message: String) -> Error {
    Error::InvalidTableFunctionArgument {
        function: function.to_string(),
        message,
    }
}
