//! [`TableFunctionScan`]: a scan over a table-valued function such as
//! `generate_series(start, stop[, step])`.
//!
//! This module holds the generic machinery: the [`TableFunction`] trait each
//! function implements, the operator that runs one, and the column projection
//! shared by all of them. The concrete functions live elsewhere: the generic
//! ones ([`series`]) here, backend-specific ones (e.g. `read_parquet`, which
//! only a catalog that can reach storage can answer) in the catalog,
//! contributed through [`CatalogTransaction::bind_default_table_function`].
//!
//! A call resolves in two steps, the way a SQL binder resolves one. The name
//! alone yields the function and its
//! [argument types](TableFunction::argument_types), which is all DuckDB needs
//! to register the overload; the arguments are then
//! [bound](TableFunction::bind), and what that produces carries the call's
//! output columns. Splitting it this way is what lets a function's schema
//! depend on its arguments: `read_parquet('…')` reads the files' footers while
//! binding and reports the columns it found there.
//!
//! Binding produces one of two things ([`BoundTableFunction`]): rows the
//! function computes, or a **table** it reads. A table is scanned exactly as a
//! catalog table is, with everything that carries (projection and filter
//! pushdown, cardinality, late materialization), so a function naming storage
//! needs no scan path of its own. Rows are emitted by [`TableFunctionScan`],
//! which asks for the function's *full* declared output and projects it down to
//! the columns the query wants: DuckDB prunes and reorders a table function's
//! output and resolves refs above the scan positionally against that, so the
//! projection is what keeps a `SELECT <subset>` returning the right columns
//! (the table-function twin of [`Input`](super::Input)'s column projection).

mod series;

use crate::catalog::{BoundTable, CatalogTransaction, Column, Result as CatalogResult};
use crate::compile::Error;
use crate::expression::Expression;
use crate::types::Type;
use arrow_array::{RecordBatch, RecordBatchOptions};
use arrow_schema::Schema;
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use duckdb_planner::ScalarValue;
use std::fmt;
use std::sync::Arc;

/// One table-valued function, as its name alone identifies it.
pub trait TableFunction: Send + Sync {
    /// The SQL name this function is invoked as (e.g. `"read_parquet"`).
    fn name(&self) -> &str;

    /// The types of the arguments a call passes, which the binder resolves the
    /// call against. Known from the name, before any argument is bound.
    fn argument_types(&self) -> Vec<Type>;

    /// Resolve a call's `arguments` into what the query will read: the rows the
    /// function computes from them, or the table they name.
    ///
    /// Whatever the arguments have to be interpreted for — opening storage,
    /// listing files, reading footers — happens here, once, and what comes back
    /// carries both the call's output columns and everything compiling it
    /// needs. An error is the statement's error, which is how an unreadable
    /// argument is reported.
    fn bind(&self, arguments: &[ScalarValue]) -> CatalogResult<BoundTableFunction>;
}

/// What binding a table-function call produced.
pub enum BoundTableFunction {
    /// Rows the function computes from its arguments (`generate_series`).
    Rows(Box<dyn TableFunctionRows>),
    /// A table the arguments named (`read_parquet`), scanned like any other.
    Table(Box<dyn BoundTable>),
}

impl BoundTableFunction {
    /// The columns this bound call emits, in order. The single place a call's
    /// schema comes from, so the binder and the scan cannot disagree about it.
    pub fn columns(&self) -> Vec<Column> {
        match self {
            Self::Rows(rows) => rows.columns(),
            Self::Table(table) => table.columns(),
        }
    }
}

/// A bound call that computes its rows. It holds whatever its arguments meant,
/// so compiling it needs nothing but the pool.
pub trait TableFunctionRows: Send + Sync {
    /// The full output columns, in declared order.
    fn columns(&self) -> Vec<Column>;

    /// Build the dataflow emitting that full output; [`TableFunctionScan`]
    /// projects it down to what the query asked for.
    fn compile(&self, dispatcher: &DataFlowDispatcher) -> Result<RecordBatchOperatorSpec, Error>;
}

/// Resolve a table function by name. The transaction is consulted first, then
/// the generic built-ins, matching the bind-side order (the bridge resolves a
/// function against the transaction before falling back to DuckDB's system
/// catalog). Keeping the two layers in the same order means a backend function
/// and a built-in of the same name can never disagree between bind and compile.
fn find_table_function(
    name: &str,
    transaction: &dyn CatalogTransaction,
) -> Option<Box<dyn TableFunction>> {
    transaction
        .bind_default_table_function(name)
        .or_else(|| builtin(name))
}

/// The generic built-in table functions, keyed by name. These depend only on
/// their arguments (no catalog), so they live in the planner rather than being
/// contributed by a backend.
fn builtin(name: &str) -> Option<Box<dyn TableFunction>> {
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
        transaction: &dyn CatalogTransaction,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let function = find_table_function(&self.function_name, transaction)
            .ok_or_else(|| Error::UnsupportedTableFunction(self.function_name.clone()))?;
        let bound = function
            .bind(&self.args)
            .map_err(|source| Error::BindTableFunction {
                function: self.function_name.clone(),
                source,
            })?;
        // A call that binds to a table is planned as a scan of that table, so
        // one reaching this operator means the binder built the wrong node for
        // it, not that the query asked for something unsupported.
        let BoundTableFunction::Rows(rows) = bound else {
            return Err(Error::TableFunctionBoundAsTable(self.function_name.clone()));
        };
        self.project(rows.compile(dispatcher)?)
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

/// The error a call whose arguments the function cannot accept raises while
/// binding. Phrased as the statement's error: the argument is the user's, so
/// what is wrong with it is what they need told.
pub(crate) fn invalid_argument(function: &str, message: String) -> crate::catalog::Error {
    crate::catalog::Error::Other(
        format!("invalid argument to table function {function}: {message}").into(),
    )
}
