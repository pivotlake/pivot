//! [`ExpressionGet`] - a scan over a list of constant rows (DuckDB's
//! `LOGICAL_EXPRESSION_GET`), the source under an `INSERT ... VALUES` (and a bare
//! `VALUES` query).
//!
//! The rows are constants known at plan time, so the operator builds the whole
//! batch once at compile and emits it from a single-batch source, over which the
//! [`Insert`](super::Insert) above reorders and casts it into the table's schema.

use crate::compile::Error;
use crate::expression::Expression;
use arrow_array::{Array, ArrayRef, Datum, RecordBatch};
use arrow_schema::{Field, Schema};
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec};
use duckdb_planner::operator as duckdb_operator;
use std::fmt;
use std::sync::Arc;

/// A list of constant rows (row-major: `rows[r][c]`).
#[derive(Debug)]
pub struct ExpressionGet {
    rows: Vec<Vec<Expression>>,
}

impl TryFrom<duckdb_operator::ExpressionGet> for ExpressionGet {
    type Error = super::Error;

    fn try_from(e: duckdb_operator::ExpressionGet) -> Result<Self, Self::Error> {
        let rows = e
            .rows
            .into_iter()
            .map(|row| -> Result<Vec<Expression>, super::Error> {
                row.into_iter()
                    .map(|cell| Ok(Expression::try_from(cell)?))
                    .collect()
            })
            .collect::<Result<Vec<_>, super::Error>>()?;
        Ok(Self { rows })
    }
}

impl fmt::Display for ExpressionGet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExpressionGet({} rows)", self.rows.len())
    }
}

impl ExpressionGet {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // The rows are constants, so build the batch once and emit it from a
        // single-batch source (the same helper a global aggregate uses to emit its
        // one precomputed result batch).
        let batch = self.build_batch()?;
        Ok(dispatch::values_input(dispatcher, [batch]).record_batches())
    }

    /// Assemble the constant rows into one [`RecordBatch`], column by column.
    /// Column names are positional placeholders (`col0`, ...) - the
    /// [`Insert`](super::Insert) above renames and casts them to the table schema.
    fn build_batch(&self) -> Result<RecordBatch, Error> {
        let n_cols = self.rows.first().map_or(0, Vec::len);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            // Each constant cell is a one-element array; concat the column's cells
            // (one per row) into the column. The `&dyn Array` borrows from the
            // owned `Scalar` in `self.rows`, which outlives this loop.
            let cells: Vec<&dyn Array> = self
                .rows
                .iter()
                .map(|row| match &row[c] {
                    Expression::Constant(scalar) => Ok(scalar.get().0),
                    other => Err(Error::UnsupportedExpression(other.clone())),
                })
                .collect::<Result<_, _>>()?;
            columns.push(arrow::compute::concat(&cells).map_err(Error::ValuesBatch)?);
        }
        let fields: Vec<Field> = columns
            .iter()
            .enumerate()
            .map(|(i, col)| Field::new(format!("col{i}"), col.data_type().clone(), true))
            .collect();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).map_err(Error::ValuesBatch)
    }
}
