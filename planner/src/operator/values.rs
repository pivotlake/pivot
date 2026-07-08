//! [`Values`] - a `VALUES` list: rows of expressions each evaluated once.

use crate::compile::{Error, ExprEvalFn, ExprFn};
use crate::expression::Expression;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{Field, Schema};
use dispatch::RecordBatchOperatorSpec;
use std::fmt;
use std::sync::Arc;

/// A `VALUES` list (DuckDB's `LogicalExpressionGet`): `rows` of expressions,
/// each row producing one output row. Its child is the single-row
/// [`DummyScan`](crate::operator::DummyScan) the expressions evaluate against,
/// exactly like a `FROM`-less `SELECT`'s projection.
#[derive(Debug)]
pub struct Values {
    pub rows: Vec<Vec<Expression>>,
}

impl fmt::Display for Values {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rows = self
            .rows
            .iter()
            .map(|row| {
                let cells = row
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("({cells})")
            })
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Values({rows})")
    }
}

impl Values {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let column_count = self.rows.first().map_or(0, Vec::len);
        let builders: Arc<Vec<Vec<ExprFn>>> = Arc::new(
            self.rows
                .iter()
                .map(|row| row.iter().map(|e| e.compile()).collect::<Result<_, _>>())
                .collect::<Result<_, _>>()?,
        );

        // The child is the single-row DummyScan, so this projection runs exactly
        // once: evaluate every cell against that one empty row and gather each
        // column across the rows, yielding one batch of `rows.len()` rows.
        Ok(input.project(move || {
            let mut evals: Vec<Vec<ExprEvalFn>> = builders
                .iter()
                .map(|row| row.iter().map(|b| b()).collect())
                .collect();
            move |batch: RecordBatch| {
                let columns: Vec<ArrayRef> = (0..column_count)
                    .map(|column| {
                        let cells: Vec<ArrayRef> = evals
                            .iter_mut()
                            .map(|row| row[column](&batch).into_array(1))
                            .collect();
                        // DuckDB bound one logical type per column, but each
                        // cell's expression picks its own physical arrow
                        // representation (e.g. `Utf8` vs `Utf8View`), so
                        // reconcile them to the first cell's before
                        // concatenating.
                        let target = cells[0].data_type().clone();
                        let cells: Vec<ArrayRef> = cells
                            .iter()
                            .map(|cell| {
                                arrow::compute::cast(cell, &target)
                                    .expect("a column's cells share one logical type")
                            })
                            .collect();
                        let cells: Vec<&dyn arrow_array::Array> =
                            cells.iter().map(|a| a.as_ref()).collect();
                        arrow::compute::concat(&cells).expect("cells were cast to one type")
                    })
                    .collect();
                let fields: Vec<Field> = columns
                    .iter()
                    .enumerate()
                    .map(|(i, c)| Field::new(format!("col{i}"), c.data_type().clone(), true))
                    .collect();
                RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
            }
        }))
    }
}
