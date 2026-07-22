//! [`Projection`] — computes a list of output expressions from its child.

use crate::compile::{Error, ExprEvalFn, ExprFn};
use crate::expression::Expression;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{Field, Schema};
use dispatch::RecordBatchOperatorSpec;
use std::fmt;
use std::sync::Arc;

/// Computes a list of output expressions from its child's columns.
#[derive(Debug)]
pub struct Projection {
    pub projections: Vec<Expression>,
}

impl fmt::Display for Projection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let exprs = self
            .projections
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Projection({exprs})")
    }
}

impl Projection {
    pub fn compile(
        &self,
        input: RecordBatchOperatorSpec,
        parameters: &crate::compile::BoundParameters,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // Fast path: every projection is a plain column reference, so we can
        // select columns zero-copy and preserve the input schema's fields. The
        // query's client-facing column names are stamped on once at the plan
        // root (see `Plan::compile`), so intermediate field names don't matter.
        if let Some(idxs) = self
            .projections
            .iter()
            .map(|e| match e {
                Expression::Ref(n) => Some(n.column_idx),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
        {
            let idxs = Arc::new(idxs);
            return Ok(input.project(move || {
                let idxs = idxs.clone();
                move |batch: RecordBatch| {
                    // A late-materialized narrow scan appends row-group metadata
                    // columns the downstream materializer reads positionally from
                    // the end. A plain column projection would drop them, so carry
                    // any such trailing pair through untouched.
                    let meta = dispatch::trailing_metadata_columns(batch.schema_ref());
                    if meta == 0 {
                        batch.project(&idxs).unwrap()
                    } else {
                        let n = batch.num_columns();
                        let mut cols: Vec<usize> = idxs.as_ref().clone();
                        cols.extend((n - meta)..n);
                        batch.project(&cols).unwrap()
                    }
                }
            }));
        }

        // General path: at least one projection is a computed expression
        // (e.g. the `sum / count` divide of an AVG). Evaluate each projection
        // per batch and assemble a new RecordBatch, deriving the output schema
        // from the produced arrays.
        let builders: Arc<Vec<ExprFn>> = Arc::new(
            self.projections
                .iter()
                .map(|e| e.compile(parameters))
                .collect::<Result<Vec<_>, _>>()?,
        );

        Ok(input.project(move || {
            let mut evals: Vec<ExprEvalFn> = builders.iter().map(|b| b()).collect();
            move |batch: RecordBatch| {
                let num_rows = batch.num_rows();
                // A constant projection (e.g. `SELECT 1`) yields a length-1 scalar;
                // `into_array` broadcasts it to the batch's row count so every output
                // column has the same length.
                let columns: Vec<ArrayRef> = evals
                    .iter_mut()
                    .map(|eval| eval(&batch).into_array(num_rows))
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
