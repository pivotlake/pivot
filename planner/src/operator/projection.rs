//! [`Projection`] — computes a list of output expressions from its child.

use crate::compile::{Error, ExprEvalFn, ExprFn, ExprResult};
use crate::expression::Expression;
use arrow_array::{ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::{Field, Schema};
use dispatch::RecordBatchOperatorSpec;
use duckdb_planner::operator as duckdb_operator;
use std::fmt;
use std::sync::Arc;

/// Computes a list of output expressions from its child's columns.
#[derive(Debug)]
pub struct Projection {
    pub projections: Vec<Expression>,
}

impl TryFrom<duckdb_operator::Projection> for Projection {
    type Error = super::Error;
    fn try_from(p: duckdb_operator::Projection) -> Result<Self, Self::Error> {
        Ok(Projection {
            projections: p
                .projections
                .into_iter()
                .map(Expression::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
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
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // Fast path: every projection is a plain column reference, so we can
        // select columns zero-copy and preserve the input schema's fields.
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
                .map(|e| e.compile())
                .collect::<Result<Vec<_>, _>>()?,
        );

        Ok(input.project(move || {
            let mut evals: Vec<ExprEvalFn> = builders.iter().map(|b| b()).collect();
            // A constant projection (e.g. `SELECT 1`) evaluates to the same scalar for
            // every batch, but each output column must match the batch's row count, so
            // the scalar has to be broadcast to a full-length array. That array depends
            // only on the row count, so build it once per projection and reuse it (a
            // cheap `Arc` clone) whenever the next batch has the same length — batches
            // are usually uniform, so it's built once and cloned thereafter.
            let mut const_cols: Vec<Option<(usize, ArrayRef)>> = vec![None; evals.len()];
            move |batch: RecordBatch| {
                let num_rows = batch.num_rows();
                let columns: Vec<ArrayRef> = evals
                    .iter_mut()
                    .enumerate()
                    .map(|(i, eval)| match eval(&batch) {
                        ExprResult::Array(a) => a,
                        ExprResult::Scalar(s) => const_cols[i]
                            .as_ref()
                            .filter(|(len, _)| *len == num_rows)
                            .map(|(_, arr)| arr.clone())
                            .unwrap_or_else(|| {
                                let indices = UInt32Array::from(vec![0u32; num_rows]);
                                let arr =
                                    arrow::compute::take(&s.into_inner(), &indices, None).unwrap();
                                const_cols[i] = Some((num_rows, arr.clone()));
                                arr
                            }),
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
