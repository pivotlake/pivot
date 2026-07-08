//! [`Filter`] — filters rows by one or more boolean conditions.

use crate::catalog::ConditionObserverFn;
use crate::compile::{Error, ExprEvalFn, ExprFn};
use crate::expression::Expression;
use arrow::compute::kernels::boolean::and;
use arrow::compute::take_record_batch;
use arrow_array::{Array, BooleanArray, RecordBatch, UInt32Array};
use dispatch::RecordBatchOperatorSpec;
use std::fmt;
use std::sync::Arc;

/// Filters rows by one or more boolean conditions (implicitly ANDed).
#[derive(Debug)]
pub struct Filter {
    pub conditions: Vec<Expression>,
}

impl fmt::Display for Filter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let conds = self
            .conditions
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(" AND ");
        write!(f, "Filter({conds})")
    }
}

impl Filter {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let filters = compile_conditions(self.conditions.iter())?;

        Ok(input.filter(|| {
            let mut eval_fns: Vec<ExprEvalFn> = filters.iter().map(|f| f()).collect();
            move |batch: &RecordBatch| evaluate_mask(&mut eval_fns, batch)
        }))
    }

    /// Compile a whole filter stack (`stages`, bottom-up, each stage one
    /// Filter node's conditions) into one filter that reports every batch and
    /// its full-length mask to `observer` before dispatch acts on the mask —
    /// so even a zero-survivor batch is observed, which the observer's
    /// row-group completeness accounting depends on.
    ///
    /// Stages evaluate like the unfused stack would: each stage's conditions
    /// run only on the rows surviving the stages below (via an intermediate
    /// filtered batch), so an expensive residual condition above a selective
    /// pushdown never evaluates on rows the pushdown already dropped. The
    /// surviving row identities are threaded through, producing one mask over
    /// the ORIGINAL batch rows for the observer.
    pub(crate) fn compile_observed(
        stages: &[&Filter],
        input: RecordBatchOperatorSpec,
        observer: ConditionObserverFn,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let compiled_stages: Vec<Arc<Vec<ExprFn>>> = stages
            .iter()
            .map(|stage| compile_conditions(stage.conditions.iter()))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(input.filter(move || {
            let observer = observer.clone();
            let mut stage_eval_fns: Vec<Vec<ExprEvalFn>> = compiled_stages
                .iter()
                .map(|stage| stage.iter().map(|f| f()).collect())
                .collect();
            move |batch: &RecordBatch| {
                let mask = evaluate_staged_mask(&mut stage_eval_fns, batch);
                observer(batch, &mask);
                mask
            }
        }))
    }
}

/// Evaluate a filter stack stage by stage, each stage running only on the
/// rows surviving the previous stages, returning the combined mask over the
/// original batch rows.
fn evaluate_staged_mask(stages: &mut [Vec<ExprEvalFn>], batch: &RecordBatch) -> BooleanArray {
    let (first, rest) = stages.split_first_mut().expect("at least one stage");
    let mut mask = evaluate_mask(first, batch);
    if rest.is_empty() {
        return mask;
    }

    // Conditions reference only the scan's data columns (positional, ahead of
    // any trailing metadata pair), so the intermediate batches carry just
    // those. Beyond dropping dead weight, this keeps arrow's take away from
    // the run-end-encoded row-group column, which it processes one logical
    // row at a time.
    let data_columns = batch.num_columns() - dispatch::trailing_metadata_columns(&batch.schema());
    let data_batch = batch
        .project(&(0..data_columns).collect::<Vec<_>>())
        .expect("in-bounds projection");

    for stage in rest {
        // A null mask slot excludes its row, exactly as filter_record_batch
        // treats it downstream.
        let survivors: Vec<u32> = mask
            .iter()
            .enumerate()
            .filter_map(|(row, keep)| (keep == Some(true)).then_some(row as u32))
            .collect();
        if survivors.is_empty() {
            return BooleanArray::from(vec![false; batch.num_rows()]);
        }
        let indices = UInt32Array::from(survivors.clone());
        let sub_batch = take_record_batch(&data_batch, &indices).expect("in-bounds take");
        let sub_mask = evaluate_mask(stage, &sub_batch);

        let mut combined = vec![false; batch.num_rows()];
        for (sub_row, row) in survivors.into_iter().enumerate() {
            if sub_mask.is_valid(sub_row) && sub_mask.value(sub_row) {
                combined[row as usize] = true;
            }
        }
        mask = BooleanArray::from(combined);
    }
    mask
}

fn compile_conditions<'a>(
    conditions: impl Iterator<Item = &'a Expression>,
) -> Result<Arc<Vec<ExprFn>>, Error> {
    let filters = Arc::new(
        conditions
            .map(|e| e.compile())
            .collect::<Result<Vec<_>, _>>()?,
    );
    assert!(!filters.is_empty());
    Ok(filters)
}

/// AND together each compiled condition's boolean mask for one batch.
fn evaluate_mask(eval_fns: &mut [ExprEvalFn], batch: &RecordBatch) -> BooleanArray {
    eval_fns
        .iter_mut()
        .map(|f| {
            let result = f(batch);
            let (arr, _) = result.as_datum().get();
            arr.as_any().downcast_ref::<BooleanArray>().unwrap().clone()
        })
        .reduce(|left, right| and(&left, &right).unwrap())
        .unwrap()
}
