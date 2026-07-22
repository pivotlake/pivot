//! [`Filter`] — filters rows by one or more boolean conditions.

use crate::compile::{Error, ExprEvalFn};
use crate::expression::Expression;
use arrow::compute::filter_record_batch;
use arrow::compute::kernels::boolean::and;
use arrow_array::{BooleanArray, RecordBatch};
use dispatch::RecordBatchOperatorSpec;
use std::fmt;
use std::sync::Arc;

/// Filters rows by one or more boolean conditions (implicitly ANDed).
#[derive(Debug, Clone)]
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
        let filters = Arc::new(
            self.conditions
                .iter()
                .map(|e| e.compile())
                .collect::<Result<Vec<_>, _>>()?,
        );
        assert!(!filters.is_empty());

        // Conditions apply progressively: each one's survivors are what the
        // next condition evaluates, so a selective early condition spares the
        // later ones most of the rows (see the shrink decision below).
        let remaining_kernels: Vec<usize> = {
            let per_condition: Vec<usize> = self
                .conditions
                .iter()
                .map(Expression::count_kernels)
                .collect();
            (0..per_condition.len())
                .map(|i| per_condition[i + 1..].iter().sum())
                .collect()
        };
        // Cycles one gathered element-copy costs, measured in kernel-pass
        // units: a copy is an L1 load + store plus gather bookkeeping (~4
        // cycles), while a vectorized kernel touches an element in ~1.
        const SHRINK_COST_RATIO: f64 = 4.0;

        Ok(input.filter(move || {
            let mut eval_fns: Vec<ExprEvalFn> = filters.iter().map(|f| f()).collect();
            let remaining_kernels = remaining_kernels.clone();
            move |batch: RecordBatch| {
                let mut current = batch;
                let mut pending: Option<BooleanArray> = None;
                for (eval, &remaining) in eval_fns.iter_mut().zip(&remaining_kernels) {
                    if current.num_rows() == 0 {
                        break;
                    }
                    let result = eval(&current);
                    let (arr, _) = result.as_datum().get();
                    let mask = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
                    let mask = match pending.take() {
                        Some(previous) => and(&previous, mask).unwrap(),
                        None => mask.clone(),
                    };
                    let kept = mask.true_count();
                    if kept == 0 {
                        // Nothing survives; skip the remaining conditions.
                        return current.slice(0, 0);
                    }
                    // Shrink or carry the mask? Arrow kernels cannot skip rows,
                    // so rows that already failed a condition ("dead", the
                    // `1.0 - alive` fraction) are still computed over by every
                    // remaining condition unless the batch is physically
                    // compacted first. Compacting is not free either: a gather
                    // copies every SURVIVING row (the `alive` fraction; dead
                    // rows are simply not written) once per column, including
                    // columns no remaining condition reads.
                    //
                    // Compare the two options in expected per-row cost (the
                    // batch's row count multiplies both sides, so it cancels):
                    //
                    //   wasted ops/row if we keep the dead rows around:
                    //       remaining kernel passes x fraction dead
                    //   copy ops/row if we compact now:
                    //       SHRINK_COST_RATIO x columns x fraction alive
                    //
                    // Copy when the copy is cheaper than the waste it removes.
                    // A selective condition with lots of work left shrinks
                    // (e.g. 86% dead with an OR tree ahead: 17 x 0.86 wasted
                    // vs 4 x 8 x 0.14 copied); a barely-selective one, or one
                    // with only a cheap tail remaining, carries the mask and
                    // gathers once at the end.
                    let alive = kept as f64 / current.num_rows() as f64;
                    if remaining as f64 * (1.0 - alive)
                        > SHRINK_COST_RATIO * current.num_columns() as f64 * alive
                    {
                        current = filter_record_batch(&current, &mask)
                            .expect("mask length matches the batch");
                    } else {
                        pending = Some(mask);
                    }
                }
                match pending {
                    Some(mask) => {
                        filter_record_batch(&current, &mask).expect("mask length matches the batch")
                    }
                    None => current,
                }
            }
        }))
    }
}
