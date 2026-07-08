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
        let filters = Arc::new(
            self.conditions
                .iter()
                .map(|e| e.compile())
                .collect::<Result<Vec<_>, _>>()?,
        );
        assert!(!filters.is_empty());

        // Conditions apply progressively: each one's survivors are what the
        // next condition evaluates, so a selective early condition spares the
        // later ones most of the rows. Gathering the surviving rows costs a
        // pass over every column, though, so the batch only shrinks while the
        // remaining conditions are expensive enough to repay it; the tail of
        // cheap conditions just ANDs masks and gathers once at the end.
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
        const SHRINK_KERNEL_THRESHOLD: usize = 4;

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
                    if remaining >= SHRINK_KERNEL_THRESHOLD {
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
