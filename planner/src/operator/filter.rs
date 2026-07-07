//! [`Filter`] — filters rows by one or more boolean conditions.

use crate::catalog::ConditionObserverFn;
use crate::compile::{Error, ExprEvalFn, ExprFn};
use crate::expression::Expression;
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
        let filters = compile_conditions(self.conditions.iter())?;

        Ok(input.filter(|| {
            let mut eval_fns: Vec<ExprEvalFn> = filters.iter().map(|f| f()).collect();
            move |batch: &RecordBatch| evaluate_mask(&mut eval_fns, batch)
        }))
    }

    /// Compile `conditions` (a whole filter stack, ANDed) into one filter that
    /// reports every batch and its mask to `observer` before dispatch acts on
    /// the mask — so even a zero-survivor batch is observed, which the
    /// observer's row-group completeness accounting depends on.
    pub(crate) fn compile_observed<'a>(
        conditions: impl Iterator<Item = &'a Expression>,
        input: RecordBatchOperatorSpec,
        observer: ConditionObserverFn,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let filters = compile_conditions(conditions)?;

        Ok(input.filter(move || {
            let observer = observer.clone();
            let mut eval_fns: Vec<ExprEvalFn> = filters.iter().map(|f| f()).collect();
            move |batch: &RecordBatch| {
                let mask = evaluate_mask(&mut eval_fns, batch);
                observer(batch, &mask);
                mask
            }
        }))
    }
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
