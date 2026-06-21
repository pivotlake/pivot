//! [`Filter`] — filters rows by one or more boolean conditions.

use crate::compile::{Error, ExprEvalFn};
use crate::expression::Expression;
use arrow::compute::kernels::boolean::and;
use arrow_array::{BooleanArray, RecordBatch};
use dispatch::RecordBatchOperatorSpec;
use duckdb_planner::operator as duckdb_operator;
use std::fmt;
use std::sync::Arc;

/// Filters rows by one or more boolean conditions (implicitly ANDed).
#[derive(Debug)]
pub struct Filter {
    pub conditions: Vec<Expression>,
}

impl TryFrom<duckdb_operator::Filter> for Filter {
    type Error = super::Error;
    fn try_from(f: duckdb_operator::Filter) -> Result<Self, Self::Error> {
        Ok(Filter {
            conditions: f
                .conditions
                .into_iter()
                .map(Expression::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
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

        Ok(input.filter(|| {
            let mut eval_fns: Vec<ExprEvalFn> = filters.iter().map(|f| f()).collect();
            move |batch: &RecordBatch| {
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
        }))
    }
}
