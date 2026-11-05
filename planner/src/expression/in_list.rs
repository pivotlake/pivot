//! [`InList`] — an `input IN (v0, v1, …)` membership test.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow::compute::kernels::boolean::or_kleene;
use arrow_array::{ArrayRef, BooleanArray, RecordBatch};
use arrow_ord::cmp::eq;
use std::fmt::{self, Display};
use std::sync::Arc;

/// An `input IN (v0, v1, …)` membership test. Evaluates to a boolean column;
/// see its compile impl, which expands it to an OR of per-value equalities.
#[derive(Debug, Clone)]
pub struct InList {
    pub input: Box<Expression>,
    pub values: Vec<Expression>,
}

impl Display for InList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let values: Vec<String> = self.values.iter().map(|v| v.to_string()).collect();
        write!(f, "{} IN ({})", self.input, values.join(", "))
    }
}

impl InList {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        // `x IN (a, b, …)` is the disjunction `x = a OR x = b OR …`. We compile
        // the tested expression and every list value once, then per batch
        // OR-reduce the equality masks. The Kleene kernel keeps SQL
        // three-valued logic: a match beside a NULL list value is TRUE, and
        // only a miss beside a NULL is NULL.
        let input_builder = self.input.compile()?;
        let value_builders = self
            .values
            .iter()
            .map(|v| v.compile())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Box::new(move || {
            let mut input_expr = input_builder();
            let mut value_exprs: Vec<ExprEvalFn> = value_builders.iter().map(|b| b()).collect();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let mask = value_exprs
                    .iter_mut()
                    .map(|v| {
                        let value = v(batch);
                        eq(input.as_datum(), value.as_datum()).expect("IN operands share a type")
                    })
                    .reduce(|left, right| or_kleene(&left, &right).unwrap())
                    // An empty list (`IN ()`) matches nothing; DuckDB folds this
                    // away before planning, so this is only a defensive fallback.
                    .unwrap_or_else(|| BooleanArray::from(vec![false; batch.num_rows()]));
                ExprResult::Array(Arc::new(mask) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    // DuckDB rewrites `x IN (..)` into an OR of equalities (see `conjunction`)
    // in projections and filters only, so an IN reaches pivot as an
    // `Expression::InList` from an aggregate's argument, or from a filter
    // DuckDB pushed into a scan.
    use super::InList;
    use crate::expression::{Expression, Ref};
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::{ArrayRef, BooleanArray, Int16Array, RecordBatch, Scalar};
    use arrow_schema::{DataType, Field, Schema};
    use rstest::rstest;
    use std::sync::Arc;

    fn constant(value: Option<i16>) -> Expression {
        Expression::Constant(Scalar::new(
            Arc::new(Int16Array::from(vec![value])) as ArrayRef
        ))
    }

    fn ts_in(values: Vec<Expression>) -> InList {
        InList {
            input: Box::new(Expression::Ref(Ref {
                column_idx: 0,
                return_type: Type::Int16,
                name: None,
            })),
            values,
        }
    }

    fn evaluate(in_list: &InList, ts: Vec<i16>) -> Vec<Option<bool>> {
        let mut eval = in_list.compile().unwrap()();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("ts", DataType::Int16, false)])),
            vec![Arc::new(Int16Array::from(ts)) as ArrayRef],
        )
        .unwrap();
        let result = eval(&batch);
        let (arr, _) = result.as_datum().get();
        let mask = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
        mask.iter().collect()
    }

    #[test]
    fn compiles_to_membership_mask() {
        let in_list = ts_in(vec![constant(Some(-1)), constant(Some(6))]);

        let mask = evaluate(&in_list, vec![-1, 6, 3, 6]);

        assert_eq!(mask, vec![Some(true), Some(true), Some(false), Some(true)]);
    }

    #[test]
    fn a_match_beside_a_null_value_is_true() {
        let in_list = ts_in(vec![constant(Some(6)), constant(None)]);

        let mask = evaluate(&in_list, vec![6, 3]);

        assert_eq!(mask, vec![Some(true), None]);
    }

    #[rstest]
    fn an_aggregate_argument_keeps_three_valued_logic(mut testing_planner: TestingPlanner) {
        // Row a = 2 has b NULL, and `2 IN (2, NULL)` is TRUE.
        let rows = run(
            &mut testing_planner,
            "SELECT count(CASE WHEN a IN (2, b) THEN 1 END) AS hits FROM nullable_table",
        );

        assert_eq!(
            rows,
            serde_json::json!([{"hits": 1}]).as_array().unwrap().clone()
        );
    }
}
