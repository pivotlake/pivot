//! [`InList`] — an `input IN (v0, v1, …)` membership test.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow::compute::kernels::boolean::or;
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
    pub fn compile(&self, parameters: &compile::BoundParameters) -> Result<ExprFn, compile::Error> {
        // `x IN (a, b, …)` is the disjunction `x = a OR x = b OR …`. We compile
        // the tested expression and every list value once, then per batch
        // OR-reduce the equality masks.
        let input_builder = self.input.compile(parameters)?;
        let value_builders = self
            .values
            .iter()
            .map(|v| v.compile(parameters))
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
                    .reduce(|left, right| or(&left, &right).unwrap())
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
    // DuckDB folds a small `x IN (..)` in a WHERE into the OR conjunction (see
    // `conjunction`), so the `Expression::InList` path isn't reachable from a
    // SQL query through the parquet harness. Exercise `InList::compile`
    // directly instead: `ts IN (-1, 6)` over `[-1, 6, 3, 6]`.
    use super::InList;
    use crate::expression::{Expression, Ref};
    use crate::types::Type;
    use arrow_array::{ArrayRef, BooleanArray, Int16Array, RecordBatch, Scalar};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn compiles_to_membership_mask() {
        let constant = |v: i16| {
            Expression::Constant(Scalar::new(Arc::new(Int16Array::from(vec![v])) as ArrayRef))
        };
        let in_list = InList {
            input: Box::new(Expression::Ref(Ref {
                column_idx: 0,
                return_type: Type::Int16,
                name: None,
            })),
            values: vec![constant(-1), constant(6)],
        };

        let mut eval = in_list.compile(&[]).unwrap()();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("ts", DataType::Int16, false)])),
            vec![Arc::new(Int16Array::from(vec![-1i16, 6, 3, 6])) as ArrayRef],
        )
        .unwrap();

        let result = eval(&batch);
        let (arr, _) = result.as_datum().get();
        let mask = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            (0..mask.len()).map(|i| mask.value(i)).collect::<Vec<_>>(),
            vec![true, true, false, true]
        );
    }
}
