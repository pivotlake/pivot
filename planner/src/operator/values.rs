//! Rows produced by a SQL `VALUES` clause.
//!
//! Every cell is an [`Expression`], evaluated per row at run time. A prepared
//! `VALUES ($1, $2)` carries its parameters as
//! [`Expression::Parameter`](crate::expression::Expression) holes like any
//! other expression; [`Plan::bind_parameters`](crate::Plan::bind_parameters)
//! substitutes the bound values as constants before compilation.

use std::fmt;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_schema::{Field, Schema};
use dispatch::{DataFlowDispatcher, RecordBatchOperatorSpec, values_input};

use crate::compile::{Error, ExprFn};
use crate::expression::Expression;

#[derive(Debug, Clone)]
pub struct Values {
    pub rows: Vec<Vec<Expression>>,
}

impl fmt::Display for Values {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Values(rows: {})", self.rows.len())
    }
}

impl Values {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // Compile on the coordinator so unsupported expressions fail before the
        // dataflow launches. The resulting builder closures are Sync and can be
        // shared; each worker creates private evaluators for the row it steals.
        let rows: Vec<Vec<Arc<ExprFn>>> = self
            .rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|expression| expression.compile().map(Arc::new))
                    .collect()
            })
            .collect::<Result<_, _>>()?;

        // A single empty row every constant expression evaluates against; the
        // columns are the same regardless of its contents. Built once here and
        // reused for every row instead of rebuilt per row.
        let single_row = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(1)),
        )
        .expect("one empty row is a valid batch");

        Ok(values_input(dispatcher, rows)
            .map_each(move |row: Vec<Arc<ExprFn>>| {
                let columns: Vec<ArrayRef> = row
                    .iter()
                    .map(|builder| builder()(&single_row).into_array(1))
                    .collect();
                // Column names carry no meaning here: the insert matches VALUES
                // columns to the table positionally, so leave them empty.
                let fields = columns
                    .iter()
                    .map(|column| Field::new("", column.data_type().clone(), true))
                    .collect::<Vec<_>>();
                RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
                    .expect("VALUES expressions in one row have equal length")
            })
            .record_batches())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn bound_values_produce_the_output_rows(mut testing_planner: TestingPlanner) {
        let plan = testing_planner
            .plan("SELECT * FROM (VALUES (CAST($1 AS INTEGER)), (CAST($2 AS INTEGER))) t(x)")
            .unwrap();
        let params: Vec<ArrayRef> = vec![
            Arc::new(arrow_array::Int32Array::from(vec![10])),
            Arc::new(arrow_array::Int32Array::from(vec![20])),
        ];

        let mut rows = run_bound(&testing_planner, &plan, &params);

        rows.sort_by_key(|row| row["x"].as_i64().unwrap());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["x"], 10);
        assert_eq!(rows[1]["x"], 20);
    }

    #[rstest]
    fn null_parameter_outside_an_insert_is_allowed(mut testing_planner: TestingPlanner) {
        let plan = testing_planner
            .plan("SELECT * FROM (VALUES (CAST($1 AS INTEGER))) t(x)")
            .unwrap();
        let params: Vec<ArrayRef> =
            vec![Arc::new(arrow_array::Int32Array::from(vec![None::<i32>]))];

        let rows = run_bound(&testing_planner, &plan, &params);

        // Nothing is stored by a SELECT, so the NULL flows through; only an
        // INSERT rejects NULL parameters (columns are non-nullable). The JSON
        // writer drops null fields, so the row is present but empty.
        assert_eq!(rows.len(), 1);
        assert!(rows[0].get("x").is_none_or(|value| value.is_null()));
    }

    #[rstest]
    fn volatile_expression_values_still_evaluate(mut testing_planner: TestingPlanner) {
        let plan = testing_planner
            .plan("SELECT * FROM (VALUES (now())) t(ts)")
            .unwrap();

        let rows = run_bound(&testing_planner, &plan, &[]);

        assert_eq!(rows.len(), 1);
    }
}
