//! [`MaybeError`] — a runtime guard that fails the query when its check trips.
//!
//! DuckDB plans `CASE WHEN check THEN error('…') ELSE value END` around every
//! scalar subquery (check = "more than one row came back"). Running that as a
//! plain CASE would evaluate `error()` eagerly on every batch, so the build
//! step lowers the shape to this expression instead: evaluate the check, fail
//! the query with the message if any row trips, otherwise yield `value`.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use arrow_array::{BooleanArray, RecordBatch};
use std::fmt::{self, Display};

/// `CASE WHEN check THEN error(message) ELSE value END`, evaluated lazily: the
/// message only renders (and the query only fails) when a row's check is true.
#[derive(Debug, Clone)]
pub struct MaybeError {
    pub check: Box<Expression>,
    /// The `error()` call's message argument, evaluated only on failure.
    pub message: Box<Expression>,
    pub value: Box<Expression>,
}

impl Display for MaybeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "maybe_error(if {} then {}, else {})",
            self.check, self.message, self.value
        )
    }
}

impl MaybeError {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let check_builder = self.check.compile()?;
        let message_builder = self.message.compile()?;
        let value_builder = self.value.compile()?;
        Ok(Box::new(move || {
            let mut check_eval = check_builder();
            let mut message_eval = message_builder();
            let mut value_eval = value_builder();
            Box::new(move |batch: &RecordBatch| {
                let check = check_eval(batch);
                let (check_arr, _) = check.as_datum().get();
                let mask = check_arr.as_any().downcast_ref::<BooleanArray>().unwrap();
                // A NULL check reads as false, exactly as a CASE arm would.
                if mask.true_count() > 0 {
                    // The panic fails the query through the dataflow's worker
                    // error path, surfacing the message to the client.
                    let message = message_eval(batch);
                    let (message_arr, _) = message.as_datum().get();
                    let rendered = ArrayFormatter::try_new(message_arr, &FormatOptions::default())
                        .map(|f| f.value(0).to_string())
                        .unwrap_or_else(|_| "error() raised".to_string());
                    panic!("{rendered}");
                }
                value_eval(batch)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn single_row_scalar_subquery_yields_its_value(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT a FROM example_table \
             WHERE a = (SELECT b - 18 FROM example_table WHERE name = 'bob')",
        );

        assert_eq!(rows.len(), 1);
        assert_eq!(*only_column(&rows[0]), 2);
    }

    #[rstest]
    fn multi_row_scalar_subquery_fails_the_query(mut testing_planner: TestingPlanner) {
        let error = run_expecting_error(
            &mut testing_planner,
            "SELECT a FROM example_table WHERE a = (SELECT b FROM example_table)",
        );

        assert!(
            error.contains("More than one row returned by a subquery"),
            "unexpected error: {error}"
        );
    }
}
