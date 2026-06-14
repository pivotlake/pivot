//! [`Case`] — a `CASE WHEN … THEN … ELSE … END` expression.

use super::{Error, Expression};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow::compute::kernels::zip::zip;
use arrow_array::{BooleanArray, RecordBatch};
use duckdb_planner::expression as duckdb_expression;
use std::fmt::{self, Display};

/// One `WHEN when THEN then` arm of a [`Case`].
#[derive(Debug, Clone)]
pub struct CaseCheck {
    pub when: Box<Expression>,
    pub then: Box<Expression>,
}

/// A `CASE WHEN … THEN … [WHEN …] ELSE … END` expression. Evaluates each
/// `when` predicate in order and yields the first matching `then`, falling back
/// to `else_expr`; see its compile impl, which folds the arms with arrow's
/// `zip` kernel.
#[derive(Debug, Clone)]
pub struct Case {
    pub checks: Vec<CaseCheck>,
    pub else_expr: Box<Expression>,
}

impl TryFrom<duckdb_expression::Case> for Case {
    type Error = Error;
    fn try_from(c: duckdb_expression::Case) -> Result<Self, Self::Error> {
        let checks = c
            .checks
            .into_iter()
            .map(|check| {
                Ok(CaseCheck {
                    when: Box::<Expression>::try_from(check.when)?,
                    then: Box::<Expression>::try_from(check.then)?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Case {
            checks,
            else_expr: Box::<Expression>::try_from(c.else_expr)?,
        })
    }
}

impl Display for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CASE")?;
        for check in &self.checks {
            write!(f, " WHEN {} THEN {}", check.when, check.then)?;
        }
        write!(f, " ELSE {} END", self.else_expr)
    }
}

impl Case {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        // Compile the ELSE branch and each (WHEN, THEN) arm once. Per batch we
        // start from the ELSE value and fold the arms back-to-front with arrow's
        // `zip` (selecting `then` where the WHEN mask is true, else the running
        // result). Applying the *first* arm last makes it win on overlap, giving
        // SQL's first-match-wins CASE semantics. A NULL WHEN reads as false.
        //
        // `zip` requires `then` and the running result share a data type; DuckDB
        // unifies all branch types when binding the CASE, so they always do.
        let else_builder = self.else_expr.compile()?;
        let arm_builders = self
            .checks
            .iter()
            .map(|c| Ok((c.when.compile()?, c.then.compile()?)))
            .collect::<Result<Vec<_>, compile::Error>>()?;
        Ok(Box::new(move || {
            let mut else_eval = else_builder();
            let mut arm_evals: Vec<(ExprEvalFn, ExprEvalFn)> = arm_builders
                .iter()
                .map(|(when, then)| (when(), then()))
                .collect();
            Box::new(move |batch: &RecordBatch| {
                let mut result = else_eval(batch);
                for (when_eval, then_eval) in arm_evals.iter_mut().rev() {
                    let when = when_eval(batch);
                    let (when_arr, _) = when.as_datum().get();
                    let mask = when_arr.as_any().downcast_ref::<BooleanArray>().unwrap();
                    let then = then_eval(batch);
                    let zipped = zip(mask, then.as_datum(), result.as_datum()).unwrap();
                    result = ExprResult::Array(zipped);
                }
                result
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn two_arm_case_in_projection(mut testing_planner: TestingPlanner) {
        // a < 3 -> 'low' (a=1,2), else 'high' (a=3,4,5).
        let mut labels = run(
            &mut testing_planner,
            "SELECT CASE WHEN a < 3 THEN 'low' ELSE 'high' END FROM example_table",
        )
        .iter()
        .map(|r| r["col0"].as_str().unwrap().to_string())
        .collect::<Vec<_>>();

        labels.sort();

        assert_eq!(labels, vec!["high", "high", "high", "low", "low"]);
    }
}
