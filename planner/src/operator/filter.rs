//! [`Filter`] — filters rows by one or more boolean conditions.

use crate::compile::{Error, ExprEvalFn, ExprFn, SelectedEvalFn, SelectedFn};
use crate::expression::Expression;
use arrow::compute::kernels::boolean::and;
use arrow_array::cast::AsArray;
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
        // Order cheap predicates (comparisons, ranges) before expensive ones
        // (substring/regex scans) so the running mask is built before a scan
        // runs. A scan that can honor the mask (see `compile_selected`) then only
        // inspects the rows that survived the cheaper predicates; anything else
        // is evaluated densely and ANDed in.
        let (expensive, cheap): (Vec<_>, Vec<_>) = self
            .conditions
            .iter()
            .partition(|c| c.is_expensive_predicate());

        let conditions = cheap
            .iter()
            .chain(expensive.iter())
            .map(|&e| CompiledCondition::compile(e))
            .collect::<Result<Vec<_>, _>>()?;
        assert!(!conditions.is_empty());
        let conditions = Arc::new(conditions);

        Ok(input.filter(move || {
            let mut evals: Vec<ConditionEval> = conditions.iter().map(|c| c.build()).collect();
            move |batch: &RecordBatch| fold_conditions(&mut evals, batch)
        }))
    }
}

/// A compiled filter condition: either a dense boolean evaluator, or a
/// selection-aware one that runs only over the rows still set in the mask.
enum CompiledCondition {
    Dense(ExprFn),
    Selected(SelectedFn),
}

impl CompiledCondition {
    fn compile(expr: &Expression) -> Result<Self, Error> {
        match expr.compile_selected() {
            Some(selected) => Ok(CompiledCondition::Selected(selected?)),
            None => Ok(CompiledCondition::Dense(expr.compile()?)),
        }
    }

    /// Build this condition's per-worker evaluator.
    fn build(&self) -> ConditionEval {
        match self {
            CompiledCondition::Dense(f) => ConditionEval::Dense(f()),
            CompiledCondition::Selected(f) => ConditionEval::Selected(f()),
        }
    }
}

/// A per-worker evaluator for one condition. Mirrors [`CompiledCondition`].
enum ConditionEval {
    Dense(ExprEvalFn),
    Selected(SelectedEvalFn),
}

/// AND the conditions together into one mask, threading the running mask through
/// each condition. A dense condition is evaluated over the whole batch and ANDed
/// in; a selection-aware condition receives the running mask and returns
/// `mask AND condition` directly, so it only scans the rows still alive.
fn fold_conditions(evals: &mut [ConditionEval], batch: &RecordBatch) -> BooleanArray {
    let mut mask: Option<BooleanArray> = None;
    for eval in evals.iter_mut() {
        mask = Some(match eval {
            ConditionEval::Dense(f) => {
                let result = f(batch);
                let (arr, _) = result.as_datum().get();
                let dense = arr.as_boolean();
                match mask.take() {
                    Some(prev) => and(dense, &prev).unwrap(),
                    None => dense.clone(),
                }
            }
            // `None` selection (the first condition) means every row is still
            // alive, so the scan runs densely; otherwise it inspects only the
            // rows set in the running mask.
            ConditionEval::Selected(f) => f(batch, mask.take().as_ref()),
        });
        // Once no row survives, ANDing the remaining conditions can only keep it
        // empty, so skip them (and any substring scans they would run).
        if mask.as_ref().is_some_and(|m| m.true_count() == 0) {
            break;
        }
    }
    mask.expect("filter has at least one condition")
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn cheap_and_expensive_predicate(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT a, name FROM example_table WHERE contains(name, 'ali') AND a = 5",
        );

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["a"], 5);
        assert_eq!(rows[0]["name"], "alice");
    }

    #[rstest]
    fn cheap_predicate_prunes_all_rows(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE a = 999 AND contains(name, 'ali')",
        );

        assert!(rows.is_empty());
    }

    #[rstest]
    fn cheap_predicate_keeps_all_rows(mut testing_planner: TestingPlanner) {
        let mut rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE a > 0 AND contains(name, 'ali')",
        );

        rows.sort_by(|l, r| l["name"].to_string().cmp(&r["name"].to_string()));
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r["name"] == "alice"));
    }

    // `a IN (2, 5)` keeps bob and alice; the selection-aware `contains` then
    // scans only those two rows and drops bob, so a row the cheap predicate kept
    // is still rejected by the expensive one.
    #[rstest]
    fn expensive_predicate_drops_a_survivor(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT a, name FROM example_table WHERE a IN (2, 5) AND contains(name, 'ali')",
        );

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["a"], 5);
        assert_eq!(rows[0]["name"], "alice");
    }
}
