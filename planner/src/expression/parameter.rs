//! [`Parameter`] - a prepared-statement placeholder (`$1`).
//!
//! A parameter is a first-class hole in the plan: planning types it (DuckDB
//! infers the type from context - the compared column, the INSERT target
//! column, an explicit cast) but leaves it unbound, so the plan can be prepared
//! once and executed many times. Each execute substitutes the bound values via
//! [`Plan::bind_parameters`](crate::Plan::bind_parameters), which replaces the
//! hole with an [`Expression::Constant`](crate::expression::Expression::Constant);
//! an unbound parameter reaching `compile` is therefore an error.

use crate::compile;
use crate::types::Type;
use std::fmt::{self, Display};

/// A typed hole for the `index`th (zero-based) statement parameter; `$1` is
/// index 0.
#[derive(Debug, Clone)]
pub struct Parameter {
    pub index: usize,
    pub return_type: Type,
}

impl Display for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${}:{}", self.index + 1, self.return_type)
    }
}

impl Parameter {
    pub fn compile(&self) -> Result<compile::ExprFn, compile::Error> {
        Err(compile::Error::UnboundParameter(self.index))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::{ArrayRef, Int32Array, StringViewArray};
    use rstest::rstest;
    use std::sync::Arc;

    fn int_param(value: i32) -> Vec<ArrayRef> {
        vec![Arc::new(Int32Array::from(vec![value]))]
    }

    #[rstest]
    fn binds_scalar_parameter_in_a_filter(mut testing_planner: TestingPlanner) {
        let plan = testing_planner
            .plan("SELECT a FROM example_table WHERE a = $1")
            .unwrap();

        let rows = run_bound(&testing_planner, &plan, &int_param(3));

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["a"], 3);
    }

    #[rstest]
    fn one_plan_binds_fresh_values_per_execute(mut testing_planner: TestingPlanner) {
        let plan = testing_planner
            .plan("SELECT name FROM example_table WHERE a = $1")
            .unwrap();

        let first = run_bound(&testing_planner, &plan, &int_param(2));
        let second = run_bound(&testing_planner, &plan, &int_param(4));

        assert_eq!(first[0]["name"], "bob");
        assert_eq!(second[0]["name"], "dave");
    }

    #[rstest]
    fn infers_parameter_types_from_the_compared_columns(mut testing_planner: TestingPlanner) {
        let plan = testing_planner
            .plan("SELECT a FROM example_table WHERE a = $1 AND name = $2")
            .unwrap();

        let types = plan.parameter_types().unwrap();

        assert_eq!(types, vec![Type::Int32, Type::Utf8]);
    }

    #[rstest]
    fn casts_a_text_bound_value_to_the_planned_type(mut testing_planner: TestingPlanner) {
        let plan = testing_planner
            .plan("SELECT a FROM example_table WHERE a = $1")
            .unwrap();
        let params: Vec<ArrayRef> = vec![Arc::new(StringViewArray::from(vec!["5"]))];

        let rows = run_bound(&testing_planner, &plan, &params);

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["a"], 5);
    }
}
