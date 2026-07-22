//! Drop bound parameter values into a prepared [`Plan`]'s holes.
//!
//! A statement with `$n` placeholders is planned once (the expensive DuckDB
//! round-trip) and the resulting [`Plan`] keeps each parameter as a typed
//! [`Expression::Parameter`] hole. Each execute calls [`Plan::bind_parameters`]
//! with that run's values, producing a plan with every hole substituted by a
//! constant; the bound plan then compiles and runs through the ordinary path,
//! and the original plan stays untouched and reusable.
//!
//! Each bound value arrives as a single-row arrow array of whatever type the
//! client sent (e.g. `Utf8` for a text-format parameter) and is cast to the
//! hole's planned type here, erroring rather than truncating on a value that
//! does not fit. A filter condition that gained its constant from a parameter
//! is also re-offered to the scan it sits on, so a prepared `WHERE id = $1`
//! keeps the same row-group pruning its inlined twin gets at plan time.

use std::collections::BTreeMap;

use arrow_array::{Array, ArrayRef, Scalar};
use arrow_schema::{ArrowError, DataType};
use thiserror::Error;

use crate::Plan;
use crate::expression::{Expression, TableFilter};
use crate::operator::Operator;
use crate::plan::PlanNode;
use crate::types::{Type, physical_arrow_type};

#[derive(Debug, Error)]
pub enum Error {
    #[error("statement takes {expected} parameters, {provided} were bound")]
    ParameterCount { expected: usize, provided: usize },
    #[error("no value bound for parameter ${}", .0 + 1)]
    MissingParameter(usize),
    #[error("parameter ${} is never used by the statement", .0 + 1)]
    UnusedParameter(usize),
    #[error("converting a bound parameter to {target}: {source}")]
    Cast {
        target: DataType,
        #[source]
        source: ArrowError,
    },
    #[error(
        "parameter ${} is NULL, but INSERT cannot store NULL (columns are non-nullable)",
        .0 + 1
    )]
    NullInsertParameter(usize),
    #[error("re-applying a bound filter to the scan: {0}")]
    Pushdown(#[from] crate::catalog::Error),
}

/// Cast a bound value to the arrow type the plan expects at its hole. Strict
/// on purpose: an out-of-range or malformed value errors instead of becoming
/// NULL.
fn cast_value(array: ArrayRef, target: &DataType) -> Result<ArrayRef, Error> {
    if array.data_type() == target {
        return Ok(array);
    }
    let options = arrow::compute::CastOptions {
        safe: false,
        ..Default::default()
    };
    arrow::compute::cast_with_options(&array, target, &options).map_err(|source| Error::Cast {
        target: target.clone(),
        source,
    })
}

/// One execute's bound values, with the bookkeeping the bind walk needs:
/// which values a hole has consumed (so leftovers are reported) and whether
/// the subtree being bound feeds an INSERT (where a NULL can never be stored,
/// every pivot column being non-nullable; rejecting it here keeps the failure
/// out of the insert sink's workers).
pub(crate) struct BoundValues<'a> {
    values: &'a [ArrayRef],
    used: Vec<bool>,
    reject_nulls: bool,
}

impl<'a> BoundValues<'a> {
    pub(crate) fn new(values: &'a [ArrayRef]) -> Self {
        Self {
            values,
            used: vec![false; values.len()],
            reject_nulls: false,
        }
    }

    /// The value for hole `index`, marked consumed and checked against the
    /// current NULL policy.
    pub(crate) fn take(&mut self, index: usize) -> Result<ArrayRef, Error> {
        let value = self
            .values
            .get(index)
            .ok_or(Error::MissingParameter(index))?;
        if self.reject_nulls && value.null_count() > 0 {
            return Err(Error::NullInsertParameter(index));
        }
        self.used[index] = true;
        Ok(value.clone())
    }

    /// A value that was bound but never consumed by any hole, if any.
    fn first_unused(&self) -> Option<usize> {
        self.used.iter().position(|used| !used)
    }
}

impl Plan {
    /// The planned type of each `$n` parameter, indexed by position (`$1`
    /// first). Errors when a position in the range never occurs in the plan
    /// (`$1, $3` with no `$2`), matching Postgres's prepare-time validation.
    pub fn parameter_types(&self) -> Result<Vec<Type>, Error> {
        let mut types = BTreeMap::new();
        collect_node_parameters(&self.root, &mut types);
        let count = types.keys().next_back().map_or(0, |last| last + 1);
        (0..count)
            .map(|index| types.remove(&index).ok_or(Error::UnusedParameter(index)))
            .collect()
    }

    /// Produce a copy of this plan with every parameter hole filled from
    /// `params` (one single-row array per parameter, by position). The plan
    /// itself is the reuse boundary: it stays untouched, so a cached prepared
    /// plan can be bound concurrently by many executes.
    pub fn bind_parameters(&self, params: &[ArrayRef]) -> Result<Plan, Error> {
        let mut root = self.root.clone();
        let mut bound = BoundValues::new(params);
        bind_node(&mut root, &mut bound)?;
        if let Some(index) = bound.first_unused() {
            return Err(Error::UnusedParameter(index));
        }
        Ok(Plan {
            catalog: self.catalog.clone(),
            root,
            output_names: self.output_names.clone(),
        })
    }
}

fn bind_node(node: &mut PlanNode, bound: &mut BoundValues<'_>) -> Result<(), Error> {
    // Everything beneath an INSERT feeds the table, so NULL values are
    // rejected there (and only there: a NULL compared or selected is fine).
    let entering_insert = matches!(node.operator, Operator::Insert(_)) && !bound.reject_nulls;
    if entering_insert {
        bound.reject_nulls = true;
    }
    for input in &mut node.inputs {
        bind_node(input, bound)?;
    }
    if entering_insert {
        bound.reject_nulls = false;
    }

    // Remember which filter conditions carry holes before substituting: those
    // are re-offered to the scan below once their constants exist.
    let parameterized_conditions: Vec<usize> = match &node.operator {
        Operator::Filter(filter) => filter
            .conditions
            .iter()
            .enumerate()
            .filter(|(_, condition)| contains_parameter(condition))
            .map(|(index, _)| index)
            .collect(),
        _ => Vec::new(),
    };
    for expression in node.operator.expressions_mut() {
        bind_expression(expression, bound)?;
    }
    push_bound_conditions(node, &parameterized_conditions)?;
    Ok(())
}

/// Re-offer a filter's freshly bound conditions to the scan they sit on, the
/// same [`Table::pushdown_filter`](crate::catalog::Table::pushdown_filter)
/// call the planner makes for inline constants. Without this a prepared
/// `WHERE id = $1` would scan every row group forever, since plan-time
/// pushdown saw a hole instead of a constant. A condition the table consumed
/// outright is dropped from the filter, exactly as at plan time.
fn push_bound_conditions(node: &mut PlanNode, condition_indices: &[usize]) -> Result<(), Error> {
    if condition_indices.is_empty() {
        return Ok(());
    }
    let Some(child) = node.inputs.first_mut() else {
        return Ok(());
    };
    let (Operator::Filter(filter), Operator::Input(input)) =
        (&mut node.operator, &mut child.operator)
    else {
        return Ok(());
    };
    let mut consumed = Vec::new();
    for &index in condition_indices {
        let condition = TableFilter::Expression(Box::new(filter.conditions[index].clone()));
        if input.table.pushdown_filter(condition)? {
            consumed.push(index);
        }
    }
    for &index in consumed.iter().rev() {
        filter.conditions.remove(index);
    }
    Ok(())
}

/// Substitute every scalar parameter hole in the tree with the bound value,
/// cast to the hole's planned type and wrapped as a constant.
fn bind_expression(expression: &mut Expression, bound: &mut BoundValues<'_>) -> Result<(), Error> {
    if let Expression::Parameter(parameter) = expression {
        let value = bound.take(parameter.index)?;
        let target = physical_arrow_type(&parameter.return_type);
        let value = cast_value(value, &target)?;
        *expression = Expression::Constant(Scalar::new(value));
        return Ok(());
    }
    for child in expression.children_mut() {
        bind_expression(child, bound)?;
    }
    Ok(())
}

fn contains_parameter(expression: &Expression) -> bool {
    matches!(expression, Expression::Parameter(_))
        || expression.children().into_iter().any(contains_parameter)
}

fn collect_node_parameters(node: &PlanNode, types: &mut BTreeMap<usize, Type>) {
    for input in &node.inputs {
        collect_node_parameters(input, types);
    }
    for expression in node.operator.expressions() {
        collect_expression_parameters(expression, types);
    }
}

fn collect_expression_parameters(expression: &Expression, types: &mut BTreeMap<usize, Type>) {
    if let Expression::Parameter(parameter) = expression {
        types.insert(parameter.index, parameter.return_type.clone());
        return;
    }
    for child in expression.children() {
        collect_expression_parameters(child, types);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;
    use arrow_array::StringViewArray;
    use rstest::rstest;
    use std::sync::Arc;

    #[rstest]
    fn rejects_an_extra_bound_value(mut testing_planner: TestingPlanner) {
        let plan = testing_planner
            .plan("SELECT a FROM example_table WHERE a = $1")
            .unwrap();
        let params: Vec<ArrayRef> = vec![
            Arc::new(arrow_array::Int32Array::from(vec![1])),
            Arc::new(arrow_array::Int32Array::from(vec![2])),
        ];

        let error = plan.bind_parameters(&params).unwrap_err();

        assert!(matches!(error, Error::UnusedParameter(1)));
    }

    #[rstest]
    fn rejects_a_missing_bound_value(mut testing_planner: TestingPlanner) {
        let plan = testing_planner
            .plan("SELECT a FROM example_table WHERE a = $1")
            .unwrap();

        let error = plan.bind_parameters(&[]).unwrap_err();

        assert!(matches!(error, Error::MissingParameter(0)));
    }

    #[rstest]
    fn rejects_a_value_that_cannot_convert_to_the_planned_type(
        mut testing_planner: TestingPlanner,
    ) {
        let plan = testing_planner
            .plan("SELECT a FROM example_table WHERE a = $1")
            .unwrap();
        let params: Vec<ArrayRef> = vec![Arc::new(StringViewArray::from(vec!["not a number"]))];

        let error = plan.bind_parameters(&params).unwrap_err();

        assert!(matches!(error, Error::Cast { .. }));
    }

    #[rstest]
    fn binding_a_filter_parameter_reaches_the_scan(mut testing_planner: TestingPlanner) {
        let plan = testing_planner
            .plan("SELECT a FROM example_table WHERE a = $1")
            .unwrap();
        let log = testing_planner.filter_log("example_table");
        let params: Vec<ArrayRef> = vec![Arc::new(arrow_array::Int32Array::from(vec![3]))];

        let _bound = plan.bind_parameters(&params).unwrap();

        // The freshly bound constant was re-offered to the scan for pruning;
        // the test table records each filter it is offered.
        let log = log.lock().unwrap();
        assert!(
            log.iter().any(|f| f.contains("3:Int32")),
            "no bound-constant filter reached the scan: {log:?}"
        );
    }
}
