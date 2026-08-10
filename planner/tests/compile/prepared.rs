//! Prepared statements at the planner level: plans that keep `$n`
//! placeholders and compile per execution with bound values, and plans that
//! replan with the values bound as constants.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{ArrayRef, Int32Array, Scalar, StringViewArray};
use planner::compile::Error as CompileError;
use planner::types::Type;
use rstest::rstest;

use crate::common::*;

fn int32_value(v: i32) -> Scalar<ArrayRef> {
    planner::types::build_scalar_value(&planner::ScalarValue::Int32(v)).unwrap()
}

fn utf8_value(v: &str) -> Scalar<ArrayRef> {
    planner::types::build_scalar_value(&planner::ScalarValue::Utf8(v.to_string())).unwrap()
}

#[rstest]
fn placeholder_plan_compiles_once_per_binding(mut testing_planner: TestingPlanner) {
    let transaction = testing_planner.transaction();
    let prepared = testing_planner
        .planner
        .plan_prepare("SELECT $1::INTEGER + $2::INTEGER", transaction)
        .unwrap();

    assert_eq!(prepared.param_types, vec![Type::Int32, Type::Int32]);
    assert_eq!(prepared.output_types.len(), 1);
    let plan = prepared.plan.expect("a table-free SELECT keeps its plan");
    assert!(plan.is_parameter_cacheable());
    for (values, expected) in [
        (vec![int32_value(40), int32_value(2)], 42),
        (vec![int32_value(1), int32_value(2)], 3),
    ] {
        let transaction = testing_planner.transaction();
        let batches = plan
            .compile_with_parameters(testing_planner.dispatcher(), transaction.as_ref(), &values)
            .unwrap()
            .collect()
            .unwrap();
        let result = batches[0].column(0).as_primitive::<Int32Type>().value(0);
        assert_eq!(result, expected);
    }
}

#[rstest]
fn prepared_insert_keeps_a_parameter_cacheable_plan(mut testing_planner: TestingPlanner) {
    let transaction = testing_planner.transaction();

    let prepared = testing_planner
        .planner
        .plan_prepare(
            "INSERT INTO example_table VALUES ($1, $2, $3, $4)",
            transaction,
        )
        .unwrap();

    assert_eq!(
        prepared.param_types,
        vec![Type::Int32, Type::Int32, Type::Int32, Type::Utf8]
    );
    let plan = prepared.plan.expect("a VALUES-only INSERT keeps its plan");
    assert!(plan.is_parameter_cacheable());
    let values = find_values_operator(&plan.root).expect("the INSERT plans a Values operator");
    assert!(
        matches!(
            values.rows[0][0],
            planner::expression::Expression::Parameter(ref p) if p.index == 1
        ),
        "expected $1 to stay a placeholder: {:?}",
        values.rows[0][0]
    );
}

/// Depth-first search for the VALUES operator in an INSERT plan.
fn find_values_operator(node: &planner::PlanNode) -> Option<&planner::operator::Values> {
    if let planner::Operator::Values(values) = &node.operator {
        return Some(values);
    }
    node.inputs.iter().find_map(find_values_operator)
}

#[rstest]
fn prepare_without_parameters_reports_shapes_but_keeps_no_plan(
    mut testing_planner: TestingPlanner,
) {
    let transaction = testing_planner.transaction();

    let prepared = testing_planner
        .planner
        .plan_prepare("SELECT a FROM example_table", transaction)
        .unwrap();

    assert!(
        prepared.plan.is_none(),
        "a parameter-free statement executes through the shared plan cache, \
         so a plan kept here would be dead weight"
    );
    assert!(prepared.param_types.is_empty());
    assert_eq!(prepared.output_names, vec!["a"]);
    assert_eq!(prepared.output_types, vec![Type::Int32]);
}

#[rstest]
fn parameterized_table_read_replans_with_values(mut testing_planner: TestingPlanner) {
    let transaction = testing_planner.transaction();
    let prepared = testing_planner
        .planner
        .plan_prepare("SELECT a FROM example_table WHERE a = $1", transaction)
        .unwrap();
    assert!(
        prepared.plan.is_none(),
        "a parameterized table read replans per execution"
    );

    let transaction = testing_planner.transaction();
    let batches = testing_planner
        .planner
        .plan_with_values(
            "SELECT a FROM example_table WHERE a = $1",
            transaction.clone(),
            &[planner::ScalarValue::Int32(3)],
        )
        .unwrap()
        .compile(testing_planner.dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap();

    let values: Vec<i32> = batches
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int32Type>().values().to_vec())
        .collect();
    assert_eq!(values, vec![3]);
}

#[rstest]
fn compiling_with_too_few_values_errors(mut testing_planner: TestingPlanner) {
    let transaction = testing_planner.transaction();
    let plan = testing_planner
        .planner
        .plan_prepare("SELECT $1::INTEGER + $2::INTEGER", transaction)
        .unwrap()
        .plan
        .unwrap();

    let transaction = testing_planner.transaction();
    let result = plan.compile_with_parameters(
        testing_planner.dispatcher(),
        transaction.as_ref(),
        &[int32_value(1)],
    );

    match result {
        Err(CompileError::MissingParameterValue {
            index: 2,
            supplied: 1,
        }) => {}
        Err(e) => panic!("unexpected error: {e}"),
        Ok(_) => panic!("expected a missing-value error"),
    }
}

#[rstest]
fn compiling_with_a_mistyped_value_errors(mut testing_planner: TestingPlanner) {
    let transaction = testing_planner.transaction();
    let plan = testing_planner
        .planner
        .plan_prepare("SELECT $1::INTEGER", transaction)
        .unwrap()
        .plan
        .unwrap();

    let transaction = testing_planner.transaction();
    let result = plan.compile_with_parameters(
        testing_planner.dispatcher(),
        transaction.as_ref(),
        &[utf8_value("not a number")],
    );

    match result {
        Err(CompileError::ParameterTypeMismatch { index: 1, .. }) => {}
        Err(e) => panic!("unexpected error: {e}"),
        Ok(_) => panic!("expected a type-mismatch error"),
    }
}

#[rstest]
fn insert_compile_detects_a_schema_change_since_planning(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "drifting",
        &[(
            "x",
            Type::Int32,
            Arc::new(Int32Array::from(vec![1])) as ArrayRef,
        )],
    );
    let plan = testing_planner
        .plan("INSERT INTO drifting VALUES (2)")
        .unwrap();

    testing_planner.add_table(
        "drifting",
        &[(
            "x",
            Type::Utf8,
            Arc::new(StringViewArray::from(vec!["a"])) as ArrayRef,
        )],
    );
    let transaction = testing_planner.transaction();
    let result = plan.compile(testing_planner.dispatcher(), transaction.as_ref());

    match result {
        Err(CompileError::InsertTargetSchemaChanged(reference)) => {
            assert_eq!(reference.table, "drifting")
        }
        Err(e) => panic!("unexpected error: {e}"),
        Ok(_) => panic!("expected a schema-changed error"),
    }
}

#[rstest]
fn insert_compile_detects_a_dropped_target(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "doomed",
        &[(
            "x",
            Type::Int32,
            Arc::new(Int32Array::from(vec![1])) as ArrayRef,
        )],
    );
    let plan = testing_planner
        .plan("INSERT INTO doomed VALUES (2)")
        .unwrap();

    testing_planner.remove_table("doomed");
    let transaction = testing_planner.transaction();
    let result = plan.compile(testing_planner.dispatcher(), transaction.as_ref());

    match result {
        Err(CompileError::InsertTargetMissing(reference)) => {
            assert_eq!(reference.table, "doomed")
        }
        Err(e) => panic!("unexpected error: {e}"),
        Ok(_) => panic!("expected a missing-target error"),
    }
}

#[rstest]
fn compiling_a_placeholder_without_values_errors(mut testing_planner: TestingPlanner) {
    let transaction = testing_planner.transaction();
    let plan = testing_planner
        .planner
        .plan_prepare("SELECT $1::INTEGER", transaction)
        .unwrap()
        .plan
        .unwrap();

    let transaction = testing_planner.transaction();
    let result = plan.compile(testing_planner.dispatcher(), transaction.as_ref());

    match result {
        Err(CompileError::UnresolvedParameter(1)) => {}
        Err(e) => panic!("unexpected error: {e}"),
        Ok(_) => panic!("expected an unresolved-parameter error"),
    }
}
