mod common;

use common::planner;
use duckdb_planner::{LogicalTypeId, Operator, PlannerContext};
use insta::assert_snapshot;
use rstest::rstest;

#[rstest]
fn catalog_resolves_table_from_provider(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT id, name FROM users")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(#0:INTEGER, #1:VARCHAR)
      Input([#0:INTEGER, #1:VARCHAR])
    ");
}

#[rstest]
fn catalog_column_types_propagate(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT id, name, age FROM users")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(#0:INTEGER, #1:VARCHAR, #2:INTEGER)
      Input([#0:INTEGER, #1:VARCHAR, #3:INTEGER])
    ");
}

/// `Input.table` carries the trait object provided by the catalog — verify it's
/// reachable from the plan and returns the original schema. (This is internal
/// trait dispatch, not visible in the printed plan, so it can't be a snapshot.)
#[rstest]
fn input_holds_catalog_table(mut planner: PlannerContext) {
    let plan = planner.plan("SELECT id FROM users").unwrap();

    let Operator::Input(input) = &plan.root.inputs[0].operator else {
        panic!("expected Input, got {}", plan.root.inputs[0].operator)
    };

    let cols = input.table.duckdb_typed_columns();
    assert_eq!(cols.len(), 5);
    assert_eq!(cols[0].name, "id");
    assert_eq!(cols[0].duckdb_logical_type_id, LogicalTypeId::INTEGER as u8);
    assert_eq!(cols[1].name, "name");
    assert_eq!(cols[1].duckdb_logical_type_id, LogicalTypeId::VARCHAR as u8);
    assert_eq!(cols[4].name, "active");
    assert_eq!(cols[4].duckdb_logical_type_id, LogicalTypeId::BOOLEAN as u8);
}

#[rstest]
fn catalog_unknown_table_returns_error(mut planner: PlannerContext) {
    let result = planner.plan("SELECT * FROM nonexistent");
    assert!(result.is_err());
}
