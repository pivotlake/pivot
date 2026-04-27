mod common;

use common::planner;
use duckdb_planner::PlannerContext;
use insta::assert_snapshot;
use rstest::rstest;

// ---- Ref expressions ----

#[rstest]
fn ref_column_index_and_type(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT id, name, score FROM users")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(#0:INTEGER, #1:VARCHAR, #2:INTEGER)
      Input([#0:INTEGER, #1:VARCHAR, #2:INTEGER])
    ");
}

// ---- Constant expressions ----

#[rstest]
fn integer_constant(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT * FROM users WHERE score <> 42")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(#1:INTEGER, #2:VARCHAR, #0:INTEGER, #3:INTEGER, #4:BOOLEAN)
      Filter(#0:INTEGER <> 42:INTEGER -> BOOLEAN)
        Input([#2:INTEGER, #0:INTEGER, #1:VARCHAR, #3:INTEGER, #4:BOOLEAN])
    ");
}

#[rstest]
fn string_constant(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT * FROM users WHERE name <> 'alice'")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(#1:INTEGER, #0:VARCHAR, #2:INTEGER, #3:INTEGER, #4:BOOLEAN)
      Filter(#0:VARCHAR <> alice:VARCHAR -> BOOLEAN)
        Input([#1:VARCHAR, #0:INTEGER, #2:INTEGER, #3:INTEGER, #4:BOOLEAN])
    ");
}

// ---- Compare expressions ----

#[rstest]
fn compare_notequal_structure(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT * FROM users WHERE score <> 0")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(#1:INTEGER, #2:VARCHAR, #0:INTEGER, #3:INTEGER, #4:BOOLEAN)
      Filter(#0:INTEGER <> 0:INTEGER -> BOOLEAN)
        Input([#2:INTEGER, #0:INTEGER, #1:VARCHAR, #3:INTEGER, #4:BOOLEAN])
    ");
}

// ---- Function expressions ----

#[rstest]
fn arithmetic_function(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT score + 10 FROM users")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(+(#0:INTEGER, 10:INTEGER) -> INTEGER)
      Input([#2:INTEGER])
    ");
}

#[rstest]
fn nested_arithmetic(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT (score + id) * 2 FROM users")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(*(+(#0:INTEGER, #1:INTEGER) -> INTEGER, 2:INTEGER) -> INTEGER)
      Input([#2:INTEGER, #0:INTEGER])
    ");
}

// ---- Aggregate function expressions ----

#[rstest]
fn count_star_aggregate(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT COUNT(*) FROM users WHERE score <> 0")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(#0:BIGINT)
      Aggregate(groups: [], exprs: [count_star() -> BIGINT])
        Filter(#0:INTEGER <> 0:INTEGER -> BOOLEAN)
          Input([#2:INTEGER])
    ");
}

#[rstest]
fn sum_aggregate(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT SUM(score) FROM users")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(#0:UNKNOWN)
      Aggregate(groups: [], exprs: [sum(#0:INTEGER) -> UNKNOWN])
        Input([#2:INTEGER])
    ");
}

#[rstest]
fn multiple_aggregates_with_group_by(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT name, COUNT(*), SUM(score) FROM users GROUP BY name")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(#0:VARCHAR, #1:BIGINT, #2:UNKNOWN)
      Aggregate(groups: [#0:VARCHAR], exprs: [count_star() -> BIGINT, sum(#1:INTEGER) -> UNKNOWN])
        Input([#1:VARCHAR, #2:INTEGER])
    ");
}