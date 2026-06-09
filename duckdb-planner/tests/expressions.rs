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

/// Equality compares deserialize as `Compare` (same variant as `<>`) thanks
/// to the multi-tag `#[type_tag]` on `Expression::Compare`.
#[rstest]
fn compare_equal_structure(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT * FROM users WHERE score = 0")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(#1:INTEGER, #2:VARCHAR, #0:INTEGER, #3:INTEGER, #4:BOOLEAN)
      Filter(#0:INTEGER = 0:INTEGER -> BOOLEAN)
        Input([#2:INTEGER, #0:INTEGER, #1:VARCHAR, #3:INTEGER, #4:BOOLEAN])
    ");
}

/// `=` between two columns also lands on `Expression::Compare`.
#[rstest]
fn compare_equal_two_columns(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT id FROM users WHERE id = score")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @"
    Projection(#0:INTEGER)
      Projection(#0:INTEGER)
        Filter(#0:INTEGER = #1:INTEGER -> BOOLEAN)
          Input([#0:INTEGER, #2:INTEGER])
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

// ---- IN / conjunction expressions ----

/// A small `IN` list is rewritten by DuckDB's optimizer into an `OR` of
/// equalities, which deserializes as `Expression::Conjunction`.
#[rstest]
fn in_list_lowers_to_or_conjunction(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT id FROM users WHERE score IN (1, 3, 5)")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(#0:INTEGER)
      Projection(#1:INTEGER)
        Filter((#0:INTEGER = 1:INTEGER -> BOOLEAN OR #0:INTEGER = 3:INTEGER -> BOOLEAN OR #0:INTEGER = 5:INTEGER -> BOOLEAN))
          Input([#2:INTEGER, #0:INTEGER])
    ");
}

/// An explicit `OR`/`AND` mix deserializes as nested `Conjunction`s.
#[rstest]
fn explicit_or_and_conjunction(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT id FROM users WHERE score = 1 OR (age = 2 AND active)")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(#0:INTEGER)
      Projection(#3:INTEGER)
        Filter((#0:INTEGER = 1:INTEGER -> BOOLEAN OR (#2:BOOLEAN AND #1:INTEGER = 2:INTEGER -> BOOLEAN)))
          Input([#2:INTEGER, #3:INTEGER, #4:BOOLEAN, #0:INTEGER])
    ");
}

// ---- CASE expressions ----

/// `CASE WHEN … THEN … ELSE … END` deserializes as `Expression::Case` and
/// renders each arm. A two-arm CASE here exercises both checks and the ELSE.
#[rstest]
fn case_expression_structure(mut planner: PlannerContext) {
    let plan = planner
        .plan(
            "SELECT CASE WHEN score < 10 THEN 'lo' WHEN score < 20 THEN 'mid' ELSE 'hi' END \
             FROM users",
        )
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(CASE WHEN #0:INTEGER < 10:INTEGER -> BOOLEAN THEN lo:VARCHAR WHEN #0:INTEGER < 20:INTEGER -> BOOLEAN THEN mid:VARCHAR ELSE hi:VARCHAR END)
      Input([#2:INTEGER])
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
