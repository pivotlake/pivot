use crate::common::*;
use insta::assert_snapshot;
use rstest::rstest;

/// `Ref` — a bound column reference, displayed as `#idx:Type`.
#[rstest]
fn ref_column(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("SELECT a FROM example_table")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r"
    Projection(#0:Int32)
      Input([#0:Int32])
    ");
}

/// `Compare(NotEqual)` between two column references.
#[rstest]
fn compare_notequal_columns(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("SELECT a FROM example_table WHERE a <> b")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r"
    Projection(#0:Int32)
      Filter(#0:Int32 <> #1:Int32 -> Boolean)
        Input([#0:Int32, #1:Int32])
    ");
}

/// `Compare(Equal)` between two column references.
#[rstest]
fn compare_equal_columns(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("SELECT a FROM example_table WHERE a = b")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r"
    Projection(#0:Int32)
      Filter(#0:Int32 = #1:Int32 -> Boolean)
        Input([#0:Int32, #1:Int32])
    ");
}

/// `Compare(Equal)` against a constant — same shape as `<>`, displayed as `=`.
#[rstest]
fn compare_equal_constant(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("SELECT a FROM example_table WHERE a = 5")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r"
    Projection(#0:Int32)
      Filter(#0:Int32 = 5:Int32 -> Boolean)
        Input([#0:Int32])
    ");
}

/// `Constant` (integer) on the RHS of a comparison.
#[rstest]
fn constant_integer(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("SELECT a FROM example_table WHERE a <> 5")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r"
    Projection(#0:Int32)
      Filter(#0:Int32 <> 5:Int32 -> Boolean)
        Input([#0:Int32])
    ");
}

/// `Constant` (string) on the RHS of a comparison — DuckDB widens string
/// literals to `Utf8View`.
#[rstest]
fn constant_string(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("SELECT name FROM example_table WHERE name <> 'alice'")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r"
    Projection(#0:Utf8)
      Filter(#0:Utf8 <> alice:Utf8View -> Boolean)
        Input([#3:Utf8])
    ");
}

/// `AggregateFunc::CountStar` — the only aggregate the planner currently
/// supports.
#[rstest]
fn aggregate_count_star(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("SELECT COUNT(*) FROM example_table WHERE a <> 0")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r"
    Projection(#0:Int64)
      Aggregate(groups: [], exprs: [count_star()])
        Filter(#0:Int32 <> 0:Int32 -> Boolean)
          Input([#0:Int32])
    ");
}

/// `Function::Contains` — the only scalar function the planner currently
/// supports.
#[rstest]
fn function_contains(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("SELECT name FROM example_table WHERE contains(name, 'ali')")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r"
    Projection(#0:Utf8)
      Filter(contains(#0:Utf8, ali:Utf8View))
        Input([#3:Utf8])
    ");
}
