use crate::common::*;
use insta::assert_snapshot;
use rstest::rstest;

/// `Ref` — a bound column reference, displayed as `#idx:Type`.
#[rstest]
fn ref_column(mut testing_planner: TestingPlanner) {
    let plan = testing_planner.plan("SELECT a FROM example_table").unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32)
      Input([a:Int32])
    ");
}

/// `Compare(NotEqual)` between two column references.
#[rstest]
fn compare_notequal_columns(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a FROM example_table WHERE a <> b")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32)
      Projection(#0:Int32)
        Filter(a:Int32 <> b:Int32 -> Boolean)
          Input([a:Int32, b:Int32])
    ");
}

/// `Compare(Equal)` between two column references.
#[rstest]
fn compare_equal_columns(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a FROM example_table WHERE a = b")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32)
      Projection(#0:Int32)
        Filter(a:Int32 = b:Int32 -> Boolean)
          Input([a:Int32, b:Int32])
    ");
}

/// `Compare(Equal)` against a constant — same shape as `<>`, displayed as `=`.
#[rstest]
fn compare_equal_constant(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a FROM example_table WHERE a = 5")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32)
      Filter(a:Int32 = 5:Int32 -> Boolean)
        Input([a:Int32])
    ");
}

/// `Constant` (integer) on the RHS of a comparison.
#[rstest]
fn constant_integer(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a FROM example_table WHERE a <> 5")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32)
      Filter(a:Int32 <> 5:Int32 -> Boolean)
        Input([a:Int32])
    ");
}

/// `Constant` (string) on the RHS of a comparison — DuckDB widens string
/// literals to `Utf8View`.
#[rstest]
fn constant_string(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT name FROM example_table WHERE name <> 'alice'")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(name:Utf8)
      Filter(name:Utf8 <> alice:Utf8View -> Boolean)
        Input([name:Utf8])
    ");
}

/// `AggregateFunc::CountStar` — the only aggregate the planner currently
/// supports.
#[rstest]
fn aggregate_count_star(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT COUNT(*) FROM example_table WHERE a <> 0")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(count_star():Int64)
      Aggregate(groups: [], exprs: [count_star()])
        Filter(a:Int32 <> 0:Int32 -> Boolean)
          Input([a:Int32])
    ");
}

/// `Function::Contains` — a substring test on a string column.
#[rstest]
fn function_contains(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT name FROM example_table WHERE contains(name, 'ali')")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(name:Utf8)
      Filter(contains(name:Utf8, ali:Utf8View))
        Input([name:Utf8])
    ");
}

/// `Function::RegexpFullMatch` via the `~` operator. The pattern has a prefix
/// range DuckDB's `regex_range` optimizer could exploit, so this also pins that
/// no extra BLOB-bounded `Filter` is layered under the plan.
#[rstest]
fn function_regexp_full_match(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT name FROM example_table WHERE name ~ 'ali.*'")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(name:Utf8)
      Filter(regexp_full_match(name:Utf8, 'ali.*'))
        Input([name:Utf8])
    ");
}

/// `Function::Minute` — `extract(minute FROM ts)` lowers to the `minute`
/// scalar function over a `Timestamp` column, here a computed GROUP BY key.
#[rstest]
fn function_minute(mut testing_planner: TestingPlanner) {
    use arrow_array::{ArrayRef, Int64Array};
    use std::sync::Arc;
    testing_planner.add_table(
        "events",
        &[(
            "EventTime",
            planner::types::Type::Timestamp(planner::types::TimestampUnit::Second),
            Arc::new(Int64Array::from(vec![0i64, 90, 150])) as ArrayRef,
        )],
    );
    let plan = testing_planner
        .plan("SELECT extract(minute FROM EventTime) AS m, COUNT(*) FROM events GROUP BY m")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(#0:Int64, count_star():Int64)
      Aggregate(groups: [minute(cast(EventTime:Timestamp(s) as Timestamp(us)))], exprs: [count_star()])
        Input([EventTime:Timestamp(s)])
    ");
}

/// `Case` — a `CASE WHEN … THEN … ELSE … END` used as a computed GROUP BY key.
#[rstest]
fn case_expression_group_key(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan(
            "SELECT CASE WHEN a < 3 THEN 'low' ELSE 'high' END AS c, COUNT(*) \
             FROM example_table GROUP BY 1",
        )
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(#0:Utf8, count_star():Int64)
      Aggregate(groups: [CASE WHEN a:Int32 < 3:Int32 -> Boolean THEN low:Utf8View ELSE high:Utf8View END], exprs: [count_star()])
        Input([a:Int32])
    ");
}
