use crate::common::*;
use insta::assert_snapshot;
use rstest::rstest;

// A user `IN`/`EXISTS` subquery lowers to a semi-join, which pivot can't
// execute. It must NOT be mistaken for the row-id semi-join DuckDB's late
// materialization produces (the bridge only collapses the latter) — it should
// surface as a plain unsupported-plan error, not a mis-collapsed Materialize.
#[rstest]
fn in_subquery_semijoin_is_unsupported_not_late_materialized(mut testing_planner: TestingPlanner) {
    let result = testing_planner
        .planner
        .plan("SELECT a FROM example_table WHERE a IN (SELECT b FROM example_table WHERE b > 20)");
    assert!(
        result.is_err(),
        "a user semi-join must error, not be collapsed as late materialization; got: {result:?}"
    );
}

#[rstest]
fn simple_select(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("SELECT a, b FROM example_table")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r"
    Projection(#0:Int32, #1:Int32)
      Input([#0:Int32, #1:Int32])
    ");
}

#[rstest]
fn filter(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("SELECT a FROM example_table WHERE a <> b")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(#0:Int32)
      Projection(#0:Int32)
        Filter(#0:Int32 <> #1:Int32 -> Boolean)
          Input([#0:Int32, #1:Int32])
    ");
}

#[rstest]
fn order_by_limit_produces_top_n(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("SELECT a FROM example_table ORDER BY a DESC LIMIT 2")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r"
    TopN(limit: 2, offset: 0, order: #0:Int32 DESC)
      Projection(#0:Int32)
        Input([#0:Int32])
    ");
}

#[rstest]
fn order_by_without_limit_produces_order_by(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("SELECT a FROM example_table ORDER BY a DESC")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r"
    OrderBy(#0:Int32 DESC)
      Projection(#0:Int32)
        Input([#0:Int32])
    ");
}

#[rstest]
fn limit_offset_over_order_by_fuses_into_top_n(mut testing_planner: TestingPlanner) {
    // limit + offset beyond DuckDB's Top-N threshold: DuckDB plans a full
    // OrderBy with a separate Limit above it; pivot re-fuses them into TopN.
    let plan = testing_planner
        .planner
        .plan("SELECT a FROM example_table ORDER BY a DESC LIMIT 2 OFFSET 100000")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r"
    TopN(limit: 2, offset: 100000, order: #0:Int32 DESC)
      Projection(#0:Int32)
        Input([#0:Int32])
    ");
}

#[rstest]
fn limit_without_order_by_stays_limit(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("SELECT a, COUNT(*) FROM example_table GROUP BY a LIMIT 3")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r"
    Projection(#0:Int32, #1:Int64)
      Limit(limit: 3, offset: 0)
        Aggregate(groups: [#0:Int32], exprs: [count_star()])
          Input([#0:Int32])
    ");
}

#[rstest]
fn simple_aggregate(mut testing_planner: TestingPlanner) {
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

#[rstest]
fn aggregate_with_single_group(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("SELECT a, COUNT(*) FROM example_table GROUP BY a")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r"
    Projection(#0:Int32, #1:Int64)
      Aggregate(groups: [#0:Int32], exprs: [count_star()])
        Input([#0:Int32])
    ");
}

#[rstest]
fn aggregate_with_multiple_groups(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("SELECT a, b, COUNT(*) FROM example_table GROUP BY a, b")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r"
    Projection(#0:Int32, #1:Int32, #2:Int64)
      Aggregate(groups: [#0:Int32, #1:Int32], exprs: [count_star()])
        Input([#0:Int32, #1:Int32])
    ");
}

#[rstest]
fn input_references_correct_columns(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("SELECT a, c FROM example_table")
        .unwrap();
    // `a` is column #0 and `c` is column #2 in the table; the Input scans
    // those two source columns, while the Projection references them by
    // their positions in the Input's output (#0 and #1).
    assert_snapshot!(plan.to_string(), @r"
    Projection(#0:Int32, #1:Int32)
      Input([#0:Int32, #2:Int32])
    ");
}

#[rstest]
fn create_table_produces_create_table_operator(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("CREATE TABLE created_table (id INTEGER, name VARCHAR)")
        .unwrap();
    assert_snapshot!(
        plan.to_string(),
        @"CreateTable(created_table, [id:Int32, name:Utf8], options: {})
    "
    );
}

#[rstest]
fn create_table_propagates_with_options(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("CREATE TABLE created_table (id INTEGER) WITH (path='/asdf', format='parquet')")
        .unwrap();
    assert_snapshot!(
        plan.to_string(),
        @r#"CreateTable(created_table, [id:Int32], options: {"format": "parquet", "path": "/asdf"})
    "#
    );
}

#[rstest]
fn combined_filter_order_limit(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .planner
        .plan("SELECT a FROM example_table WHERE a <> b ORDER BY a DESC LIMIT 2")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r"
    TopN(limit: 2, offset: 0, order: #0:Int32 DESC)
      Projection(#0:Int32)
        Projection(#0:Int32)
          Filter(#0:Int32 <> #1:Int32 -> Boolean)
            Input([#0:Int32, #1:Int32])
    ");
}
