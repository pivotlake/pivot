use crate::common::*;
use dispatch::RowDelivery;
use insta::assert_snapshot;
use rstest::rstest;

// A user `IN`/`EXISTS` subquery lowers to a semi-join, which pivot can't
// execute. It must NOT be mistaken for the row-id semi-join DuckDB's late
// materialization produces (the bridge only collapses the latter) — it should
// surface as a plain unsupported-plan error, not a mis-collapsed Materialize.
#[rstest]
fn in_subquery_semijoin_is_unsupported_not_late_materialized(mut testing_planner: TestingPlanner) {
    let result = testing_planner
        .plan("SELECT a FROM example_table WHERE a IN (SELECT b FROM example_table WHERE b > 20)");
    assert!(
        result.is_err(),
        "a user semi-join must error, not be collapsed as late materialization; got: {result:?}"
    );
}

#[rstest]
fn simple_select(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a, b FROM example_table")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32, b:Int32)
      Input([a:Int32, b:Int32])
    ");
}

#[rstest]
fn explain_wraps_the_explained_plan(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("EXPLAIN SELECT a, b FROM example_table")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Explain
      Projection(a:Int32, b:Int32)
        Input([a:Int32, b:Int32])
    ");
}

#[rstest]
fn filter(mut testing_planner: TestingPlanner) {
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

// DuckDB pushes the simple `a > 5` into the scan's table_filters while the
// column-to-column `a <> b` stays in the plan's Filter; the walk must fold
// both into ONE Filter (pushed condition first), never a stack of two - a
// stack materializes survivors once per level.
#[rstest]
fn pushed_and_residual_conditions_fold_into_one_filter(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a FROM example_table WHERE a > 5 AND a <> b")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32)
      Projection(#0:Int32)
        Filter(a:Int32 > 5:Int32 -> Boolean AND a:Int32 <> b:Int32 -> Boolean)
          Input([a:Int32, b:Int32])
    ");
}

#[rstest]
fn order_by_limit_produces_top_n(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a FROM example_table ORDER BY a DESC LIMIT 2")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r#"
    TopN(limit: 2, offset: 0, order: "default".main.example_table.a:Int32 DESC)
      Projection(a:Int32)
        Input([a:Int32])
    "#);
}

#[rstest]
fn order_by_without_limit_produces_order_by(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a FROM example_table ORDER BY a DESC")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r#"
    OrderBy("default".main.example_table.a:Int32 DESC)
      Projection(a:Int32)
        Input([a:Int32])
    "#);
}

#[rstest]
fn simple_aggregate(mut testing_planner: TestingPlanner) {
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

#[rstest]
fn aggregate_with_single_group(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a, COUNT(*) FROM example_table GROUP BY a")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32, count_star():Int64)
      Aggregate(groups: [a:Int32], exprs: [count_star()])
        Input([a:Int32])
    ");
}

#[rstest]
fn aggregate_with_multiple_groups(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a, b, COUNT(*) FROM example_table GROUP BY a, b")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32, b:Int32, count_star():Int64)
      Aggregate(groups: [a:Int32, b:Int32], exprs: [count_star()])
        Input([a:Int32, b:Int32])
    ");
}

#[rstest]
fn input_references_correct_columns(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a, c FROM example_table")
        .unwrap();
    // `a` is column #0 and `c` is column #2 in the table; the Input scans
    // those two source columns, while the Projection references them by
    // their positions in the Input's output (#0 and #1).
    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32, c:Int32)
      Input([a:Int32, c:Int32])
    ");
}

#[rstest]
fn create_table_produces_create_table_operator(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
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
        .plan("CREATE TABLE created_table (id INTEGER) WITH (path='/asdf', format='parquet')")
        .unwrap();
    assert_snapshot!(
        plan.to_string(),
        @r#"CreateTable(created_table, [id:Int32], options: {"format": "parquet", "path": "/asdf"})
    "#
    );
}

#[rstest]
fn values_produces_values_operator(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("VALUES (1, 'one'), (2, 'two')")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(col0:Int32, col1:Utf8)
      Values(rows: 2)
        DummyScan
    ");
}

#[rstest]
fn insert_values_produces_insert_and_values_operators(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("INSERT INTO example_table VALUES (6, 60, 600, 'eve'), (7, 70, 700, 'bob')")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Insert
      Projection(a:Int32, b:Int32, c:Int32, name:Utf8)
        Values(rows: 2)
          DummyScan
    ");
}

#[rstest]
fn insert_select_keeps_distinct_target_and_scan_bindings(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("INSERT INTO example_table SELECT a, b, c, name FROM example_table")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Insert
      Projection(a:Int32, b:Int32, c:Int32, name:Utf8)
        Input([a:Int32, b:Int32, c:Int32, name:Utf8])
    ");
}

#[rstest]
fn combined_filter_order_limit(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a FROM example_table WHERE a <> b ORDER BY a DESC LIMIT 2")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r#"
    TopN(limit: 2, offset: 0, order: "default".main.example_table.a:Int32 DESC)
      Projection(a:Int32)
        Projection(#0:Int32)
          Filter(a:Int32 <> b:Int32 -> Boolean)
            Input([a:Int32, b:Int32])
    "#);
}

#[rstest]
fn set_variable_is_parsed_by_duckdb(mut testing_planner: TestingPlanner) {
    let plan = testing_planner.plan("SET pivot_stats = true").unwrap();

    let set = plan
        .as_set_variable()
        .expect("SET should not compile to a query");

    assert_eq!(set.name, "pivot_stats");
    assert_eq!(set.value.as_deref(), Some("true"));
}

#[rstest]
fn reset_variable_carries_no_value(mut testing_planner: TestingPlanner) {
    let plan = testing_planner.plan("RESET pivot_stats").unwrap();

    let set = plan
        .as_set_variable()
        .expect("RESET should not compile to a query");

    assert_eq!(set.name, "pivot_stats");
    assert_eq!(set.value, None);
}

fn add_join_tables(planner: &TestingPlanner) {
    use arrow_array::{ArrayRef, Int64Array};
    use planner::types::Type;
    use std::sync::Arc;

    let int64_col = |values: Vec<i64>| -> ArrayRef { Arc::new(Int64Array::from(values)) };
    planner.add_table(
        "orders",
        &[
            ("o_key", Type::Int64, int64_col(vec![1, 2])),
            ("o_total", Type::Int64, int64_col(vec![10, 20])),
        ],
    );
    planner.add_table(
        "items",
        &[
            ("i_order", Type::Int64, int64_col(vec![1, 1, 2])),
            ("i_qty", Type::Int64, int64_col(vec![5, 6, 7])),
        ],
    );
}

#[rstest]
fn count_star_join_keeps_all_columns(mut testing_planner: TestingPlanner) {
    add_join_tables(&testing_planner);

    let plan = testing_planner
        .plan("SELECT count(*) FROM items JOIN orders ON i_order = o_key")
        .unwrap();

    assert_snapshot!(plan.to_string(), @"
    Projection(count_star():Int64)
      Aggregate(groups: [], exprs: [count_star()])
        Join(probe_key: 0, build_key: 0, probe_output: [0], build_output: [0])
          Input([i_order:Int64])
          Input([o_key:Int64])
    ");
}

#[rstest]
fn join_output_folds_projection_maps(mut testing_planner: TestingPlanner) {
    add_join_tables(&testing_planner);

    let plan = testing_planner
        .plan("SELECT i_qty, o_total FROM items JOIN orders ON i_order = o_key")
        .unwrap();

    assert_snapshot!(plan.to_string(), @"
    Projection(i_qty:Int64, o_total:Int64)
      Join(probe_key: 0, build_key: 0, probe_output: [1], build_output: [1])
        Input([i_order:Int64, i_qty:Int64])
        Input([o_key:Int64, o_total:Int64])
    ");
}

#[rstest]
fn join_with_unread_build_side_keeps_it_anyway(mut testing_planner: TestingPlanner) {
    add_join_tables(&testing_planner);

    let plan = testing_planner
        .plan("SELECT sum(i_order) FROM items JOIN orders ON i_order = o_key")
        .unwrap();

    assert_snapshot!(plan.to_string(), @"
    Projection(sum(i_order):Int128)
      Aggregate(groups: [], exprs: [sum(i_order:Int64)])
        Join(probe_key: 0, build_key: 0, probe_output: [0], build_output: [0])
          Input([i_order:Int64])
          Input([o_key:Int64])
    ");
}

/// The delivery of every filter in `plan`, in tree order.
fn filter_deliveries(node: &planner::plan::PlanNode) -> Vec<RowDelivery> {
    let mut found = match &node.operator {
        planner::operator::Operator::Filter(f) => vec![f.delivery],
        _ => vec![],
    };
    for child in &node.inputs {
        found.extend(filter_deliveries(child));
    }
    found
}

// A filter feeding a LIMIT must hand rows over as it selects them: the LIMIT
// cancels its input once it has enough, and rows held back to fill a batch
// keep the scan reading. The projections in between are transparent, so the
// choice reaches the filter through them.
#[rstest]
fn filter_under_a_limit_delivers_immediately(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a FROM example_table WHERE a <> b LIMIT 10")
        .unwrap();

    assert_eq!(filter_deliveries(&plan.root), vec![RowDelivery::Immediate]);
}

// An ORDER BY with a LIMIT is a Top-N, which publishes the boundary that
// prunes row groups from the scan, so it wants rows just as early.
#[rstest]
fn filter_under_a_top_n_delivers_immediately(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a FROM example_table WHERE a <> b ORDER BY a LIMIT 10")
        .unwrap();

    assert_eq!(filter_deliveries(&plan.root), vec![RowDelivery::Immediate]);
}

// A group-by reads its whole input before emitting anything, so its filter
// coalesces into full batches instead.
#[rstest]
fn filter_under_a_group_by_coalesces(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a, count(*) FROM example_table WHERE a <> b GROUP BY a")
        .unwrap();

    assert_eq!(filter_deliveries(&plan.root), vec![RowDelivery::Coalesced]);
}

// The nearest consumer decides: a LIMIT above a group-by changes nothing for
// the filter, which still feeds the group-by.
#[rstest]
fn group_by_under_a_limit_still_coalesces_its_filter(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a, count(*) FROM example_table WHERE a <> b GROUP BY a LIMIT 10")
        .unwrap();

    assert_eq!(filter_deliveries(&plan.root), vec![RowDelivery::Coalesced]);
}
