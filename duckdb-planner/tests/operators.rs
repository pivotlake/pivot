mod common;

use common::planner;
use duckdb_planner::PlannerContext;
use insta::assert_snapshot;
use rstest::rstest;

#[rstest]
fn seq_scan(mut planner: PlannerContext) {
    let plan = planner.plan("SELECT * FROM users").unwrap().to_string();
    assert_snapshot!(plan, @r"
    Projection(#0:INTEGER, #1:VARCHAR, #2:INTEGER, #3:INTEGER, #4:BOOLEAN)
      Input([#0:INTEGER, #1:VARCHAR, #2:INTEGER, #3:INTEGER, #4:BOOLEAN])
    ");
}

#[rstest]
fn get_with_column_subset(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT score, age FROM users")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(#0:INTEGER, #1:INTEGER)
      Input([#2:INTEGER, #3:INTEGER])
    ");
}

#[rstest]
fn projection(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT score, age + 1 FROM users")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(#0:INTEGER, +(#1:INTEGER, 1:INTEGER) -> INTEGER)
      Input([#2:INTEGER, #3:INTEGER])
    ");
}

#[rstest]
fn filter(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT * FROM users WHERE id + score <> 0")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(#0:INTEGER, #2:VARCHAR, #1:INTEGER, #3:INTEGER, #4:BOOLEAN)
      Filter(+(#0:INTEGER, #1:INTEGER) -> INTEGER <> 0:INTEGER -> BOOLEAN)
        Input([#0:INTEGER, #2:INTEGER, #1:VARCHAR, #3:INTEGER, #4:BOOLEAN])
    ");
}

#[rstest]
fn aggregate(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT age, COUNT(*) FROM users GROUP BY age")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Projection(#0:INTEGER, #1:BIGINT)
      Aggregate(groups: [#0:INTEGER], exprs: [count_star() -> BIGINT])
        Input([#3:INTEGER])
    ");
}

#[rstest]
fn order_by(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT * FROM users ORDER BY age DESC")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    OrderBy(#3:INTEGER DESC)
      Projection(#0:INTEGER, #1:VARCHAR, #2:INTEGER, #3:INTEGER, #4:BOOLEAN)
        Input([#0:INTEGER, #1:VARCHAR, #2:INTEGER, #3:INTEGER, #4:BOOLEAN])
    ");
}

#[rstest]
fn top_n(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT * FROM users ORDER BY age LIMIT 5")
        .unwrap()
        .to_string();
    // DuckDB's late_materialization optimizer fires for `SELECT * ... ORDER BY
    // ... LIMIT`: it scans only the sort column (`age` = #3) up front, then the
    // bridge collapses its row-id SEMI join into a Materialize that re-reads the
    // full row for the surviving rows.
    assert_snapshot!(plan, @"
    OrderBy(#3:INTEGER ASC)
      Projection(#0:INTEGER, #1:VARCHAR, #2:INTEGER, #3:INTEGER, #4:BOOLEAN)
        Materialize([#0, #1, #2, #3, #4])
          TopN(limit: 5, offset: 0, order: #0:INTEGER ASC)
            Projection(#0:INTEGER)
              Input([#3:INTEGER])
    ");
}

#[rstest]
fn limit_without_order_by(mut planner: PlannerContext) {
    // No ORDER BY, so the Top-N optimizer can't fire: the plan keeps a plain
    // LogicalLimit, which arrives as a Limit node.
    let plan = planner
        .plan("SELECT id FROM users LIMIT 3")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @"
    Projection(#0:INTEGER)
      Limit(limit: 3, offset: 0)
        Input([#0:INTEGER])
    ");
}

#[rstest]
fn limit_over_aggregate(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT age, COUNT(*) FROM users GROUP BY age LIMIT 2")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @"
    Projection(#0:INTEGER, #1:BIGINT)
      Limit(limit: 2, offset: 0)
        Aggregate(groups: [#0:INTEGER], exprs: [count_star() -> BIGINT])
          Input([#3:INTEGER])
    ");
}

#[rstest]
fn offset_without_limit_is_unbounded(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT id FROM users OFFSET 2")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Limit(limit: unbounded, offset: 2)
      Projection(#0:INTEGER)
        Input([#0:INTEGER])
    ");
}

#[rstest]
fn large_offset_keeps_order_by_and_limit_separate(mut planner: PlannerContext) {
    // When limit + offset exceeds DuckDB's Top-N threshold it plans a full
    // sort with a separate limit instead of a LogicalTopN.
    let plan = planner
        .plan("SELECT age FROM users ORDER BY age DESC LIMIT 10 OFFSET 100000")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Limit(limit: 10, offset: 100000)
      OrderBy(#0:INTEGER DESC)
        Projection(#0:INTEGER)
          Input([#3:INTEGER])
    ");
}

#[rstest]
fn create_table(mut planner: PlannerContext) {
    let plan = planner
        .plan("CREATE TABLE created_table (id INTEGER, name VARCHAR)")
        .unwrap()
        .to_string();
    assert_snapshot!(
        plan,
        @"CreateTable(created_table, [id:INTEGER, name:VARCHAR], options: {})"
    );
}

#[rstest]
fn create_table_with_options(mut planner: PlannerContext) {
    let plan = planner
        .plan("CREATE TABLE created_table (id INTEGER) WITH (path='/asdf', format='parquet')")
        .unwrap()
        .to_string();
    assert_snapshot!(
        plan,
        @r#"CreateTable(created_table, [id:INTEGER], options: {"format": "parquet", "path": "/asdf"})"#
    );
}
