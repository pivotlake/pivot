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
    assert_snapshot!(plan, @"
    TopN(limit: 5, offset: 0, order: #3:INTEGER ASC)
      Projection(#0:INTEGER, #1:VARCHAR, #2:INTEGER, #3:INTEGER, #4:BOOLEAN)
        Input([#0:INTEGER, #1:VARCHAR, #2:INTEGER, #3:INTEGER, #4:BOOLEAN])
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
