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
fn explain_wraps_the_optimized_plan(mut planner: PlannerContext) {
    let plan = planner
        .plan("EXPLAIN SELECT score, age FROM users")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    Explain
      Projection(#0:INTEGER, #1:INTEGER)
        Input([#2:INTEGER, #3:INTEGER])
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
fn derived_groups_removed(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT score, score - 1, score - 2, COUNT(*) AS c FROM users GROUP BY score, score - 1, score - 2 ORDER BY c DESC LIMIT 10")
        .unwrap()
        .to_string();

    // `score - 1` and `score - 2` are functions of `score`: the aggregate keeps the single grouping
    // key, and the derived keys are recomputed in a projection above it.
    assert_snapshot!(plan, @"
    TopN(limit: 10, offset: 0, order: #3:BIGINT DESC)
      Projection(#0:INTEGER, #1:INTEGER, #2:INTEGER, #3:BIGINT)
        Projection(#0:INTEGER, -(#0:INTEGER, 1:INTEGER) -> INTEGER, -(#0:INTEGER, 2:INTEGER) -> INTEGER, #1:BIGINT)
          Aggregate(groups: [#0:INTEGER], exprs: [count_star() -> BIGINT])
            Input([#2:INTEGER])
    ");
}

#[rstest]
fn derived_groups_removed_from_multiple_determinants(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT score, age, score + age, score - age, score + 1, age + 1, COUNT(*) FROM users GROUP BY score, age, score + age, score - age, score + 1, age + 1")
        .unwrap()
        .to_string();

    // `score` and `age` determine all four remaining keys, so the aggregate groups by just the two of
    // them and the rest are recomputed in the projection.
    assert_snapshot!(plan, @"
    Projection(#0:INTEGER, #1:INTEGER, #2:INTEGER, #3:INTEGER, #4:INTEGER, #5:INTEGER, #6:BIGINT)
      Projection(#0:INTEGER, #1:INTEGER, +(#0:INTEGER, #1:INTEGER) -> INTEGER, -(#0:INTEGER, #1:INTEGER) -> INTEGER, +(#0:INTEGER, 1:INTEGER) -> INTEGER, +(#1:INTEGER, 1:INTEGER) -> INTEGER, #2:BIGINT)
        Aggregate(groups: [#0:INTEGER, #1:INTEGER], exprs: [count_star() -> BIGINT])
          Input([#2:INTEGER, #3:INTEGER])
    ");
}

#[rstest]
fn derived_groups_kept_when_independent(mut planner: PlannerContext) {
    let plan = planner
        .plan("SELECT score, age, COUNT(*) AS c FROM users GROUP BY score, age")
        .unwrap()
        .to_string();

    // `score` and `age` are independent, so both remain grouping keys.
    assert_snapshot!(plan, @"
    Projection(#0:INTEGER, #1:INTEGER, #2:BIGINT)
      Aggregate(groups: [#0:INTEGER, #1:INTEGER], exprs: [count_star() -> BIGINT])
        Input([#2:INTEGER, #3:INTEGER])
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
fn large_limit_offset_still_fuses_into_top_n(mut planner: PlannerContext) {
    // `limit + offset` here (10010) is past DuckDB's stock Top-N threshold, so
    // upstream would emit a LogicalLimit over a LogicalOrder. Our optimizer
    // patch (src/optimizer/topn_optimizer.cpp) drops that bail-out, so it stays
    // a TopN — the only shape Pivot's bridge translates.
    let plan = planner
        .plan("SELECT score, age, COUNT(*) AS c FROM users GROUP BY score, age ORDER BY c DESC LIMIT 10 OFFSET 10000")
        .unwrap()
        .to_string();
    assert_snapshot!(plan, @r"
    TopN(limit: 10, offset: 10000, order: #2:BIGINT DESC)
      Projection(#0:INTEGER, #1:INTEGER, #2:BIGINT)
        Aggregate(groups: [#0:INTEGER, #1:INTEGER], exprs: [count_star() -> BIGINT])
          Input([#2:INTEGER, #3:INTEGER])
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
