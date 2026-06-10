use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, Int32Array, RecordBatch};
use dispatch::Dispatch;

use crate::common::*;
use planner::Error as PlannerError;
use planner::Planner;
use planner::catalog::{Catalog, CreateTableRequest, Table};
use planner::types::Type;
use rstest::rstest;

fn int_col(values: Vec<i32>) -> ArrayRef {
    Arc::new(Int32Array::from(values))
}

#[rstest]
fn select_column_subset(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a, b FROM example_table")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["a"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1, "b": 10},
            {"a": 2, "b": 20},
            {"a": 3, "b": 30},
            {"a": 4, "b": 40},
            {"a": 5, "b": 50},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn select_all_columns(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a, b, c FROM example_table")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["a"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1, "b": 10, "c": 100},
            {"a": 2, "b": 20, "c": 200},
            {"a": 3, "b": 30, "c": 300},
            {"a": 4, "b": 40, "c": 400},
            {"a": 5, "b": 50, "c": 500},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn select_single_column(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT c FROM example_table")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["c"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"c": 100},
            {"c": 200},
            {"c": 300},
            {"c": 400},
            {"c": 500},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn filter_not_equal_columns(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "pairs_with_dup",
        &[
            ("a", Type::Int32, int_col(vec![1, 2, 3, 10])),
            ("b", Type::Int32, int_col(vec![10, 20, 30, 10])),
        ],
    );

    let results = testing_planner
        .planner
        .plan("SELECT a, b FROM pairs_with_dup WHERE a <> b")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["a"].as_i64().unwrap());

    // (10, 10) is excluded.
    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1, "b": 10},
            {"a": 2, "b": 20},
            {"a": 3, "b": 30},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn filter_not_equal_no_matches(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "pairs_all_equal",
        &[
            ("a", Type::Int32, int_col(vec![1, 2, 3])),
            ("b", Type::Int32, int_col(vec![1, 2, 3])),
        ],
    );

    let results = testing_planner
        .planner
        .plan("SELECT a FROM pairs_all_equal WHERE a <> b")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert!(rows.is_empty());
}

#[rstest]
fn filter_not_equal_all_pass(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a, b FROM example_table WHERE a <> b")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["a"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1, "b": 10},
            {"a": 2, "b": 20},
            {"a": 3, "b": 30},
            {"a": 4, "b": 40},
            {"a": 5, "b": 50},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn filter_not_equal_constant(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a FROM example_table WHERE a <> 2")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["a"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1},
            {"a": 3},
            {"a": 4},
            {"a": 5},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn filter_equal_constant(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a FROM example_table WHERE a = 3")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([{"a": 3}]).as_array().unwrap().clone()
    );
}

#[rstest]
fn filter_equal_no_match(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a FROM example_table WHERE a = 999")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert!(rows.is_empty(), "expected no rows, got: {rows:?}");
}

// ---------------------------------------------------------------------------
// OrderBy tests
// ---------------------------------------------------------------------------

#[rstest]
fn order_by_ascending(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a, b FROM example_table ORDER BY a ASC")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1, "b": 10},
            {"a": 2, "b": 20},
            {"a": 3, "b": 30},
            {"a": 4, "b": 40},
            {"a": 5, "b": 50},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn order_by_descending(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a, b FROM example_table ORDER BY a DESC")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 5, "b": 50},
            {"a": 4, "b": 40},
            {"a": 3, "b": 30},
            {"a": 2, "b": 20},
            {"a": 1, "b": 10},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn top_n_limit_1(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a FROM example_table ORDER BY a DESC LIMIT 1")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["a"], 5);
}

#[rstest]
fn top_n_limit_exceeds_row_count(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a FROM example_table ORDER BY a LIMIT 100")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 5);
}

#[rstest]
fn top_n_limit_2_ascending(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT b FROM example_table ORDER BY a ASC LIMIT 2")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([
            {"b": 10},
            {"b": 20},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

// `SELECT *` over a filtered Top-N is exactly the late-materialization shape:
// DuckDB scans only `a`/`name` for the predicate+sort, then materializes the
// full row for the survivors. Exercises multi-column materialize + reordering
// back to schema order, plus that metadata survives the narrow projection.
#[rstest]
fn select_star_filtered_top_n_late_materializes(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT * FROM example_table WHERE name <> 'bob' ORDER BY a ASC LIMIT 2")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1, "b": 10, "c": 100, "name": "alice"},
            {"a": 3, "b": 30, "c": 300, "name": "charlie"},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn group_by_int_column(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a, COUNT(*) FROM example_table GROUP BY a")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_i64().unwrap());

    // Each value of a (1..=5) appears exactly once.
    assert_eq!(rows.len(), 5);
    for row in &rows {
        assert_eq!(row["v0"], 1);
    }
}

#[rstest]
fn group_by_string_column_with_duplicates(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT name, COUNT(*) FROM example_table GROUP BY name")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_str().unwrap().to_string());

    // alice appears twice, bob/charlie/dave each once.
    assert_eq!(rows.len(), 4);
    let alice = rows.iter().find(|r| r["key"] == "alice").unwrap();
    assert_eq!(alice["v0"], 2);
    let bob = rows.iter().find(|r| r["key"] == "bob").unwrap();
    assert_eq!(bob["v0"], 1);
}

#[rstest]
fn group_by_count_distinct(mut testing_planner: TestingPlanner) {
    // g: 1,1,1,2,2,3   x: 10,10,20,30,30,40
    // distinct x per g: g=1 -> {10,20}=2, g=2 -> {30}=1, g=3 -> {40}=1
    testing_planner.add_table(
        "gx",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 1, 2, 2, 3])),
            ("x", Type::Int32, int_col(vec![10, 10, 20, 30, 30, 40])),
        ],
    );
    let results = testing_planner
        .planner
        .plan("SELECT g, COUNT(DISTINCT x) FROM gx GROUP BY g")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_i64().unwrap());
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["key"], 1);
    assert_eq!(rows[0]["v0"], 2);
    assert_eq!(rows[1]["key"], 2);
    assert_eq!(rows[1]["v0"], 1);
    assert_eq!(rows[2]["key"], 3);
    assert_eq!(rows[2]["v0"], 1);
}

#[rstest]
fn global_count_distinct_string(mut testing_planner: TestingPlanner) {
    // example_table.name = alice, bob, charlie, dave, alice -> 4 distinct.
    let results = testing_planner
        .planner
        .plan("SELECT COUNT(DISTINCT name) FROM example_table")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    let only_value = rows[0].as_object().unwrap().values().next().unwrap();
    assert_eq!(only_value, 4);
}

#[rstest]
fn global_count_distinct_int(mut testing_planner: TestingPlanner) {
    // distinct {7, 8, 9, 0} = 4 (includes 0, exercising the HashOnly extractor's
    // mix64(0)==0 / empty-sentinel edge through the keys-only count path).
    testing_planner.add_table(
        "ints",
        &[("v", Type::Int32, int_col(vec![7, 7, 7, 8, 9, 9, 0, 0]))],
    );
    let results = testing_planner
        .planner
        .plan("SELECT COUNT(DISTINCT v) FROM ints")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    let only_value = rows[0].as_object().unwrap().values().next().unwrap();
    assert_eq!(only_value, 4);
}

#[rstest]
fn group_by_mixed_distinct(mut testing_planner: TestingPlanner) {
    // g: 1,1,2  x: 10,10,20  v: 5,5,7
    // per g: SUM(v), COUNT(*), COUNT(DISTINCT x)
    //   g=1 -> sum=10, count=2, distinct=1 ; g=2 -> sum=7, count=1, distinct=1
    testing_planner.add_table(
        "mixed",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 2])),
            ("x", Type::Int32, int_col(vec![10, 10, 20])),
            ("v", Type::Int32, int_col(vec![5, 5, 7])),
        ],
    );
    let results = testing_planner
        .planner
        .plan("SELECT g, SUM(v), COUNT(*), COUNT(DISTINCT x) FROM mixed GROUP BY g")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_i64().unwrap());
    assert_eq!(rows.len(), 2);
    // Value columns are v0/v1/v2 in DuckDB's expression order; assert the
    // multiset so the test is robust to that order (all three values differ).
    let vals = |r: &serde_json::Value| -> Vec<i64> {
        ["v0", "v1", "v2"]
            .iter()
            .map(|c| r[*c].as_i64().unwrap())
            .collect()
    };
    assert_eq!(rows[0]["key"], 1);
    let v1 = vals(&rows[0]);
    assert!(
        v1.contains(&10) && v1.contains(&2) && v1.contains(&1),
        "g=1 {v1:?}"
    );
    assert_eq!(rows[1]["key"], 2);
    let v2 = vals(&rows[1]);
    assert!(v2.contains(&7) && v2.contains(&1), "g=2 {v2:?}");
}

// ---------------------------------------------------------------------------
// Combined operator tests
// ---------------------------------------------------------------------------

#[rstest]
fn filter_then_order_by(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a, b FROM example_table WHERE a <> b ORDER BY a DESC")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 5, "b": 50},
            {"a": 4, "b": 40},
            {"a": 3, "b": 30},
            {"a": 2, "b": 20},
            {"a": 1, "b": 10},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn filter_then_count(mut testing_planner: TestingPlanner) {
    // Every row has a == b, so WHERE a <> b yields 0 rows.
    testing_planner.add_table(
        "pairs_all_equal",
        &[
            ("a", Type::Int32, int_col(vec![1, 2, 3, 4, 5])),
            ("b", Type::Int32, int_col(vec![1, 2, 3, 4, 5])),
        ],
    );

    let results = testing_planner
        .planner
        .plan("SELECT COUNT(*) FROM pairs_all_equal WHERE a <> b")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["count"], 0);
}

#[rstest]
fn global_avg_is_lowered_to_sum_and_count(mut testing_planner: TestingPlanner) {
    // AVG never reaches the global aggregate operator as a dedicated kind:
    // DuckDB lowers it to sum/count plus a divide projection. After the
    // `AggKind::Avg` removal this must still compile (no
    // `UnsupportedAggregateExpression`) and produce the right average through
    // the sum + count slots. avg(a) over [1,2,3,4,5] = 3.0.
    let results = testing_planner
        .planner
        .plan("SELECT AVG(a) FROM example_table")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    let avg = rows[0].as_object().unwrap().values().next().unwrap();
    assert_eq!(avg.as_f64().unwrap(), 3.0);
}

#[rstest]
fn global_sum_count_avg_together(mut testing_planner: TestingPlanner) {
    // The multi-aggregate global path with a SUM, a COUNT(*) and an AVG mixed
    // (q02's shape). Must compile and run end-to-end; count = 5 and avg(b) over
    // [10,20,30,40,50] = 30.0 (sum(a) is a Decimal128, skipped by the f64 scan).
    let results = testing_planner
        .planner
        .plan("SELECT SUM(a), COUNT(*), AVG(b) FROM example_table")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    let vals: Vec<f64> = rows[0]
        .as_object()
        .unwrap()
        .values()
        .filter_map(|v| v.as_f64())
        .collect();
    assert!(vals.contains(&5.0), "expected count 5 in {vals:?}");
    assert!(vals.contains(&30.0), "expected avg(b) 30.0 in {vals:?}");
}

#[rstest]
fn filter_then_top_n(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a FROM example_table WHERE a <> b ORDER BY a DESC LIMIT 2")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["a"], 5);
    assert_eq!(rows[1]["a"], 4);
}

#[derive(Debug, Default)]
struct RecordingCatalog {
    created_tables: Mutex<Vec<CreateTableRequest>>,
}

impl Catalog for RecordingCatalog {
    fn table(&self, _name: &str) -> Option<Box<dyn Table>> {
        None
    }

    fn create_table(
        &self,
        request: CreateTableRequest,
        dispatcher: &dispatch::DataFlowDispatcher,
    ) -> planner::catalog::Result<dispatch::RecordBatchOperatorSpec> {
        self.created_tables.lock().unwrap().push(request);
        // CREATE TABLE yields no rows; a real catalog would commit the table here.
        Ok(dispatch::RecordBatchOperatorSpec::from_nullary(
            dispatcher,
            (0..dispatcher.worker_count()).map(|_| NoRowsNullary::default()),
        ))
    }
}

/// A nullary that emits nothing and finishes immediately — backs the empty
/// result of this test's `CREATE TABLE`.
#[derive(Default)]
struct NoRowsNullary {
    ran: bool,
}

impl dispatch::NullaryFactory<RecordBatch> for NoRowsNullary {
    type Nullary = NoRowsNullary;

    fn build_nullary(self) -> NoRowsNullary {
        self
    }
}

impl dispatch::Nullary<RecordBatch> for NoRowsNullary {
    fn run<S: dispatch::Sender<RecordBatch>>(
        &mut self,
        _sender: &mut S,
    ) -> dispatch::NullaryResult<dispatch::WorkStatus> {
        if self.ran {
            return Ok(dispatch::WorkStatus::Pending);
        }
        self.ran = true;
        Ok(dispatch::WorkStatus::Ran)
    }

    fn finish<S: dispatch::Sender<RecordBatch>>(
        &mut self,
        _sender: &mut S,
    ) -> dispatch::NullaryResult<bool> {
        Ok(self.ran)
    }
}

// CREATE TABLE tests use a custom recording catalog (which the shared
// `TestCatalog` can't impersonate without adding state we don't otherwise
// need), so they construct their own `Planner` rather than going through the
// `testing_planner` fixture.
#[test]
fn create_table_calls_catalog_once() {
    let dispatch = Dispatch::spin_up(1, 32);
    let catalog = Arc::new(RecordingCatalog::default());
    let mut planner = Planner::new(catalog.clone());

    let results = planner
        .plan("CREATE TABLE created_table (id INTEGER, name VARCHAR)")
        .unwrap()
        .compile(dispatch.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    assert!(results.is_empty());

    let created = catalog.created_tables.lock().unwrap();
    assert_eq!(created.len(), 1);
    assert_eq!(created[0].name, "created_table");
    assert_eq!(created[0].columns.len(), 2);
    assert_eq!(created[0].columns[0].name, "id");
    assert_eq!(created[0].columns[0].col_type, Type::Int32);
    assert_eq!(created[0].columns[1].name, "name");
    assert_eq!(created[0].columns[1].col_type, Type::Utf8);
    assert!(created[0].options.is_empty());
    assert!(!created[0].if_not_exists);
}

#[test]
fn create_table_passes_with_options_to_catalog() {
    let dispatch = Dispatch::spin_up(1, 32);
    let catalog = Arc::new(RecordingCatalog::default());
    let mut planner = Planner::new(catalog.clone());

    let results = planner
        .plan("CREATE TABLE created_table (id INTEGER) WITH (existing_path='/asdf')")
        .unwrap()
        .compile(dispatch.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    assert!(results.is_empty());

    let created = catalog.created_tables.lock().unwrap();
    assert_eq!(created.len(), 1);
    assert_eq!(created[0].name, "created_table");
    assert_eq!(created[0].columns.len(), 1);
    assert_eq!(created[0].columns[0].name, "id");
    assert_eq!(created[0].columns[0].col_type, Type::Int32);
    assert_eq!(
        created[0].options.get("existing_path").map(String::as_str),
        Some("/asdf")
    );
}

#[rstest]
fn unsupported_aggregate_returns_error(mut testing_planner: TestingPlanner) {
    let result = testing_planner
        .planner
        .plan("SELECT median(b) FROM example_table");
    assert!(matches!(result, Err(PlannerError::PlanConversion(_))));
}

// ---- MIN / MAX aggregates ----

#[rstest]
fn global_min_max(mut testing_planner: TestingPlanner) {
    // The in-memory test table exposes no metadata bounds, so this runs the
    // scan-based global path.
    let results = testing_planner
        .planner
        .plan("SELECT MIN(b), MAX(b), COUNT(*) FROM example_table")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    // The select-list projection passes the aggregate's columns through, so
    // the output keeps the operator's own field names.
    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["min"].as_i64(), Some(10));
    assert_eq!(rows[0]["max"].as_i64(), Some(50));
    assert_eq!(rows[0]["count"].as_i64(), Some(5));
}

#[rstest]
fn grouped_numeric_min_max(mut testing_planner: TestingPlanner) {
    // alice spans b=10 and b=50; everyone else has one row.
    let results = testing_planner
        .planner
        .plan("SELECT name, MIN(b), MAX(b), COUNT(*) FROM example_table GROUP BY name")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows: Vec<(String, i64, i64, i64)> = batches_to_json(&results)
        .iter()
        .map(|r| {
            (
                r["key"].as_str().unwrap().to_string(),
                r["v0"].as_i64().unwrap(),
                r["v1"].as_i64().unwrap(),
                r["v2"].as_i64().unwrap(),
            )
        })
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            ("alice".to_string(), 10, 50, 2),
            ("bob".to_string(), 20, 20, 1),
            ("charlie".to_string(), 30, 30, 1),
            ("dave".to_string(), 40, 40, 1),
        ]
    );
}

#[rstest]
fn grouped_string_min(mut testing_planner: TestingPlanner) {
    // MIN over a string column, with candidates long enough to live in the
    // arena and a key whose later row improves on its first.
    testing_planner.add_table(
        "pages",
        &[
            ("k", Type::Int32, int_col(vec![1, 1, 2, 1])),
            (
                "url",
                Type::Utf8,
                crate::common::str_col(vec![
                    "http://example.com/zzz/very-long-path",
                    "http://example.com/aaa/very-long-path",
                    "http://other.org/x",
                    "http://example.com/mmm",
                ]),
            ),
        ],
    );
    let results = testing_planner
        .planner
        .plan("SELECT k, MIN(url), COUNT(*) FROM pages GROUP BY k")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows: Vec<(i64, String, i64)> = batches_to_json(&results)
        .iter()
        .map(|r| {
            (
                r["key"].as_i64().unwrap(),
                r["v0"].as_str().unwrap().to_string(),
                r["v1"].as_i64().unwrap(),
            )
        })
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (1, "http://example.com/aaa/very-long-path".to_string(), 3),
            (2, "http://other.org/x".to_string(), 1),
        ]
    );
}

// ---- Multi-key grouping (row-encoded key extractor) ----

#[rstest]
fn group_by_int_and_string_keys(mut testing_planner: TestingPlanner) {
    // Two-key (int, string) grouping with duplicates that fold: region 1
    // pairs with "x" twice.
    testing_planner.add_table(
        "visits",
        &[
            ("region", Type::Int32, int_col(vec![1, 1, 2, 1])),
            (
                "site",
                Type::Utf8,
                crate::common::str_col(vec!["x", "x", "x", "y"]),
            ),
            ("v", Type::Int32, int_col(vec![10, 20, 30, 40])),
        ],
    );
    let results = testing_planner
        .planner
        .plan("SELECT region, site, COUNT(*), SUM(v) FROM visits GROUP BY region, site")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows: Vec<(i64, String, i64, i64)> = batches_to_json(&results)
        .iter()
        .map(|r| {
            (
                r["k0"].as_i64().unwrap(),
                r["k1"].as_str().unwrap().to_string(),
                r["v0"].as_i64().unwrap(),
                r["v1"].as_i64().unwrap(),
            )
        })
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (1, "x".to_string(), 2, 30),
            (1, "y".to_string(), 1, 40),
            (2, "x".to_string(), 1, 30),
        ]
    );
}

#[rstest]
fn group_by_string_key_with_sum(mut testing_planner: TestingPlanner) {
    // A single string key with a SUM slot (not just COUNT): alice spans rows
    // with b=10 and b=50.
    let results = testing_planner
        .planner
        .plan("SELECT name, SUM(b), COUNT(*) FROM example_table GROUP BY name")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows: Vec<(String, i64, i64)> = batches_to_json(&results)
        .iter()
        .map(|r| {
            (
                r["key"].as_str().unwrap().to_string(),
                r["v0"].as_i64().unwrap(),
                r["v1"].as_i64().unwrap(),
            )
        })
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            ("alice".to_string(), 60, 2),
            ("bob".to_string(), 20, 1),
            ("charlie".to_string(), 30, 1),
            ("dave".to_string(), 40, 1),
        ]
    );
}

#[rstest]
fn group_by_constant_and_string(mut testing_planner: TestingPlanner) {
    // `GROUP BY 1, name` — the positional 1 resolves to the constant select
    // item, so one key column is a broadcast constant.
    let results = testing_planner
        .planner
        .plan("SELECT 1, name, COUNT(*) AS c FROM example_table GROUP BY 1, name")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows: Vec<(i64, String, i64)> = batches_to_json(&results)
        .iter()
        .map(|r| {
            (
                r["col0"].as_i64().unwrap(),
                r["col1"].as_str().unwrap().to_string(),
                r["col2"].as_i64().unwrap(),
            )
        })
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (1, "alice".to_string(), 2),
            (1, "bob".to_string(), 1),
            (1, "charlie".to_string(), 1),
            (1, "dave".to_string(), 1),
        ]
    );
}

#[rstest]
fn group_by_derived_int_keys(mut testing_planner: TestingPlanner) {
    // Computed sibling keys (`a, a - 1`) — both keys materialised by the
    // key projection, grouped as an int pair.
    let results = testing_planner
        .planner
        .plan("SELECT a, a - 1, COUNT(*) FROM example_table GROUP BY a, a - 1")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows: Vec<(i64, i64, i64)> = batches_to_json(&results)
        .iter()
        .map(|r| {
            (
                r["col0"].as_i64().unwrap(),
                r["col1"].as_i64().unwrap(),
                r["col2"].as_i64().unwrap(),
            )
        })
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![(1, 0, 1), (2, 1, 1), (3, 2, 1), (4, 3, 1), (5, 4, 1)]
    );
}

#[rstest]
fn group_by_three_keys_mixed(mut testing_planner: TestingPlanner) {
    // Three keys, two ints and a string — beyond the packed-pair extractor.
    // Rows 0 and 2 share (1, 7, "x") and fold into one group.
    testing_planner.add_table(
        "triples",
        &[
            ("g1", Type::Int32, int_col(vec![1, 1, 1, 2])),
            ("g2", Type::Int32, int_col(vec![7, 8, 7, 7])),
            (
                "site",
                Type::Utf8,
                crate::common::str_col(vec!["x", "x", "x", "x"]),
            ),
        ],
    );
    let results = testing_planner
        .planner
        .plan("SELECT g1, g2, site, COUNT(*) FROM triples GROUP BY g1, g2, site")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows: Vec<(i64, i64, String, i64)> = batches_to_json(&results)
        .iter()
        .map(|r| {
            (
                r["k0"].as_i64().unwrap(),
                r["k1"].as_i64().unwrap(),
                r["k2"].as_str().unwrap().to_string(),
                r["v0"].as_i64().unwrap(),
            )
        })
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (1, 7, "x".to_string(), 2),
            (1, 8, "x".to_string(), 1),
            (2, 7, "x".to_string(), 1),
        ]
    );
}

#[rstest]
fn derived_group_keys_are_recomputed(mut testing_planner: TestingPlanner) {
    // `a - 1` and `a + 1` are pure functions of the key `a`: the grouping
    // must collapse to `a` alone and recompute them per group.
    let results = testing_planner
        .planner
        .plan("SELECT a, a - 1, a + 1, COUNT(*) FROM example_table GROUP BY 1, 2, 3")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows: Vec<(i64, i64, i64, i64)> = batches_to_json(&results)
        .iter()
        .map(|r| {
            (
                r["col0"].as_i64().unwrap(),
                r["col1"].as_i64().unwrap(),
                r["col2"].as_i64().unwrap(),
                r["col3"].as_i64().unwrap(),
            )
        })
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (1, 0, 2, 1),
            (2, 1, 3, 1),
            (3, 2, 4, 1),
            (4, 3, 5, 1),
            (5, 4, 6, 1)
        ]
    );
}

#[rstest]
fn constant_group_key_is_derived(mut testing_planner: TestingPlanner) {
    // The constant key contributes nothing to grouping; it must be dropped
    // from the hash key and broadcast back into the output.
    let results = testing_planner
        .planner
        .plan("SELECT 7, name, COUNT(*) FROM example_table GROUP BY 1, 2")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows: Vec<(i64, String, i64)> = batches_to_json(&results)
        .iter()
        .map(|r| {
            (
                r["col0"].as_i64().unwrap(),
                r["col1"].as_str().unwrap().to_string(),
                r["col2"].as_i64().unwrap(),
            )
        })
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (7, "alice".to_string(), 2),
            (7, "bob".to_string(), 1),
            (7, "charlie".to_string(), 1),
            (7, "dave".to_string(), 1),
        ]
    );
}

#[rstest]
fn wide_sum_with_extremes_accumulates_exactly(mut testing_planner: TestingPlanner) {
    // SUM + MIN over an Int64 column in one grouped aggregate: the mixed-slot
    // extractor accumulates its sum slots in i128 and narrows checked to Int64
    // at output. Group 1's two i64::MAX/2 values sum to i64::MAX - 1 —
    // exactly representable, far beyond i32 — and must come out exact (the old
    // guard rejected this shape outright with UnsupportedWideSumWithExtremes).
    use arrow_array::Int64Array;
    const HALF: i64 = i64::MAX / 2;
    testing_planner.add_table(
        "wide",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 2])),
            (
                "v",
                Type::Int64,
                Arc::new(Int64Array::from(vec![HALF, HALF, 7])) as ArrayRef,
            ),
        ],
    );
    let results = testing_planner
        .planner
        .plan("SELECT g, SUM(v), MIN(v) FROM wide GROUP BY g")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows: Vec<(i64, i64, i64)> = batches_to_json(&results)
        .iter()
        .map(|r| {
            (
                r["key"].as_i64().unwrap(),
                r["v0"].as_i64().unwrap(),
                r["v1"].as_i64().unwrap(),
            )
        })
        .collect();
    rows.sort();
    assert_eq!(rows, vec![(1, i64::MAX - 1, HALF), (2, 7, 7)]);
}

#[rstest]
fn grouped_avg_length_count_min_over_computed_key(mut testing_planner: TestingPlanner) {
    // The q28 shape: GROUP BY a computed (regexp-extracted) key with
    // AVG(length(s)) — which DuckDB lowers to SUM + COUNT over a projected
    // Int64 length column — plus COUNT(*) and MIN over the string itself.
    // The Int64-typed SUM input alongside the MIN(string) extreme must
    // compile (the old wide-sum guard rejected it) and produce exact values.
    testing_planner.add_table(
        "refs",
        &[
            ("i", Type::Int32, int_col(vec![1, 2, 3, 4])),
            (
                "url",
                Type::Utf8,
                crate::common::str_col(vec![
                    "http://example.com/a",   // 20 chars, key example.com
                    "http://example.com/abc", // 22 chars, key example.com
                    "http://other.org/xy",    // 19 chars, key other.org
                    "",                       // filtered out
                ]),
            ),
        ],
    );
    let results = testing_planner
        .planner
        .plan(
            r"SELECT regexp_replace(url, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k,
                     AVG(length(url)) AS l, COUNT(*) AS c, MIN(url) AS m
              FROM refs WHERE url <> '' GROUP BY k ORDER BY l DESC LIMIT 25",
        )
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows: Vec<(String, f64, i64, String)> = batches_to_json(&results)
        .iter()
        .map(|r| {
            (
                r["col0"].as_str().unwrap().to_string(),
                r["col1"].as_f64().unwrap(),
                r["col2"].as_i64().unwrap(),
                r["col3"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    assert_eq!(
        rows,
        vec![
            (
                "example.com".to_string(),
                21.0,
                2,
                "http://example.com/a".to_string()
            ),
            (
                "other.org".to_string(),
                19.0,
                1,
                "http://other.org/xy".to_string()
            ),
        ]
    );
}
