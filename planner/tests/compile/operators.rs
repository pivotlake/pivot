use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, Int32Array};

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
        .compile()
        .unwrap()
        .collect();

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
        .compile()
        .unwrap()
        .collect();

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
        .compile()
        .unwrap()
        .collect();

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
    testing_planner.catalog.add_table(
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
        .compile()
        .unwrap()
        .collect();

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
    testing_planner.catalog.add_table(
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
        .compile()
        .unwrap()
        .collect();

    let rows = batches_to_json(&results);
    assert!(rows.is_empty());
}

#[rstest]
fn filter_not_equal_all_pass(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a, b FROM example_table WHERE a <> b")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

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
        .compile()
        .unwrap()
        .collect();

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

// ---------------------------------------------------------------------------
// OrderBy tests
// ---------------------------------------------------------------------------

#[rstest]
fn order_by_ascending(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a, b FROM example_table ORDER BY a ASC")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

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
        .compile()
        .unwrap()
        .collect();

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
        .compile()
        .unwrap()
        .collect();

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
        .compile()
        .unwrap()
        .collect();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 5);
}

#[rstest]
fn top_n_limit_2_ascending(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT b FROM example_table ORDER BY a ASC LIMIT 2")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

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

#[rstest]
fn group_by_int_column(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a, COUNT(*) FROM example_table GROUP BY a")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_i64().unwrap());

    // Each value of a (1..=5) appears exactly once.
    assert_eq!(rows.len(), 5);
    for row in &rows {
        assert_eq!(row["value"], 1);
    }
}

#[rstest]
fn group_by_string_column_with_duplicates(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT name, COUNT(*) FROM example_table GROUP BY name")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_str().unwrap().to_string());

    // alice appears twice, bob/charlie/dave each once.
    assert_eq!(rows.len(), 4);
    let alice = rows.iter().find(|r| r["key"] == "alice").unwrap();
    assert_eq!(alice["value"], 2);
    let bob = rows.iter().find(|r| r["key"] == "bob").unwrap();
    assert_eq!(bob["value"], 1);
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
        .compile()
        .unwrap()
        .collect();

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
    testing_planner.catalog.add_table(
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
        .compile()
        .unwrap()
        .collect();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["count"], 0);
}

#[rstest]
fn filter_then_top_n(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT a FROM example_table WHERE a <> b ORDER BY a DESC LIMIT 2")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

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
    fn table(&self, _name: &str) -> Option<Arc<dyn Table>> {
        None
    }

    fn create_table(&self, request: CreateTableRequest) -> planner::catalog::Result<()> {
        self.created_tables.lock().unwrap().push(request);
        Ok(())
    }
}

// CREATE TABLE tests use a custom recording catalog (which the shared
// `TestCatalog` can't impersonate without adding state we don't otherwise
// need), so they construct their own `Planner` rather than going through the
// `testing_planner` fixture.
#[test]
fn create_table_calls_catalog_once() {
    init();
    let catalog = Arc::new(RecordingCatalog::default());
    let mut planner = Planner::new(catalog.clone());

    let results = planner
        .plan("CREATE TABLE created_table (id INTEGER, name VARCHAR)")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

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
    init();
    let catalog = Arc::new(RecordingCatalog::default());
    let mut planner = Planner::new(catalog.clone());

    let results = planner
        .plan("CREATE TABLE created_table (id INTEGER) WITH (existing_path='/asdf')")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

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
        .plan("SELECT SUM(b) FROM example_table");
    assert!(matches!(result, Err(PlannerError::PlanConversion(_))));
}
