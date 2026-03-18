use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, Int32Array};

use crate::common::*;
use planner::Error as PlannerError;
use planner::Planner;
use planner::catalog::{Catalog, CreateTableRequest, Table};
use planner::types::Type;

#[test]
fn select_column_subset() {
    init();
    let mut planner = int_table();

    let results = planner
        .plan("SELECT a, b FROM test")
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
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[test]
fn select_all_columns() {
    init();
    let mut planner = int_table();

    let results = planner
        .plan("SELECT a, b, c FROM test")
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
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[test]
fn select_single_column() {
    init();
    let mut planner = int_table();

    let results = planner
        .plan("SELECT c FROM test")
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
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[test]
fn filter_not_equal_columns() {
    init();
    let mut planner = make_planner_with_table(
        "test",
        &[
            (
                "a",
                Type::Int32,
                Arc::new(Int32Array::from(vec![1, 2, 3, 10])) as ArrayRef,
            ),
            (
                "b",
                Type::Int32,
                Arc::new(Int32Array::from(vec![10, 20, 30, 10])),
            ),
        ],
    );

    let results = planner
        .plan("SELECT a, b FROM test WHERE a <> b")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["a"].as_i64().unwrap());

    // (10, 10) should be excluded
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

#[test]
fn filter_not_equal_no_matches() {
    init();
    let mut planner = make_planner_with_table(
        "test",
        &[
            (
                "a",
                Type::Int32,
                Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
            ),
            ("b", Type::Int32, Arc::new(Int32Array::from(vec![1, 2, 3]))),
        ],
    );

    let results = planner
        .plan("SELECT a FROM test WHERE a <> b")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let rows = batches_to_json(&results);
    assert!(rows.is_empty());
}

#[test]
fn filter_not_equal_all_pass() {
    init();
    let mut planner = int_table();

    let results = planner
        .plan("SELECT a, b FROM test WHERE a <> b")
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
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[test]
fn filter_not_equal_constant() {
    init();
    let mut planner = int_table();

    let results = planner
        .plan("SELECT a FROM test WHERE a <> 2")
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
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

// ---------------------------------------------------------------------------
// OrderBy tests
// ---------------------------------------------------------------------------

#[test]
fn order_by_ascending() {
    init();
    let mut planner = int_table();

    let results = planner
        .plan("SELECT a, b FROM test ORDER BY a ASC")
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
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[test]
fn order_by_descending() {
    init();
    let mut planner = int_table();

    let results = planner
        .plan("SELECT a, b FROM test ORDER BY a DESC")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 3, "b": 30},
            {"a": 2, "b": 20},
            {"a": 1, "b": 10},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[test]
fn top_n_limit_1() {
    init();
    let mut planner = int_table();

    let results = planner
        .plan("SELECT a FROM test ORDER BY a DESC LIMIT 1")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["a"], 3);
}

#[test]
fn top_n_limit_exceeds_row_count() {
    init();
    let mut planner = int_table();

    let results = planner
        .plan("SELECT a FROM test ORDER BY a LIMIT 100")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 3);
}

#[test]
fn top_n_limit_2_ascending() {
    init();
    let mut planner = int_table();

    let results = planner
        .plan("SELECT b FROM test ORDER BY a ASC LIMIT 2")
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

#[test]
fn group_by_int_column() {
    init();
    let mut planner = int_table();

    let results = planner
        .plan("SELECT a, COUNT(*) FROM test GROUP BY a")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_i64().unwrap());

    // Each value of a (1, 2, 3) appears once
    assert_eq!(rows.len(), 3);
    for row in &rows {
        assert_eq!(row["value"], 1);
    }
}

#[test]
fn group_by_string_column_with_duplicates() {
    init();
    let mut planner = string_table();

    let results = planner
        .plan("SELECT name, COUNT(*) FROM test GROUP BY name")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_str().unwrap().to_string());

    // alice appears twice, bob/charlie/dave each once
    assert_eq!(rows.len(), 4);
    let alice = rows.iter().find(|r| r["key"] == "alice").unwrap();
    assert_eq!(alice["value"], 2);
    let bob = rows.iter().find(|r| r["key"] == "bob").unwrap();
    assert_eq!(bob["value"], 1);
}

// ---------------------------------------------------------------------------
// Combined operator tests
// ---------------------------------------------------------------------------

#[test]
fn filter_then_order_by() {
    init();
    let mut planner = int_table();

    let results = planner
        .plan("SELECT a, b FROM test WHERE a <> b ORDER BY a DESC")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 3, "b": 30},
            {"a": 2, "b": 20},
            {"a": 1, "b": 10},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[test]
fn filter_then_count() {
    init();
    let mut planner = make_planner_with_table(
        "test",
        &[
            (
                "a",
                Type::Int32,
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])) as ArrayRef,
            ),
            (
                "b",
                Type::Int32,
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            ),
        ],
    );

    // All rows have a == b, so WHERE a <> b yields 0 rows
    let results = planner
        .plan("SELECT COUNT(*) FROM test WHERE a <> b")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["count"], 0);
}

#[test]
fn filter_then_top_n() {
    init();
    let mut planner = int_table();

    let results = planner
        .plan("SELECT a FROM test WHERE a <> b ORDER BY a DESC LIMIT 2")
        .unwrap()
        .compile()
        .unwrap()
        .collect();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["a"], 3);
    assert_eq!(rows[1]["a"], 2);
}

#[derive(Debug, Default)]
struct RecordingCatalog {
    created_tables: Mutex<Vec<CreateTableRequest>>,
}

impl Catalog for RecordingCatalog {
    fn table(&self, _name: &str) -> Option<Arc<dyn Table>> {
        None
    }

    fn create_table(&self, request: CreateTableRequest) {
        self.created_tables.lock().unwrap().push(request);
    }
}

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

#[test]
fn unsupported_aggregate_returns_error() {
    init();
    let mut planner = string_table();

    let result = planner.plan("SELECT SUM(value) FROM test");
    assert!(matches!(result, Err(PlannerError::PlanConversion(_))));
}
