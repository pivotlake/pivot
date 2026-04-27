use duckdb_planner::{DuckDBBind, DuckDBColumn, DuckDBTable, Error, LogicalTypeId, PlannerContext};
use std::sync::Arc;

struct TTable;

impl DuckDBTable for TTable {
    fn duckdb_typed_columns(&self) -> Vec<DuckDBColumn> {
        vec![
            DuckDBColumn {
                name: "id".to_string(),
                duckdb_logical_type_id: LogicalTypeId::INTEGER as u8,
            },
            DuckDBColumn {
                name: "name".to_string(),
                duckdb_logical_type_id: LogicalTypeId::VARCHAR as u8,
            },
        ]
    }
}

struct TestCatalog;

impl DuckDBBind for TestCatalog {
    fn try_bind(&self, table_name: &str) -> Option<Arc<dyn DuckDBTable>> {
        match table_name {
            "t" => Some(Arc::new(TTable)),
            _ => None,
        }
    }
}

fn create_simple_context() -> PlannerContext {
    PlannerContext::new(Arc::new(TestCatalog))
}

#[test]
fn invalid_syntax() {
    let mut p = create_simple_context();
    let result = p.plan("SELECTTT * FROM t");
    match result {
        Ok(_) => panic!("Expected error"),
        Err(Error::DuckDBPlanning(e)) => {
            assert!(e.exception_message.contains("syntax error at or near"))
        }
        Err(e) => panic!("Unexpected error: {e}"),
    }
}

#[test]
fn nonexistent_column() {
    let mut p = create_simple_context();
    let result = p.plan("SELECT nonexistent_column FROM t");
    match result {
        Ok(_) => panic!("Expected error"),
        Err(Error::DuckDBPlanning(e)) => {
            assert!(e.exception_message.contains("nonexistent_column"))
        }
        Err(e) => panic!("Unexpected error: {e}"),
    }
}

#[test]
fn nonexistent_table() {
    let mut p = create_simple_context();
    let result = p.plan("SELECT * FROM nonexistent_table");
    match result {
        Ok(_) => panic!("Expected error"),
        Err(Error::DuckDBPlanning(e)) => assert!(e.exception_message.contains("nonexistent_table")),
        Err(e) => panic!("Unexpected error: {e}"),
    }
}

#[test]
fn exception_location() {
    let mut p = create_simple_context();
    let result = p.plan("SELECT * FROM nonexistent_table");
    match result {
        Ok(_) => panic!("Expected error"),
        Err(Error::DuckDBPlanning(e)) => assert_eq!(e.position.unwrap(), "14"),
        Err(e) => panic!("Unexpected error: {e}"),
    }
}

#[test]
fn unsupported_plan_returns_error() {
    let mut p = create_simple_context();
    let result = p.plan("SELECT CAST(id AS BIGINT) FROM t");
    match result {
        Ok(_) => panic!("Expected error"),
        Err(Error::UnsupportedPlan(message)) => {
            assert!(message.contains("Unsupported expression"))
        }
        Err(e) => panic!("Unexpected error: {e}"),
    }
}
