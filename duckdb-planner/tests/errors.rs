use duckdb_planner::{
    DuckDBBind, DuckDBColumn, DuckDBTable, DuckDBTransaction, Error, LogicalTypeId, PlannerContext,
};
use std::sync::Arc;

struct TTable;

impl DuckDBTable for TTable {
    fn clone_box(&self) -> Box<dyn DuckDBTable> {
        Box::new(TTable)
    }

    fn duckdb_typed_columns(&self) -> Vec<DuckDBColumn> {
        vec![
            DuckDBColumn {
                name: "id".to_string(),
                duckdb_logical_type_id: LogicalTypeId::INTEGER as u8,
                decimal_width: 0,
                decimal_scale: 0,
            },
            DuckDBColumn {
                name: "name".to_string(),
                duckdb_logical_type_id: LogicalTypeId::VARCHAR as u8,
                decimal_width: 0,
                decimal_scale: 0,
            },
        ]
    }
}

struct TestCatalog;

impl DuckDBBind for TestCatalog {}

struct TestTransaction;

impl DuckDBTransaction for TestTransaction {
    fn table(&self, table_name: &str) -> Option<Box<dyn DuckDBTable>> {
        match table_name {
            "t" => Some(Box::new(TTable)),
            _ => None,
        }
    }
}

fn create_simple_context() -> PlannerContext {
    PlannerContext::new(Arc::new(TestCatalog))
}

fn plan(p: &mut PlannerContext, query: &str) -> Result<duckdb_planner::Plan, Error> {
    p.plan(query, Arc::new(TestTransaction))
}

#[test]
fn invalid_syntax() {
    let mut p = create_simple_context();
    let result = plan(&mut p, "SELECTTT * FROM t");
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
    let result = plan(&mut p, "SELECT nonexistent_column FROM t");
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
    let result = plan(&mut p, "SELECT * FROM nonexistent_table");
    match result {
        Ok(_) => panic!("Expected error"),
        Err(Error::DuckDBPlanning(e)) => assert!(e.exception_message.contains("nonexistent_table")),
        Err(e) => panic!("Unexpected error: {e}"),
    }
}

#[test]
fn exception_location() {
    let mut p = create_simple_context();
    let result = plan(&mut p, "SELECT * FROM nonexistent_table");
    match result {
        Ok(_) => panic!("Expected error"),
        Err(Error::DuckDBPlanning(e)) => assert_eq!(e.position.unwrap(), "14"),
        Err(e) => panic!("Unexpected error: {e}"),
    }
}
