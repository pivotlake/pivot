#![allow(dead_code)]

use duckdb_planner::{DuckDBBind, DuckDBColumn, DuckDBTable, LogicalTypeId, PlannerContext};
use rstest::fixture;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
struct ColTable {
    columns: Vec<(String, u8)>,
}

impl DuckDBTable for ColTable {
    fn clone_box(&self) -> Box<dyn DuckDBTable> {
        Box::new(self.clone())
    }

    fn duckdb_typed_columns(&self) -> Vec<DuckDBColumn> {
        self.columns
            .iter()
            .map(|(name, type_id)| DuckDBColumn {
                name: name.clone(),
                duckdb_logical_type_id: *type_id,
            })
            .collect()
    }
}

struct TestCatalog {
    tables: HashMap<String, Box<ColTable>>,
}

impl DuckDBBind for TestCatalog {
    fn try_bind(&self, name: &str) -> Option<Box<dyn DuckDBTable>> {
        self.tables.get(name).cloned().map(|t| t as _)
    }
}

/// Shared `users` schema used by all test files:
///   #0 id      INTEGER
///   #1 name    VARCHAR
///   #2 score   INTEGER
///   #3 age     INTEGER
///   #4 active  BOOLEAN
#[fixture]
pub fn planner() -> PlannerContext {
    let cols: &[(&str, LogicalTypeId)] = &[
        ("id", LogicalTypeId::INTEGER),
        ("name", LogicalTypeId::VARCHAR),
        ("score", LogicalTypeId::INTEGER),
        ("age", LogicalTypeId::INTEGER),
        ("active", LogicalTypeId::BOOLEAN),
    ];
    let columns = cols
        .iter()
        .map(|(n, t)| (n.to_string(), t.clone() as u8))
        .collect();
    let mut tables: HashMap<String, Box<ColTable>> = HashMap::new();
    tables.insert("users".to_string(), Box::new(ColTable { columns }));
    PlannerContext::new(Arc::new(TestCatalog { tables }))
}
