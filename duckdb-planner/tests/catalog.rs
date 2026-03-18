use duckdb_planner::{
    DuckDBBind, DuckDBColumn, GetDuckDBTypedColumns, LogicalTypeId, Operator, PlannerContext,
    expression::Expression,
};
use std::sync::Arc;

struct UsersTable;

impl GetDuckDBTypedColumns for UsersTable {
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
            DuckDBColumn {
                name: "age".to_string(),
                duckdb_logical_type_id: LogicalTypeId::SMALLINT as u8,
            },
        ]
    }
}

struct TestCatalog;

impl DuckDBBind for TestCatalog {
    fn try_bind(&self, table_name: &str) -> Option<Arc<dyn GetDuckDBTypedColumns>> {
        match table_name {
            "users" => Some(Arc::new(UsersTable)),
            _ => None,
        }
    }
}

#[test]
fn catalog_resolves_table_from_provider() {
    let mut ctx = PlannerContext::new(Arc::new(TestCatalog));
    let plan = ctx.plan("SELECT id, name FROM users").unwrap();

    match &plan.operator {
        Operator::Projection(p) => assert_eq!(p.projections.len(), 2),
        other => panic!("expected Projection, got {other}"),
    }
    assert_eq!(plan.inputs.len(), 1);
    match &plan.inputs[0].operator {
        Operator::Input(input) => {
            assert_eq!(input.columns.len(), 2);
        }
        other => panic!("expected Input, got {other}"),
    }
}

#[test]
fn catalog_column_types_propagate() {
    let mut ctx = PlannerContext::new(Arc::new(TestCatalog));
    let plan = ctx.plan("SELECT id, name, age FROM users").unwrap();

    match &plan.operator {
        Operator::Projection(p) => {
            assert_eq!(p.projections.len(), 3);
        }
        other => panic!("expected Projection, got {other}"),
    }
    match &plan.inputs[0].operator {
        Operator::Input(input) => {
            assert_eq!(input.columns.len(), 3);
            // Verify the types and indexes came through from the catalog provider
            let Expression::Ref(col0) = &input.columns[0] else {
                panic!("expected Ref")
            };
            let Expression::Ref(col1) = &input.columns[1] else {
                panic!("expected Ref")
            };
            let Expression::Ref(col2) = &input.columns[2] else {
                panic!("expected Ref")
            };
            assert_eq!(col0.column_idx, 0);
            assert!(col0.return_type == LogicalTypeId::INTEGER);
            assert_eq!(col1.column_idx, 1);
            assert!(col1.return_type == LogicalTypeId::VARCHAR);
            assert_eq!(col2.column_idx, 2);
            assert!(col2.return_type == LogicalTypeId::SMALLINT);
        }
        other => panic!("expected Input, got {other}"),
    }
}

#[test]
fn input_holds_catalog_table() {
    let mut ctx = PlannerContext::new(Arc::new(TestCatalog));
    let plan = ctx.plan("SELECT id FROM users").unwrap();

    let input = match &plan.inputs[0].operator {
        Operator::Input(input) => input,
        other => panic!("expected Input, got {other}"),
    };

    let cols = input.table.duckdb_typed_columns();
    assert_eq!(cols.len(), 3);
    assert_eq!(cols[0].name, "id");
    assert_eq!(cols[0].duckdb_logical_type_id, LogicalTypeId::INTEGER as u8);
    assert_eq!(cols[1].name, "name");
    assert_eq!(cols[1].duckdb_logical_type_id, LogicalTypeId::VARCHAR as u8);
    assert_eq!(cols[2].name, "age");
    assert_eq!(
        cols[2].duckdb_logical_type_id,
        LogicalTypeId::SMALLINT as u8
    );
}

#[test]
fn catalog_unknown_table_returns_error() {
    let mut ctx = PlannerContext::new(Arc::new(TestCatalog));
    let result = ctx.plan("SELECT * FROM nonexistent");
    assert!(result.is_err());
}
