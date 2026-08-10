//! Prepared-statement planning: prepare mode (typed placeholders) and
//! bind-values mode (values folded in as constants).

use duckdb_planner::duckdb_bridge::duckdb_types::LogicalOperatorType;
use duckdb_planner::handle::{Expression, LogicalOp, Operator};
use duckdb_planner::{
    DuckDBBind, DuckDBColumn, DuckDBTable, DuckDBTransaction, Error, LogicalTypeId, PlannerContext,
    ScalarValue,
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
    fn does_schema_exist(&self, _datastore: &str, schema: &str) -> bool {
        schema == "main"
    }

    fn bind_table(
        &self,
        _datastore: &str,
        _schema: &str,
        table_name: &str,
    ) -> Option<Box<dyn DuckDBTable>> {
        match table_name {
            "t" => Some(Box::new(TTable)),
            _ => None,
        }
    }
}

fn create_context() -> PlannerContext {
    PlannerContext::new(
        Arc::new(TestCatalog),
        vec!["db".to_string()],
        "db".to_string(),
    )
    .unwrap()
}

/// Depth-first search for the VALUES operator under an INSERT plan.
fn find_values(op: LogicalOp<'_>) -> Option<LogicalOp<'_>> {
    if op.op_type().unwrap() == LogicalOperatorType::LOGICAL_EXPRESSION_GET {
        return Some(op);
    }
    op.children()
        .unwrap()
        .into_iter()
        .find_map(|child| find_values(child))
}

#[test]
fn prepare_insert_keeps_typed_placeholders_in_the_plan() {
    let mut p = create_context();

    let prepared = p
        .plan_prepare("INSERT INTO t VALUES ($1, $2)", Arc::new(TestTransaction))
        .unwrap();

    let types: Vec<_> = prepared.param_types.iter().map(|t| t.id.clone()).collect();
    assert_eq!(types, vec![LogicalTypeId::INTEGER, LogicalTypeId::VARCHAR]);
    let plan = prepared.plan.expect("a scan-free INSERT keeps its plan");
    let root = plan.root().unwrap();
    assert_eq!(root.op_type().unwrap(), LogicalOperatorType::LOGICAL_INSERT);
    let values = find_values(root).expect("INSERT ... VALUES plans a VALUES operator");
    let Operator::Values(values) = values.operator().unwrap() else {
        panic!("expected a Values view");
    };
    let Expression::Parameter(param) = values.expression(0, 0).unwrap().expression().unwrap()
    else {
        panic!("expected $1 to stay a placeholder");
    };
    assert_eq!(param.index().unwrap(), 1);
    assert_eq!(param.return_type().unwrap().id, LogicalTypeId::INTEGER);
}

#[test]
fn prepare_parameterized_select_withholds_the_plan() {
    let mut p = create_context();

    let prepared = p
        .plan_prepare(
            "SELECT name FROM t WHERE id = $1",
            Arc::new(TestTransaction),
        )
        .unwrap();

    assert!(
        prepared.plan.is_none(),
        "a parameterized plan that reads a table is not reusable"
    );
    let types: Vec<_> = prepared.param_types.iter().map(|t| t.id.clone()).collect();
    assert_eq!(types, vec![LogicalTypeId::INTEGER]);
    assert_eq!(prepared.output_names, vec!["name"]);
    assert_eq!(prepared.output_types[0].id, LogicalTypeId::VARCHAR);
}

#[test]
fn prepare_without_parameters_keeps_the_plan() {
    let mut p = create_context();

    let prepared = p
        .plan_prepare("SELECT id FROM t", Arc::new(TestTransaction))
        .unwrap();

    assert!(prepared.plan.is_some());
    assert!(prepared.param_types.is_empty());
    assert_eq!(prepared.output_names, vec!["id"]);
    assert_eq!(prepared.output_types[0].id, LogicalTypeId::INTEGER);
}

#[test]
fn plan_with_values_binds_parameters_as_constants() {
    let mut p = create_context();

    let plan = p
        .plan_with_values(
            "INSERT INTO t VALUES ($1, $2)",
            Arc::new(TestTransaction),
            &[ScalarValue::Int32(7), ScalarValue::Utf8("x".to_string())],
        )
        .unwrap();

    let root = plan.root().unwrap();
    let values = find_values(root).expect("INSERT ... VALUES plans a VALUES operator");
    let Operator::Values(values) = values.operator().unwrap() else {
        panic!("expected a Values view");
    };
    let Expression::Constant(constant) = values.expression(0, 0).unwrap().expression().unwrap()
    else {
        panic!("expected $1 to bind as a constant");
    };
    let ScalarValue::Int32(v) = constant.value().unwrap() else {
        panic!("expected an INTEGER constant");
    };
    assert_eq!(v, 7);
}

#[test]
fn plan_with_values_accepts_a_typed_null() {
    let mut p = create_context();

    let plan = p.plan_with_values(
        "INSERT INTO t VALUES ($1, $2)",
        Arc::new(TestTransaction),
        &[
            ScalarValue::Null(duckdb_planner::BoundLogicalType::plain(
                LogicalTypeId::INTEGER,
            )),
            ScalarValue::Utf8("x".to_string()),
        ],
    );

    assert!(plan.is_ok(), "unexpected error: {:?}", plan.err());
}

#[test]
fn statement_mode_rejects_parameters() {
    let mut p = create_context();

    let result = p.plan("SELECT id FROM t WHERE id = $1", Arc::new(TestTransaction));

    match result {
        Err(Error::DuckDBPlanning(e)) => assert!(
            e.exception_message.contains("extended query protocol"),
            "unexpected message: {}",
            e.exception_message
        ),
        Ok(_) => panic!("expected a planning error"),
        Err(e) => panic!("unexpected error: {e}"),
    }
}

#[test]
fn prepare_rejects_a_parameter_with_no_inferable_type() {
    let mut p = create_context();

    let result = p.plan_prepare("SELECT $1", Arc::new(TestTransaction));

    match result {
        Err(Error::DuckDBPlanning(e)) => assert!(
            e.exception_message.contains("could not infer"),
            "unexpected message: {}",
            e.exception_message
        ),
        Ok(_) => panic!("expected a planning error"),
        Err(e) => panic!("unexpected error: {e}"),
    }
}

#[test]
fn plan_with_values_rejects_a_wrong_value_count() {
    let mut p = create_context();

    let result = p.plan_with_values(
        "SELECT name FROM t WHERE id = $1",
        Arc::new(TestTransaction),
        &[],
    );

    match result {
        Err(Error::DuckDBPlanning(e)) => assert!(
            e.exception_message.contains("1 parameters but 0"),
            "unexpected message: {}",
            e.exception_message
        ),
        Ok(_) => panic!("expected a planning error"),
        Err(e) => panic!("unexpected error: {e}"),
    }
}
