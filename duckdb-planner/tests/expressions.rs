use duckdb_planner::duckdb_bridge::duckdb_types::ExpressionType;
use duckdb_planner::expression::Expression;
use duckdb_planner::operator::Operator;
use duckdb_planner::{
    DuckDBBind, DuckDBColumn, GetDuckDBTypedColumns, LogicalTypeId, PlannerContext,
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
                name: "score".to_string(),
                duckdb_logical_type_id: LogicalTypeId::INTEGER as u8,
            },
            DuckDBColumn {
                name: "active".to_string(),
                duckdb_logical_type_id: LogicalTypeId::BOOLEAN as u8,
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

fn create_context() -> PlannerContext {
    PlannerContext::new(Arc::new(TestCatalog))
}

// ---- Ref expressions ----

#[test]
fn ref_column_index_and_type() {
    let mut p = create_context();
    let plan = p.plan("SELECT id, name, score FROM users").unwrap();

    let Operator::Projection(proj) = &plan.operator else {
        panic!("expected Projection")
    };
    assert_eq!(proj.projections.len(), 3);

    let Expression::Ref(r0) = &proj.projections[0] else {
        panic!("expected Ref")
    };
    assert_eq!(r0.column_idx, 0);
    assert!(r0.return_type == LogicalTypeId::INTEGER);

    let Expression::Ref(r1) = &proj.projections[1] else {
        panic!("expected Ref")
    };
    assert_eq!(r1.column_idx, 1);
    assert!(r1.return_type == LogicalTypeId::VARCHAR);

    let Expression::Ref(r2) = &proj.projections[2] else {
        panic!("expected Ref")
    };
    assert_eq!(r2.column_idx, 2);
    assert!(r2.return_type == LogicalTypeId::INTEGER);
}

// ---- Constant expressions ----

#[test]
fn integer_constant() {
    let mut p = create_context();
    // "score <> 42" produces a Compare whose right side is a constant
    let plan = p.plan("SELECT * FROM users WHERE score <> 42").unwrap();

    let filter_node = &plan.inputs[0];
    let Operator::Filter(filter) = &filter_node.operator else {
        panic!("expected Filter")
    };
    let Expression::Compare(cmp) = &filter.conditions[0] else {
        panic!("expected Compare")
    };
    let Expression::Constant(c) = cmp.right.as_ref() else {
        panic!("expected Constant on RHS, got {}", cmp.right)
    };
    assert!(c.logical_type == LogicalTypeId::INTEGER);
    assert_eq!(c.raw_value, "42");
}

#[test]
fn string_constant() {
    let mut p = create_context();
    let plan = p.plan("SELECT * FROM users WHERE name <> 'alice'").unwrap();

    let filter_node = &plan.inputs[0];
    let Operator::Filter(filter) = &filter_node.operator else {
        panic!("expected Filter")
    };
    let Expression::Compare(cmp) = &filter.conditions[0] else {
        panic!("expected Compare")
    };
    let Expression::Constant(c) = cmp.right.as_ref() else {
        panic!("expected Constant on RHS, got {}", cmp.right)
    };
    assert!(c.logical_type == LogicalTypeId::VARCHAR);
    assert_eq!(c.raw_value, "alice");
}

// ---- Compare expressions ----

#[test]
fn compare_notequal_structure() {
    let mut p = create_context();
    let plan = p.plan("SELECT * FROM users WHERE score <> 0").unwrap();

    let filter_node = &plan.inputs[0];
    let Operator::Filter(filter) = &filter_node.operator else {
        panic!("expected Filter")
    };
    let Expression::Compare(cmp) = &filter.conditions[0] else {
        panic!("expected Compare")
    };
    assert!(matches!(cmp.compare_type, ExpressionType::COMPARE_NOTEQUAL));
    assert!(cmp.return_type == LogicalTypeId::BOOLEAN);

    // LHS is a column ref, RHS is a constant
    let Expression::Ref(lhs) = cmp.left.as_ref() else {
        panic!("expected Ref on LHS")
    };
    assert_eq!(lhs.column_idx, 0); // score is projected at index 0 in the scan
    let Expression::Constant(_) = cmp.right.as_ref() else {
        panic!("expected Constant on RHS")
    };
}

// ---- Function expressions ----

#[test]
fn arithmetic_function() {
    let mut p = create_context();
    let plan = p.plan("SELECT score + 10 FROM users").unwrap();

    let Operator::Projection(proj) = &plan.operator else {
        panic!("expected Projection")
    };
    let Expression::Function(func) = &proj.projections[0] else {
        panic!("expected Function, got {}", proj.projections[0])
    };
    assert_eq!(func.function, "+");
    assert_eq!(func.params.len(), 2);
    assert!(func.return_type == LogicalTypeId::INTEGER);

    // First param: column ref, second param: constant
    assert!(matches!(&func.params[0], Expression::Ref(_)));
    assert!(matches!(&func.params[1], Expression::Constant(_)));
}

#[test]
fn nested_arithmetic() {
    let mut p = create_context();
    let plan = p.plan("SELECT (score + id) * 2 FROM users").unwrap();

    let Operator::Projection(proj) = &plan.operator else {
        panic!("expected Projection")
    };
    let Expression::Function(mul) = &proj.projections[0] else {
        panic!("expected Function, got {}", proj.projections[0])
    };
    assert_eq!(mul.function, "*");
    assert_eq!(mul.params.len(), 2);

    // LHS is the inner (score + id) function
    let Expression::Function(add) = &mul.params[0] else {
        panic!("expected nested Function, got {}", mul.params[0])
    };
    assert_eq!(add.function, "+");
    assert_eq!(add.params.len(), 2);
    assert!(matches!(&add.params[0], Expression::Ref(_)));
    assert!(matches!(&add.params[1], Expression::Ref(_)));

    // RHS is the constant 2
    let Expression::Constant(c) = &mul.params[1] else {
        panic!("expected Constant")
    };
    assert_eq!(c.raw_value, "2");
}

// ---- Aggregate function expressions ----

#[test]
fn count_star_aggregate() {
    let mut p = create_context();
    let plan = p.plan("SELECT COUNT(*) FROM users").unwrap();

    let Operator::Projection(_) = &plan.operator else {
        panic!("expected Projection")
    };
    let agg_node = &plan.inputs[0];
    let Operator::Aggregate(agg) = &agg_node.operator else {
        panic!("expected Aggregate")
    };
    let Expression::AggregateFunc(func) = &agg.expressions[0] else {
        panic!("expected AggregateFunc")
    };
    assert_eq!(func.aggregate_function, "count_star");
    assert!(func.params.is_empty());
    assert!(func.return_type == LogicalTypeId::BIGINT);
}

#[test]
fn sum_aggregate() {
    let mut p = create_context();
    let plan = p.plan("SELECT SUM(score) FROM users").unwrap();

    let Operator::Projection(_) = &plan.operator else {
        panic!("expected Projection")
    };
    let agg_node = &plan.inputs[0];
    let Operator::Aggregate(agg) = &agg_node.operator else {
        panic!("expected Aggregate")
    };
    let Expression::AggregateFunc(func) = &agg.expressions[0] else {
        panic!("expected AggregateFunc")
    };
    assert_eq!(func.aggregate_function, "sum");
    assert_eq!(func.params.len(), 1);
    assert!(matches!(&func.params[0], Expression::Ref(_)));
}

#[test]
fn multiple_aggregates_with_group_by() {
    let mut p = create_context();
    let plan = p
        .plan("SELECT name, COUNT(*), SUM(score) FROM users GROUP BY name")
        .unwrap();

    let Operator::Projection(_) = &plan.operator else {
        panic!("expected Projection")
    };
    let agg_node = &plan.inputs[0];
    let Operator::Aggregate(agg) = &agg_node.operator else {
        panic!("expected Aggregate")
    };

    // One group-by key
    assert_eq!(agg.groups.len(), 1);
    assert!(matches!(&agg.groups[0], Expression::Ref(_)));

    // Two aggregate expressions
    assert_eq!(agg.expressions.len(), 2);
    let Expression::AggregateFunc(f0) = &agg.expressions[0] else {
        panic!("expected AggregateFunc")
    };
    let Expression::AggregateFunc(f1) = &agg.expressions[1] else {
        panic!("expected AggregateFunc")
    };
    assert_eq!(f0.aggregate_function, "count_star");
    assert_eq!(f1.aggregate_function, "sum");
}

// ---- Compare in filter with function on LHS ----

#[test]
fn compare_with_function_lhs() {
    let mut p = create_context();
    let plan = p
        .plan("SELECT * FROM users WHERE id + score <> 100")
        .unwrap();

    let filter_node = &plan.inputs[0];
    let Operator::Filter(filter) = &filter_node.operator else {
        panic!("expected Filter")
    };
    let Expression::Compare(cmp) = &filter.conditions[0] else {
        panic!("expected Compare")
    };

    // LHS is a function (id + score)
    let Expression::Function(func) = cmp.left.as_ref() else {
        panic!("expected Function on LHS, got {}", cmp.left)
    };
    assert_eq!(func.function, "+");
    assert_eq!(func.params.len(), 2);

    // RHS is the constant 100
    let Expression::Constant(c) = cmp.right.as_ref() else {
        panic!("expected Constant on RHS")
    };
    assert_eq!(c.raw_value, "100");
}
