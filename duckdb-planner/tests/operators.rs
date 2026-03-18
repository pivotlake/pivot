use duckdb_planner::duckdb_bridge::duckdb_types::ExpressionType;
use duckdb_planner::expression::Expression;
use duckdb_planner::operator::{Operator, OrderByDirection};
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
                name: "score".to_string(),
                duckdb_logical_type_id: LogicalTypeId::INTEGER as u8,
            },
            DuckDBColumn {
                name: "age".to_string(),
                duckdb_logical_type_id: LogicalTypeId::INTEGER as u8,
            },
            DuckDBColumn {
                name: "active".to_string(),
                duckdb_logical_type_id: LogicalTypeId::INTEGER as u8,
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

#[test]
fn seq_scan() {
    let mut p = create_context();
    let node = p.plan("SELECT * FROM users").unwrap();

    // DuckDB wraps the scan in an identity projection
    let Operator::Projection(proj) = &node.operator else {
        panic!("expected Projection, got {}", node.operator)
    };

    assert_eq!(proj.projections.len(), 4);

    assert_eq!(node.inputs.len(), 1);
    let Operator::Input(scan) = &node.inputs[0].operator else {
        panic!("expected Input, got {}", node.inputs[0].operator)
    };
    assert_eq!(scan.columns.len(), 4);
    let col_idxs: Vec<usize> = scan
        .columns
        .iter()
        .map(|c| {
            let Expression::Ref(r) = c else {
                panic!("expected Ref")
            };
            r.column_idx
        })
        .collect();
    assert_eq!(col_idxs, vec![0, 1, 2, 3]);
    assert!(scan.filters.is_empty());
}

#[test]
fn get_with_column_subset() {
    let mut p = create_context();
    let node = p.plan("SELECT score, age FROM users").unwrap();

    // DuckDB wraps the scan in an identity projection
    let Operator::Projection(proj) = &node.operator else {
        panic!("expected Projection, got {}", node.operator)
    };
    assert_eq!(proj.projections.len(), 2);

    assert_eq!(node.inputs.len(), 1);
    let Operator::Input(scan) = &node.inputs[0].operator else {
        panic!("expected Input, got {}", node.inputs[0].operator)
    };
    let col_idxs: Vec<usize> = scan
        .columns
        .iter()
        .map(|c| {
            let Expression::Ref(r) = c else {
                panic!("expected Ref")
            };
            r.column_idx
        })
        .collect();
    assert_eq!(col_idxs, vec![1, 2]);
}

#[test]
fn projection() {
    let mut p = create_context();
    let node = p.plan("SELECT score, age + 1 FROM users").unwrap();

    let Operator::Projection(proj) = &node.operator else {
        panic!("expected Projection, got {}", node.operator)
    };
    assert_eq!(proj.projections.len(), 2);
    let Expression::Ref(r0) = &proj.projections[0] else {
        panic!("expected Ref")
    };
    assert_eq!(r0.column_idx, 0);
    assert!(r0.return_type == LogicalTypeId::INTEGER);
    let Expression::Function(_) = &proj.projections[1] else {
        panic!("expected Function, got {}", proj.projections[1])
    };

    assert_eq!(node.inputs.len(), 1);
    let Operator::Input(scan) = &node.inputs[0].operator else {
        panic!("expected Input")
    };
    let col_idxs: Vec<usize> = scan
        .columns
        .iter()
        .map(|c| {
            let Expression::Ref(r) = c else {
                panic!("expected Ref")
            };
            r.column_idx
        })
        .collect();
    assert_eq!(col_idxs, vec![1, 2]);
}

#[test]
fn filter() {
    let mut p = create_context();
    let node = p.plan("SELECT * FROM users WHERE id + score <> 0").unwrap();

    // DuckDB wraps the result in an identity projection over the filter
    let Operator::Projection(_) = &node.operator else {
        panic!("expected Projection, got {}", node.operator)
    };
    assert_eq!(node.inputs.len(), 1);
    let filter_node = &node.inputs[0];
    let Operator::Filter(filter) = &filter_node.operator else {
        panic!("expected Filter, got {}", filter_node.operator)
    };
    assert_eq!(filter.conditions.len(), 1);
    let Expression::Compare(cmp) = &filter.conditions[0] else {
        panic!("expected Compare")
    };

    assert!(matches!(cmp.compare_type, ExpressionType::COMPARE_NOTEQUAL));
    assert!(matches!(cmp.left.as_ref(), Expression::Function(_)));
    assert!(matches!(cmp.right.as_ref(), Expression::Constant(_)));
    assert!(cmp.return_type == LogicalTypeId::BOOLEAN);

    assert_eq!(filter_node.inputs.len(), 1);
    assert!(matches!(
        &filter_node.inputs[0].operator,
        Operator::Input(_)
    ));
}

#[test]
fn aggregate() {
    let mut p = create_context();
    let node = p
        .plan("SELECT age, COUNT(*) FROM users GROUP BY age")
        .unwrap();

    // DuckDB wraps the aggregate in an identity projection
    let Operator::Projection(_) = &node.operator else {
        panic!("expected Projection, got {}", node.operator)
    };
    assert_eq!(node.inputs.len(), 1);
    let agg_node = &node.inputs[0];
    let Operator::Aggregate(agg) = &agg_node.operator else {
        panic!("expected Aggregate, got {}", agg_node.operator)
    };
    assert_eq!(agg.groups.len(), 1);
    assert!(matches!(&agg.groups[0], Expression::Ref(_)));
    assert_eq!(agg.expressions.len(), 1);
    let Expression::AggregateFunc(func) = &agg.expressions[0] else {
        panic!("expected AggregateFunc")
    };
    assert_eq!(func.aggregate_function, "count_star");
    assert!(func.return_type == LogicalTypeId::BIGINT);
}

#[test]
fn order_by() {
    let mut p = create_context();
    let node = p.plan("SELECT * FROM users ORDER BY age DESC").unwrap();

    let Operator::OrderBy(order) = &node.operator else {
        panic!("expected OrderBy, got {}", node.operator)
    };
    assert_eq!(order.order_bys.len(), 1);
    assert!(matches!(
        order.order_bys[0].direction,
        OrderByDirection::Desc
    ));
    let Expression::Ref(r) = &order.order_bys[0].expression else {
        panic!("expected Ref")
    };
    assert_eq!(r.column_idx, 2);
    assert!(r.return_type == LogicalTypeId::INTEGER);

    // DuckDB inserts an identity projection between OrderBy and Input
    assert_eq!(node.inputs.len(), 1);
    let Operator::Projection(_) = &node.inputs[0].operator else {
        panic!("expected Projection, got {}", node.inputs[0].operator)
    };
    assert_eq!(node.inputs[0].inputs.len(), 1);
    assert!(matches!(
        &node.inputs[0].inputs[0].operator,
        Operator::Input(_)
    ));
}

#[test]
fn top_n() {
    let mut p = create_context();
    let node = p.plan("SELECT * FROM users ORDER BY age LIMIT 5").unwrap();

    let Operator::TopN(top) = &node.operator else {
        panic!("expected TopN, got {}", node.operator)
    };
    assert_eq!(top.limit, 5);
    assert_eq!(top.order_bys.len(), 1);
    assert!(matches!(top.order_bys[0].direction, OrderByDirection::Asc));
    let Expression::Ref(r) = &top.order_bys[0].expression else {
        panic!("expected Ref")
    };
    assert_eq!(r.column_idx, 2);

    // DuckDB inserts an identity projection between TopN and Input
    assert_eq!(node.inputs.len(), 1);
    let Operator::Projection(_) = &node.inputs[0].operator else {
        panic!("expected Projection, got {}", node.inputs[0].operator)
    };
    assert_eq!(node.inputs[0].inputs.len(), 1);
    assert!(matches!(
        &node.inputs[0].inputs[0].operator,
        Operator::Input(_)
    ));
}

#[test]
fn create_table() {
    let mut p = create_context();
    let node = p
        .plan("CREATE TABLE created_table (id INTEGER, name VARCHAR)")
        .unwrap();

    let Operator::CreateTable(create) = &node.operator else {
        panic!("expected CreateTable, got {}", node.operator)
    };

    assert_eq!(create.name, "created_table");
    assert_eq!(create.columns.len(), 2);
    assert!(create.options.is_empty());
    assert!(!create.if_not_exists);
    assert!(!create.or_replace);
    assert!(!create.temporary);
    assert!(!create.has_query);
    assert_eq!(create.constraint_count, 0);
    assert!(node.inputs.is_empty());
}

#[test]
fn create_table_with_options() {
    let mut p = create_context();
    let node = p
        .plan("CREATE TABLE created_table (id INTEGER) WITH (path='/asdf', format='parquet')")
        .unwrap();

    let Operator::CreateTable(create) = &node.operator else {
        panic!("expected CreateTable, got {}", node.operator)
    };

    assert_eq!(
        create.options.get("path").map(String::as_str),
        Some("/asdf")
    );
    assert_eq!(
        create.options.get("format").map(String::as_str),
        Some("parquet")
    );
}
