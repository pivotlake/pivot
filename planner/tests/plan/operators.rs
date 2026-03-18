use crate::common::*;
use planner::Operator;

#[test]
fn select_has_projection_over_input() {
    init();
    let mut planner = int_table();

    let plan = planner.plan("SELECT a, b FROM test").unwrap().root;

    assert!(matches!(plan.operator, Operator::Projection(_)));
    assert_eq!(plan.inputs.len(), 1);
    assert!(matches!(plan.inputs[0].operator, Operator::Input(_)));
}

#[test]
fn filter_sits_between_projection_and_input() {
    init();
    let mut planner = int_table();

    let plan = planner
        .plan("SELECT a FROM test WHERE a <> b")
        .unwrap()
        .root;

    assert!(matches!(plan.operator, Operator::Projection(_)));
    let filter = &plan.inputs[0];
    assert!(matches!(filter.operator, Operator::Filter(_)));
    let input = &filter.inputs[0];
    assert!(matches!(input.operator, Operator::Input(_)));
}

#[test]
fn order_by_limit_produces_top_n() {
    init();
    let mut planner = int_table();

    let plan = planner
        .plan("SELECT a FROM test ORDER BY a DESC LIMIT 2")
        .unwrap()
        .root;

    fn has_top_n(plan: &planner::PlanNode) -> bool {
        matches!(plan.operator, Operator::TopN(_)) || plan.inputs.iter().any(has_top_n)
    }
    assert!(has_top_n(&plan), "Expected a TopN operator in the plan");
}

#[test]
fn order_by_without_limit_produces_order_by() {
    init();
    let mut planner = int_table();

    let plan = planner
        .plan("SELECT a FROM test ORDER BY a DESC")
        .unwrap()
        .root;

    fn has_order_by(plan: &planner::PlanNode) -> bool {
        matches!(plan.operator, Operator::OrderBy(_)) || plan.inputs.iter().any(has_order_by)
    }
    assert!(
        has_order_by(&plan),
        "Expected an OrderBy operator in the plan"
    );
}

#[test]
fn count_star_produces_aggregate() {
    init();
    let mut planner = int_table();

    let plan = planner.plan("SELECT COUNT(*) FROM test").unwrap().root;

    fn has_aggregate(plan: &planner::PlanNode) -> bool {
        matches!(plan.operator, Operator::Aggregate(_)) || plan.inputs.iter().any(has_aggregate)
    }
    assert!(
        has_aggregate(&plan),
        "Expected an Aggregate operator in the plan"
    );
}

#[test]
fn group_by_produces_aggregate_with_one_group() {
    init();
    let mut planner = int_table();

    let plan = planner
        .plan("SELECT a, COUNT(*) FROM test GROUP BY a")
        .unwrap()
        .root;

    fn find_aggregate(plan: &planner::PlanNode) -> Option<&planner::PlanNode> {
        if matches!(plan.operator, Operator::Aggregate(_)) {
            return Some(plan);
        }
        plan.inputs.iter().find_map(find_aggregate)
    }
    let agg_node = find_aggregate(&plan).expect("Expected an Aggregate node");
    if let Operator::Aggregate(agg) = &agg_node.operator {
        assert_eq!(agg.groups.len(), 1, "Expected exactly one group-by column");
    }
}

#[test]
fn input_references_correct_columns() {
    init();
    let mut planner = int_table();

    let plan = planner.plan("SELECT a, c FROM test").unwrap().root;

    fn find_input(plan: &planner::PlanNode) -> Option<&planner::PlanNode> {
        if matches!(plan.operator, Operator::Input(_)) {
            return Some(plan);
        }
        plan.inputs.iter().find_map(find_input)
    }
    let input_node = find_input(&plan).expect("Expected an Input node");
    if let Operator::Input(input) = &input_node.operator {
        assert!(!input.columns.is_empty());
    }
}

#[test]
fn create_table_produces_create_table_operator() {
    init();
    let mut planner = int_table();

    let plan = planner
        .plan("CREATE TABLE created_table (id INTEGER, name VARCHAR)")
        .unwrap()
        .root;

    let Operator::CreateTable(create) = &plan.operator else {
        panic!("expected CreateTable, got {:?}", plan.operator);
    };

    assert_eq!(create.request.name, "created_table");
    assert_eq!(create.request.columns.len(), 2);
    assert!(create.request.options.is_empty());
    assert!(plan.inputs.is_empty());
}

#[test]
fn create_table_propagates_with_options() {
    init();
    let mut planner = int_table();

    let plan = planner
        .plan("CREATE TABLE created_table (id INTEGER) WITH (path='/asdf', format='parquet')")
        .unwrap()
        .root;

    let Operator::CreateTable(create) = &plan.operator else {
        panic!("expected CreateTable, got {:?}", plan.operator);
    };

    assert_eq!(
        create.request.options.get("path").map(String::as_str),
        Some("/asdf")
    );
    assert_eq!(
        create.request.options.get("format").map(String::as_str),
        Some("parquet")
    );
}

#[test]
fn combined_filter_order_limit() {
    init();
    let mut planner = int_table();

    let plan = planner
        .plan("SELECT a FROM test WHERE a <> b ORDER BY a DESC LIMIT 2")
        .unwrap()
        .root;

    fn collect_operator_types(plan: &planner::PlanNode, types: &mut Vec<String>) {
        let t = match &plan.operator {
            Operator::Input(_) => "Input",
            Operator::Projection(_) => "Projection",
            Operator::Filter(_) => "Filter",
            Operator::Aggregate(_) => "Aggregate",
            Operator::OrderBy(_) => "OrderBy",
            Operator::TopN(_) => "TopN",
            Operator::CreateTable(_) => "CreateTable",
        };
        types.push(t.to_string());
        for input in &plan.inputs {
            collect_operator_types(input, types);
        }
    }
    let mut types = Vec::new();
    collect_operator_types(&plan, &mut types);

    assert!(
        types.contains(&"TopN".to_string()),
        "Expected TopN, got {:?}",
        types
    );
    assert!(
        types.contains(&"Filter".to_string()),
        "Expected Filter, got {:?}",
        types
    );
    assert!(
        types.contains(&"Input".to_string()),
        "Expected Input, got {:?}",
        types
    );
}
