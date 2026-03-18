use crate::common::*;
use planner::Operator;

#[test]
fn filter_with_contains_has_filter_node() {
    init();
    let mut planner = string_table();

    let plan = planner
        .plan("SELECT name FROM test WHERE contains(name, 'ali')")
        .unwrap()
        .root;

    fn has_filter(plan: &planner::PlanNode) -> bool {
        matches!(plan.operator, Operator::Filter(_)) || plan.inputs.iter().any(has_filter)
    }
    assert!(has_filter(&plan), "Expected a Filter operator in the plan");
}
