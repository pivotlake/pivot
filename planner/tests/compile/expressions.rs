use std::sync::Arc;

use arrow_array::{ArrayRef, StringViewArray};

use crate::common::*;
use planner::Error as PlannerError;
use planner::types::Type;
use rstest::rstest;

#[rstest]
fn filter_contains_substring(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT name, b FROM example_table WHERE contains(name, 'ali')")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["b"].as_i64().unwrap());

    // "alice" appears at rows with b=10 and b=50.
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["name"], "alice");
    assert_eq!(rows[1]["name"], "alice");
}

#[rstest]
fn filter_contains_no_match(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .planner
        .plan("SELECT name FROM example_table WHERE contains(name, 'zzz')")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert!(rows.is_empty());
}

#[rstest]
fn filter_contains_matches_all(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "substrings",
        &[(
            "s",
            Type::Utf8,
            Arc::new(StringViewArray::from(vec!["aaa", "baab", "caaac"])) as ArrayRef,
        )],
    );

    let results = testing_planner
        .planner
        .plan("SELECT s FROM substrings WHERE contains(s, 'aa')")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 3);
}

#[rstest]
fn contains_then_group_by(mut testing_planner: TestingPlanner) {
    // Filter to names containing "a" (alice, charlie, dave), then group by name.
    let results = testing_planner
        .planner
        .plan("SELECT name, COUNT(*) FROM example_table WHERE contains(name, 'a') GROUP BY name")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_str().unwrap().to_string());

    // alice (2 rows), charlie (1), dave (1) all contain "a"; bob does not.
    assert_eq!(rows.len(), 3);
    let alice = rows.iter().find(|r| r["key"] == "alice").unwrap();
    assert_eq!(alice["v0"], 2);
    assert!(rows.iter().all(|r| r["key"] != "bob"));
}

#[rstest]
fn filter_in_list_int(mut testing_planner: TestingPlanner) {
    // a IN (2, 4): rows with a=2 and a=4 survive.
    let results = testing_planner
        .planner
        .plan("SELECT a FROM example_table WHERE a IN (2, 4)")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["a"].as_i64().unwrap());
    assert_eq!(
        rows.iter().map(|r| r["a"].as_i64().unwrap()).collect::<Vec<_>>(),
        vec![2, 4]
    );
}

#[rstest]
fn filter_in_list_string(mut testing_planner: TestingPlanner) {
    // name IN ('alice', 'charlie'): alice (×2) and charlie (×1) survive.
    let results = testing_planner
        .planner
        .plan("SELECT name FROM example_table WHERE name IN ('alice', 'charlie')")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut names = batches_to_json(&results)
        .iter()
        .map(|r| r["name"].as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, vec!["alice", "alice", "charlie"]);
}

/// DuckDB's `InFilter::ToExpression` (the IN pushed into a scan) reconstructs
/// the membership test as a `COMPARE_IN` operator, which lowers to
/// `Expression::InList` rather than the `OR` conjunction the optimizer emits
/// for an un-pushed list. That path isn't reachable from the parquet-backed
/// test harness, so exercise `InList::compile` directly: `ts IN (-1, 6)` over
/// `[-1, 6, 3, 6]` should yield `[true, true, false, true]`.
#[test]
fn in_list_compare_in_compiles_to_membership_mask() {
    use arrow_array::{BooleanArray, Int16Array, RecordBatch, Scalar};
    use arrow_schema::{DataType, Field, Schema};
    use planner::expression::{Expression, InList, Ref};

    let constant = |v: i16| {
        Expression::Constant(Scalar::new(Arc::new(Int16Array::from(vec![v])) as ArrayRef))
    };
    let in_list = InList {
        input: Box::new(Expression::Ref(Ref {
            column_idx: 0,
            return_type: Type::Int16,
        })),
        values: vec![constant(-1), constant(6)],
    };

    let mut eval = in_list.compile().unwrap()();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("ts", DataType::Int16, false)])),
        vec![Arc::new(Int16Array::from(vec![-1i16, 6, 3, 6])) as ArrayRef],
    )
    .unwrap();

    let result = eval(&batch);
    let (arr, _) = result.as_datum().get();
    let mask = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
    assert_eq!(
        (0..mask.len()).map(|i| mask.value(i)).collect::<Vec<_>>(),
        vec![true, true, false, true]
    );
}

#[rstest]
fn unsupported_scalar_function_returns_error(mut testing_planner: TestingPlanner) {
    let result = testing_planner
        .planner
        .plan("SELECT lower(name) FROM example_table");
    assert!(matches!(result, Err(PlannerError::PlanConversion(_))));
}
