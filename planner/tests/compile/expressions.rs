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
fn group_by_minute_of_timestamp(mut testing_planner: TestingPlanner) {
    use arrow_array::Int64Array;
    // EventTime as epoch seconds. Minute-of-hour = (t mod 3600) / 60:
    //   0 -> 0, 90 -> 1, 150 -> 2, 3690 -> 1 (3690 mod 3600 = 90).
    testing_planner.add_table(
        "events",
        &[(
            "EventTime",
            Type::Timestamp,
            Arc::new(Int64Array::from(vec![0i64, 90, 150, 3690])) as ArrayRef,
        )],
    );

    let results = testing_planner
        .planner
        .plan("SELECT extract(minute FROM EventTime) AS m, COUNT(*) FROM events GROUP BY m")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_i64().unwrap());

    // Minutes 0 and 2 occur once; minute 1 occurs twice.
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows.iter()
            .map(|r| r["key"].as_i64().unwrap())
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    let minute_one = rows.iter().find(|r| r["key"] == 1).unwrap();
    assert_eq!(minute_one["v0"], 2);
}

/// Every `extract(<part> FROM ts)` part, validated against values produced by
/// the DuckDB CLI at four timestamps (incl. a pre-epoch negative one and dates
/// that exercise the ISO-week year-boundary rollover):
///   t0 = 1704067200  2024-01-01 00:00:00 (Mon)
///   t1 = 1700000000  2023-11-14 22:13:20 (Tue)
///   t2 = 1262304000  2010-01-01 00:00:00 (Fri, ISO week 53 of 2009)
///   t3 =     -100000  1969-12-30 20:13:20 (Tue, ISO week 1 of 1970)
#[rstest]
fn extract_all_date_parts(mut testing_planner: TestingPlanner) {
    use arrow_array::Int64Array;
    let timestamps = [1_704_067_200i64, 1_700_000_000, 1_262_304_000, -100_000];
    testing_planner.add_table(
        "ts",
        &[(
            "EventTime",
            Type::Timestamp,
            Arc::new(Int64Array::from(timestamps.to_vec())) as ArrayRef,
        )],
    );

    // (part, [expected for t0, t1, t2, t3]).
    let cases: &[(&str, [i64; 4])] = &[
        (
            "epoch",
            [1_704_067_200, 1_700_000_000, 1_262_304_000, -100_000],
        ),
        ("second", [0, 20, 0, 20]),
        ("millisecond", [0, 20_000, 0, 20_000]),
        ("microsecond", [0, 20_000_000, 0, 20_000_000]),
        ("minute", [0, 13, 0, 13]),
        ("hour", [0, 22, 0, 20]),
        ("day", [1, 14, 1, 30]),
        ("month", [1, 11, 1, 12]),
        ("quarter", [1, 4, 1, 4]),
        ("year", [2024, 2023, 2010, 1969]),
        ("decade", [202, 202, 201, 196]),
        ("century", [21, 21, 21, 20]),
        ("millennium", [3, 3, 3, 2]),
        ("dayofweek", [1, 2, 5, 2]),
        ("isodow", [1, 2, 5, 2]),
        ("dayofyear", [1, 318, 1, 364]),
        ("week", [1, 46, 53, 1]),
    ];

    for (part, expected) in cases {
        let results = testing_planner
            .planner
            .plan(&format!(
                "SELECT extract({part} FROM EventTime) AS v, EventTime AS t FROM ts"
            ))
            .unwrap()
            .compile(testing_planner.dispatcher())
            .unwrap()
            .collect()
            .unwrap();

        // The computed projection names its columns col0 (the extracted value)
        // and col1 (the source EventTime). Map each row's timestamp to its
        // value, then compare against the expectation (row order isn't fixed).
        let rows = batches_to_json(&results);
        for (i, &t) in timestamps.iter().enumerate() {
            let row = rows
                .iter()
                .find(|r| r["col1"].as_i64().unwrap() == t)
                .unwrap_or_else(|| panic!("{part}: no row for ts {t}"));
            assert_eq!(
                row["col0"].as_i64().unwrap(),
                expected[i],
                "extract({part} FROM {t})"
            );
        }
    }
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
        rows.iter()
            .map(|r| r["a"].as_i64().unwrap())
            .collect::<Vec<_>>(),
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

    let constant =
        |v: i16| Expression::Constant(Scalar::new(Arc::new(Int16Array::from(vec![v])) as ArrayRef));
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
fn case_expression_in_projection(mut testing_planner: TestingPlanner) {
    // a = [1,2,3,4,5]; a < 3 -> 'low' (a=1,2), else 'high' (a=3,4,5).
    let results = testing_planner
        .planner
        .plan("SELECT CASE WHEN a < 3 THEN 'low' ELSE 'high' END AS c FROM example_table")
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut labels = batches_to_json(&results)
        .iter()
        .map(|r| r["col0"].as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    labels.sort();
    assert_eq!(labels, vec!["high", "high", "high", "low", "low"]);
}

#[rstest]
fn case_expression_multi_arm_group_key(mut testing_planner: TestingPlanner) {
    // Three-arm CASE bucketing a into lo/mid/hi, grouped and counted:
    //   a=1     -> "lo"  (1 row)
    //   a=2,3   -> "mid" (2 rows)
    //   a=4,5   -> "hi"  (2 rows)
    let results = testing_planner
        .planner
        .plan(
            "SELECT CASE WHEN a < 2 THEN 'lo' WHEN a < 4 THEN 'mid' ELSE 'hi' END AS c, COUNT(*) \
             FROM example_table GROUP BY 1",
        )
        .unwrap()
        .compile(testing_planner.dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["key"].as_str().unwrap().to_string());
    let counts: Vec<(String, i64)> = rows
        .iter()
        .map(|r| {
            (
                r["key"].as_str().unwrap().to_string(),
                r["v0"].as_i64().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        counts,
        vec![
            ("hi".to_string(), 2),
            ("lo".to_string(), 1),
            ("mid".to_string(), 2),
        ]
    );
}

#[rstest]
fn unsupported_scalar_function_returns_error(mut testing_planner: TestingPlanner) {
    let result = testing_planner
        .planner
        .plan("SELECT lower(name) FROM example_table");
    assert!(matches!(result, Err(PlannerError::PlanConversion(_))));
}
