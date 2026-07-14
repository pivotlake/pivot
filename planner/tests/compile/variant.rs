//! SQL over a variant (JSON) column stored in real parquet files: the planner
//! projects the `doc` column, the catalog scan reassembles its leaves (however
//! the file shredded them), and the `->`/cast expressions read the paths.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Schema};
use parquet_variant_compute::{ShreddedSchemaBuilder, json_to_variant, shred_variant};

use crate::common::*;
use planner::catalog::Column;
use planner::types::Type;
use rstest::rstest;

/// Register a parquet-backed `docs(d VARIANT)` table holding `rows` as JSON.
fn docs_table(planner: &mut TestingPlanner, rows: Vec<&str>) {
    let json: ArrayRef = Arc::new(StringArray::from(rows));
    let docs = json_to_variant(&json).unwrap().into_inner();
    planner.add_table("docs", &[("d", Type::Variant, Arc::new(docs) as ArrayRef)]);
}

/// Like [`docs_table`], but with `path` shredded into a typed leaf, so the file
/// carries the extra `typed_value` chunks a real ingest would write.
fn shredded_docs_table(planner: &mut TestingPlanner, rows: Vec<&str>, path: &str, ty: &DataType) {
    let json: ArrayRef = Arc::new(StringArray::from(rows));
    let shred = ShreddedSchemaBuilder::new()
        .with_path(path, ty)
        .unwrap()
        .build();
    let shredded = shred_variant(&json_to_variant(&json).unwrap(), &shred).unwrap();
    let docs = shredded.into_inner();
    planner.add_table("docs", &[("d", Type::Variant, Arc::new(docs) as ArrayRef)]);
}

#[rstest]
fn casts_a_path_from_parquet(mut testing_planner: TestingPlanner) {
    docs_table(&mut testing_planner, vec![r#"{"age":30}"#, r#"{"age":25}"#]);

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d->'age' AS BIGINT) AS a FROM docs ORDER BY a",
    );

    let got: Vec<i64> = rows
        .iter()
        .map(|r| only_column(r).as_i64().unwrap())
        .collect();
    assert_eq!(got, vec![25, 30]);
}

#[rstest]
fn casts_a_shredded_path_from_parquet(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![r#"{"age":30}"#, r#"{"name":"bob"}"#],
        "age",
        &DataType::Int64,
    );

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d->'age' AS BIGINT) AS a FROM docs",
    );

    // The JSON writer omits null fields, so the missing row has no "a" key
    // (indexing yields JSON null).
    let mut got: Vec<Option<i64>> = rows.iter().map(|r| r["a"].as_i64()).collect();
    got.sort();
    assert_eq!(got, vec![None, Some(30)]);
}

/// A single-column `docs(d)` batch holding `rows`, with `shred_age` deciding
/// whether the file gets a typed `age` leaf or stays unshredded.
fn docs_batch(rows: Vec<&str>, shred_age: bool) -> RecordBatch {
    let json: ArrayRef = Arc::new(StringArray::from(rows));
    let variant = json_to_variant(&json).unwrap();
    let (field, array) = if shred_age {
        let shred = ShreddedSchemaBuilder::new()
            .with_path("age", &DataType::Int64)
            .unwrap()
            .build();
        let shredded = shred_variant(&variant, &shred).unwrap();
        (shredded.field("d"), shredded.into_inner())
    } else {
        (variant.field("d"), variant.into_inner())
    };
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![field])),
        vec![Arc::new(array) as _],
    )
    .unwrap()
}

/// `SELECT d.age` across two files where only one shreds `age`: each file's
/// leaves resolve on their own, and both render as the same JSON text.
#[rstest]
fn dot_access_across_mixed_shredding(mut testing_planner: TestingPlanner) {
    let shredded = docs_batch(vec![r#"{"age":30}"#, r#"{"age":25}"#], true);
    let unshredded = docs_batch(vec![r#"{"age":40}"#, r#"{"name":"bob"}"#], false);
    testing_planner.add_table_files(
        "docs",
        vec![Column {
            name: "d".to_string(),
            col_type: Type::Variant,
        }],
        &[shredded, unshredded],
    );

    let rows = run(&mut testing_planner, "SELECT d.age AS a FROM docs");

    let mut got: Vec<Option<String>> = rows
        .iter()
        .map(|r| r["a"].as_str().map(str::to_string))
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            None,
            Some("25".to_string()),
            Some("30".to_string()),
            Some("40".to_string()),
        ]
    );
}

/// The `d.age` dot syntax over a real shredded parquet file.
#[rstest]
fn dot_access_over_shredded_parquet(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![r#"{"age":30}"#, r#"{"name":"bob"}"#],
        "age",
        &DataType::Int64,
    );

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d.age AS BIGINT) AS a FROM docs",
    );

    let mut got: Vec<Option<i64>> = rows.iter().map(|r| r["a"].as_i64()).collect();
    got.sort();
    assert_eq!(got, vec![None, Some(30)]);
}

#[rstest]
fn groups_by_a_cast_path_from_parquet(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![r#"{"age":30}"#, r#"{"age":30}"#, r#"{"age":25}"#],
        "age",
        &DataType::Int64,
    );

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d->'age' AS BIGINT) AS a, count(*) AS n FROM docs GROUP BY a ORDER BY a",
    );

    let got: Vec<(i64, i64)> = rows
        .iter()
        .map(|r| (r["a"].as_i64().unwrap(), r["n"].as_i64().unwrap()))
        .collect();
    assert_eq!(got, vec![(25, 1), (30, 2)]);
}

#[rstest]
fn filters_on_a_cast_path_from_parquet(mut testing_planner: TestingPlanner) {
    docs_table(
        &mut testing_planner,
        vec![r#"{"age":30}"#, r#"{"age":25}"#, r#"{"name":"bob"}"#],
    );

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d->'age' AS BIGINT) AS a FROM docs \
         WHERE CAST(d->'age' AS BIGINT) > 27",
    );

    let got: Vec<i64> = rows
        .iter()
        .map(|r| only_column(r).as_i64().unwrap())
        .collect();
    assert_eq!(got, vec![30]);
}
