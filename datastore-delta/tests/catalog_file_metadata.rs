//! What the catalog keeps per data file. A table gains a file on every commit
//! and holds it until compaction retires it, so anything held per file is held
//! for as long as the table is open and multiplied by however many files ingest
//! has made. These pin down the two places that would otherwise carry a copy
//! each: the file's schema, and long string values in its statistics.

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Datum, Int64Array, RecordBatch, Scalar, StringArray, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use dispatch::DataFlowDispatcher;
use tempfile::TempDir;

use common::{commit_datastore_transaction, current_parquet, insert_batches, shared_dispatcher};
use datastore::DatastoreTransaction;
use datastore_delta::{ColumnStatFilter, DeltaDatastore};
use planner::catalog::{Column, CreateTableRequest, SchemaQualifiedTableName};
use planner::expression::CompareType;
use planner::types::Type;

const TABLE: &str = "events";

fn dispatcher() -> DataFlowDispatcher {
    shared_dispatcher(2, 64)
}

fn batch_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("body", DataType::Utf8View, false),
    ]))
}

/// A datastore holding one empty `events (id BIGINT, body VARCHAR)` table.
fn open_datastore_with_empty_table() -> (TempDir, Arc<DeltaDatastore>) {
    let database = TempDir::new().unwrap();
    let datastore =
        DeltaDatastore::open(&database.path().to_string_lossy(), &dispatcher()).unwrap();
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_create_table(CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: TABLE.to_string(),
            columns: vec![
                Column {
                    name: "id".to_string(),
                    col_type: Type::Int64,
                },
                Column {
                    name: "body".to_string(),
                    col_type: Type::Utf8,
                },
            ],
            options: HashMap::new(),
            if_not_exists: false,
        })
        .unwrap()
        .compile(&dispatcher())
        .unwrap()
        .execute()
        .collect()
        .unwrap();
    commit_datastore_transaction(transaction).unwrap();
    (database, datastore)
}

/// INSERT one file's worth of rows in a commit of its own.
fn insert_file(datastore: &Arc<DeltaDatastore>, ids: Vec<i64>, body: &str) {
    let bodies: Vec<&str> = ids.iter().map(|_| body).collect();
    let batch = RecordBatch::try_new(
        batch_schema(),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringViewArray::from(bodies)),
        ],
    )
    .unwrap();
    insert_batches(
        &dispatcher(),
        datastore.clone().begin_transaction(),
        TABLE,
        vec![batch],
    );
}

/// Every file of a table parses to the same schema, so they share one
/// allocation rather than each holding a copy for the life of the table.
#[test]
fn files_of_a_table_share_one_schema() {
    let (_database, datastore) = open_datastore_with_empty_table();

    insert_file(&datastore, vec![1, 2, 3], "first");
    insert_file(&datastore, vec![4, 5, 6], "second");
    insert_file(&datastore, vec![7, 8, 9], "third");

    let parquet = current_parquet(&datastore, TABLE);
    let row_groups = parquet.row_groups();
    assert_eq!(row_groups.len(), 3, "one row group per inserted file");
    let first = &row_groups[0].schema;
    for other in &row_groups[1..] {
        assert!(
            Arc::ptr_eq(first, &other.schema),
            "each file kept its own copy of an identical schema"
        );
    }
}

/// A long text value would otherwise be recorded whole as a bound, twice per
/// column per file. The recorded bounds stay small while still enclosing the
/// column, so pruning is coarser but never wrong.
#[test]
fn long_text_bounds_are_recorded_shortened() {
    let (_database, datastore) = open_datastore_with_empty_table();
    let body = format!("prefix-{}", "y".repeat(4000));

    insert_file(&datastore, vec![1], &body);

    let parquet = current_parquet(&datastore, TABLE);
    let stats = parquet.row_groups()[0]
        .column_statistics(1)
        .expect("the body column records bounds");
    let text = |bound: &Scalar<ArrayRef>| {
        arrow::compute::cast(bound.get().0, &DataType::Utf8)
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0)
            .to_owned()
    };
    let min = text(stats.min.as_ref().unwrap());
    let max = text(stats.max.as_ref().unwrap());

    assert!(
        min.len() < 128,
        "min was recorded whole: {} bytes",
        min.len()
    );
    assert!(
        max.len() < 128,
        "max was recorded whole: {} bytes",
        max.len()
    );
    assert!(
        min.as_str() <= body.as_str(),
        "min must not exceed the value"
    );
    assert!(max.as_str() >= body.as_str(), "max must not fall below it");
}

/// The scan still returns every row after the bounds are shortened: a filter
/// that matches the (long) value must not be pruned away by a coarse bound.
#[test]
fn a_row_with_long_text_survives_a_filtered_scan() {
    let (_database, datastore) = open_datastore_with_empty_table();
    let body = format!("prefix-{}", "y".repeat(4000));

    insert_file(&datastore, vec![7], &body);

    let mut table = datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema(TABLE))
        .unwrap();
    table.refresh().unwrap();
    let kept =
        table
            .build_scan_view(
                &[],
                &[ColumnStatFilter {
                    column: "body".to_string(),
                    compare_type: CompareType::Equal,
                    value: Scalar::new(
                        Arc::new(StringViewArray::from(vec![body.as_str()])) as ArrayRef
                    ),
                }],
            )
            .unwrap();

    assert_eq!(kept.row_groups().len(), 1);
}
