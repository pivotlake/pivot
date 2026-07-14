//! Reading files whose schema says less than the table's declaration.
//!
//! Parquet files in the wild often store text as *unannotated* BYTE_ARRAY,
//! which is binary as far as the file is concerned. The table's declared
//! schema is what says those bytes are text: a column declared VARCHAR is
//! retyped from binary to string when each file's footer is loaded.

use std::sync::Arc;

use arrow_array::{ArrayRef, BinaryViewArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use crate::common::*;
use planner::catalog::Column;
use planner::types::Type;
use rstest::rstest;

/// One file with a single BYTE_ARRAY column carrying text but no string
/// annotation (written from an arrow binary array).
fn unannotated_text_batch(values: &[&[u8]]) -> RecordBatch {
    let array: ArrayRef = Arc::new(BinaryViewArray::from(values.to_vec()));
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "s",
            DataType::BinaryView,
            false,
        )])),
        vec![array],
    )
    .unwrap()
}

/// A column declared VARCHAR reads as a real string column even though the
/// file only claims binary: values decode as text and string predicates work.
#[rstest]
fn declared_varchar_reads_unannotated_binary_as_text(mut testing_planner: TestingPlanner) {
    testing_planner.add_table_files(
        "pages",
        vec![Column {
            name: "s".to_string(),
            col_type: Type::Utf8,
        }],
        &[unannotated_text_batch(&[b"alpha", b"beta"])],
    );

    let rows = run(&mut testing_planner, "SELECT s FROM pages WHERE s = 'beta'");

    assert_eq!(rows.len(), 1);
    assert_eq!(only_column(&rows[0]).as_str().unwrap(), "beta");
}

/// The scan's output column is pivot's string type (`Utf8View`), not binary.
#[rstest]
fn declared_varchar_surfaces_a_string_column(mut testing_planner: TestingPlanner) {
    testing_planner.add_table_files(
        "pages",
        vec![Column {
            name: "s".to_string(),
            col_type: Type::Utf8,
        }],
        &[unannotated_text_batch(&[b"alpha"])],
    );

    let batches = run_batches(&mut testing_planner, "SELECT s FROM pages");

    assert_eq!(*batches[0].column(0).data_type(), DataType::Utf8View);
}
