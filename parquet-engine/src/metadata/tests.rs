use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, RecordBatch, Scalar};
use arrow_schema::{DataType, Field, Schema};
use dispatch::Dispatch;
use object_storage::DataFile;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use planner::expression::CompareType;

use crate::TableColumns;
use pruning::ColumnPredicate;

use super::load_file_row_groups;

fn row_group_eliminated(
    group: &crate::RowGroupMetadata,
    column_idx: usize,
    compare: CompareType,
    value: &Scalar<ArrayRef>,
) -> Result<bool, arrow_schema::ArrowError> {
    let predicate = ColumnPredicate {
        column_idx,
        path: Vec::new(),
        as_type: None,
        compare_type: compare.into(),
        value: value.clone(),
    };
    Ok(!group
        .statistics
        .bounds
        .prune_row(group.file_row_group_idx, &[predicate])?)
}

#[test]
fn file_nan_proof_enables_row_group_pruning_without_changing_other_columns_or_views() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("floats.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Float64Array::from(vec![1.0, 1.0, 3.0, 3.0])),
            Arc::new(Float64Array::from(vec![1.0, 1.0, 3.0, 3.0])),
        ],
    )
    .unwrap();
    let properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(2))
        .build();
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(&path).unwrap(),
        schema,
        Some(properties),
    )
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let file = DataFile::local(path.clone(), std::fs::metadata(path).unwrap().len());
    let dispatch = Dispatch::spin_up(1, 32, None);
    let mut loaded = load_file_row_groups(
        dispatch.dispatcher(),
        &[file],
        TableColumns::by_name(vec![]),
    )
    .unwrap()
    .pop()
    .unwrap();
    let original = loaded.row_groups.clone();

    loaded.mark_columns_nan_free(&[0]);

    for (compare, value, expected) in [
        (CompareType::Equal, 1.0, [false, true]),
        (CompareType::NotEqual, 1.0, [true, false]),
        (CompareType::Less, 2.0, [false, true]),
        (CompareType::LessEqual, 1.0, [false, true]),
        (CompareType::Greater, 2.0, [true, false]),
        (CompareType::GreaterEqual, 3.0, [true, false]),
    ] {
        let constant = Scalar::new(Arc::new(Float64Array::from(vec![value])) as ArrayRef);
        let eliminated: Vec<_> = loaded
            .row_groups
            .iter()
            .map(|group| row_group_eliminated(group, 0, compare, &constant).unwrap())
            .collect();
        assert_eq!(eliminated, expected, "{compare:?} {value}");
    }
    let two = Scalar::new(Arc::new(Float64Array::from(vec![2.0])) as ArrayRef);
    assert!(!row_group_eliminated(&loaded.row_groups[0], 1, CompareType::Greater, &two).unwrap());
    assert!(!row_group_eliminated(&original[0], 0, CompareType::Greater, &two).unwrap());
    assert!(row_group_eliminated(&original[0], 0, CompareType::Equal, &two).unwrap());

    // Adding another proof preserves the first one and leaves captured views
    // unchanged. Scalar consumers still see the original bounds and counts.
    let first_proof = loaded.clone();
    loaded.mark_columns_nan_free(&[1]);
    assert!(row_group_eliminated(&loaded.row_groups()[0], 0, CompareType::Greater, &two).unwrap());
    assert!(row_group_eliminated(&loaded.row_groups()[0], 1, CompareType::Greater, &two).unwrap());
    assert!(
        !row_group_eliminated(&first_proof.row_groups()[0], 1, CompareType::Greater, &two).unwrap()
    );
    for (current, original) in loaded.row_groups().iter().zip(&original) {
        for column in 0..2 {
            let current = current.column_statistics(column).unwrap();
            let original = original.column_statistics(column).unwrap();
            assert_eq!(current.null_count, original.null_count);
            assert_eq!(current.distinct_count, original.distinct_count);
            assert_eq!(
                current.min().unwrap().into_inner().to_data(),
                original.min().unwrap().into_inner().to_data()
            );
            assert_eq!(
                current.max().unwrap().into_inner().to_data(),
                original.max().unwrap().into_inner().to_data()
            );
        }
    }
}
