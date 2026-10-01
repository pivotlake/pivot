use std::sync::Arc;

use arrow_array::{ArrayRef, Float32Array, Float64Array, Int64Array, RecordBatch, Scalar};
use dispatch::{BoundarySlot, Dispatch};
use object_storage::DataFile;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use planner::catalog::DynamicScanPredicate;
use planner::expression::CompareType;

use super::{PushedPredicate, RowGroupMetadata, prune_file_row_groups};
use crate::metadata::{FileRowGroups, load_file_row_groups};
use crate::{ParquetTable, TableColumns, prune_parquet, row_group_filter_from};

fn write_and_load_file(
    dispatch: &Dispatch,
    columns: Vec<(&str, ArrayRef)>,
    statistics: EnabledStatistics,
) -> FileRowGroups {
    let batch = RecordBatch::try_from_iter(columns).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    let properties = WriterProperties::builder()
        .set_statistics_enabled(statistics)
        .set_max_row_group_row_count(Some(1))
        .build();
    let mut writer =
        ArrowWriter::try_new(file.reopen().unwrap(), batch.schema(), Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let source = DataFile::local(
        file.path().to_path_buf(),
        file.as_file().metadata().unwrap().len(),
    );
    load_file_row_groups(
        dispatch.dispatcher(),
        &[source],
        TableColumns::by_name(vec![]),
    )
    .unwrap()
    .pop()
    .unwrap()
}

fn predicate(column_idx: usize, compare_type: CompareType, value: ArrayRef) -> PushedPredicate {
    PushedPredicate {
        column_idx,
        path: vec![],
        as_type: None,
        compare_type,
        value: Scalar::new(value),
    }
}

fn int_predicate(column: usize, compare: CompareType, value: i64) -> PushedPredicate {
    predicate(column, compare, Arc::new(Int64Array::from(vec![value])))
}

fn assert_groups(groups: &[Arc<RowGroupMetadata>], expected: &[&Arc<RowGroupMetadata>]) {
    assert_eq!(groups.len(), expected.len());
    for (actual, expected) in groups.iter().zip(expected) {
        assert!(Arc::ptr_eq(actual, expected));
    }
}

#[test]
fn pruning_preserves_order_and_sparse_file_indexes() {
    let dispatch = Dispatch::spin_up(1, 32, None);
    let groups = write_and_load_file(
        &dispatch,
        vec![
            ("x", Arc::new(Int64Array::from(vec![10, 20, 30, 40]))),
            ("y", Arc::new(Int64Array::from(vec![0, 50, 0, 0]))),
        ],
        EnabledStatistics::Chunk,
    )
    .row_groups;
    let selected = vec![groups[3].clone(), groups[1].clone(), groups[2].clone()];

    let pruned = prune_file_row_groups(
        &selected,
        &[
            int_predicate(0, CompareType::GreaterEqual, 15),
            int_predicate(1, CompareType::Less, 10),
        ],
    );
    let pruned_again = prune_file_row_groups(&pruned, &[int_predicate(0, CompareType::Less, 40)]);
    let empty = prune_file_row_groups(&pruned_again, &[int_predicate(0, CompareType::Less, 0)]);

    assert_groups(&pruned, &[&groups[3], &groups[2]]);
    assert_groups(&pruned_again, &[&groups[2]]);
    assert!(empty.is_empty());
    assert_eq!(selected.len(), 3);
}

#[test]
fn pruning_distinguishes_unknown_bounds_from_all_null_groups() {
    let dispatch = Dispatch::spin_up(1, 32, None);
    let known = write_and_load_file(
        &dispatch,
        vec![(
            "x",
            Arc::new(Int64Array::from(vec![None, Some(1), Some(5), Some(8)])),
        )],
        EnabledStatistics::Chunk,
    )
    .row_groups;
    let unknown = write_and_load_file(
        &dispatch,
        vec![("x", Arc::new(Int64Array::from(vec![None, Some(1)])))],
        EnabledStatistics::None,
    )
    .row_groups;

    let predicates = [int_predicate(0, CompareType::Greater, 3)];
    let pruned_known = prune_file_row_groups(&known, &predicates);
    let pruned_unknown = prune_file_row_groups(&unknown, &predicates);

    assert_groups(&pruned_known, &[&known[2], &known[3]]);
    assert_groups(&pruned_unknown, &[&unknown[0], &unknown[1]]);
}

#[test]
fn pruning_keeps_nan_proofs_local_to_their_metadata_view() {
    let dispatch = Dispatch::spin_up(1, 32, None);
    let mut file = write_and_load_file(
        &dispatch,
        vec![(
            "x",
            Arc::new(Float64Array::from(vec![Some(1.0), Some(3.0), None])),
        )],
        EnabledStatistics::Chunk,
    );
    let unknown = file.row_groups.clone();
    file.mark_columns_nan_free(&[0]);
    let proven = file.row_groups;
    let range = [predicate(
        0,
        CompareType::Greater,
        Arc::new(Float64Array::from(vec![2.0])),
    )];
    let equality = [predicate(
        0,
        CompareType::Equal,
        Arc::new(Float64Array::from(vec![2.0])),
    )];

    let pruned_proven = prune_file_row_groups(&proven, &range);
    let pruned_unknown = prune_file_row_groups(&unknown, &range);
    let equal_proven = prune_file_row_groups(&proven, &equality);
    let equal_unknown = prune_file_row_groups(&unknown, &equality);

    assert_groups(&pruned_proven, &[&proven[1]]);
    assert_groups(&pruned_unknown, &[&unknown[0], &unknown[1]]);
    assert!(equal_proven.is_empty());
    assert!(equal_unknown.is_empty());
}

#[test]
fn scalar_pruning_still_uses_float_bounds_without_nan_proofs() {
    let dispatch = Dispatch::spin_up(1, 32, None);
    let groups = write_and_load_file(
        &dispatch,
        vec![(
            "x",
            Arc::new(Float64Array::from(vec![Some(1.0), Some(3.0), None])),
        )],
        EnabledStatistics::Chunk,
    )
    .row_groups;
    let table = ParquetTable::new(groups.clone());
    let constant = Arc::new(Float64Array::from(vec![2.0])) as ArrayRef;
    let predicates = [predicate(0, CompareType::Greater, constant.clone())];
    let slot = Arc::new(BoundarySlot::new());
    let dynamic_predicate = || DynamicScanPredicate {
        column_idx: 0,
        compare_type: CompareType::Greater,
        slot: slot.clone(),
    };

    let scalar = prune_parquet(&table, &predicates);
    let batched = prune_file_row_groups(&groups, &predicates);
    let scalar_filter = row_group_filter_from(vec![dynamic_predicate()]).unwrap();
    let bounds_filter = crate::pruning::row_group_filter_from(vec![dynamic_predicate()]).unwrap();
    let unarmed: Vec<_> = groups.iter().map(|group| scalar_filter(group)).collect();
    slot.publish_value(constant);
    let scalar_mask: Vec<_> = groups.iter().map(|group| scalar_filter(group)).collect();
    let bounds_mask: Vec<_> = groups.iter().map(|group| bounds_filter(group)).collect();

    assert_groups(scalar.row_groups(), &[&groups[1]]);
    assert_groups(&batched, &[&groups[0], &groups[1]]);
    assert_eq!(unarmed, [true, true, true]);
    assert_eq!(scalar_mask, [false, true, false]);
    assert_eq!(bounds_mask, [true, true, false]);
    assert_groups(table.row_groups(), &[&groups[0], &groups[1], &groups[2]]);
}

#[test]
fn conservative_decoder_predicates_leave_nan_equality_to_sql() {
    let values: Vec<ArrayRef> = vec![
        Arc::new(Float32Array::from(vec![f32::NAN])),
        Arc::new(Float64Array::from(vec![f64::NAN])),
        Arc::new(Float32Array::from(vec![1.0])),
        Arc::new(Float64Array::from(vec![-0.0])),
        Arc::new(Float64Array::from(vec![0.0])),
        Arc::new(Int64Array::from(vec![1])),
    ];
    let predicates: Vec<_> = values
        .into_iter()
        .enumerate()
        .map(|(column, value)| predicate(column, CompareType::Equal, value))
        .collect();

    let scalar = crate::equality_predicates(&predicates);
    let conservative = crate::pruning::equality_predicates(&predicates);

    assert_eq!(scalar.len(), 6);
    assert_eq!(
        conservative
            .iter()
            .map(|predicate| predicate.column_idx)
            .collect::<Vec<_>>(),
        [2, 3, 4, 5],
    );
}
