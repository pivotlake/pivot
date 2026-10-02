use ::pruning::ColumnPredicate;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, Float32Array, Float64Array, Int64Array, RecordBatch, Scalar, StringArray,
    StringViewArray,
};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{BoundarySlot, Dispatch};
use object_storage::DataFile;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use planner::catalog::DynamicScanPredicate;
use planner::expression::CompareType;

use super::RowGroupMetadata;
use crate::metadata::{FileRowGroups, load_file_row_groups};
use crate::{TableColumns, row_group_filter_from};

fn write_and_load_file(
    dispatch: &Dispatch,
    columns: Vec<(&str, ArrayRef)>,
    statistics: EnabledStatistics,
) -> FileRowGroups {
    let batch = RecordBatch::try_from_iter(columns).unwrap();
    write_and_load_batch(dispatch, batch, statistics)
}

fn write_and_load_batch(
    dispatch: &Dispatch,
    batch: RecordBatch,
    statistics: EnabledStatistics,
) -> FileRowGroups {
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

fn predicate(column_idx: usize, compare_type: CompareType, value: ArrayRef) -> ColumnPredicate {
    ColumnPredicate {
        column_idx,
        path: vec![],
        as_type: None,
        compare_type: compare_type.into(),
        value: Scalar::new(value),
    }
}

fn int_predicate(column: usize, compare: CompareType, value: i64) -> ColumnPredicate {
    predicate(column, compare, Arc::new(Int64Array::from(vec![value])))
}

fn assert_groups(groups: &[Arc<RowGroupMetadata>], expected: &[&Arc<RowGroupMetadata>]) {
    assert_eq!(groups.len(), expected.len());
    for (actual, expected) in groups.iter().zip(expected) {
        assert!(Arc::ptr_eq(actual, expected));
    }
}

#[test]
fn pruning_preserves_order_and_file_indexes() {
    let dispatch = Dispatch::spin_up(1, 32, None);
    let file = write_and_load_file(
        &dispatch,
        vec![
            ("x", Arc::new(Int64Array::from(vec![10, 20, 30, 40]))),
            ("y", Arc::new(Int64Array::from(vec![0, 50, 0, 0]))),
        ],
        EnabledStatistics::Chunk,
    );
    let groups = file.row_groups();
    let mut predicates = vec![
        int_predicate(0, CompareType::GreaterEqual, 15),
        int_predicate(1, CompareType::Less, 10),
    ];
    let pruned = file.prune(&predicates);
    predicates.push(int_predicate(0, CompareType::Less, 40));
    let narrower = file.prune(&predicates);
    let empty = file.prune(&[int_predicate(0, CompareType::Less, 0)]);

    assert_groups(&pruned, &[&groups[2], &groups[3]]);
    assert_groups(&narrower, &[&groups[2]]);
    assert_eq!(
        pruned
            .iter()
            .map(|group| group.file_row_group_idx)
            .collect::<Vec<_>>(),
        [2, 3]
    );
    assert!(empty.is_empty());
    assert_eq!(file.row_groups().len(), 4);
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
    );
    let unknown = write_and_load_file(
        &dispatch,
        vec![("x", Arc::new(Int64Array::from(vec![None, Some(1)])))],
        EnabledStatistics::None,
    );

    let predicates = [int_predicate(0, CompareType::Greater, 3)];
    let pruned_known = known.prune(&predicates);
    let pruned_unknown = unknown.prune(&predicates);

    assert_groups(
        &pruned_known,
        &[&known.row_groups()[2], &known.row_groups()[3]],
    );
    assert_groups(
        &pruned_unknown,
        &[&unknown.row_groups()[0], &unknown.row_groups()[1]],
    );
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
    let unknown = file.clone();
    file.mark_columns_nan_free(&[0]);
    let proven = file;
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

    let pruned_proven = proven.prune(&range);
    let pruned_unknown = unknown.prune(&range);
    let equal_proven = proven.prune(&equality);
    let equal_unknown = unknown.prune(&equality);

    assert_groups(&pruned_proven, &[&proven.row_groups()[1]]);
    assert_groups(
        &pruned_unknown,
        &[&unknown.row_groups()[0], &unknown.row_groups()[1]],
    );
    assert!(equal_proven.is_empty());
    assert!(equal_unknown.is_empty());
}

#[test]
fn static_and_dynamic_pruning_share_conservative_float_semantics() {
    let dispatch = Dispatch::spin_up(1, 32, None);
    let file = write_and_load_file(
        &dispatch,
        vec![(
            "x",
            Arc::new(Float64Array::from(vec![Some(1.0), Some(3.0), None])),
        )],
        EnabledStatistics::Chunk,
    );
    let groups = file.row_groups();
    let constant = Arc::new(Float64Array::from(vec![2.0])) as ArrayRef;
    let predicates = [predicate(0, CompareType::Greater, constant.clone())];
    let slot = Arc::new(BoundarySlot::new());
    let dynamic_predicate = || DynamicScanPredicate {
        column_idx: 0,
        compare_type: CompareType::Greater,
        slot: slot.clone(),
    };

    let per_file = file.prune(&predicates);
    let dynamic_filter = row_group_filter_from(vec![dynamic_predicate()]).unwrap();
    let unarmed: Vec<_> = groups.iter().map(|group| dynamic_filter(group)).collect();
    slot.publish_value(constant);
    let dynamic_mask: Vec<_> = groups.iter().map(|group| dynamic_filter(group)).collect();

    assert_groups(&per_file, &[&groups[0], &groups[1]]);
    assert_eq!(unarmed, [true, true, true]);
    assert_eq!(dynamic_mask, [true, true, false]);
    assert_groups(file.row_groups(), &[&groups[0], &groups[1], &groups[2]]);
}

#[test]
fn pruning_uses_each_files_bounds_and_preserves_file_order() {
    let dispatch = Dispatch::spin_up(1, 32, None);
    let first = write_and_load_file(
        &dispatch,
        vec![("x", Arc::new(Int64Array::from(vec![1, 30])))],
        EnabledStatistics::Chunk,
    );
    let second = write_and_load_file(
        &dispatch,
        vec![("x", Arc::new(Int64Array::from(vec![100, 200])))],
        EnabledStatistics::Chunk,
    );
    let a = first.row_groups();
    let b = second.row_groups();
    let predicates = [int_predicate(0, CompareType::Greater, 15)];

    let selected: Vec<_> = [&second, &first]
        .into_iter()
        .flat_map(|file| file.prune(&predicates))
        .collect();
    assert_groups(&selected, &[&b[0], &b[1], &a[1]]);
    assert_groups(&first.prune(&predicates), &[&a[1]]);
    assert_groups(&second.prune(&predicates), &[&b[0], &b[1]]);
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

    let conservative = crate::equality_predicates(&predicates);

    assert_eq!(
        conservative
            .iter()
            .map(|predicate| predicate.column_idx)
            .collect::<Vec<_>>(),
        [2, 3, 4, 5],
    );
}

#[test]
fn logical_columns_and_paths_work_across_different_physical_leaf_layouts() {
    use parquet_variant_compute::{ShreddedSchemaBuilder, json_to_variant, shred_variant};

    let dispatch = Dispatch::spin_up(1, 32, None);
    let docs = Arc::new(StringArray::from(vec![
        r#"{"item":{"price":10}}"#,
        r#"{"item":{"price":20}}"#,
        r#"{"item":{"price":"30"}}"#,
    ])) as ArrayRef;
    let variant = json_to_variant(&docs).unwrap();
    let slot = Arc::new(BoundarySlot::new());
    slot.publish_value(Arc::new(Float64Array::from(vec![15.0])) as ArrayRef);
    let dynamic = row_group_filter_from(vec![DynamicScanPredicate {
        column_idx: 1,
        compare_type: CompareType::Greater,
        slot,
    }])
    .unwrap();

    for with_extra in [false, true] {
        let mut schema = ShreddedSchemaBuilder::new()
            .with_path("item.price", &DataType::Int64)
            .unwrap();
        if with_extra {
            // Shredding an absent sibling still adds physical leaves before
            // price and id, without putting sibling data in the root fallback.
            schema = schema.with_path("extra", &DataType::Int64).unwrap();
        }
        let shredded = shred_variant(&variant, &schema.build()).unwrap();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                shredded.field("doc"),
                Field::new("id", DataType::Float64, false),
            ])),
            vec![
                Arc::new(shredded.into_inner()),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            ],
        )
        .unwrap();
        let mut file = write_and_load_batch(&dispatch, batch, EnabledStatistics::Chunk);
        // Adding a NaN proof must retain logical columns and the
        // VARIANT validity masks for the other columns.
        file.mark_columns_nan_free(&[1]);
        let groups = file.row_groups();
        let price_path = vec!["item".into(), "price".into()];
        let ::pruning::ColumnStatistics::Variant(paths) =
            &groups[0].statistics.bounds.column_stats()[&0]
        else {
            panic!("VARIANT path bounds belong to their parent column");
        };
        let typed_leaf = super::variant_shredded_leaves(groups[0].schema.fields(), 0, &price_path)
            .unwrap()
            .typed_leaf;
        let physical = groups[0]
            .leaf_statistics(typed_leaf)
            .unwrap()
            .min()
            .unwrap()
            .into_inner();
        let logical = paths[&price_path].bounds.lower.as_ref().unwrap();
        assert_eq!(
            physical.to_data().buffers()[0].as_ptr(),
            logical.to_data().buffers()[0].as_ptr()
        );
        let path = ColumnPredicate {
            column_idx: 0,
            path: price_path,
            as_type: Some(DataType::Int64),
            ..int_predicate(0, CompareType::Equal, 10)
        };
        assert_groups(&file.prune(&[path]), &[&groups[0], &groups[2]]);
        assert_groups(
            &file.prune(&[predicate(
                1,
                CompareType::Greater,
                Arc::new(Float64Array::from(vec![15.0])),
            )]),
            &[&groups[1], &groups[2]],
        );
        assert_eq!(
            groups
                .iter()
                .map(|group| dynamic(group))
                .collect::<Vec<_>>(),
            [false, true, true]
        );
    }
}

#[test]
fn variant_string_bounds_keep_terminal_json_null_and_unknown_fallbacks() {
    use parquet_variant_compute::{ShreddedSchemaBuilder, json_to_variant, shred_variant};

    let dispatch = Dispatch::spin_up(1, 32, None);
    let docs = Arc::new(StringArray::from(vec![
        r#"{"name":"A"}"#,
        r#"{"name":null}"#,
        "null",
        "{}",
        r#"{"name":"null"}"#,
        r#"{"name":7}"#,
    ])) as ArrayRef;
    let variant = json_to_variant(&docs).unwrap();
    let schema = ShreddedSchemaBuilder::new()
        .with_path("name", &DataType::Utf8)
        .unwrap()
        .build();
    let shredded = shred_variant(&variant, &schema).unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![shredded.field("doc")])),
        vec![Arc::new(shredded.into_inner())],
    )
    .unwrap();
    let file = write_and_load_batch(&dispatch, batch, EnabledStatistics::Chunk);
    let groups = file.row_groups();
    let predicate = ColumnPredicate {
        column_idx: 0,
        path: vec!["name".into()],
        as_type: Some(DataType::Utf8View),
        compare_type: CompareType::Equal.into(),
        value: Scalar::new(Arc::new(StringViewArray::from(vec!["null"])) as ArrayRef),
    };

    // Terminal JSON null casts to the string "null". An ancestor JSON null
    // and a missing name cannot supply a value at this path. A numeric
    // fallback is unknown to these string bounds, so it must survive too.
    assert_groups(
        &file.prune(&[predicate]),
        &[&groups[1], &groups[4], &groups[5]],
    );
}
