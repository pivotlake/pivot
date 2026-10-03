use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int64Array, RecordBatch, Scalar, StringArray,
    StringViewArray,
};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{BoundarySlot, Dispatch};
use object_storage::DataFile;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use planner::catalog::DynamicScanPredicate;
use planner::expression::CompareType;
use pruning::{ColumnPath, Predicate};

use super::{RowGroupMetadata, prune_row_groups};
use crate::metadata::load_file_row_groups;
use crate::{TableColumns, row_group_filter_from};

/// Write `columns` one row per row group and load the file's row groups.
fn write_and_load_file(
    dispatch: &Dispatch,
    columns: Vec<(&str, ArrayRef)>,
    statistics: EnabledStatistics,
) -> Vec<Arc<RowGroupMetadata>> {
    let batch = RecordBatch::try_from_iter(columns).unwrap();
    write_and_load_batch(dispatch, batch, statistics)
}

fn write_and_load_batch(
    dispatch: &Dispatch,
    batch: RecordBatch,
    statistics: EnabledStatistics,
) -> Vec<Arc<RowGroupMetadata>> {
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
    .row_groups
}

/// A VARIANT column `doc` shredding `path` as `data_type`, one document per
/// row group, next to a plain `id` column.
fn write_and_load_shredded(
    dispatch: &Dispatch,
    documents: impl Into<StringArray>,
    shredded_paths: &[(&str, DataType)],
) -> Vec<Arc<RowGroupMetadata>> {
    use parquet_variant_compute::{ShreddedSchemaBuilder, json_to_variant, shred_variant};

    let documents: StringArray = documents.into();
    let ids: Vec<f64> = (1..=documents.len()).map(|id| id as f64 * 10.0).collect();
    let documents = Arc::new(documents) as ArrayRef;
    let variant = json_to_variant(&documents).unwrap();
    let mut schema = ShreddedSchemaBuilder::new();
    for (path, data_type) in shredded_paths {
        schema = schema.with_path(*path, data_type).unwrap();
    }
    let shredded = shred_variant(&variant, &schema.build()).unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            shredded.field("doc"),
            Field::new("id", DataType::Float64, false),
        ])),
        vec![
            Arc::new(shredded.into_inner()),
            Arc::new(Float64Array::from(ids)),
        ],
    )
    .unwrap();
    write_and_load_batch(dispatch, batch, EnabledStatistics::Chunk)
}

fn predicate(column: impl Into<ColumnPath>, compare: CompareType, value: ArrayRef) -> Predicate {
    Predicate {
        column: column.into(),
        comparison: compare.into(),
        value: Scalar::new(value),
    }
}

fn int_predicate(column: impl Into<ColumnPath>, compare: CompareType, value: i64) -> Predicate {
    predicate(column, compare, Arc::new(Int64Array::from(vec![value])))
}

fn field(column_idx: usize, path: &[&str]) -> ColumnPath {
    ColumnPath {
        column_idx,
        path: path.iter().map(|name| name.to_string()).collect(),
    }
}

fn assert_groups(groups: &[Arc<RowGroupMetadata>], expected: &[&Arc<RowGroupMetadata>]) {
    assert_eq!(groups.len(), expected.len());
    for (actual, expected) in groups.iter().zip(expected) {
        assert!(Arc::ptr_eq(actual, expected));
    }
}

#[test]
fn pruning_keeps_the_groups_every_predicate_may_match_in_order() {
    let dispatch = Dispatch::spin_up(1, 32, None);
    let groups = write_and_load_file(
        &dispatch,
        vec![
            ("x", Arc::new(Int64Array::from(vec![10, 20, 30, 40]))),
            ("y", Arc::new(Int64Array::from(vec![0, 50, 0, 0]))),
        ],
        EnabledStatistics::Chunk,
    );
    let predicates = [
        int_predicate(0, CompareType::GreaterEqual, 15),
        int_predicate(1, CompareType::Less, 10),
    ];

    let pruned = prune_row_groups(&groups, &predicates);
    let none = prune_row_groups(&groups, &[int_predicate(0, CompareType::Less, 0)]);

    assert_groups(&pruned, &[&groups[2], &groups[3]]);
    assert!(none.is_empty());
}

#[test]
fn pruning_distinguishes_unknown_bounds_from_all_null_groups() {
    let dispatch = Dispatch::spin_up(1, 32, None);
    let rows = Arc::new(Int64Array::from(vec![None, Some(1), Some(5)])) as ArrayRef;
    let known = write_and_load_file(
        &dispatch,
        vec![("x", rows.clone())],
        EnabledStatistics::Chunk,
    );
    let unknown = write_and_load_file(&dispatch, vec![("x", rows)], EnabledStatistics::None);
    let predicates = [int_predicate(0, CompareType::Greater, 3)];

    let pruned_known = prune_row_groups(&known, &predicates);
    let pruned_unknown = prune_row_groups(&unknown, &predicates);

    assert_groups(&pruned_known, &[&known[2]]);
    assert_groups(&pruned_unknown, &[&unknown[0], &unknown[1], &unknown[2]]);
}

#[test]
fn each_group_is_pruned_by_its_own_file_in_any_order() {
    let dispatch = Dispatch::spin_up(1, 32, None);
    let load = |values: Vec<i64>| {
        write_and_load_file(
            &dispatch,
            vec![("x", Arc::new(Int64Array::from(values)))],
            EnabledStatistics::Chunk,
        )
    };
    let a = load(vec![1, 30, 2]);
    let b = load(vec![100, 3]);
    let mixed = [&b[0], &b[1], &a[2], &a[0], &a[1]].map(Arc::clone);

    let pruned = prune_row_groups(&mixed, &[int_predicate(0, CompareType::Greater, 15)]);

    assert_groups(&pruned, &[&b[0], &a[1]]);
}

#[test]
fn static_and_dynamic_pruning_agree_on_float_ranges() {
    let dispatch = Dispatch::spin_up(1, 32, None);
    let groups = write_and_load_file(
        &dispatch,
        vec![(
            "x",
            Arc::new(Float64Array::from(vec![Some(1.0), Some(3.0), None])),
        )],
        EnabledStatistics::Chunk,
    );
    let constant = Arc::new(Float64Array::from(vec![2.0])) as ArrayRef;
    let slot = Arc::new(BoundarySlot::new());
    let dynamic_filter = row_group_filter_from(vec![DynamicScanPredicate {
        column_idx: 0,
        compare_type: CompareType::Greater,
        slot: slot.clone(),
    }])
    .unwrap();

    let pruned = prune_row_groups(
        &groups,
        &[predicate(0, CompareType::Greater, constant.clone())],
    );
    let unarmed: Vec<_> = groups.iter().map(|group| dynamic_filter(group)).collect();
    slot.publish_value(constant);
    let armed: Vec<_> = groups.iter().map(|group| dynamic_filter(group)).collect();

    assert_groups(&pruned, &[&groups[1]]);
    assert_eq!(unarmed, [true, true, true]);
    assert_eq!(armed, [false, true, false]);
}

#[test]
fn decoder_predicates_leave_nan_equality_to_sql() {
    let values: Vec<ArrayRef> = vec![
        Arc::new(Float32Array::from(vec![f32::NAN])),
        Arc::new(Float64Array::from(vec![f64::NAN])),
        Arc::new(Float32Array::from(vec![1.0])),
        Arc::new(Float64Array::from(vec![-0.0])),
        Arc::new(Int64Array::from(vec![1])),
    ];
    let predicates: Vec<_> = values
        .into_iter()
        .enumerate()
        .map(|(column, value)| predicate(column, CompareType::Equal, value))
        .collect();

    let decoder_predicates = crate::equality_predicates(&predicates);

    assert_eq!(
        decoder_predicates
            .iter()
            .map(|predicate| predicate.column_idx)
            .collect::<Vec<_>>(),
        [2, 3, 4],
    );
}

#[test]
fn a_variant_field_is_pruned_by_its_typed_leaf_wherever_the_file_puts_it() {
    let dispatch = Dispatch::spin_up(1, 32, None);
    let documents = vec![
        r#"{"item":{"price":10}}"#,
        r#"{"item":{"price":20}}"#,
        r#"{"item":{"price":"30"}}"#,
    ];
    let price = field(0, &["item", "price"]);

    // Shredding another field puts more leaves before the price and the id.
    for extra in [None, Some(("extra", DataType::Int64))] {
        let mut shredded_paths = vec![("item.price", DataType::Int64)];
        shredded_paths.extend(extra);
        let groups = write_and_load_shredded(&dispatch, documents.clone(), &shredded_paths);

        let by_price = prune_row_groups(
            &groups,
            &[int_predicate(price.clone(), CompareType::Equal, 10)],
        );
        let by_id = prune_row_groups(
            &groups,
            &[predicate(
                1,
                CompareType::Greater,
                Arc::new(Float64Array::from(vec![15.0])),
            )],
        );

        // The string price lives in the untyped fallback, which no typed
        // bound describes.
        assert_groups(&by_price, &[&groups[0], &groups[2]]);
        assert_groups(&by_id, &[&groups[1], &groups[2]]);
    }
}

#[test]
fn a_variant_field_cast_to_another_type_than_its_leaf_is_not_pruned() {
    let dispatch = Dispatch::spin_up(1, 32, None);
    let groups = write_and_load_shredded(
        &dispatch,
        vec![r#"{"price":10}"#, r#"{"other":1}"#],
        &[("price", DataType::Int64)],
    );
    let as_double = predicate(
        field(0, &["price"]),
        CompareType::Equal,
        Arc::new(Float64Array::from(vec![99.0])),
    );

    let pruned = prune_row_groups(&groups, &[as_double]);

    assert_groups(&pruned, &[&groups[0], &groups[1]]);
}

#[test]
fn a_null_variant_is_pruned_whatever_the_path_and_whether_or_not_it_is_shredded() {
    let dispatch = Dispatch::spin_up(1, 32, None);
    let groups = write_and_load_shredded(
        &dispatch,
        vec![Some(r#"{"price":10,"tag":7}"#), None],
        &[("price", DataType::Int64)],
    );
    let whole = int_predicate(0, CompareType::Equal, 5);
    let unshredded_field = int_predicate(field(0, &["tag"]), CompareType::Equal, 7);
    let shredded_as_another_type = predicate(
        field(0, &["price"]),
        CompareType::Equal,
        Arc::new(Float64Array::from(vec![10.0])),
    );

    for predicate in [whole, unshredded_field, shredded_as_another_type] {
        let pruned = prune_row_groups(&groups, &[predicate]);

        assert_groups(&pruned, &[&groups[0]]);
    }
}

#[test]
fn variant_string_bounds_keep_terminal_json_null_and_unknown_fallbacks() {
    let dispatch = Dispatch::spin_up(1, 32, None);
    let groups = write_and_load_shredded(
        &dispatch,
        vec![
            r#"{"name":"A"}"#,
            r#"{"name":null}"#,
            "null",
            "{}",
            r#"{"name":"null"}"#,
            r#"{"name":7}"#,
        ],
        &[("name", DataType::Utf8)],
    );
    let predicate = predicate(
        field(0, &["name"]),
        CompareType::Equal,
        Arc::new(StringViewArray::from(vec!["null"])),
    );

    let pruned = prune_row_groups(&groups, &[predicate]);

    // Terminal JSON null casts to the string "null". An ancestor JSON null
    // and a missing name cannot supply a value at this path. A numeric
    // fallback is unknown to these string bounds, so it must survive too.
    assert_groups(&pruned, &[&groups[1], &groups[4], &groups[5]]);
}
