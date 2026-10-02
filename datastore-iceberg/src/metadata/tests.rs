use std::collections::HashMap;
use std::sync::Arc;

use super::*;
use planner::expression::CompareType;

use arrow_array::{ArrayRef, Int64Array, Scalar};
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, FormatVersion, ManifestContentType,
    NestedField, PartitionSpec, PrimitiveType, Schema, Type,
};
use planner::expression::{Compare, Expression, Ref, TableFilter};
use planner::types::Type as PivotType;

fn schema(id: i32, name: &str, primitive: PrimitiveType) -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_fields(vec![Arc::new(NestedField::optional(
                id,
                name,
                Type::Primitive(primitive),
            ))])
            .build()
            .unwrap(),
    )
}

fn prune(
    current: &SchemaRef,
    stored: SchemaRef,
    compare: CompareType,
    constant: i64,
) -> impl Fn(&DataFile) -> bool {
    let predicates = TableFilter::Expression(Box::new(Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx: 0,
            return_type: PivotType::Int64,
            name: None,
        })),
        right: Box::new(Expression::Constant(Scalar::new(
            Arc::new(Int64Array::from(vec![constant])) as ArrayRef,
        ))),
        compare_type: compare,
        return_type: PivotType::Boolean,
    })))
    .pruning_predicates();
    let metadata = ManifestMetadata {
        schema: stored.clone(),
        schema_id: stored.schema_id(),
        partition_spec: PartitionSpec::builder(stored).build().unwrap(),
        format_version: FormatVersion::V2,
        content: ManifestContentType::Data,
    };
    let current = current.clone();
    move |file| {
        file_statistics(&current, &[(&metadata, file)])
            .unwrap()
            .prune(&predicates)
            .unwrap()
            .value(0)
    }
}

fn file(value: Option<Datum>, null_count: Option<u64>) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path("file.parquet".into())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(1)
        .record_count(1)
        .value_counts(HashMap::from([(1, 1)]))
        .null_value_counts(null_count.map(|count| (1, count)).into_iter().collect())
        .lower_bounds(value.clone().map(|value| (1, value)).into_iter().collect())
        .upper_bounds(value.map(|value| (1, value)).into_iter().collect())
        .build()
        .unwrap()
}

#[test]
fn renamed_and_promoted_fields_compare_old_metrics_in_the_current_type() {
    let current = schema(1, "renamed", PrimitiveType::Long);
    let stored = schema(1, "original", PrimitiveType::Int);
    let matching = file(Some(Datum::int(5)), Some(0));
    let other = file(Some(Datum::int(3)), Some(0));
    let pruner = prune(&current, stored, CompareType::Greater, 4);

    let kept = pruner(&matching);
    let excluded = pruner(&other);

    assert!(kept);
    assert!(!excluded);
}

#[test]
fn constants_outside_an_older_types_range_prune_safely() {
    let current = schema(1, "id", PrimitiveType::Long);
    let stored = schema(1, "id", PrimitiveType::Int);
    let file = file(Some(Datum::int(5)), Some(0));
    let greater = prune(
        &current,
        stored.clone(),
        CompareType::Greater,
        i64::from(i32::MAX) + 1,
    );
    let less = prune(&current, stored, CompareType::Less, i64::from(i32::MAX) + 1);

    let above = greater(&file);
    let below = less(&file);

    assert!(!above);
    assert!(below);
}

#[test]
fn missing_metrics_and_null_values_have_different_meanings() {
    let schema = schema(1, "id", PrimitiveType::Long);
    let pruner = prune(&schema, schema.clone(), CompareType::Equal, 5);
    let unknown = file(None, None);
    let all_null = file(None, Some(1));

    let unknown_matches = pruner(&unknown);
    let null_matches = pruner(&all_null);

    assert!(unknown_matches);
    assert!(!null_matches);
}

#[test]
fn promoted_float_partition_predicates_do_not_round_the_query_constant() {
    let current = schema(1, "value", PrimitiveType::Double);
    let stored = schema(1, "value", PrimitiveType::Float);
    let partition_spec = PartitionSpec::builder(stored.clone())
        .add_partition_field(
            "value",
            "value_partition",
            iceberg::spec::Transform::Identity,
        )
        .unwrap()
        .build()
        .unwrap();
    let metadata = ManifestMetadata {
        schema: stored.clone(),
        schema_id: stored.schema_id(),
        partition_spec,
        format_version: FormatVersion::V2,
        content: ManifestContentType::Data,
    };
    let predicates = TableFilter::Expression(Box::new(Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx: 0,
            return_type: PivotType::Float64,
            name: None,
        })),
        right: Box::new(Expression::Constant(Scalar::new(
            Arc::new(arrow_array::Float64Array::from(vec![1.00000005])) as ArrayRef,
        ))),
        compare_type: CompareType::NotEqual,
        return_type: PivotType::Boolean,
    })))
    .pruning_predicates();
    let file = DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_path("file.parquet".into())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(1)
        .record_count(1)
        .partition(iceberg::spec::Struct::from_iter([Some(
            iceberg::spec::Literal::Primitive(Datum::float(1.0_f32).literal().clone()),
        )]))
        .lower_bounds(HashMap::from([(1, Datum::float(1.0_f32))]))
        .upper_bounds(HashMap::from([(1, Datum::float(1.0_f32))]))
        .nan_value_counts(HashMap::from([(1, 0)]))
        .build()
        .unwrap();

    let keep = file_statistics(&current, &[(&metadata, &file)])
        .unwrap()
        .prune(&predicates)
        .unwrap();

    assert!(keep.value(0));
}

#[test]
fn a_column_added_after_a_manifest_was_written_is_null() {
    let current = schema(2, "added", PrimitiveType::Long);
    let stored = schema(1, "original", PrimitiveType::Long);
    let pruner = prune(&current, stored, CompareType::Equal, 5);
    let file = file(Some(Datum::long(5)), Some(0));

    let matches = pruner(&file);

    assert!(!matches);
}

fn manifest_entry(
    spec: i32,
    row: usize,
    partitions: Option<Vec<iceberg::spec::FieldSummary>>,
) -> ManifestFile {
    ManifestFile {
        manifest_path: format!("manifest-{row}.avro"),
        manifest_length: 100 + row as i64,
        partition_spec_id: spec,
        content: ManifestContentType::Data,
        sequence_number: 1,
        min_sequence_number: 1,
        added_snapshot_id: 1,
        added_files_count: Some(1),
        existing_files_count: Some(0),
        deleted_files_count: Some(0),
        added_rows_count: Some(row as u64 + 1),
        existing_rows_count: Some(2),
        deleted_rows_count: Some(0),
        partitions,
        key_metadata: None,
        first_row_id: None,
    }
}

fn summary(bytes: Option<Vec<u8>>) -> iceberg::spec::FieldSummary {
    iceberg::spec::FieldSummary {
        contains_null: bytes.is_none(),
        contains_nan: Some(false),
        lower_bound: bytes.clone().map(Into::into),
        upper_bound: bytes.map(Into::into),
    }
}

#[test]
fn manifest_descriptors_stay_aligned_with_bounds_across_evolved_specs() {
    use iceberg::spec::{SortOrder, TableMetadataBuilder, Transform, UnboundPartitionSpec};
    let schema = Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::optional(
                1,
                "at",
                Type::Primitive(PrimitiveType::Timestamp),
            )),
            Arc::new(NestedField::optional(
                2,
                "region",
                Type::Primitive(PrimitiveType::String),
            )),
        ])
        .build()
        .unwrap();
    let monthly = UnboundPartitionSpec::builder()
        .add_partition_field(2, "region", Transform::Identity)
        .unwrap()
        .add_partition_field(1, "at_month", Transform::Month)
        .unwrap()
        .build();
    let daily = UnboundPartitionSpec::builder()
        .add_partition_field(1, "at_day", Transform::Day)
        .unwrap()
        .build();
    let reordered = UnboundPartitionSpec::builder()
        .add_partition_field(1, "at_month", Transform::Month)
        .unwrap()
        .add_partition_field(2, "region", Transform::Identity)
        .unwrap()
        .build();
    let metadata = TableMetadataBuilder::new(
        schema,
        monthly,
        SortOrder::unsorted_order(),
        "s3://test/table".into(),
        FormatVersion::V2,
        HashMap::new(),
    )
    .unwrap()
    .add_partition_spec(daily)
    .unwrap()
    .add_partition_spec(reordered)
    .unwrap()
    .add_partition_spec(UnboundPartitionSpec::builder().build())
    .unwrap()
    .build()
    .unwrap()
    .metadata;
    let integer = |value: i32| summary(Some(value.to_le_bytes().to_vec()));
    let region = || summary(Some(b"west".to_vec()));
    let manifests = vec![
        manifest_entry(0, 0, Some(vec![region(), integer(600)])),
        manifest_entry(1, 1, Some(vec![integer(18_263)])),
        manifest_entry(2, 2, Some(vec![integer(601), region()])),
        manifest_entry(1, 3, Some(vec![integer(18_262)])),
        manifest_entry(3, 4, None),
        manifest_entry(0, 5, None),
        manifest_entry(0, 6, Some(vec![region(), summary(None)])),
    ];
    let list = ManifestList::new(&metadata, &manifests).unwrap();
    // Month is shared across reordered specs; only three distinct expressions.
    assert_eq!(list.statistics.partition_stats().len(), 3);
    assert_eq!(list.statistics.len(), list.manifests().len());
    assert_eq!(list.row_count(), Some(42));
    let predicate = ColumnPredicate {
        column_idx: 0,
        path: vec![],
        as_type: None,
        compare_type: ::pruning::Comparison::Less,
        value: Scalar::new(Arc::new(arrow_array::TimestampMicrosecondArray::from(vec![
            18_263 * 86_400_000_000_i64,
        ])) as ArrayRef),
    };
    let selected = list.select(&[predicate]).unwrap();
    assert_eq!(
        selected
            .iter()
            .map(|manifest| manifest.path.as_str())
            .collect::<Vec<_>>(),
        vec![
            "manifest-0.avro",
            "manifest-3.avro",
            "manifest-4.avro",
            "manifest-5.avro"
        ]
    );
    assert_eq!(
        selected
            .iter()
            .map(|manifest| manifest.length)
            .collect::<Vec<_>>(),
        vec![100, 103, 104, 105]
    );
    assert_eq!(
        selected
            .iter()
            .map(|manifest| manifest.added_rows_count)
            .collect::<Vec<_>>(),
        vec![Some(1), Some(4), Some(5), Some(6)]
    );
    assert!(
        selected
            .iter()
            .all(|manifest| manifest.existing_rows_count == Some(2))
    );
    assert!(std::ptr::eq(selected[1], &list.manifests()[3]));
    assert_eq!(list.select(&[]).unwrap().len(), 7);
    assert_eq!(list.row_count(), Some(42));

    let mut missing = manifests.clone();
    missing[0].added_rows_count = None;
    assert_eq!(
        ManifestList::new(&metadata, &missing).unwrap().row_count(),
        None
    );
    missing[0].added_rows_count = Some(u64::MAX);
    assert_eq!(
        ManifestList::new(&metadata, &missing).unwrap().row_count(),
        None
    );
    assert_eq!(
        ManifestList::new(&metadata, &[]).unwrap().row_count(),
        Some(0)
    );
}

#[test]
fn file_descriptors_stay_aligned_when_an_older_partition_type_cannot_match() {
    use iceberg::spec::{Literal, ManifestEntry, ManifestStatus, Struct, Transform};
    let old = schema(1, "id", PrimitiveType::Int);
    let new = Arc::new(
        Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![Arc::new(NestedField::optional(
                1,
                "id",
                Type::Primitive(PrimitiveType::Long),
            ))])
            .build()
            .unwrap(),
    );
    let manifest = |schema: SchemaRef, row: usize, value: i64| {
        let spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(row as i32)
            .add_partition_field("id", "id_partition", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();
        let value = if row == 0 {
            Datum::int(value as i32)
        } else {
            Datum::long(value)
        };
        let data = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_format(DataFileFormat::Parquet)
            .file_path(format!("file-{row}.parquet"))
            .file_size_in_bytes(100 + row as u64)
            .record_count(1)
            .partition(Struct::from_iter([Some(Literal::from(value))]))
            .nan_value_counts(HashMap::from([(1, 0)]))
            .build()
            .unwrap();
        Manifest::new(
            ManifestMetadata {
                schema_id: schema.schema_id(),
                schema,
                partition_spec: spec,
                format_version: FormatVersion::V2,
                content: ManifestContentType::Data,
            },
            vec![
                ManifestEntry::builder()
                    .status(ManifestStatus::Added)
                    .data_file(data)
                    .build(),
            ],
        )
    };
    let larger = i64::from(i32::MAX) + 1;
    let manifests = vec![manifest(old, 0, 5), manifest(new.clone(), 1, larger)];
    let files = DataFiles::new(&new, &manifests).unwrap();
    drop(manifests);
    let sliced = files.slice(1, 1);
    // File bounds were omitted: only partition metadata can decide this.
    let selected = files
        .select(&[ColumnPredicate {
            column_idx: 0,
            path: vec![],
            as_type: None,
            compare_type: ::pruning::Comparison::Equal,
            value: Scalar::new(Arc::new(Int64Array::from(vec![larger])) as ArrayRef),
        }])
        .unwrap();
    assert_eq!(selected.files().len(), 1);
    assert_eq!(selected.files()[0].path, "file-1.parquet");
    assert_eq!(selected.files()[0].length, 101);
    assert_eq!(selected.nan_free_columns(0), vec![0]);
    assert_eq!(sliced.files()[0].path, selected.files()[0].path);
    assert_eq!(sliced.nan_free_columns(0), selected.nan_free_columns(0));
    assert_eq!(selected.statistics.len(), selected.files().len());
}

#[test]
fn file_selection_and_slicing_preserve_order_bounds_and_nan_proofs() {
    use iceberg::spec::{ManifestEntry, ManifestStatus};
    let schema = schema(1, "value", PrimitiveType::Double);
    let entries = [
        (0, Some(10.0), Some(0), ManifestStatus::Added),
        (1, Some(10.0), Some(0), ManifestStatus::Added),
        (1, Some(20.0), Some(0), ManifestStatus::Existing),
        (1, Some(10.0), Some(0), ManifestStatus::Deleted),
        (1, Some(10.0), None, ManifestStatus::Existing),
        (1, None, Some(0), ManifestStatus::Added),
    ]
    .into_iter()
    .enumerate()
    .map(|(row, (count, value, nan_count, status))| {
        let bounds: HashMap<_, _> = value.map(|v| (1, Datum::double(v))).into_iter().collect();
        let data = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_format(DataFileFormat::Parquet)
            .file_path(format!("file-{row}.parquet"))
            .file_size_in_bytes(100 + row as u64)
            .record_count(count)
            .lower_bounds(bounds.clone())
            .upper_bounds(bounds)
            .nan_value_counts(nan_count.map(|count| (1, count)).into_iter().collect())
            .build()
            .unwrap();
        ManifestEntry::builder()
            .status(status)
            .data_file(data)
            .build()
    })
    .collect();
    let manifest = Manifest::new(
        ManifestMetadata {
            schema: schema.clone(),
            schema_id: schema.schema_id(),
            partition_spec: PartitionSpec::builder(schema.clone()).build().unwrap(),
            format_version: FormatVersion::V2,
            content: ManifestContentType::Data,
        },
        entries,
    );
    let equal = |value| ColumnPredicate {
        column_idx: 0,
        path: vec![],
        as_type: None,
        compare_type: ::pruning::Comparison::Equal,
        value: Scalar::new(Arc::new(arrow_array::Float64Array::from(vec![value])) as ArrayRef),
    };
    let files = DataFiles::new(&schema, &[manifest])
        .unwrap()
        .select(&[equal(10.0)])
        .unwrap();
    // Deleted, empty and non-matching files are excluded. Missing bounds stay.
    assert_eq!(
        files
            .files()
            .iter()
            .map(|file| file.path.as_str())
            .collect::<Vec<_>>(),
        ["file-1.parquet", "file-4.parquet", "file-5.parquet"]
    );
    assert_eq!(files.nan_free_columns(0), [0]);
    assert!(files.nan_free_columns(1).is_empty());
    assert_eq!(files.nan_free_columns(2), [0]);

    let tail = files.slice(1, 2);
    assert!(tail.nan_free_columns(0).is_empty());
    assert_eq!(tail.nan_free_columns(1), [0]);
    let unknown = tail.select(&[equal(30.0)]).unwrap();
    assert_eq!(unknown.files().len(), 1);
    assert_eq!(unknown.files()[0].path, "file-5.parquet");
    assert_eq!(unknown.files()[0].length, 105);
    assert_eq!(unknown.nan_free_columns(0), [0]);
}
