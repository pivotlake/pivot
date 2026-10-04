use std::collections::HashMap;
use std::sync::Arc;

use super::*;
use arrow_array::{ArrayRef, Float64Array, Int64Array, Scalar, TimestampMicrosecondArray};
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, FieldSummary, FormatVersion, Literal,
    ManifestContentType, ManifestEntry, ManifestMetadata, ManifestStatus, NestedField,
    PartitionSpec, PrimitiveType, Schema, Struct, Transform as IcebergTransform, Type,
};
use planner::expression::{Compare, CompareType, Ref};
use planner::types::{Type as PivotType, type_from_physical};

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

fn predicate(compare_type: CompareType, value: ArrayRef) -> Vec<Expression> {
    let constant_type = type_from_physical(value.data_type()).unwrap();
    vec![Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx: 0,
            return_type: constant_type,
            name: None,
        })),
        right: Box::new(Expression::Constant(Scalar::new(value))),
        compare_type,
        return_type: PivotType::Boolean,
    })]
}

fn long_predicate(compare_type: CompareType, value: i64) -> Vec<Expression> {
    predicate(compare_type, Arc::new(Int64Array::from(vec![value])))
}

fn double_predicate(compare_type: CompareType, value: f64) -> Vec<Expression> {
    predicate(compare_type, Arc::new(Float64Array::from(vec![value])))
}

/// A one-row data file named `path` whose metrics describe field 1.
fn file(path: &str) -> DataFileBuilder {
    let mut builder = DataFileBuilder::default();
    builder
        .content(DataContentType::Data)
        .file_path(path.into())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(1)
        .record_count(1);
    builder
}

/// A file whose field 1 holds exactly `value`, without NULLs.
fn file_holding(path: &str, value: Datum) -> DataFile {
    file(path)
        .null_value_counts(HashMap::from([(1, 0)]))
        .lower_bounds(HashMap::from([(1, value.clone())]))
        .upper_bounds(HashMap::from([(1, value)]))
        .build()
        .unwrap()
}

/// A manifest written under `stored` and `spec`, listing `files` as added.
fn manifest(stored: &SchemaRef, spec: PartitionSpec, files: Vec<DataFile>) -> Manifest {
    let entries = files.into_iter().map(|file| (ManifestStatus::Added, file));
    manifest_of_entries(stored, spec, entries.collect())
}

fn manifest_of_entries(
    stored: &SchemaRef,
    spec: PartitionSpec,
    entries: Vec<(ManifestStatus, DataFile)>,
) -> Manifest {
    Manifest::new(
        ManifestMetadata {
            schema: stored.clone(),
            schema_id: stored.schema_id(),
            partition_spec: spec,
            format_version: FormatVersion::V2,
            content: ManifestContentType::Data,
        },
        entries
            .into_iter()
            .map(|(status, file)| {
                ManifestEntry::builder()
                    .status(status)
                    .data_file(file)
                    .build()
            })
            .collect(),
    )
}

fn unpartitioned(schema: &SchemaRef) -> PartitionSpec {
    PartitionSpec::builder(schema.clone()).build().unwrap()
}

fn partitioned_by(schema: &SchemaRef, spec_id: i32, transform: IcebergTransform) -> PartitionSpec {
    let column = schema.as_struct().fields()[0].name.clone();
    PartitionSpec::builder(schema.clone())
        .with_spec_id(spec_id)
        .add_partition_field(column, "partition", transform)
        .unwrap()
        .build()
        .unwrap()
}

/// The paths of the files `select_files` keeps.
fn selected(current: &SchemaRef, manifests: &[Manifest], filters: &[Expression]) -> Vec<String> {
    select_files(current, manifests, filters)
        .unwrap()
        .into_iter()
        .map(|file| file.path)
        .collect()
}

#[test]
fn renamed_and_promoted_fields_compare_old_metrics_in_the_current_type() {
    let current = schema(1, "renamed", PrimitiveType::Long);
    let stored = schema(1, "original", PrimitiveType::Int);
    let files = vec![
        file_holding("three", Datum::int(3)),
        file_holding("five", Datum::int(5)),
    ];
    let manifests = [manifest(&stored, unpartitioned(&stored), files)];

    let above_four = selected(
        &current,
        &manifests,
        &long_predicate(CompareType::Greater, 4),
    );
    let beyond_int = selected(
        &current,
        &manifests,
        &long_predicate(CompareType::Greater, i64::from(i32::MAX) + 1),
    );

    assert_eq!(above_four, ["five"]);
    assert!(beyond_int.is_empty());
}

#[test]
fn missing_metrics_are_unknown_and_an_all_null_column_matches_nothing() {
    let schema = schema(1, "id", PrimitiveType::Long);
    let unknown = file("unknown").build().unwrap();
    let all_null = file("all null")
        .null_value_counts(HashMap::from([(1, 1)]))
        .build()
        .unwrap();
    let manifests = [manifest(
        &schema,
        unpartitioned(&schema),
        vec![unknown, all_null],
    )];

    let kept = selected(&schema, &manifests, &long_predicate(CompareType::Equal, 5));

    assert_eq!(kept, ["unknown"]);
}

#[test]
fn a_column_added_after_a_manifest_was_written_is_null() {
    let current = schema(2, "added", PrimitiveType::Long);
    let stored = schema(1, "original", PrimitiveType::Long);
    let files = vec![file_holding("older", Datum::long(5))];
    let manifests = [manifest(&stored, unpartitioned(&stored), files)];

    let kept = selected(&current, &manifests, &long_predicate(CompareType::Equal, 5));

    assert!(kept.is_empty());
}

#[test]
fn only_live_non_empty_files_are_selected_in_manifest_order() {
    let schema = schema(1, "id", PrimitiveType::Long);
    let empty = file("empty").record_count(0).build().unwrap();
    let entries = vec![
        (ManifestStatus::Added, file_holding("added", Datum::long(1))),
        (
            ManifestStatus::Deleted,
            file_holding("deleted", Datum::long(1)),
        ),
        (ManifestStatus::Added, empty),
        (
            ManifestStatus::Existing,
            file_holding("existing", Datum::long(1)),
        ),
        (
            ManifestStatus::Existing,
            file_holding("other", Datum::long(2)),
        ),
    ];
    let manifests = [manifest_of_entries(
        &schema,
        unpartitioned(&schema),
        entries,
    )];

    let unfiltered = selected(&schema, &manifests, &[]);
    let ones = selected(&schema, &manifests, &long_predicate(CompareType::Equal, 1));

    assert_eq!(unfiltered, ["added", "existing", "other"]);
    assert_eq!(ones, ["added", "existing"]);
}

#[test]
fn a_file_is_pruned_by_the_partition_value_of_its_own_spec() {
    let stored = schema(1, "id", PrimitiveType::Int);
    let current = schema(1, "id", PrimitiveType::Long);
    let larger = i64::from(i32::MAX) + 1;
    let in_partition = |path: &str, value: Datum| {
        file(path)
            .partition(Struct::from_iter([Some(Literal::from(value))]))
            .build()
            .unwrap()
    };
    let manifests = [
        // Written while the column was an INT.
        manifest(
            &stored,
            partitioned_by(&stored, 0, IcebergTransform::Identity),
            vec![in_partition("five", Datum::int(5))],
        ),
        manifest(
            &current,
            partitioned_by(&current, 1, IcebergTransform::Identity),
            vec![in_partition("larger", Datum::long(larger))],
        ),
        manifest(
            &current,
            partitioned_by(&current, 2, IcebergTransform::Truncate(10)),
            vec![in_partition("fifties", Datum::long(50))],
        ),
        manifest(
            &current,
            PartitionSpec::builder(current.clone())
                .with_spec_id(3)
                .build()
                .unwrap(),
            vec![file("unpartitioned").build().unwrap()],
        ),
    ];

    let five = selected(&current, &manifests, &long_predicate(CompareType::Equal, 5));
    let fifty_five = selected(
        &current,
        &manifests,
        &long_predicate(CompareType::Equal, 55),
    );
    let large = selected(
        &current,
        &manifests,
        &long_predicate(CompareType::Equal, larger),
    );

    assert_eq!(five, ["five", "unpartitioned"]);
    assert_eq!(fifty_five, ["fifties", "unpartitioned"]);
    assert_eq!(large, ["larger", "unpartitioned"]);
}

#[test]
fn a_promoted_float_partition_value_is_compared_exactly() {
    let current = schema(1, "value", PrimitiveType::Double);
    let stored = schema(1, "value", PrimitiveType::Float);
    let one = file("one")
        .partition(Struct::from_iter([Some(Literal::from(Datum::float(
            1.0_f32,
        )))]))
        .build()
        .unwrap();
    let spec = partitioned_by(&stored, 0, IcebergTransform::Identity);
    let manifests = [manifest(&stored, spec, vec![one])];

    // 1.00000005 rounds to 1.0 as a FLOAT, but the stored 1.0 differs from it.
    let differs = selected(
        &current,
        &manifests,
        &double_predicate(CompareType::NotEqual, 1.00000005),
    );
    let equals = selected(
        &current,
        &manifests,
        &double_predicate(CompareType::Equal, 1.00000005),
    );

    assert_eq!(differs, ["one"]);
    assert!(equals.is_empty());
}

fn manifest_entry(spec_id: i32, row: usize, partitions: Option<Vec<FieldSummary>>) -> ManifestFile {
    ManifestFile {
        manifest_path: format!("manifest-{row}.avro"),
        manifest_length: 100 + row as i64,
        partition_spec_id: spec_id,
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

/// The summary of a partition field holding one value, or only NULLs.
fn summary(bytes: Option<Vec<u8>>) -> FieldSummary {
    FieldSummary {
        contains_null: bytes.is_none(),
        contains_nan: Some(false),
        lower_bound: bytes.clone().map(Into::into),
        upper_bound: bytes.map(Into::into),
    }
}

/// A table of `at` and `region` whose spec changed three times: by region and
/// month, by day, by month and region, and unpartitioned.
fn evolved_table() -> TableMetadata {
    use iceberg::spec::{SortOrder, TableMetadataBuilder, UnboundPartitionSpec};
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
    let spec = |fields: &[(i32, &str, IcebergTransform)]| {
        let mut spec = UnboundPartitionSpec::builder();
        for (source_id, name, transform) in fields {
            spec = spec
                .add_partition_field(*source_id, name.to_string(), *transform)
                .unwrap();
        }
        spec.build()
    };
    let by_region_and_month = spec(&[
        (2, "region", IcebergTransform::Identity),
        (1, "at_month", IcebergTransform::Month),
    ]);
    TableMetadataBuilder::new(
        schema,
        by_region_and_month,
        SortOrder::unsorted_order(),
        "s3://test/table".into(),
        FormatVersion::V2,
        HashMap::new(),
    )
    .unwrap()
    .add_partition_spec(spec(&[(1, "at_day", IcebergTransform::Day)]))
    .unwrap()
    .add_partition_spec(spec(&[
        (1, "at_month", IcebergTransform::Month),
        (2, "region", IcebergTransform::Identity),
    ]))
    .unwrap()
    .add_partition_spec(spec(&[]))
    .unwrap()
    .build()
    .unwrap()
    .metadata
}

#[test]
fn a_manifest_is_pruned_by_the_partition_summaries_of_its_own_spec() {
    let metadata = evolved_table();
    let integer = |value: i32| summary(Some(value.to_le_bytes().to_vec()));
    let region = || summary(Some(b"west".to_vec()));
    // Day 18263 is 2020-01-02, in month 600.
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
    let before_day_18263 = predicate(
        CompareType::Less,
        Arc::new(TimestampMicrosecondArray::from(vec![
            18_263 * 86_400_000_000_i64,
        ])),
    );

    let selected = list.select(&before_day_18263);

    let paths: Vec<_> = selected.iter().map(|manifest| &manifest.path).collect();
    let lengths: Vec<_> = selected.iter().map(|manifest| manifest.length).collect();
    assert_eq!(
        paths,
        [
            "manifest-0.avro",
            "manifest-3.avro",
            "manifest-4.avro",
            "manifest-5.avro"
        ]
    );
    assert_eq!(lengths, [100, 103, 104, 105]);
    assert_eq!(list.select(&[]).len(), 7);
}

#[test]
fn the_row_count_is_known_only_when_every_manifest_counts_its_rows() {
    let metadata = evolved_table();
    let counted = vec![manifest_entry(3, 0, None), manifest_entry(3, 1, None)];
    let mut uncounted = counted.clone();
    uncounted[0].added_rows_count = None;
    let mut overflowing = counted.clone();
    overflowing[0].added_rows_count = Some(u64::MAX);

    let row_count =
        |manifests: &[ManifestFile]| ManifestList::new(&metadata, manifests).unwrap().row_count();

    assert_eq!(row_count(&counted), Some(7));
    assert_eq!(row_count(&uncounted), None);
    assert_eq!(row_count(&overflowing), None);
    assert_eq!(row_count(&[]), Some(0));
}
