//! What a table reads of its manifest list and its manifests: the delete
//! manifests a scan cannot skip, told by the partition summaries of each
//! one's own spec, and of the data manifests' entries the live, non-empty
//! files, each under the schema its manifest was written with.

mod support;

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, Int64Array, Scalar, TimestampMicrosecondArray};
use iceberg::io::FileIO;
use iceberg::spec::{
    FieldSummary, FormatVersion, MAIN_BRANCH, ManifestContentType, ManifestFile,
    ManifestListWriter, NestedField, Operation, PrimitiveType, Schema, Snapshot, SortOrder,
    Summary, TableMetadata, TableMetadataBuilder, Transform as IcebergTransform, Type,
    UnboundPartitionSpec,
};
use iceberg_tables::files::data_files::DataFileEntry;
use iceberg_tables::{ManifestList, block_on};
use object_storage::{FileRef, ObjectPath};
use parquet_engine::FileStats;
use planner::expression::{Compare, CompareType, Expression, Ref};
use planner::types::{Type as PivotType, type_from_physical};
use support::{
    append, column, create_table, create_table_of, entry, int_scalar, live_entry, paths_in_order,
    publish_changed, selected, v_equals,
};

/// Where the manifest list of the hand-made table's one snapshot is.
const MANIFEST_LIST: &str = "memory:///metadata/snap-1.avro";

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

/// The manifest-list entry of a delete manifest, otherwise as
/// [`manifest_entry`] makes one.
fn delete_manifest_entry(
    spec_id: i32,
    row: usize,
    partitions: Option<Vec<FieldSummary>>,
) -> ManifestFile {
    ManifestFile {
        content: ManifestContentType::Deletes,
        ..manifest_entry(spec_id, row, partitions)
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
/// month, by day, by month and region, and unpartitioned. Its one snapshot
/// names the manifest list at [`MANIFEST_LIST`].
fn evolved_table() -> TableMetadata {
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
    let snapshot = Snapshot::builder()
        .with_snapshot_id(1)
        .with_sequence_number(1)
        .with_timestamp_ms(0)
        .with_manifest_list(MANIFEST_LIST)
        .with_summary(Summary {
            operation: Operation::Append,
            additional_properties: HashMap::new(),
        })
        .with_schema_id(0)
        .build();
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
    .set_branch_snapshot(snapshot, MAIN_BRANCH)
    .unwrap()
    .build()
    .unwrap()
    .metadata
}

/// The manifest list of `manifests` as the table `metadata` describes
/// reads it: written the way a commit writes one, and parsed.
fn list_manifests(metadata: &TableMetadata, manifests: Vec<ManifestFile>) -> ManifestList {
    let file_io = FileIO::new_with_memory();
    let bytes = block_on(async {
        let output = file_io.new_output(MANIFEST_LIST).unwrap();
        let mut writer = ManifestListWriter::v2(output.writer().await.unwrap(), 1, None, 1);
        writer.add_manifests(manifests.into_iter()).unwrap();
        writer.close().await.unwrap();
        let input = file_io.new_input(MANIFEST_LIST).unwrap();
        input.read().await.unwrap()
    });
    ManifestList::parse("table", metadata, &bytes).unwrap()
}

/// The filter that keeps `at` before the start of day 18263 after the
/// epoch.
fn before_day_18263() -> Vec<Expression> {
    predicate(
        CompareType::Less,
        Arc::new(TimestampMicrosecondArray::from(vec![
            18_263 * 86_400_000_000_i64,
        ])),
    )
}

#[test]
fn a_delete_manifest_is_pruned_by_the_partition_summaries_of_its_own_spec() {
    let metadata = evolved_table();
    let integer = |value: i32| summary(Some(value.to_le_bytes().to_vec()));
    let region = || summary(Some(b"west".to_vec()));
    // Day 18263 is 2020-01-02, in month 600.
    let manifests = vec![
        delete_manifest_entry(0, 0, Some(vec![region(), integer(600)])),
        delete_manifest_entry(1, 1, Some(vec![integer(18_263)])),
        delete_manifest_entry(2, 2, Some(vec![integer(601), region()])),
        delete_manifest_entry(1, 3, Some(vec![integer(18_262)])),
        delete_manifest_entry(3, 4, None),
        delete_manifest_entry(0, 5, None),
        delete_manifest_entry(0, 6, Some(vec![region(), summary(None)])),
    ];
    let list = list_manifests(&metadata, manifests);

    let selected = list.select_delete_manifests(&before_day_18263()).unwrap();

    let paths: Vec<_> = selected
        .iter()
        .map(|manifest| &manifest.manifest_path)
        .collect();
    let lengths: Vec<_> = selected
        .iter()
        .map(|manifest| manifest.manifest_length)
        .collect();
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
    assert_eq!(list.select_delete_manifests(&[]).unwrap().len(), 7);
}

#[test]
fn delete_manifests_are_listed_apart_from_data_manifests() {
    let metadata = evolved_table();
    let integer = |value: i32| summary(Some(value.to_le_bytes().to_vec()));
    let deletes_of_day_18263 = delete_manifest_entry(1, 1, Some(vec![integer(18_263)]));
    let deletes_of_day_18262 = delete_manifest_entry(1, 3, Some(vec![integer(18_262)]));
    let manifests = vec![
        manifest_entry(3, 0, None),
        deletes_of_day_18263,
        deletes_of_day_18262,
    ];
    let list = list_manifests(&metadata, manifests);

    let selected = list.select_delete_manifests(&before_day_18263()).unwrap();

    let paths = |manifests: &[&ManifestFile]| -> Vec<String> {
        manifests
            .iter()
            .map(|manifest| manifest.manifest_path.clone())
            .collect()
    };
    assert_eq!(paths(&selected), ["manifest-3.avro"]);
    assert_eq!(list.data_manifests().count(), 1);
    assert_eq!(list.delete_manifests().count(), 2);
    assert_eq!(list.row_count(), Some(3));
}

#[test]
fn a_delete_manifest_of_a_spec_the_table_does_not_list_is_kept() {
    let metadata = evolved_table();
    let day = |value: i32| summary(Some(value.to_le_bytes().to_vec()));
    let manifests = vec![
        delete_manifest_entry(1, 0, Some(vec![day(18_264)])),
        delete_manifest_entry(99, 1, Some(vec![day(18_264)])),
    ];
    let list = list_manifests(&metadata, manifests);

    let selected = list.select_delete_manifests(&before_day_18263()).unwrap();

    let paths: Vec<_> = selected
        .iter()
        .map(|manifest| &manifest.manifest_path)
        .collect();
    assert_eq!(paths, ["manifest-1.avro"]);
}

#[test]
fn only_live_non_empty_files_are_selected() {
    let (fixture, created) = create_table(&[]);
    let write =
        |name: &str, value: i64| fixture.write_data_file(entry(name, 10, value, value), value);
    let appended = append(
        &created,
        vec![
            entry("deleted", 10, 1, 1),
            write("existing", 1),
            write("other", 2),
        ],
    );
    let deleted = live_entry(&appended, "deleted.parquet");

    let replaced = appended
        .commit(&[deleted], vec![write("added", 1), entry("empty", 0, 1, 1)])
        .unwrap()
        .expect("no other writer");
    let unfiltered = selected(&replaced, &[]);
    let ones = selected(&replaced, &v_equals(1));

    // The replace lists `deleted` as deleted and the files it carried as
    // existing, in the manifest it rewrote, and its own files as added.
    assert_eq!(
        unfiltered,
        ["existing.parquet", "other.parquet", "added.parquet"]
    );
    assert_eq!(ones, ["existing.parquet", "added.parquet"]);
}

/// A schema of the `long` column `other` and the column `id` of
/// `id_type`.
fn schema_with_id(id_type: PrimitiveType) -> Schema {
    Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::optional(
                1,
                "other",
                Type::Primitive(PrimitiveType::Long),
            )),
            Arc::new(NestedField::optional(2, "id", Type::Primitive(id_type))),
        ])
        .build()
        .unwrap()
}

/// An entry for `name`, a file of ten rows whose `column` is `value`, the
/// single value of the array.
fn entry_holding(name: &str, column: &str, value: ArrayRef) -> DataFileEntry {
    DataFileEntry {
        file: FileRef {
            path: ObjectPath::new(format!("{name}.parquet")),
            size: 10,
        },
        partition: None,
        partition_spec_id: None,
        stats: Some(Arc::new(FileStats {
            num_records: Some(10),
            min_values: HashMap::from([(column.to_string(), value.clone())]),
            max_values: HashMap::from([(column.to_string(), value)]),
            null_counts: HashMap::from([(column.to_string(), 0)]),
        })),
        sort_order_id: None,
        first_row_id: None,
        manifest_path: None,
    }
}

/// The comparison of the column `id`, the table's second, with `value`.
fn id_compared(compare_type: CompareType, value: i64) -> Vec<Expression> {
    vec![Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx: 1,
            return_type: PivotType::Int64,
            name: None,
        })),
        right: Box::new(Expression::Constant(int_scalar(value))),
        compare_type,
        return_type: PivotType::Boolean,
    })]
}

#[test]
fn files_of_several_schemas_are_each_read_as_their_own_schema_recorded_them() {
    let (fixture, created) = create_table_of(&[column("other", PivotType::Int64)], &[], &[]);
    let one: ArrayRef = Arc::new(Int64Array::from(vec![1]));
    append(
        &created,
        vec![entry_holding("without-the-column", "other", one)],
    );
    let with_narrow_id = publish_changed(&fixture, |builder| {
        let narrow = schema_with_id(PrimitiveType::Int);
        builder
            .add_schema(narrow)
            .unwrap()
            .set_current_schema(-1)
            .unwrap()
    });
    let narrow = |name: &str, id: i32| {
        let entry = entry_holding(name, "id", Arc::new(Int32Array::from(vec![id])));
        fixture.write_data_file(entry, i64::from(id))
    };
    append(
        &with_narrow_id,
        vec![narrow("narrow-three", 3), narrow("narrow-nine", 9)],
    );
    let with_wide_id = publish_changed(&fixture, |builder| {
        let wide = schema_with_id(PrimitiveType::Long);
        builder
            .add_schema(wide)
            .unwrap()
            .set_current_schema(-1)
            .unwrap()
    });
    let wide = |name: &str, id: i64| {
        let entry = entry_holding(name, "id", Arc::new(Int64Array::from(vec![id])));
        fixture.write_data_file(entry, id)
    };
    let written = append(
        &with_wide_id,
        vec![wide("wide-three", 3), wide("wide-nine", 9)],
    );

    let above_four = selected(&written, &id_compared(CompareType::Greater, 4));
    let below_four = selected(&written, &id_compared(CompareType::Less, 4));

    assert_eq!(above_four, ["narrow-nine.parquet", "wide-nine.parquet"]);
    assert_eq!(below_four, ["narrow-three.parquet", "wide-three.parquet"]);
}

#[test]
fn a_file_recorded_before_its_sort_column_was_widened_stands_by_value_among_the_files_since() {
    let columns = [column("id", PivotType::Int32)];
    let (fixture, created) = create_table_of(&columns, &[], &["id".to_string()]);
    let twenty: ArrayRef = Arc::new(Int32Array::from(vec![20]));
    append(&created, vec![entry_holding("narrow-twenty", "id", twenty)]);
    let widened = publish_changed(&fixture, |builder| {
        let wide_id = NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long));
        let wide = Schema::builder().with_fields(vec![Arc::new(wide_id)]);
        builder
            .add_schema(wide.build().unwrap())
            .unwrap()
            .set_current_schema(-1)
            .unwrap()
    });
    let wide =
        |name: &str, id: i64| entry_holding(name, "id", Arc::new(Int64Array::from(vec![id])));

    let table = append(
        &widened,
        vec![wide("wide-thirty", 30), wide("wide-ten", 10)],
    );

    let by_value = [
        "wide-ten.parquet",
        "narrow-twenty.parquet",
        "wide-thirty.parquet",
    ];
    assert_eq!(paths_in_order(&table), by_value);
}
