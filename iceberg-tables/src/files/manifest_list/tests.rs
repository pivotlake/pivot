use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::ops::RangeInclusive;
use std::sync::Arc;

use super::*;
use crate::files::manifest_parts::GroupedManifest;
use crate::files::pruning::ComparedColumn;
use crate::metadata::layout::TableLocation;
use crate::table::FileSelector;
use crate::table::manifest_reader::PartitionSlice;
use arrow_array::{
    ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array, Int32Array,
    Int64Array, Scalar, StringViewArray, TimestampMicrosecondArray,
};
use iceberg::expr::BoundPredicate;
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FieldSummary, FormatVersion,
    Literal, Manifest, ManifestContentType, ManifestEntry, ManifestMetadata, ManifestStatus,
    NestedField, PartitionSpec, PrimitiveType, Schema, SchemaRef, Struct,
    Transform as IcebergTransform, Type,
};
use planner::catalog::SchemaQualifiedTableName;
use planner::expression::{
    Between, Compare, CompareType, Conjunction, ConjunctionOp, Expression, InList, IsNull, Ref,
};
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

/// A one-row data file named `path` at the root of its store, whose metrics
/// describe field 1.
fn file(path: &str) -> DataFileBuilder {
    let mut builder = DataFileBuilder::default();
    builder
        .content(DataContentType::Data)
        .file_path(format!("/{path}"))
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

/// The metadata of a table whose current schema is the first of `schemas`,
/// each under its position as its id, and that lists `partition_specs`.
fn table_metadata(schemas: &[SchemaRef], partition_specs: &[PartitionSpec]) -> TableMetadata {
    let schemas: Vec<Schema> = schemas
        .iter()
        .enumerate()
        .map(|(schema_id, schema)| {
            Schema::clone(schema)
                .into_builder()
                .with_schema_id(schema_id as i32)
                .build()
                .unwrap()
        })
        .collect();
    serde_json::from_value(serde_json::json!({
        "format-version": 2,
        "table-uuid": "9c12d441-03fe-4693-9a96-a0705ddf69c1",
        "location": "file:///table",
        "last-sequence-number": 0,
        "last-updated-ms": 0,
        "last-column-id": 2,
        "schemas": schemas,
        "current-schema-id": 0,
        "partition-specs": partition_specs,
        "default-spec-id": partition_specs[0].spec_id(),
        "last-partition-id": 1000,
        "sort-orders": [{"order-id": 0, "fields": []}],
        "default-sort-order-id": 0,
    }))
    .unwrap()
}

/// How `bound`, recorded for a column whose type is now `column_type`,
/// orders against `constant`, a value of that type, by the spec crate's
/// comparison of the two; `None` for a bound of a type whose bounds rule no
/// file out.
fn compare_bound(bound: &Datum, column_type: &PrimitiveType, constant: &Datum) -> Option<Ordering> {
    let is_ordered = match (bound.data_type(), column_type) {
        (PrimitiveType::Int, PrimitiveType::Long)
        | (PrimitiveType::Float, PrimitiveType::Double) => true,
        (
            PrimitiveType::Decimal {
                scale: bound_scale, ..
            },
            PrimitiveType::Decimal { scale, .. },
        ) => bound_scale == scale,
        (bound_type, column_type) => {
            bound_type == column_type
                && matches!(
                    bound_type,
                    PrimitiveType::Boolean
                        | PrimitiveType::Int
                        | PrimitiveType::Date
                        | PrimitiveType::Long
                        | PrimitiveType::Timestamp
                        | PrimitiveType::Timestamptz
                        | PrimitiveType::Float
                        | PrimitiveType::Double
                        | PrimitiveType::String
                )
        }
    };
    if !is_ordered {
        return None;
    }
    pruning::literal_to_datum(bound.literal(), column_type)?.partial_cmp(constant)
}

/// What a file's manifest entry says of one column the filters compare,
/// read off the entry itself: what the parts a manifest is cut into are
/// checked against.
struct EntryRange<'a> {
    /// The column in the table's current schema.
    column: &'a NestedField,
    /// The schema the file's manifest was written under; `None` for one
    /// the table does not list.
    file_schema: Option<&'a Schema>,
    data: &'a DataFile,
}

impl ValueRange for EntryRange<'_> {
    fn field_id(&self) -> i32 {
        self.column.id
    }

    fn may_match_comparison(&self, operator: PredicateOperator, constant: &Datum) -> bool {
        let field_id = self.column.id;
        let stored_field = match self.file_schema {
            // A file of a schema the table does not list is taken to have
            // the column, with bounds of a type that is not known.
            None => None,
            Some(file_schema) => match file_schema.field_by_id(field_id) {
                None => return false,
                Some(stored_field) => Some(stored_field),
            },
        };
        if self.data.null_value_counts().get(&field_id) == Some(&self.data.record_count()) {
            return false;
        }
        let column_type = self.column.field_type.as_primitive_type().unwrap();
        let is_bounded = stored_field.is_some_and(|field| field.field_type.is_primitive());
        let compare = |bound: Option<&Datum>| {
            is_bounded
                .then(|| compare_bound(bound?, column_type, constant))
                .flatten()
        };
        pruning::may_bounds_match_comparison(
            operator,
            || compare(self.data.lower_bounds().get(&field_id)),
            || compare(self.data.upper_bounds().get(&field_id)),
        )
    }
}

/// The manifests of a table as the table reads them, with what reading
/// them takes.
struct ListedManifests {
    table_metadata: TableMetadata,
    location: TableLocation,
    /// Each manifest's entry in the manifest list, with the manifest under
    /// its schema's id, its entries numbered by the manifest's place.
    manifests: Vec<(ManifestFile, Manifest)>,
    _directory: tempfile::TempDir,
}

impl ListedManifests {
    /// `manifests`, in order, as the data manifests of a table of
    /// `current` whose partition specs are the ones they were written
    /// under.
    fn new(current: &SchemaRef, manifests: &[Manifest]) -> Self {
        let mut schemas = vec![current.clone()];
        let mut partition_specs: Vec<PartitionSpec> = Vec::new();
        for manifest in manifests {
            let metadata = manifest.metadata();
            if !schemas.contains(&metadata.schema) {
                schemas.push(metadata.schema.clone());
            }
            if !partition_specs.contains(&metadata.partition_spec) {
                partition_specs.push(metadata.partition_spec.clone());
            }
        }
        let directory = tempfile::tempdir().unwrap();
        let store: Arc<dyn object_storage::ObjectStore> =
            Arc::new(object_storage::LocalStore::new(directory.path()).unwrap());
        let location = TableLocation::in_datastore_store(
            &SchemaQualifiedTableName::new("main", "t"),
            store,
            object_storage::ObjectPath::new("table"),
        )
        .unwrap();
        let manifests = manifests
            .iter()
            .enumerate()
            .map(|(position, manifest)| {
                let file =
                    manifest_entry(manifest.metadata().partition_spec.spec_id(), position, None);
                let schema_id = schemas
                    .iter()
                    .position(|schema| schema == &manifest.metadata().schema)
                    .unwrap();
                let entries = manifest
                    .entries()
                    .iter()
                    .map(|entry| ManifestEntry {
                        sequence_number: Some(position as i64),
                        ..ManifestEntry::clone(entry)
                    })
                    .collect();
                let manifest = Manifest::new(
                    ManifestMetadata {
                        schema_id: schema_id as i32,
                        ..manifest.metadata().clone()
                    },
                    entries,
                );
                (file, manifest)
            })
            .collect();
        Self {
            table_metadata: table_metadata(&schemas, &partition_specs),
            location,
            manifests,
            _directory: directory,
        }
    }

    /// The names of the files `selector` keeps, by partition, then by
    /// manifest and then by name, each file checked by what its manifest
    /// entry says of it against `row_filters`, the filters of `selector`
    /// bound to the table's current schema.
    fn select_through_entries(
        &self,
        selector: &FileSelector,
        row_filters: &[BoundPredicate],
    ) -> iceberg::Result<Vec<String>> {
        let compared = pruning::list_compared_fields(row_filters);
        // The files the filters keep of each partition, each with its
        // manifest's place among the table's.
        type KeptFiles = (Partition, Vec<(usize, String)>);
        let mut partitions: BTreeMap<Arc<Partition>, KeptFiles> = BTreeMap::new();
        for (manifest_position, (_, manifest)) in self.manifests.iter().enumerate() {
            let spec_id = manifest.metadata().partition_spec().spec_id();
            let file_schema = self
                .table_metadata
                .schema_by_id(manifest.metadata().schema_id());
            for entry in manifest.entries().iter().filter(|entry| entry.is_alive()) {
                let data = entry.data_file();
                let (_, kept) = partitions
                    .entry(Arc::new(Partition::new(spec_id, data.partition().clone())))
                    .or_insert_with(|| {
                        let partition = Partition::new(spec_id, data.partition().clone());
                        (partition, Vec::new())
                    });
                let ranges: Vec<EntryRange<'_>> = compared
                    .iter()
                    .map(|column| EntryRange {
                        column,
                        file_schema: file_schema.map(|schema| &**schema),
                        data,
                    })
                    .collect();
                let may_match = row_filters
                    .iter()
                    .all(|row_filter| pruning::may_match_predicate(row_filter, &ranges));
                if may_match {
                    let path = self.location.find_path_for_uri(data.file_path()).unwrap();
                    kept.push((manifest_position, path.as_str().to_string()));
                }
            }
        }

        let mut tuple_values = Vec::new();
        let mut selected = Vec::new();
        for (partition, kept) in partitions.values_mut() {
            if !selector.may_partition_match(partition, &mut tuple_values)? {
                continue;
            }
            kept.sort_unstable();
            selected.extend(kept.iter().map(|(_, path)| name_file(path)));
        }
        Ok(selected)
    }

    /// The names of the files `selector` keeps, by partition, then by
    /// manifest and then by name, each manifest cut into the parts a scan
    /// selects its files through.
    fn select_through_parts(&self, selector: &FileSelector) -> iceberg::Result<Vec<String>> {
        // The files of each partition in each manifest, by the manifest's
        // place among the table's, with what the manifest records of the
        // columns the filters compare.
        type PartitionSlices = (Arc<Partition>, Vec<(usize, PartitionSlice)>);
        let mut partitions: BTreeMap<Arc<Partition>, PartitionSlices> = BTreeMap::new();
        for (manifest_position, (file, manifest)) in self.manifests.iter().enumerate() {
            let manifest = Manifest::new(manifest.metadata().clone(), entries_of(manifest));
            let grouped = GroupedManifest::group("table", manifest).unwrap();
            for partition in grouped.partitions() {
                let files = partition
                    .build_files(file, &self.table_metadata, &self.location)
                    .unwrap();
                let mut compared: Vec<i32> = selector
                    .compared_field_ids()
                    .filter(|field_id| files.find_stats_field(*field_id).is_some())
                    .collect();
                compared.sort_unstable();
                let bounds = partition.build_columns(&compared);
                let slice = PartitionSlice::new(
                    Arc::clone(partition.partition()),
                    Arc::from(file.manifest_path.as_str()),
                    Arc::new(files),
                    Arc::from(compared),
                    bounds
                        .into_iter()
                        .map(|bounds| Some(Arc::new(bounds)))
                        .collect(),
                );
                partitions
                    .entry(Arc::clone(partition.partition()))
                    .or_insert_with(|| (Arc::clone(partition.partition()), Vec::new()))
                    .1
                    .push((manifest_position, slice));
            }
        }

        let mut tuple_values = Vec::new();
        let mut selected = Vec::new();
        for (partition, slices) in partitions.values() {
            if !selector.may_partition_match(partition, &mut tuple_values)? {
                continue;
            }
            let mut selected_of_partition = Vec::new();
            for (manifest_position, slice) in slices {
                let is_selected = selector.select_files(slice);
                for position in is_selected.set_indices() {
                    selected_of_partition.push((*manifest_position, slice.files.path(position)));
                }
            }
            selected_of_partition.sort_unstable();
            selected.extend(
                selected_of_partition
                    .into_iter()
                    .map(|(_, path)| name_file(path)),
            );
        }
        Ok(selected)
    }
}

/// The entries of `manifest`, each its own.
fn entries_of(manifest: &Manifest) -> Vec<ManifestEntry> {
    manifest
        .entries()
        .iter()
        .map(|entry| ManifestEntry::clone(entry))
        .collect()
}

/// The name a test gave the file a table names `path`.
fn name_file(path: &str) -> String {
    path.trim_start_matches('/').to_string()
}

/// The names of the files of `manifests` that `filters` keep, by partition
/// and then in the order a partition's files stand in (by manifest, then by
/// name): those of a table of `current` whose data manifests they are, in
/// order, and whose partition specs are the ones the manifests were written
/// under. The files are selected through the parts each manifest is cut
/// into, which must keep exactly the files that checking every manifest
/// entry on its own keeps.
fn select_files(
    current: &SchemaRef,
    manifests: &[Manifest],
    filters: &[Expression],
) -> iceberg::Result<Vec<String>> {
    let listed = ListedManifests::new(current, manifests);
    let selector = FileSelector::new(&listed.table_metadata, filters)?;
    let through_parts = listed.select_through_parts(&selector)?;
    let row_filters = pruning::bind_filters(listed.table_metadata.current_schema(), filters);
    let through_entries = listed.select_through_entries(&selector, &row_filters)?;
    assert_eq!(through_parts, through_entries);
    Ok(through_parts)
}

/// The names of the files of `manifests` that `filters` keep.
fn selected(current: &SchemaRef, manifests: &[Manifest], filters: &[Expression]) -> Vec<String> {
    select_files(current, manifests, filters).unwrap()
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

fn id_column() -> Box<Expression> {
    Box::new(Expression::Ref(Ref {
        column_idx: 0,
        return_type: PivotType::Int64,
        name: None,
    }))
}

fn long_constant(value: Option<i64>) -> Expression {
    Expression::Constant(Scalar::new(
        Arc::new(Int64Array::from(vec![value])) as ArrayRef
    ))
}

fn equals(value: i64) -> Expression {
    long_predicate(CompareType::Equal, value).remove(0)
}

/// A table of one BIGINT column `id`, and a manifest of three files holding
/// 10, 20 and 30.
fn ten_twenty_thirty() -> (SchemaRef, [Manifest; 1]) {
    let schema = schema(1, "id", PrimitiveType::Long);
    let files = vec![
        file_holding("ten", Datum::long(10)),
        file_holding("twenty", Datum::long(20)),
        file_holding("thirty", Datum::long(30)),
    ];
    let manifests = [manifest(&schema, unpartitioned(&schema), files)];
    (schema, manifests)
}

#[test]
fn an_or_keeps_the_files_either_side_may_match() {
    let (schema, manifests) = ten_twenty_thirty();
    let ten_or_thirty = Expression::Conjunction(Conjunction {
        op: ConjunctionOp::Or,
        children: vec![equals(10), equals(30)],
    });

    let kept = selected(&schema, &manifests, &[ten_or_thirty]);

    assert_eq!(kept, ["ten", "thirty"]);
}

#[test]
fn an_or_with_a_side_metadata_cannot_answer_keeps_every_file() {
    let (schema, manifests) = ten_twenty_thirty();
    let is_null = Expression::IsNull(IsNull {
        negated: false,
        input: id_column(),
    });
    let ten_or_null = Expression::Conjunction(Conjunction {
        op: ConjunctionOp::Or,
        children: vec![equals(10), is_null],
    });

    let kept = selected(&schema, &manifests, &[ten_or_null]);

    assert_eq!(kept, ["ten", "thirty", "twenty"]);
}

#[test]
fn an_in_list_keeps_the_files_that_may_hold_one_of_its_values() {
    let (schema, manifests) = ten_twenty_thirty();
    let in_ten_or_thirty = Expression::InList(InList {
        input: id_column(),
        values: vec![long_constant(Some(10)), long_constant(Some(30))],
    });

    let kept = selected(&schema, &manifests, &[in_ten_or_thirty]);

    assert_eq!(kept, ["ten", "thirty"]);
}

#[test]
fn a_between_keeps_the_files_that_overlap_its_range() {
    let (schema, manifests) = ten_twenty_thirty();
    let from_fifteen_to_thirty = Expression::Between(Between {
        input: id_column(),
        lower: Box::new(long_constant(Some(15))),
        upper: Box::new(long_constant(Some(30))),
        lower_inclusive: true,
        upper_inclusive: false,
    });

    let kept = selected(&schema, &manifests, &[from_fifteen_to_thirty]);

    assert_eq!(kept, ["twenty"]);
}

#[test]
fn every_filter_has_to_keep_a_file() {
    let (schema, manifests) = ten_twenty_thirty();
    let above_ten = long_predicate(CompareType::Greater, 10).remove(0);
    let below_thirty = long_predicate(CompareType::Less, 30).remove(0);

    let kept = selected(&schema, &manifests, &[above_ten, below_thirty]);

    assert_eq!(kept, ["twenty"]);
}

#[test]
fn a_comparison_with_null_matches_no_file() {
    let (schema, manifests) = ten_twenty_thirty();
    let equals_null = Expression::Compare(Compare {
        left: id_column(),
        right: Box::new(long_constant(None)),
        compare_type: CompareType::Equal,
        return_type: PivotType::Boolean,
    });

    let kept = selected(&schema, &manifests, &[equals_null]);

    assert!(kept.is_empty());
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
fn the_row_count_is_known_only_when_every_manifest_counts_its_rows() {
    let metadata = evolved_table();
    let counted = vec![manifest_entry(3, 0, None), manifest_entry(3, 1, None)];
    let mut uncounted = counted.clone();
    uncounted[0].added_rows_count = None;
    let mut overflowing = counted.clone();
    overflowing[0].added_rows_count = Some(u64::MAX);

    let row_count = |manifests: &[ManifestFile]| {
        ManifestList::new(&metadata, manifests.to_vec())
            .unwrap()
            .count_rows()
    };

    assert_eq!(row_count(&counted), Some(7));
    assert_eq!(row_count(&uncounted), None);
    assert_eq!(row_count(&overflowing), None);
    assert_eq!(row_count(&[]), Some(0));
}

#[test]
fn the_estimated_row_count_discounts_the_delete_manifests_row_totals() {
    let metadata = evolved_table();
    let mut deletes = manifest_entry(1, 1, None);
    deletes.content = ManifestContentType::Deletes;
    let mut too_many_deletes = deletes.clone();
    too_many_deletes.added_rows_count = Some(100);
    let data = manifest_entry(1, 6, None);

    let estimate = |manifests: Vec<ManifestFile>| {
        ManifestList::new(&metadata, manifests)
            .unwrap()
            .estimate_row_count()
    };

    assert_eq!(estimate(vec![data.clone(), deletes]), Some(5));
    assert_eq!(estimate(vec![data.clone(), too_many_deletes]), Some(0));
    assert_eq!(estimate(vec![data]), Some(9));
}

#[test]
fn a_partition_requirement_iceberg_cannot_express_keeps_the_file() {
    let price = PrimitiveType::Decimal {
        precision: 10,
        scale: 2,
    };
    let schema = schema(1, "price", price);
    let truncated = Datum::decimal_from_str("12.30").unwrap();
    let priced = file("priced")
        .partition(Struct::from_iter([Some(Literal::from(truncated))]))
        .build()
        .unwrap();
    let spec = partitioned_by(&schema, 0, IcebergTransform::Truncate(10));
    let manifests = [manifest(&schema, spec, vec![priced])];
    let below_fifty = Decimal128Array::from(vec![5000])
        .with_precision_and_scale(10, 2)
        .unwrap();

    let kept = selected(
        &schema,
        &manifests,
        &predicate(CompareType::Less, Arc::new(below_fifty)),
    );

    assert_eq!(kept, ["priced"]);
}

/// A file named "bounded" whose field 1 lies between `lower` and `upper`,
/// without NULLs.
fn file_between(lower: Datum, upper: Datum) -> DataFile {
    file("bounded")
        .null_value_counts(HashMap::from([(1, 0)]))
        .lower_bounds(HashMap::from([(1, lower)]))
        .upper_bounds(HashMap::from([(1, upper)]))
        .build()
        .unwrap()
}

#[test]
fn a_file_is_kept_exactly_when_a_value_within_its_bounds_would_match() {
    type Matches = fn(i64, i64) -> bool;
    let comparisons: [(CompareType, Matches); 6] = [
        (CompareType::Equal, |value, constant| value == constant),
        (CompareType::NotEqual, |value, constant| value != constant),
        (CompareType::Less, |value, constant| value < constant),
        (CompareType::LessEqual, |value, constant| value <= constant),
        (CompareType::Greater, |value, constant| value > constant),
        (CompareType::GreaterEqual, |value, constant| {
            value >= constant
        }),
    ];
    let schema = schema(1, "id", PrimitiveType::Long);

    for (lower, upper) in [(10, 20), (15, 15)] {
        let bounded = file_between(Datum::long(lower), Datum::long(upper));
        let manifests = [manifest(&schema, unpartitioned(&schema), vec![bounded])];
        for (compare_type, matches) in comparisons {
            for constant in lower - 1..=upper + 1 {
                let kept = selected(&schema, &manifests, &long_predicate(compare_type, constant));

                let may_match = (lower..=upper).any(|value| matches(value, constant));
                assert_eq!(
                    !kept.is_empty(),
                    may_match,
                    "[{lower}, {upper}] {compare_type:?} {constant}"
                );
            }
        }
    }
}

#[test]
fn a_bound_only_some_files_record_rules_out_none_of_the_others() {
    let schema = schema(1, "id", PrimitiveType::Long);
    let from_zero = file("from zero")
        .lower_bounds(HashMap::from([(1, Datum::long(0))]))
        .build()
        .unwrap();
    let up_to_zero = file("up to zero")
        .upper_bounds(HashMap::from([(1, Datum::long(0))]))
        .build()
        .unwrap();
    let files = vec![from_zero, up_to_zero];
    let manifests = [manifest(&schema, unpartitioned(&schema), files)];
    let select = |compare_type| {
        [-5, 0, 5].map(|constant| {
            selected(&schema, &manifests, &long_predicate(compare_type, constant)).join(", ")
        })
    };

    let equal = select(CompareType::Equal);
    let not_equal = select(CompareType::NotEqual);
    let less = select(CompareType::Less);
    let less_or_equal = select(CompareType::LessEqual);
    let greater = select(CompareType::Greater);
    let greater_or_equal = select(CompareType::GreaterEqual);

    let both = "from zero, up to zero";
    assert_eq!(equal, ["up to zero", both, "from zero"]);
    assert_eq!(not_equal, [both, both, both]);
    assert_eq!(less, ["up to zero", "up to zero", both]);
    assert_eq!(less_or_equal, ["up to zero", both, both]);
    assert_eq!(greater, [both, "from zero", "from zero"]);
    assert_eq!(greater_or_equal, [both, both, "from zero"]);
}

#[test]
fn bounds_of_every_column_type_compare_with_constants_of_that_type() {
    let decimal = PrimitiveType::Decimal {
        precision: 10,
        scale: 2,
    };
    let utc = |micros: i64| TimestampMicrosecondArray::from(vec![micros]).with_timezone("UTC");
    let decimal_constant = |unscaled: i128| {
        Decimal128Array::from(vec![unscaled])
            .with_precision_and_scale(10, 2)
            .unwrap()
    };
    // Each column holds values from `lower` to `upper`; `inside` is one of
    // them and `outside` is not.
    let cases: Vec<(PrimitiveType, Datum, Datum, ArrayRef, ArrayRef)> = vec![
        (
            PrimitiveType::Boolean,
            Datum::bool(false),
            Datum::bool(false),
            Arc::new(BooleanArray::from(vec![false])),
            Arc::new(BooleanArray::from(vec![true])),
        ),
        (
            PrimitiveType::Int,
            Datum::int(10),
            Datum::int(20),
            Arc::new(Int32Array::from(vec![15])),
            Arc::new(Int32Array::from(vec![25])),
        ),
        (
            PrimitiveType::Float,
            Datum::float(1.0_f32),
            Datum::float(2.0_f32),
            Arc::new(Float32Array::from(vec![1.5])),
            Arc::new(Float32Array::from(vec![2.5])),
        ),
        (
            PrimitiveType::Double,
            Datum::double(1.0),
            Datum::double(2.0),
            Arc::new(Float64Array::from(vec![1.5])),
            Arc::new(Float64Array::from(vec![2.5])),
        ),
        (
            PrimitiveType::Date,
            Datum::date(10),
            Datum::date(20),
            Arc::new(Date32Array::from(vec![15])),
            Arc::new(Date32Array::from(vec![25])),
        ),
        (
            PrimitiveType::Timestamp,
            Datum::timestamp_micros(10),
            Datum::timestamp_micros(20),
            Arc::new(TimestampMicrosecondArray::from(vec![15])),
            Arc::new(TimestampMicrosecondArray::from(vec![25])),
        ),
        (
            PrimitiveType::Timestamptz,
            Datum::timestamptz_micros(10),
            Datum::timestamptz_micros(20),
            Arc::new(utc(15)),
            Arc::new(utc(25)),
        ),
        (
            decimal,
            Datum::decimal_from_str("1.00").unwrap(),
            Datum::decimal_from_str("2.00").unwrap(),
            Arc::new(decimal_constant(150)),
            Arc::new(decimal_constant(250)),
        ),
        (
            PrimitiveType::String,
            Datum::string("b"),
            Datum::string("d"),
            Arc::new(StringViewArray::from(vec!["c"])),
            Arc::new(StringViewArray::from(vec!["e"])),
        ),
    ];

    for (column_type, lower, upper, inside, outside) in cases {
        let schema = schema(1, "value", column_type.clone());
        let manifests = [manifest(
            &schema,
            unpartitioned(&schema),
            vec![file_between(lower, upper)],
        )];

        let inside = selected(&schema, &manifests, &predicate(CompareType::Equal, inside));
        let outside = selected(&schema, &manifests, &predicate(CompareType::Equal, outside));

        assert_eq!(inside, ["bounded"], "{column_type}");
        assert!(outside.is_empty(), "{column_type}");
    }
}

#[test]
fn an_in_list_longer_than_the_limit_keeps_every_file() {
    let (schema, manifests) = ten_twenty_thirty();
    let in_many_other_values = Expression::InList(InList {
        input: id_column(),
        values: (100..301).map(|value| long_constant(Some(value))).collect(),
    });

    let kept = selected(&schema, &manifests, &[in_many_other_values]);

    assert_eq!(kept, ["ten", "thirty", "twenty"]);
}

#[test]
fn an_in_list_within_the_limit_prunes_by_each_of_its_values() {
    let (schema, manifests) = ten_twenty_thirty();
    // The odd numbers from 15 to 413, and 30.
    let mut values: Vec<_> = (0..199)
        .map(|step| long_constant(Some(15 + 2 * step)))
        .collect();
    values.push(long_constant(Some(30)));
    let in_many_odd_values_and_thirty = Expression::InList(InList {
        input: id_column(),
        values,
    });

    let kept = selected(&schema, &manifests, &[in_many_odd_values_and_thirty]);

    assert_eq!(kept, ["thirty"]);
}

#[test]
fn a_constant_on_the_left_keeps_every_file() {
    let (schema, manifests) = ten_twenty_thirty();
    let hundred_equals_id = Expression::Compare(Compare {
        left: Box::new(long_constant(Some(100))),
        right: id_column(),
        compare_type: CompareType::Equal,
        return_type: PivotType::Boolean,
    });

    let kept = selected(&schema, &manifests, &[hundred_equals_id]);

    assert_eq!(kept, ["ten", "thirty", "twenty"]);
}

#[test]
fn a_filter_the_column_type_cannot_read_leaves_the_others_to_prune() {
    let (schema, manifests) = ten_twenty_thirty();
    let equals_text = predicate(
        CompareType::Equal,
        Arc::new(StringViewArray::from(vec!["ten"])),
    )
    .remove(0);
    let above_fifteen = long_predicate(CompareType::Greater, 15).remove(0);

    let kept = selected(&schema, &manifests, &[equals_text, above_fifteen]);

    assert_eq!(kept, ["thirty", "twenty"]);
}

#[test]
fn a_constant_beyond_the_column_type_matches_every_value_or_none() {
    let schema = schema(1, "id", PrimitiveType::Int);
    let files = vec![file_holding("five", Datum::int(5))];
    let manifests = [manifest(&schema, unpartitioned(&schema), files)];
    let beyond_int = i64::from(i32::MAX) + 1;

    let above = selected(
        &schema,
        &manifests,
        &long_predicate(CompareType::Greater, beyond_int),
    );
    let below = selected(
        &schema,
        &manifests,
        &long_predicate(CompareType::Less, beyond_int),
    );

    assert!(above.is_empty());
    assert_eq!(below, ["five"]);
}

#[test]
fn bounds_recorded_before_a_promotion_compare_with_a_constant_their_type_cannot_hold() {
    let current = schema(1, "id", PrimitiveType::Long);
    let stored = schema(1, "id", PrimitiveType::Int);
    let files = vec![
        file_holding("five", Datum::int(5)),
        file("unknown").build().unwrap(),
    ];
    let manifests = [manifest(&stored, unpartitioned(&stored), files)];
    let select = |compare_type| {
        [i64::from(i32::MIN) - 1, i64::from(i32::MAX) + 1].map(|constant| {
            selected(
                &current,
                &manifests,
                &long_predicate(compare_type, constant),
            )
            .join(", ")
        })
    };

    let equal = select(CompareType::Equal);
    let not_equal = select(CompareType::NotEqual);
    let less = select(CompareType::Less);
    let less_or_equal = select(CompareType::LessEqual);
    let greater = select(CompareType::Greater);
    let greater_or_equal = select(CompareType::GreaterEqual);

    let both = "five, unknown";
    assert_eq!(equal, ["unknown", "unknown"]);
    assert_eq!(not_equal, [both, both]);
    assert_eq!(less, ["unknown", both]);
    assert_eq!(less_or_equal, ["unknown", both]);
    assert_eq!(greater, [both, "unknown"]);
    assert_eq!(greater_or_equal, [both, "unknown"]);
}

#[test]
fn a_void_partition_field_prunes_nothing() {
    let schema = schema(1, "id", PrimitiveType::Long);
    let voided = file("voided")
        .partition(Struct::from_iter([None::<Literal>]))
        .null_value_counts(HashMap::from([(1, 0)]))
        .lower_bounds(HashMap::from([(1, Datum::long(5))]))
        .upper_bounds(HashMap::from([(1, Datum::long(5))]))
        .build()
        .unwrap();
    let spec = partitioned_by(&schema, 0, IcebergTransform::Void);
    let manifests = [manifest(&schema, spec, vec![voided])];

    let five = selected(&schema, &manifests, &long_predicate(CompareType::Equal, 5));
    let six = selected(&schema, &manifests, &long_predicate(CompareType::Equal, 6));

    assert_eq!(five, ["voided"]);
    assert!(six.is_empty());
}

#[test]
fn a_date_before_the_epoch_also_keeps_the_month_after_its_own() {
    let schema = schema(1, "day", PrimitiveType::Date);
    let in_month = |path: &str, months_since_epoch: i32| {
        file(path)
            .partition(Struct::from_iter([Some(Literal::int(months_since_epoch))]))
            .build()
            .unwrap()
    };
    let files = vec![
        in_month("october 1969", -3),
        in_month("november 1969", -2),
        in_month("december 1969", -1),
        in_month("january 1970", 0),
        in_month("february 1970", 1),
    ];
    let spec = partitioned_by(&schema, 0, IcebergTransform::Month);
    let manifests = [manifest(&schema, spec, files)];
    // 1969-12-15 is 17 days before the epoch. Iceberg projects a date before
    // the epoch onto its own month and the next: some writers put it there.
    let mid_december = predicate(CompareType::Equal, Arc::new(Date32Array::from(vec![-17])));

    let kept = selected(&schema, &manifests, &mid_december);

    assert_eq!(kept, ["december 1969", "january 1970"]);
}

#[test]
fn a_summary_that_does_not_match_its_spec_is_an_error() {
    let metadata = evolved_table();
    let region_only = vec![summary(Some(b"west".to_vec()))];
    let manifests = vec![manifest_entry(0, 0, Some(region_only))];

    let list = ManifestList::new(&metadata, manifests);

    assert!(list.is_err());
}

#[test]
fn a_partition_tuple_that_does_not_match_its_spec_is_an_error() {
    let schema = schema(1, "id", PrimitiveType::Long);
    let without_a_value = file("short").build().unwrap();
    let spec = partitioned_by(&schema, 0, IcebergTransform::Identity);
    let manifests = [manifest(&schema, spec, vec![without_a_value])];

    let selected = select_files(&schema, &manifests, &long_predicate(CompareType::Equal, 5));

    assert!(selected.is_err());
}

#[test]
fn a_wide_or_is_checked_like_a_narrow_one() {
    let (schema, manifests) = ten_twenty_thirty();
    let mut children: Vec<_> = (1_000..11_000).map(equals).collect();
    children.push(equals(20));
    let twenty_or_many_others = Expression::Conjunction(Conjunction {
        op: ConjunctionOp::Or,
        children,
    });

    let kept = selected(&schema, &manifests, &[twenty_or_many_others]);

    assert_eq!(kept, ["twenty"]);
}

#[test]
fn a_file_is_kept_exactly_when_a_value_within_its_bounds_would_match_whatever_the_columns_type() {
    type Matches = fn(usize, usize) -> bool;
    let comparisons: [(CompareType, Matches); 6] = [
        (CompareType::Equal, |value, constant| value == constant),
        (CompareType::NotEqual, |value, constant| value != constant),
        (CompareType::Less, |value, constant| value < constant),
        (CompareType::LessEqual, |value, constant| value <= constant),
        (CompareType::Greater, |value, constant| value > constant),
        (CompareType::GreaterEqual, |value, constant| {
            value >= constant
        }),
    ];
    let decimal = PrimitiveType::Decimal {
        precision: 38,
        scale: 2,
    };
    let decimal_bound =
        |unscaled: i128| Datum::try_from_bytes(&unscaled.to_be_bytes(), decimal.clone()).unwrap();
    let decimal_constant = |unscaled: i128| -> ArrayRef {
        let constant = Decimal128Array::from(vec![unscaled]);
        Arc::new(constant.with_precision_and_scale(38, 2).unwrap())
    };
    let float_constant = |value: f32| -> ArrayRef { Arc::new(Float32Array::from(vec![value])) };
    let double_constant = |value: f64| -> ArrayRef { Arc::new(Float64Array::from(vec![value])) };
    let string_constant =
        |value: &str| -> ArrayRef { Arc::new(StringViewArray::from(vec![value])) };
    let widest = 10_i128.pow(37);
    // Each column holds values from the second to the fourth of five
    // constants in the type's order: negative zero before zero, a decimal by
    // its value however many digits it has, a string by its bytes with a
    // prefix first.
    let cases: Vec<(PrimitiveType, Datum, Datum, Vec<ArrayRef>)> = vec![
        (
            PrimitiveType::Float,
            Datum::float(-0.0_f32),
            Datum::float(1.0_f32),
            [-1.0, -0.0, 0.0, 1.0, f32::INFINITY]
                .map(float_constant)
                .to_vec(),
        ),
        (
            PrimitiveType::Double,
            Datum::double(-0.0),
            Datum::double(1.0),
            [-1.0, -0.0, 0.0, 1.0, f64::INFINITY]
                .map(double_constant)
                .to_vec(),
        ),
        (
            decimal.clone(),
            decimal_bound(-1),
            decimal_bound(widest - 1),
            [-widest, -1, 0, widest - 1, widest]
                .map(decimal_constant)
                .to_vec(),
        ),
        (
            PrimitiveType::String,
            Datum::string("ab"),
            Datum::string("ac"),
            ["a", "ab", "abz", "ac", "ad"].map(string_constant).to_vec(),
        ),
    ];

    for (column_type, lower, upper, constants) in cases {
        let schema = schema(1, "value", column_type.clone());
        let bounded = file_between(lower, upper);
        let manifests = [manifest(&schema, unpartitioned(&schema), vec![bounded])];
        for (compare_type, matches) in comparisons {
            for (constant, value) in constants.iter().enumerate() {
                let filter = predicate(compare_type, Arc::clone(value));

                let kept = selected(&schema, &manifests, &filter);

                let may_match = (1..=3).any(|held| matches(held, constant));
                assert_eq!(
                    !kept.is_empty(),
                    may_match,
                    "{column_type} {compare_type:?} constant {constant}"
                );
            }
        }
    }
}

#[test]
fn bounds_of_a_type_no_constant_compares_with_rule_no_file_out() {
    let cases = [
        (
            PrimitiveType::Uuid,
            Datum::uuid(uuid::Uuid::from_u128(1)),
            Datum::uuid(uuid::Uuid::from_u128(u128::MAX)),
        ),
        (
            PrimitiveType::Fixed(2),
            Datum::fixed([0, 1]),
            Datum::fixed([255, 0]),
        ),
        (
            PrimitiveType::Binary,
            Datum::binary([0]),
            Datum::binary([255, 0, 1]),
        ),
        (
            PrimitiveType::Time,
            Datum::time_micros(1).unwrap(),
            Datum::time_micros(9).unwrap(),
        ),
        (
            PrimitiveType::TimestampNs,
            Datum::timestamp_nanos(1),
            Datum::timestamp_nanos(9),
        ),
        (
            PrimitiveType::TimestamptzNs,
            Datum::timestamptz_nanos(1),
            Datum::timestamptz_nanos(9),
        ),
    ];

    let kept = cases.map(|(column_type, lower, upper)| {
        let schema = schema(1, "value", column_type);
        let bounded = file_between(lower, upper.clone());
        let manifest = manifest(&schema, unpartitioned(&schema), vec![bounded]);
        let grouped = GroupedManifest::group("table", manifest).unwrap();
        let partition = grouped.partitions().next().unwrap();
        let bounds = partition.build_columns(&[1]).remove(0);
        let column = ComparedColumn::resolve(schema.field_by_id(1).unwrap(), Some(&schema));
        let above_every_value =
            bounds.may_match_comparison(&column, PredicateOperator::GreaterThan, &upper);
        above_every_value.value(0)
    });

    assert_eq!(kept, [true; 6]);
}

/// Checks every comparison of field 1, as `current` has it, with each of
/// `constants`, values of that type in ascending order, against a manifest
/// written under `stored` that lists `bounded` beside a file it records no
/// bound of. `held` is where the bounds of `bounded` stand among the
/// constants, from its lower to its upper: the constant at `position` stands
/// at `2 * position`, and a bound between two constants at the odd place
/// between theirs. The file without bounds must always be kept, and `bounded`
/// exactly when a value from its lower bound to its upper would match.
fn assert_kept_when_a_held_value_matches(
    current: &SchemaRef,
    stored: &SchemaRef,
    bounded: DataFile,
    constants: &[ArrayRef],
    held: RangeInclusive<i32>,
) {
    type Matches = fn(i32, i32) -> bool;
    let comparisons: [(CompareType, Matches); 6] = [
        (CompareType::Equal, |value, constant| value == constant),
        (CompareType::NotEqual, |value, constant| value != constant),
        (CompareType::Less, |value, constant| value < constant),
        (CompareType::LessEqual, |value, constant| value <= constant),
        (CompareType::Greater, |value, constant| value > constant),
        (CompareType::GreaterEqual, |value, constant| {
            value >= constant
        }),
    ];
    let unknown = file("unknown").build().unwrap();
    let files = vec![bounded, unknown];
    let manifests = [manifest(stored, unpartitioned(stored), files)];

    for (compare_type, matches) in comparisons {
        for (position, constant) in constants.iter().enumerate() {
            let filter = predicate(compare_type, Arc::clone(constant));

            let kept = selected(current, &manifests, &filter);

            let place = 2 * position as i32;
            let may_match = held.clone().any(|value| matches(value, place));
            let expected: &[&str] = if may_match {
                &["bounded", "unknown"]
            } else {
                &["unknown"]
            };
            let column_type = &current.as_struct().fields()[0].field_type;
            assert_eq!(
                kept, expected,
                "{column_type} bounds at {held:?} {compare_type:?} constant {position}"
            );
        }
    }
}

#[test]
fn bounds_of_every_orderable_type_keep_a_file_exactly_when_a_held_value_would_match() {
    let booleans = || -> Vec<ArrayRef> {
        [false, true]
            .map(|value| Arc::new(BooleanArray::from(vec![value])) as ArrayRef)
            .to_vec()
    };
    let ints = [i32::MIN, -5, 0, 5, i32::MAX]
        .map(|value| Arc::new(Int32Array::from(vec![value])) as ArrayRef)
        .to_vec();
    let dates = [-20, -5, 0, 5, 20]
        .map(|value| Arc::new(Date32Array::from(vec![value])) as ArrayRef)
        .to_vec();
    let timestamps = [-20, -5, 0, 5, 20]
        .map(|value| Arc::new(TimestampMicrosecondArray::from(vec![value])) as ArrayRef)
        .to_vec();
    let zoned_timestamps = [-20, -5, 0, 5, 20]
        .map(|value| {
            let constant = TimestampMicrosecondArray::from(vec![value]).with_timezone("UTC");
            Arc::new(constant) as ArrayRef
        })
        .to_vec();
    type Case = (
        PrimitiveType,
        Datum,
        Datum,
        Vec<ArrayRef>,
        RangeInclusive<i32>,
    );
    let cases: Vec<Case> = vec![
        (
            PrimitiveType::Boolean,
            Datum::bool(false),
            Datum::bool(false),
            booleans(),
            0..=0,
        ),
        (
            PrimitiveType::Boolean,
            Datum::bool(false),
            Datum::bool(true),
            booleans(),
            0..=2,
        ),
        (
            PrimitiveType::Boolean,
            Datum::bool(true),
            Datum::bool(true),
            booleans(),
            2..=2,
        ),
        (
            PrimitiveType::Int,
            Datum::int(-5),
            Datum::int(5),
            ints,
            2..=6,
        ),
        (
            PrimitiveType::Date,
            Datum::date(-5),
            Datum::date(5),
            dates,
            2..=6,
        ),
        (
            PrimitiveType::Timestamp,
            Datum::timestamp_micros(-5),
            Datum::timestamp_micros(5),
            timestamps,
            2..=6,
        ),
        (
            PrimitiveType::Timestamptz,
            Datum::timestamptz_micros(-5),
            Datum::timestamptz_micros(5),
            zoned_timestamps,
            2..=6,
        ),
    ];

    for (column_type, lower, upper, constants, held) in cases {
        let schema = schema(1, "value", column_type);
        let bounded = file_between(lower, upper);

        assert_kept_when_a_held_value_matches(&schema, &schema, bounded, &constants, held);
    }
}

#[test]
fn bounds_recorded_in_a_narrower_type_keep_a_file_exactly_when_a_held_value_would_match() {
    let longs = |values: [i64; 5]| -> Vec<ArrayRef> {
        values
            .map(|value| Arc::new(Int64Array::from(vec![value])) as ArrayRef)
            .to_vec()
    };
    let doubles = |values: [f64; 5]| -> Vec<ArrayRef> {
        values
            .map(|value| Arc::new(Float64Array::from(vec![value])) as ArrayRef)
            .to_vec()
    };
    let narrow_decimal = PrimitiveType::Decimal {
        precision: 10,
        scale: 2,
    };
    let wide_decimal = PrimitiveType::Decimal {
        precision: 38,
        scale: 2,
    };
    let decimal_bound = |unscaled: i128| {
        Datum::try_from_bytes(&unscaled.to_be_bytes(), narrow_decimal.clone()).unwrap()
    };
    let wide_decimals = [-(10_i128.pow(37)), 100, 150, 200, 10_i128.pow(37)]
        .map(|unscaled| {
            let constant = Decimal128Array::from(vec![unscaled]);
            Arc::new(constant.with_precision_and_scale(38, 2).unwrap()) as ArrayRef
        })
        .to_vec();
    // A tenth as a FLOAT is a little more than a tenth as a DOUBLE.
    let float_tenth = f64::from(0.1_f32);
    type Case = (
        PrimitiveType,
        PrimitiveType,
        Datum,
        Datum,
        Vec<ArrayRef>,
        RangeInclusive<i32>,
    );
    let cases: Vec<Case> = vec![
        (
            PrimitiveType::Int,
            PrimitiveType::Long,
            Datum::int(-5),
            Datum::int(5),
            longs([-6, -5, 0, 5, 6]),
            2..=6,
        ),
        (
            PrimitiveType::Int,
            PrimitiveType::Long,
            Datum::int(i32::MIN),
            Datum::int(i32::MAX),
            longs([
                i64::from(i32::MIN) - 1,
                i64::from(i32::MIN),
                0,
                i64::from(i32::MAX),
                i64::from(i32::MAX) + 1,
            ]),
            2..=6,
        ),
        (
            PrimitiveType::Float,
            PrimitiveType::Double,
            Datum::float(1.0_f32),
            Datum::float(2.0_f32),
            doubles([0.99999995, 1.0, 1.00000005, 2.0, 2.00000005]),
            2..=6,
        ),
        (
            PrimitiveType::Float,
            PrimitiveType::Double,
            Datum::float(0.1_f32),
            Datum::float(0.1_f32),
            doubles([0.0, 0.1, float_tenth, 0.11, 1.0]),
            4..=4,
        ),
        (
            narrow_decimal.clone(),
            wide_decimal,
            decimal_bound(100),
            decimal_bound(200),
            wide_decimals,
            2..=6,
        ),
    ];

    for (stored_type, current_type, lower, upper, constants, held) in cases {
        let stored = schema(1, "value", stored_type);
        let current = schema(1, "value", current_type);
        let bounded = file_between(lower, upper);

        assert_kept_when_a_held_value_matches(&current, &stored, bounded, &constants, held);
    }
}

#[test]
fn float_bounds_order_zeros_infinities_and_nans_as_the_total_order_has_them() {
    let floats = |values: [f32; 4]| -> Vec<ArrayRef> {
        values
            .map(|value| Arc::new(Float32Array::from(vec![value])) as ArrayRef)
            .to_vec()
    };
    let doubles = |values: [f64; 4]| -> Vec<ArrayRef> {
        values
            .map(|value| Arc::new(Float64Array::from(vec![value])) as ArrayRef)
            .to_vec()
    };
    // The bounds of each file with where they stand among the constants
    // beside them: a NaN is past every constant, on the side of its sign.
    let ranges: [(f32, f32, [f32; 4], RangeInclusive<i32>); 5] = [
        (
            f32::NEG_INFINITY,
            -0.0,
            [f32::NEG_INFINITY, -1.0, -0.0, 0.0],
            0..=4,
        ),
        (0.0, f32::INFINITY, [-0.0, 0.0, 1.0, f32::INFINITY], 2..=6),
        (1.0, f32::NAN, [0.5, 1.0, 2.0, f32::INFINITY], 2..=7),
        (-f32::NAN, 1.0, [f32::NEG_INFINITY, 0.5, 1.0, 2.0], -1..=4),
        (f32::NAN, f32::NAN, [0.5, 1.0, 2.0, f32::INFINITY], 7..=7),
    ];
    let float = schema(1, "value", PrimitiveType::Float);
    let double = schema(1, "value", PrimitiveType::Double);

    for (lower, upper, constants, held) in ranges {
        let narrow = || file_between(Datum::float(lower), Datum::float(upper));
        let wide = file_between(
            Datum::double(f64::from(lower)),
            Datum::double(f64::from(upper)),
        );
        let narrow_constants = floats(constants);
        let wide_constants = doubles(constants.map(f64::from));

        assert_kept_when_a_held_value_matches(
            &float,
            &float,
            narrow(),
            &narrow_constants,
            held.clone(),
        );
        assert_kept_when_a_held_value_matches(
            &double,
            &double,
            wide,
            &wide_constants,
            held.clone(),
        );
        assert_kept_when_a_held_value_matches(&double, &float, narrow(), &wide_constants, held);
    }
}

#[test]
fn bounds_recorded_in_a_type_the_columns_is_not_a_widening_of_rule_no_file_out() {
    let decimal = |scale: u32| PrimitiveType::Decimal {
        precision: 10,
        scale,
    };
    let thousandths = Decimal128Array::from(vec![900_000])
        .with_precision_and_scale(10, 3)
        .unwrap();
    // Each file holds one value, far below the constant its column is
    // compared with.
    let cases: Vec<(PrimitiveType, PrimitiveType, Datum, ArrayRef)> = vec![
        (
            PrimitiveType::Long,
            PrimitiveType::Int,
            Datum::long(5),
            Arc::new(Int32Array::from(vec![900])),
        ),
        (
            PrimitiveType::Double,
            PrimitiveType::Float,
            Datum::double(5.0),
            Arc::new(Float32Array::from(vec![900.0])),
        ),
        (
            decimal(2),
            decimal(3),
            Datum::try_from_bytes(&5_i128.to_be_bytes(), decimal(2)).unwrap(),
            Arc::new(thousandths),
        ),
        (
            PrimitiveType::Date,
            PrimitiveType::Timestamp,
            Datum::date(5),
            Arc::new(TimestampMicrosecondArray::from(vec![900])),
        ),
        (
            PrimitiveType::Int,
            PrimitiveType::Date,
            Datum::int(5),
            Arc::new(Date32Array::from(vec![900])),
        ),
        (
            PrimitiveType::Timestamp,
            PrimitiveType::Timestamptz,
            Datum::timestamp_micros(5),
            Arc::new(TimestampMicrosecondArray::from(vec![900]).with_timezone("UTC")),
        ),
    ];

    let kept: Vec<Vec<String>> = cases
        .into_iter()
        .map(|(stored_type, current_type, held, constant)| {
            let stored = schema(1, "value", stored_type);
            let current = schema(1, "value", current_type);
            let files = vec![file_holding("held", held)];
            let manifests = [manifest(&stored, unpartitioned(&stored), files)];
            selected(
                &current,
                &manifests,
                &predicate(CompareType::Equal, constant),
            )
        })
        .collect();

    assert!(kept.iter().all(|kept| kept == &["held"]), "{kept:?}");
}
