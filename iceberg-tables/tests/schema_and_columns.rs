//! A table's declared shape as Iceberg metadata and back: its columns as a
//! schema, its partition and sort columns as a spec and a sort order, the
//! Pivot types Iceberg has no spelling for as a property, and the partition
//! keys an INSERT's files are cut by.

mod support;

use std::collections::HashMap;
use std::sync::Arc;

use iceberg::spec::{
    FormatVersion, NestedField, NullOrder, PrimitiveType, Schema, SortDirection, SortField,
    SortOrder, TableMetadata, TableMetadataBuilder, Transform, Type as IcebergType,
    UnboundPartitionSpec,
};
use iceberg_tables::types::schema::{
    LOGICAL_TYPES_PROPERTY, build_partition_spec, build_schema, build_sort_order,
    find_writer_sort_order_id, logical_types_property, name_mapping_property,
};
use iceberg_tables::{Error, PartitionKeys};
use planner::catalog::Column;
use planner::types::Type;
use support::{Fixture, column, create_table_of, load};

#[test]
fn columns_become_optional_fields_numbered_in_order_with_variant_leaves_after() {
    let columns = [
        column("id", Type::UInt64),
        column("doc", Type::Variant),
        column("day", Type::Date),
    ];

    let schema = build_schema(&columns).unwrap();

    let fields = schema.as_struct().fields();
    assert_eq!(
        fields
            .iter()
            .map(|field| (field.id, field.name.as_str(), field.required))
            .collect::<Vec<_>>(),
        [(1, "id", false), (2, "doc", false), (3, "day", false)]
    );
    assert_eq!(
        fields[0].field_type.as_ref(),
        &IcebergType::Primitive(PrimitiveType::Long)
    );
    let IcebergType::Struct(variant) = fields[1].field_type.as_ref() else {
        panic!("a variant column is declared as a struct");
    };
    assert_eq!(
        variant
            .fields()
            .iter()
            .map(|leaf| (leaf.id, leaf.name.as_str()))
            .collect::<Vec<_>>(),
        [(4, "metadata"), (5, "value")]
    );
}

#[test]
fn types_without_an_iceberg_form_are_refused_naming_the_column() {
    let error = build_schema(&[column("big", Type::Int128)]).unwrap_err();

    assert!(
        matches!(&error, Error::UnsupportedPivotType { column, .. } if column == "big"),
        "{error}"
    );
}

#[test]
fn logical_types_round_trip_through_the_property() {
    let columns = [
        column("tiny", Type::Int8),
        column("plain", Type::Int64),
        column("doc", Type::Variant),
    ];
    let schema = build_schema(&columns).unwrap();

    let property = logical_types_property(&schema, &columns).unwrap();
    let (fixture, created) = create_table_of(&columns, &[], &[]);
    let loaded = load(&fixture);

    assert_eq!(property, r#"{"1":"Int8","3":"Variant"}"#);
    assert_eq!(
        created.metadata().properties()[LOGICAL_TYPES_PROPERTY],
        property
    );
    assert_eq!(loaded.columns(), columns);
    assert!(logical_types_property(&schema, &[column("plain", Type::Int64)]).is_none());
}

#[test]
fn partition_and_sort_columns_bind_to_their_fields_by_identity() {
    let columns = [column("day", Type::Date), column("id", Type::Int64)];
    let schema = build_schema(&columns).unwrap();

    let spec = build_partition_spec(&schema, &["day".to_string()]).unwrap();
    let order = build_sort_order(&schema, &["id".to_string()]).unwrap();

    assert_eq!(spec.fields().len(), 1);
    assert_eq!(spec.fields()[0].source_id, 1);
    assert_eq!(spec.fields()[0].transform, Transform::Identity);
    assert_eq!(order.order_id, 1);
    assert_eq!(order.fields[0].source_id, 2);
    assert!(build_sort_order(&schema, &[]).unwrap().is_unsorted());
}

/// The metadata of a table of `columns` whose default sort order is the
/// one field `sort_field`.
fn build_metadata_sorted_by(columns: &[Column], sort_field: SortField) -> TableMetadata {
    let sort_order = SortOrder::builder()
        .with_order_id(1)
        .with_sort_field(sort_field)
        .build_unbound()
        .unwrap();
    TableMetadataBuilder::new(
        build_schema(columns).unwrap(),
        UnboundPartitionSpec::builder().build(),
        sort_order,
        "file:///t".to_string(),
        FormatVersion::V3,
        HashMap::new(),
    )
    .unwrap()
    .build()
    .unwrap()
    .metadata
}

#[test]
fn only_an_ascending_nulls_first_order_over_plain_columns_is_the_writers() {
    let signed = [column("id", Type::Int64)];
    let unsigned = [column("id", Type::UInt64)];
    let field = |transform, direction, null_order| SortField {
        source_id: 1,
        transform,
        direction,
        null_order,
    };
    let (ascending, descending) = (SortDirection::Ascending, SortDirection::Descending);
    let written = field(Transform::Identity, ascending, NullOrder::First);

    let ids = [
        find_writer_sort_order_id(&build_metadata_sorted_by(&signed, written.clone()), &signed),
        find_writer_sort_order_id(
            &build_metadata_sorted_by(
                &signed,
                field(Transform::Identity, descending, NullOrder::First),
            ),
            &signed,
        ),
        find_writer_sort_order_id(
            &build_metadata_sorted_by(
                &signed,
                field(Transform::Identity, ascending, NullOrder::Last),
            ),
            &signed,
        ),
        find_writer_sort_order_id(
            &build_metadata_sorted_by(
                &signed,
                field(Transform::Bucket(4), ascending, NullOrder::First),
            ),
            &signed,
        ),
        find_writer_sort_order_id(&build_metadata_sorted_by(&unsigned, written), &unsigned),
    ];

    assert_eq!(ids, [Some(1), None, None, None, None]);
}

#[test]
fn the_name_mapping_lists_every_top_level_field() {
    let schema = build_schema(&[column("a", Type::Int64), column("b", Type::Utf8)]).unwrap();

    assert_eq!(
        name_mapping_property(&schema),
        r#"[{"field-id":1,"names":["a"]},{"field-id":2,"names":["b"]}]"#
    );
}

/// The metadata of a table whose schema is `fields` and whose properties
/// are `properties`.
fn build_metadata(fields: Vec<NestedField>, properties: HashMap<String, String>) -> TableMetadata {
    let schema = Schema::builder()
        .with_fields(fields.into_iter().map(Arc::new))
        .build()
        .unwrap();
    TableMetadataBuilder::new(
        schema,
        UnboundPartitionSpec::builder().build(),
        SortOrder::unsorted_order(),
        "file:///t".to_string(),
        FormatVersion::V3,
        properties,
    )
    .unwrap()
    .build()
    .unwrap()
    .metadata
}

#[test]
fn a_recorded_logical_type_overrides_the_fields_iceberg_type() {
    let fields = vec![
        NestedField::optional(1, "tiny", IcebergType::Primitive(PrimitiveType::Int)),
        NestedField::optional(2, "wide", IcebergType::Primitive(PrimitiveType::Long)),
    ];
    let logical_types = HashMap::from([(
        LOGICAL_TYPES_PROPERTY.to_string(),
        r#"{"1":"Int8"}"#.to_string(),
    )]);
    let fixture = Fixture::new();

    let table = fixture.build_table_at(build_metadata(fields, logical_types));

    assert_eq!(table.columns()[0].col_type, Type::Int8);
    assert_eq!(table.columns()[1].col_type, Type::Int64);
}

#[test]
fn a_property_naming_an_unknown_type_is_an_error() {
    let fields = vec![NestedField::optional(
        1,
        "big",
        IcebergType::Primitive(PrimitiveType::Long),
    )];
    let unknown_type = HashMap::from([(
        LOGICAL_TYPES_PROPERTY.to_string(),
        r#"{"1":"Int128"}"#.to_string(),
    )]);
    let fixture = Fixture::new();

    let error = fixture
        .try_build_table_at(build_metadata(fields, unknown_type))
        .err()
        .expect("a table recording a type Pivot never records is refused");

    assert!(matches!(error, Error::MalformedProperty { .. }), "{error}");
}

#[test]
fn columns_named_by_a_table_key_identity_fields_with_nothing_derived() {
    let keys = PartitionKeys::by_columns(&["region".to_string()]);

    let partitioning = keys.partitioning();

    assert_eq!(partitioning.key_columns.as_ref(), ["region".to_string()]);
    assert_eq!(partitioning.unwritten_trailing_columns, 0);
    assert_eq!(keys.derived_fields().count(), 0);
}
