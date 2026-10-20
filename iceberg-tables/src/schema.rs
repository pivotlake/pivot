//! A managed table's declared shape as Iceberg metadata, and back: its columns
//! as an Iceberg schema, its partition and sort columns as a spec and a sort
//! order, and the Pivot types Iceberg has no spelling for as a table property.

use std::collections::BTreeMap;
use std::sync::Arc;

use iceberg::spec::{
    NestedField, NullOrder, PrimitiveType, Schema, SortDirection, SortField, SortOrder, StructType,
    Transform, Type as IcebergType, UnboundPartitionSpec,
};
use planner::catalog::Column;
use planner::types::Type;

use crate::{Error, Result};

pub use crate::columns::LOGICAL_TYPES_PROPERTY;

/// The Iceberg name-mapping property, written when a table adopts Parquet
/// files that carry no field ids so readers resolve their columns by name.
pub const NAME_MAPPING_PROPERTY: &str = "schema.name-mapping.default";

/// `columns` as an Iceberg schema: every field optional (the catalog carries
/// no NOT NULL), field ids 1.. in column order, a variant column's two leaves
/// numbered after every top-level column.
pub fn build_schema(columns: &[Column]) -> Result<Schema> {
    let mut next_nested_id = i32::try_from(columns.len()).expect("a table has few columns") + 1;
    let mut fields = Vec::with_capacity(columns.len());
    for (position, column) in columns.iter().enumerate() {
        let id = i32::try_from(position + 1).expect("a table has few columns");
        let field_type = match &column.col_type {
            Type::Variant => {
                let metadata_id = next_nested_id;
                let value_id = next_nested_id + 1;
                next_nested_id += 2;
                variant_struct(metadata_id, value_id)
            }
            pivot_type => {
                IcebergType::Primitive(to_iceberg_primitive(pivot_type).ok_or_else(|| {
                    Error::UnsupportedPivotType {
                        column: column.name.clone(),
                        pivot_type: pivot_type.to_string(),
                    }
                })?)
            }
        };
        fields.push(Arc::new(NestedField::optional(
            id,
            &column.name,
            field_type,
        )));
    }
    Schema::builder()
        .with_schema_id(0)
        .with_fields(fields)
        .build()
        .map_err(Error::spec)
}

/// The Iceberg primitive a Pivot column type is stored as, or `None` for a
/// type with no Iceberg form (`Int128`, `Interval`; `Variant` is a struct).
pub fn to_iceberg_primitive(pivot_type: &Type) -> Option<PrimitiveType> {
    Some(match pivot_type {
        Type::Boolean => PrimitiveType::Boolean,
        Type::Int8 | Type::Int16 | Type::Int32 | Type::UInt8 | Type::UInt16 => PrimitiveType::Int,
        Type::Int64 | Type::UInt32 | Type::UInt64 => PrimitiveType::Long,
        Type::Float32 => PrimitiveType::Float,
        Type::Float64 => PrimitiveType::Double,
        Type::Decimal { precision, scale } => PrimitiveType::Decimal {
            precision: u32::from(*precision),
            scale: u32::try_from(*scale).ok()?,
        },
        Type::Utf8 => PrimitiveType::String,
        Type::Date => PrimitiveType::Date,
        Type::Timestamp => PrimitiveType::Timestamp,
        Type::TimestampTz => PrimitiveType::Timestamptz,
        Type::Int128 | Type::Interval | Type::Variant => return None,
    })
}

/// The physical shape a variant column is declared as: the binary pair the
/// Parquet VARIANT group is made of.
fn variant_struct(metadata_id: i32, value_id: i32) -> IcebergType {
    IcebergType::Struct(StructType::new(vec![
        Arc::new(NestedField::optional(
            metadata_id,
            "metadata",
            IcebergType::Primitive(PrimitiveType::Binary),
        )),
        Arc::new(NestedField::optional(
            value_id,
            "value",
            IcebergType::Primitive(PrimitiveType::Binary),
        )),
    ]))
}

/// Whether `pivot_type` is stored under a different Iceberg type and so must
/// be recorded in [`LOGICAL_TYPES_PROPERTY`] to be read back as itself.
fn needs_logical_type_record(pivot_type: &Type) -> bool {
    matches!(
        pivot_type,
        Type::Int8
            | Type::Int16
            | Type::UInt8
            | Type::UInt16
            | Type::UInt32
            | Type::UInt64
            | Type::Variant
    )
}

/// The [`LOGICAL_TYPES_PROPERTY`] value for `columns` under `schema`: a JSON
/// object from field id to Pivot type name, or `None` when every column's
/// Iceberg type reads back as itself. Keyed by the ids the built schema
/// carries, so it survives any reassignment the metadata builder applies.
pub fn logical_types_property(schema: &Schema, columns: &[Column]) -> Option<String> {
    let recorded: BTreeMap<String, String> = columns
        .iter()
        .filter(|column| needs_logical_type_record(&column.col_type))
        .filter_map(|column| {
            let id = schema.field_id_by_name(&column.name)?;
            Some((id.to_string(), column.col_type.to_string()))
        })
        .collect();
    (!recorded.is_empty())
        .then(|| serde_json::to_string(&recorded).expect("a string map serializes"))
}

/// An identity partition spec over `partition_by`, in order. Partition field
/// ids start at 1000, the Iceberg convention; names are the columns' own.
pub fn build_partition_spec(
    schema: &Schema,
    partition_by: &[String],
) -> Result<UnboundPartitionSpec> {
    let mut builder = UnboundPartitionSpec::builder().with_spec_id(0);
    for column in partition_by {
        let source_id =
            schema
                .field_id_by_name(column)
                .ok_or_else(|| Error::UnknownSpecColumn {
                    option: "partition_by".to_string(),
                    column: column.clone(),
                })?;
        builder = builder
            .add_partition_field(source_id, column, Transform::Identity)
            .map_err(Error::spec)?;
    }
    Ok(builder.build())
}

/// An ascending, nulls-first identity sort order over `sort_by`, in order;
/// the unsorted order when there are no sort columns.
pub fn build_sort_order(schema: &Schema, sort_by: &[String]) -> Result<SortOrder> {
    if sort_by.is_empty() {
        return Ok(SortOrder::unsorted_order());
    }
    let mut builder = SortOrder::builder();
    builder.with_order_id(1);
    for column in sort_by {
        let source_id =
            schema
                .field_id_by_name(column)
                .ok_or_else(|| Error::UnknownSpecColumn {
                    option: "sort_by".to_string(),
                    column: column.clone(),
                })?;
        builder.with_sort_field(SortField {
            source_id,
            transform: Transform::Identity,
            direction: SortDirection::Ascending,
            null_order: NullOrder::First,
        });
    }
    builder.build_unbound().map_err(Error::spec)
}

/// The [`NAME_MAPPING_PROPERTY`] value for `schema`: every top-level field
/// resolvable by its name, for files written without field ids.
pub fn name_mapping_property(schema: &Schema) -> String {
    let mapping: Vec<serde_json::Value> = schema
        .as_struct()
        .fields()
        .iter()
        .map(|field| {
            serde_json::json!({
                "field-id": field.id,
                "names": [field.name.clone()],
            })
        })
        .collect();
    serde_json::Value::Array(mapping).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn column(name: &str, col_type: Type) -> Column {
        Column {
            name: name.to_string(),
            col_type,
        }
    }

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
        let read = crate::columns::read_logical_types(&HashMap::from([(
            LOGICAL_TYPES_PROPERTY.to_string(),
            property.clone(),
        )]))
        .unwrap();

        assert_eq!(property, r#"{"1":"Int8","3":"Variant"}"#);
        assert_eq!(read, HashMap::from([(1, Type::Int8), (3, Type::Variant)]));
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

    #[test]
    fn the_name_mapping_lists_every_top_level_field() {
        let schema = build_schema(&[column("a", Type::Int64), column("b", Type::Utf8)]).unwrap();

        assert_eq!(
            name_mapping_property(&schema),
            r#"[{"field-id":1,"names":["a"]},{"field-id":2,"names":["b"]}]"#
        );
    }
}
