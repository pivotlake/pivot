//! An Iceberg schema as the planner's columns.

use std::collections::{BTreeMap, HashMap};

use iceberg::spec::{PrimitiveType, Schema, Type as IcebergType};
use parquet_engine::TableColumns;
use planner::catalog::Column;
use planner::types::Type;

use crate::{Error, Result};

/// The table property naming, by field id, the Pivot type of every column
/// whose Iceberg type is wider than or different from it: the narrow and
/// unsigned integers, which Iceberg stores as `int`/`long`, and `Variant`,
/// which is declared as its physical struct. A reader without the property
/// sees the Iceberg types; Pivot restores its own.
pub const LOGICAL_TYPES_PROPERTY: &str = "pivot.logical-types";

/// The Pivot types [`LOGICAL_TYPES_PROPERTY`] records, by field id. A
/// property that does not parse, or names a type Pivot never records, is an
/// error: the table was written by something this reader does not
/// understand, and guessing a type would read its columns wrongly.
pub fn read_logical_types(properties: &HashMap<String, String>) -> Result<HashMap<i32, Type>> {
    let Some(raw) = properties.get(LOGICAL_TYPES_PROPERTY) else {
        return Ok(HashMap::new());
    };
    let recorded: BTreeMap<String, String> =
        serde_json::from_str(raw).map_err(|source| Error::MalformedProperty {
            property: LOGICAL_TYPES_PROPERTY.to_string(),
            message: source.to_string(),
        })?;
    recorded
        .into_iter()
        .map(|(id, name)| {
            let id = id.parse::<i32>().map_err(|_| Error::MalformedProperty {
                property: LOGICAL_TYPES_PROPERTY.to_string(),
                message: format!("`{id}` is not a field id"),
            })?;
            let pivot_type =
                parse_recorded_type(&name).ok_or_else(|| Error::MalformedProperty {
                    property: LOGICAL_TYPES_PROPERTY.to_string(),
                    message: format!("`{name}` is not a recorded Pivot type"),
                })?;
            Ok((id, pivot_type))
        })
        .collect()
}

/// The Pivot type `name` spells, among those the property records.
fn parse_recorded_type(name: &str) -> Option<Type> {
    [
        Type::Int8,
        Type::Int16,
        Type::UInt8,
        Type::UInt16,
        Type::UInt32,
        Type::UInt64,
        Type::Variant,
    ]
    .into_iter()
    .find(|candidate| candidate.to_string() == name)
}

/// `schema`'s top-level fields as the table's declared columns, each matched
/// to a file's columns by its Iceberg field id. A field `logical_types`
/// names (a Pivot type the table recorded as narrower than, or different
/// from, the field's Iceberg type) reads as that type. A field whose type
/// Pivot cannot represent refuses the whole table: a schema quietly missing
/// a column is a wrong answer to `SELECT *`. So does a field with an initial
/// default, which files predating it would have to read instead of NULL.
pub fn table_columns(
    table: &str,
    schema: &Schema,
    logical_types: &HashMap<i32, Type>,
) -> Result<TableColumns> {
    let mut columns = Vec::new();
    let mut field_ids = Vec::new();
    for field in schema.as_struct().fields() {
        let col_type = match logical_types.get(&field.id) {
            Some(recorded) => recorded.clone(),
            None => {
                to_pivot_type(&field.field_type).ok_or_else(|| Error::UnsupportedColumnType {
                    table: table.to_string(),
                    column: field.name.clone(),
                    iceberg_type: field.field_type.to_string(),
                })?
            }
        };
        if let Some(default) = &field.initial_default {
            return Err(Error::UnsupportedColumnDefault {
                table: table.to_string(),
                column: field.name.clone(),
                default: format!("{default:?}"),
            });
        }
        columns.push(Column {
            name: field.name.clone(),
            col_type,
        });
        field_ids.push(field.id);
    }
    Ok(TableColumns::by_field_id(columns, field_ids))
}

/// The Pivot type an Iceberg type reads as, or `None` for one Pivot has no
/// column type for (nested types, time, nanosecond timestamps, uuid, fixed,
/// binary).
pub fn to_pivot_type(iceberg_type: &IcebergType) -> Option<Type> {
    let IcebergType::Primitive(primitive) = iceberg_type else {
        return None;
    };
    Some(match primitive {
        PrimitiveType::Boolean => Type::Boolean,
        PrimitiveType::Int => Type::Int32,
        PrimitiveType::Long => Type::Int64,
        PrimitiveType::Float => Type::Float32,
        PrimitiveType::Double => Type::Float64,
        PrimitiveType::Decimal { precision, scale } => Type::Decimal {
            precision: u8::try_from(*precision).ok()?,
            scale: i8::try_from(*scale).ok()?,
        },
        PrimitiveType::Date => Type::Date,
        PrimitiveType::Timestamp => Type::Timestamp,
        PrimitiveType::Timestamptz => Type::TimestampTz,
        PrimitiveType::String => Type::Utf8,
        PrimitiveType::Time
        | PrimitiveType::TimestampNs
        | PrimitiveType::TimestamptzNs
        | PrimitiveType::Uuid
        | PrimitiveType::Fixed(_)
        | PrimitiveType::Binary => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use iceberg::spec::{ListType, Literal, NestedField};
    use parquet_engine::ColumnResolution;
    use std::sync::Arc;

    fn schema(fields: Vec<NestedField>) -> Schema {
        Schema::builder()
            .with_fields(fields.into_iter().map(Arc::new))
            .build()
            .unwrap()
    }

    #[test]
    fn primitive_fields_become_columns_matched_by_their_ids() {
        let schema = schema(vec![
            NestedField::required(7, "id", IcebergType::Primitive(PrimitiveType::Long)),
            NestedField::optional(3, "name", IcebergType::Primitive(PrimitiveType::String)),
            NestedField::optional(
                5,
                "price",
                IcebergType::Primitive(PrimitiveType::Decimal {
                    precision: 10,
                    scale: 2,
                }),
            ),
        ]);

        let columns = table_columns("t", &schema, &HashMap::new()).unwrap();

        let names: Vec<&str> = columns
            .columns()
            .iter()
            .map(|column| column.name.as_str())
            .collect();
        assert_eq!(names, ["id", "name", "price"]);
        assert!(
            matches!(columns.resolution(), ColumnResolution::ByFieldId(ids) if **ids == [7, 3, 5])
        );
        assert_eq!(
            columns.columns()[2].col_type,
            Type::Decimal {
                precision: 10,
                scale: 2
            }
        );
    }

    #[test]
    fn a_field_with_an_initial_default_refuses_the_table_naming_the_column() {
        let schema = schema(vec![
            NestedField::required(1, "id", IcebergType::Primitive(PrimitiveType::Long)),
            NestedField::optional(2, "score", IcebergType::Primitive(PrimitiveType::Long))
                .with_initial_default(Literal::long(7)),
        ]);

        let error = table_columns("t", &schema, &HashMap::new()).unwrap_err();

        assert!(
            matches!(&error, Error::UnsupportedColumnDefault { column, .. } if column == "score"),
            "{error}"
        );
    }

    #[test]
    fn a_recorded_logical_type_overrides_the_fields_iceberg_type() {
        let schema = schema(vec![
            NestedField::optional(1, "tiny", IcebergType::Primitive(PrimitiveType::Int)),
            NestedField::optional(2, "wide", IcebergType::Primitive(PrimitiveType::Long)),
        ]);
        let logical_types = HashMap::from([(1, Type::Int8)]);

        let columns = table_columns("t", &schema, &logical_types).unwrap();

        assert_eq!(columns.columns()[0].col_type, Type::Int8);
        assert_eq!(columns.columns()[1].col_type, Type::Int64);
    }

    #[test]
    fn a_property_naming_an_unknown_type_is_an_error() {
        let error = read_logical_types(&HashMap::from([(
            LOGICAL_TYPES_PROPERTY.to_string(),
            r#"{"1":"Int128"}"#.to_string(),
        )]))
        .unwrap_err();

        assert!(matches!(error, Error::MalformedProperty { .. }), "{error}");
    }

    #[test]
    fn a_nested_field_refuses_the_table_naming_the_column() {
        let schema = schema(vec![
            NestedField::required(1, "id", IcebergType::Primitive(PrimitiveType::Long)),
            NestedField::optional(
                2,
                "tags",
                IcebergType::List(ListType::new(Arc::new(NestedField::list_element(
                    3,
                    IcebergType::Primitive(PrimitiveType::String),
                    true,
                )))),
            ),
        ]);

        let error = table_columns("t", &schema, &HashMap::new()).unwrap_err();

        assert!(
            matches!(&error, Error::UnsupportedColumnType { column, .. } if column == "tags"),
            "{error}"
        );
    }
}
