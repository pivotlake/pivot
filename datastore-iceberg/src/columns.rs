//! An Iceberg schema as the planner's columns.

use iceberg::spec::{PrimitiveType, Schema, Type as IcebergType};
use parquet_engine::TableColumns;
use planner::catalog::Column;
use planner::types::Type;

use crate::{Error, Result};

/// `schema`'s top-level fields as the table's declared columns, each matched
/// to a file's columns by its Iceberg field id. A field whose type Pivot
/// cannot represent refuses the whole table: a schema quietly missing a column
/// is a wrong answer to `SELECT *`.
pub(crate) fn table_columns(table: &str, schema: &Schema) -> Result<TableColumns> {
    let mut columns = Vec::new();
    let mut field_ids = Vec::new();
    for field in schema.as_struct().fields() {
        let col_type =
            to_pivot_type(&field.field_type).ok_or_else(|| Error::UnsupportedColumnType {
                table: table.to_string(),
                column: field.name.clone(),
                iceberg_type: field.field_type.to_string(),
            })?;
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
fn to_pivot_type(iceberg_type: &IcebergType) -> Option<Type> {
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
    use iceberg::spec::{ListType, NestedField};
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

        let columns = table_columns("t", &schema).unwrap();

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

        let error = table_columns("t", &schema).unwrap_err();

        assert!(
            matches!(&error, Error::UnsupportedColumnType { column, .. } if column == "tags"),
            "{error}"
        );
    }
}
