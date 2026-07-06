//! Mapping an Iceberg schema (as iceberg-rust models it) to planner columns.

use iceberg::spec::{PrimitiveType, Schema, Type as IcebergType};
use planner::catalog::Column;
use planner::types::Type;

use crate::{Error, Result};

/// A schema's top-level fields as planner [`Column`]s, in field order. `table`
/// names the table in errors.
pub(crate) fn map_columns(schema: &Schema, table: &str) -> Result<Vec<Column>> {
    schema
        .as_struct()
        .fields()
        .iter()
        .map(|field| {
            let col_type =
                map_field_type(&field.field_type).ok_or_else(|| Error::UnsupportedColumnType {
                    table: table.to_string(),
                    column: field.name.clone(),
                    iceberg_type: field.field_type.to_string(),
                })?;
            Ok(Column {
                name: field.name.clone(),
                col_type,
            })
        })
        .collect()
}

/// Map one Iceberg field type to the planner [`Type`] a column of it scans as,
/// or `None` for a type pivot has no counterpart for (which fails the resolve,
/// naming the column - never a column that decodes wrong).
fn map_field_type(field_type: &IcebergType) -> Option<Type> {
    let IcebergType::Primitive(primitive) = field_type else {
        // Nested types (struct/list/map) have no columnar counterpart.
        return None;
    };
    match primitive {
        PrimitiveType::Boolean => Some(Type::Boolean),
        PrimitiveType::Int => Some(Type::Int32),
        PrimitiveType::Long => Some(Type::Int64),
        PrimitiveType::Double => Some(Type::Float64),
        PrimitiveType::String => Some(Type::Utf8),
        PrimitiveType::Date => Some(Type::Date),
        // Not supported: timestamp flavors (iceberg stores microseconds, but
        // the engine decodes parquet timestamps as epoch seconds - mapping
        // them would silently misread every value by a factor of a million),
        // float (pivot has no 32-bit float type), decimal, uuid, time, binary,
        // fixed - and anything this list does not know.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iceberg::spec::NestedField;

    fn schema_of(fields: Vec<NestedField>) -> Schema {
        Schema::builder()
            .with_fields(fields.into_iter().map(Into::into))
            .build()
            .unwrap()
    }

    #[test]
    fn maps_supported_primitives_to_planner_columns() {
        let schema = schema_of(vec![
            NestedField::required(1, "id", IcebergType::Primitive(PrimitiveType::Long)),
            NestedField::optional(2, "name", IcebergType::Primitive(PrimitiveType::String)),
            NestedField::optional(3, "score", IcebergType::Primitive(PrimitiveType::Double)),
        ]);

        let columns = map_columns(&schema, "events").unwrap();

        let mapped: Vec<(&str, &Type)> = columns
            .iter()
            .map(|c| (c.name.as_str(), &c.col_type))
            .collect();
        assert_eq!(
            mapped,
            vec![
                ("id", &Type::Int64),
                ("name", &Type::Utf8),
                ("score", &Type::Float64),
            ]
        );
    }

    // The engine decodes parquet timestamps as epoch seconds while iceberg
    // stores microseconds, so timestamp columns must fail the resolve loudly
    // instead of silently misreading every value.
    #[test]
    fn rejects_timestamp_columns_naming_the_column() {
        let schema = schema_of(vec![NestedField::required(
            1,
            "event_ts",
            IcebergType::Primitive(PrimitiveType::Timestamptz),
        )]);

        let error = map_columns(&schema, "events").unwrap_err().to_string();

        assert!(error.contains("event_ts"), "unexpected error: {error}");
    }

    #[test]
    fn rejects_decimal_and_nested_column_types() {
        let decimal = schema_of(vec![NestedField::required(
            1,
            "amount",
            IcebergType::Primitive(PrimitiveType::Decimal {
                precision: 10,
                scale: 2,
            }),
        )]);
        let nested = schema_of(vec![NestedField::optional(
            1,
            "tags",
            IcebergType::List(iceberg::spec::ListType::new(
                NestedField::list_element(2, IcebergType::Primitive(PrimitiveType::String), false)
                    .into(),
            )),
        )]);

        let decimal_error = map_columns(&decimal, "orders").unwrap_err().to_string();
        assert!(
            decimal_error.contains("amount"),
            "unexpected error: {decimal_error}"
        );
        assert!(map_columns(&nested, "t").is_err());
    }
}
