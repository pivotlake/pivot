//! How a table's partition spec keys an INSERT's files.
//!
//! An Iceberg partition field is a transform of a source column: identity,
//! or one the spec crate computes (a day of a timestamp, a bucket of an id,
//! a truncated string). The encoder cuts files by the values of named batch
//! columns, so every partition field becomes one key column named after the
//! field: an identity field named after its source column is keyed by the
//! column itself, and every other field's key is derived per row, appended
//! to the batch after the written columns, read for the key and the file's
//! partition values, and dropped before the file is written.

use std::sync::Arc;

use arrow_array::{ArrayRef, new_empty_array};
use arrow_schema::{DataType, Field};
use iceberg::arrow::type_to_arrow_type;
use iceberg::spec::{PartitionSpec, Schema as IcebergSchema, Transform, Type};
use iceberg::transform::{TransformFunction, create_transform_function};
use parquet_engine::writing::Partitioning;
use planner::catalog::Column;

use crate::{Error, Result};

/// The key columns an INSERT's files are cut by, one per partition field in
/// spec order, each named after its field.
#[derive(Clone)]
pub struct PartitionKeys {
    key_columns: Vec<String>,
    /// The keys computed per row, in the order they are appended to a batch.
    derived: Vec<DerivedKey>,
}

/// A partition key computed from a column by the field's transform.
#[derive(Clone)]
struct DerivedKey {
    name: String,
    source_column: usize,
    /// The Arrow type the transform takes its input in: the Iceberg type's
    /// own. A source column of another physical type (a string view, a
    /// narrow decimal) is cast to it first.
    input_type: DataType,
    result_type: DataType,
    transform: Arc<dyn TransformFunction>,
}

impl PartitionKeys {
    /// Files keyed by the values of the named columns, each the identity of
    /// a partition field named after it: what a table partitions by when its
    /// spec was built from column names.
    pub fn by_columns(columns: &[String]) -> Self {
        Self {
            key_columns: columns.to_vec(),
            derived: Vec::new(),
        }
    }

    /// The keys for `spec` over a table of `columns`, or why the INSERT cannot
    /// partition by it: a source that is not one of the columns, a field
    /// named like a column (its key could not be told from the column), or a
    /// transform the spec crate cannot compute for the source's type.
    pub fn plan(
        table: &str,
        schema: &IcebergSchema,
        spec: &PartitionSpec,
        columns: &[Column],
    ) -> Result<Self> {
        let mut key_columns = Vec::with_capacity(spec.fields().len());
        let mut derived = Vec::new();
        for field in spec.fields() {
            let no_source = || Error::UnsupportedPartitionSource {
                table: table.to_string(),
                field: field.name.clone(),
            };
            let source_name = schema
                .name_by_field_id(field.source_id)
                .ok_or_else(no_source)?;
            let source_column = columns
                .iter()
                .position(|column| column.name == source_name)
                .ok_or_else(no_source)?;
            let source_type = schema
                .field_by_name(source_name)
                .and_then(|source| source.field_type.as_primitive_type().cloned())
                .ok_or_else(no_source)?;
            if field.transform == Transform::Identity && field.name == source_name {
                key_columns.push(field.name.clone());
                continue;
            }
            if columns.iter().any(|column| column.name == field.name) {
                return Err(Error::PartitionFieldNamedLikeColumn {
                    table: table.to_string(),
                    field: field.name.clone(),
                });
            }
            let unsupported = |message: String| Error::UnsupportedPartitionTransform {
                table: table.to_string(),
                field: field.name.clone(),
                transform: field.transform.to_string(),
                message,
            };
            let transform = create_transform_function(&field.transform)
                .map_err(|error| unsupported(error.to_string()))?;
            let input_type = type_to_arrow_type(&Type::Primitive(source_type))
                .map_err(|error| unsupported(error.to_string()))?;
            // Run the transform once on no rows: what it refuses, it refuses
            // now, as the INSERT's error, and what it produces names the key
            // column's type.
            let result_type = transform
                .transform(new_empty_array(&input_type))
                .map_err(|error| unsupported(error.to_string()))?
                .data_type()
                .clone();
            derived.push(DerivedKey {
                name: field.name.clone(),
                source_column,
                input_type,
                result_type,
                transform: Arc::from(transform),
            });
            key_columns.push(field.name.clone());
        }
        Ok(Self {
            key_columns,
            derived,
        })
    }

    /// How the encoder splits the files: by every key column, the derived
    /// ones (appended last) read for the key and not written.
    pub fn partitioning(&self) -> Partitioning {
        Partitioning {
            key_columns: self.key_columns.clone().into(),
            derived_key_columns: self.derived.len(),
        }
    }

    /// The fields of the derived key columns, in the order they are appended.
    pub fn derived_fields(&self) -> impl Iterator<Item = Field> + '_ {
        self.derived
            .iter()
            .map(|key| Field::new(key.name.clone(), key.result_type.clone(), true))
    }

    /// The derived key columns of a batch whose columns are `columns`, in the
    /// order they are appended.
    pub fn derive(&self, columns: &[ArrayRef]) -> iceberg::Result<Vec<ArrayRef>> {
        self.derived
            .iter()
            .map(|key| {
                let source = &columns[key.source_column];
                let input = if source.data_type() == &key.input_type {
                    source.clone()
                } else {
                    arrow_cast::cast(source, &key.input_type)?
                };
                key.transform.transform(input)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::TimestampMicrosecondArray;
    use arrow_array::{Array, Date32Array, Int32Array, Int64Array, StringViewArray};
    use iceberg::spec::{NestedField, PrimitiveType};
    use planner::types::Type as PivotType;

    use super::*;

    fn events_schema() -> IcebergSchema {
        IcebergSchema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "at", Type::Primitive(PrimitiveType::Timestamp)).into(),
                NestedField::optional(3, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap()
    }

    fn events_columns() -> Vec<Column> {
        vec![
            Column {
                name: "id".to_string(),
                col_type: PivotType::Int64,
            },
            Column {
                name: "at".to_string(),
                col_type: PivotType::Timestamp,
            },
            Column {
                name: "name".to_string(),
                col_type: PivotType::Utf8,
            },
        ]
    }

    fn spec(fields: &[(&str, &str, Transform)]) -> PartitionSpec {
        let mut builder = PartitionSpec::builder(Arc::new(events_schema()));
        for (source, name, transform) in fields {
            builder = builder
                .add_partition_field(*source, *name, *transform)
                .unwrap();
        }
        builder.build().unwrap()
    }

    /// A spec as a catalog could hand it over, unchecked against the schema:
    /// what the builder refuses to build, the metadata may still say.
    fn unchecked_spec(source_id: i32, name: &str, transform: &str) -> PartitionSpec {
        serde_json::from_str(&format!(
            r#"{{"spec-id":0,"fields":[{{"source-id":{source_id},"field-id":1000,"name":"{name}","transform":"{transform}"}}]}}"#
        ))
        .unwrap()
    }

    #[test]
    fn a_day_of_a_timestamp_and_a_bucket_of_a_string_are_derived_per_row() {
        let keys = PartitionKeys::plan(
            "db.events",
            &events_schema(),
            &spec(&[
                ("at", "at_day", Transform::Day),
                ("name", "name_bucket", Transform::Bucket(8)),
                ("id", "id", Transform::Identity),
            ]),
            &events_columns(),
        )
        .unwrap();
        let day_micros = 86_400_000_000i64;
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(TimestampMicrosecondArray::from(vec![0, 3 * day_micros + 5])),
            Arc::new(StringViewArray::from(vec!["ann", "bob"])),
        ];

        let derived = keys.derive(&columns).unwrap();

        assert_eq!(keys.key_columns, ["at_day", "name_bucket", "id"]);
        assert_eq!(keys.partitioning().derived_key_columns, 2);
        assert_eq!(
            keys.derived_fields()
                .map(|field| (field.name().clone(), field.data_type().clone()))
                .collect::<Vec<_>>(),
            [
                ("at_day".to_string(), DataType::Date32),
                ("name_bucket".to_string(), DataType::Int32),
            ]
        );
        let days = derived[0].as_any().downcast_ref::<Date32Array>().unwrap();
        assert_eq!(days.values().as_ref(), [0, 3]);
        let buckets = derived[1].as_any().downcast_ref::<Int32Array>().unwrap();
        assert!(
            buckets
                .values()
                .iter()
                .all(|bucket| (0..8).contains(bucket))
        );
    }

    #[test]
    fn columns_named_by_a_table_key_identity_fields_with_nothing_derived() {
        let keys = PartitionKeys::by_columns(&["region".to_string()]);

        let partitioning = keys.partitioning();

        assert_eq!(partitioning.key_columns.as_ref(), ["region".to_string()]);
        assert_eq!(partitioning.derived_key_columns, 0);
        assert_eq!(keys.derived_fields().count(), 0);
    }

    #[test]
    fn a_partition_field_named_like_a_column_is_refused() {
        let error = PartitionKeys::plan(
            "db.events",
            &events_schema(),
            &unchecked_spec(2, "name", "day"),
            &events_columns(),
        )
        .err()
        .expect("the plan is refused");

        assert!(
            matches!(error, Error::PartitionFieldNamedLikeColumn { ref field, .. } if field == "name"),
            "{error}"
        );
    }

    #[test]
    fn a_transform_the_source_type_does_not_take_is_refused_when_planned() {
        let error = PartitionKeys::plan(
            "db.events",
            &events_schema(),
            &unchecked_spec(3, "name_day", "day"),
            &events_columns(),
        )
        .err()
        .expect("the plan is refused");

        assert!(
            matches!(error, Error::UnsupportedPartitionTransform { ref field, .. } if field == "name_day"),
            "{error}"
        );
    }
}
