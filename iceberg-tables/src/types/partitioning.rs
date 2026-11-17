//! How a table's partitioning is spelled, and how its partition spec keys an
//! INSERT's files.
//!
//! `CREATE TABLE` names what a table is partitioned by as terms, each a
//! column or a transform of one under Iceberg's name for it (`day(ts)`,
//! `bucket(16, id)`), and a table shows the fields of its partition spec as
//! the same terms.
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

use crate::types::schema::{build_partition_spec, build_schema};
use crate::{Error, Result};

/// One term of a table's partitioning as `CREATE TABLE` is given it: a
/// column, or a transform of one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionTerm {
    pub transform: Transform,
    /// The column the transform is of.
    pub column: String,
    /// The name of the partition field the term is recorded as, by the
    /// convention Iceberg's writers share: the column's own for a column,
    /// and for a transform the column's with the transform's suffix
    /// (`ts_day`, `id_bucket`, `name_trunc`).
    pub field_name: String,
}

impl PartitionTerm {
    /// Parse `term`: a column name, or `year(c)`, `month(c)`, `day(c)`,
    /// `hour(c)`, `bucket(n, c)` or `truncate(w, c)`, the transform named in
    /// any case, `n` and `w` positive. The plurals `years`, `months`, `days`
    /// and `hours`, as Spark spells the time transforms, name the same
    /// transforms.
    pub fn parse(term: &str) -> Result<Self> {
        let Some((transform_name, arguments)) = term.split_once('(') else {
            return Ok(Self {
                transform: Transform::Identity,
                column: term.to_string(),
                field_name: term.to_string(),
            });
        };
        let refuse = || {
            Error::InvalidPartitionTerm {
            term: term.to_string(),
            message: "it is neither a column nor a transform of one, which is written \
                      `year(c)`, `month(c)`, `day(c)`, `hour(c)`, `bucket(n, c)` or `truncate(w, c)`"
                .to_string(),
        }
        };
        let arguments: Vec<&str> = arguments
            .strip_suffix(')')
            .ok_or_else(refuse)?
            .split(',')
            .map(str::trim)
            .collect();
        // A bucket count and a truncation width are positive Iceberg `int`s.
        let parse_parameter = |parameter: &str| {
            parameter
                .parse::<u32>()
                .ok()
                .filter(|value| (1..=i32::MAX as u32).contains(value))
                .ok_or_else(refuse)
        };
        let transform_name = transform_name.trim().to_ascii_lowercase();
        let (transform, column, suffix) = match (transform_name.as_str(), arguments.as_slice()) {
            ("year" | "years", [column]) => (Transform::Year, column, "year"),
            ("month" | "months", [column]) => (Transform::Month, column, "month"),
            ("day" | "days", [column]) => (Transform::Day, column, "day"),
            ("hour" | "hours", [column]) => (Transform::Hour, column, "hour"),
            ("bucket", [buckets, column]) => (
                Transform::Bucket(parse_parameter(buckets)?),
                column,
                "bucket",
            ),
            ("truncate", [width, column]) => (
                Transform::Truncate(parse_parameter(width)?),
                column,
                "trunc",
            ),
            _ => return Err(refuse()),
        };
        if column.is_empty() {
            return Err(refuse());
        }
        Ok(Self {
            transform,
            column: column.to_string(),
            field_name: format!("{column}_{suffix}"),
        })
    }
}

/// A partition field that applies `transform` to `column`, as the term
/// `CREATE TABLE` spells it: the column itself for an identity field,
/// whatever the field is named, and a transform under Iceberg's name for
/// it, whichever spelling made the field.
fn format_partition_term(transform: &Transform, column: &str) -> String {
    match transform {
        Transform::Identity => column.to_string(),
        Transform::Bucket(buckets) => format!("bucket({buckets}, {column})"),
        Transform::Truncate(width) => format!("truncate({width}, {column})"),
        // The transforms that take no parameter are spelled by their names.
        Transform::Year
        | Transform::Month
        | Transform::Day
        | Transform::Hour
        | Transform::Void
        | Transform::Unknown => format!("{transform}({column})"),
    }
}

/// The fields of `spec`, a partition spec of a table whose current schema is
/// `schema`, as the terms `CREATE TABLE` spells them, in spec order. A field
/// whose source the schema no longer has names no column, and is left out.
pub(crate) fn format_partition_fields(schema: &IcebergSchema, spec: &PartitionSpec) -> Vec<String> {
    spec.fields()
        .iter()
        .filter_map(|field| {
            let column = schema.name_by_field_id(field.source_id)?;
            Some(format_partition_term(&field.transform, column))
        })
        .collect()
}

/// Refuse `partition_by`, the terms the new table `table` of `columns` is to
/// be partitioned by, unless each is a column or a transform Iceberg takes
/// over its column's type, and an INSERT can compute the key of each.
pub fn check_partition_terms(
    table: &str,
    columns: &[Column],
    partition_by: &[String],
) -> Result<()> {
    let schema = Arc::new(build_schema(columns)?);
    let spec = build_partition_spec(&schema, partition_by)?
        .bind(Arc::clone(&schema))
        .map_err(Error::spec)?;
    PartitionKeys::plan(table, &schema, &spec, columns)?;
    Ok(())
}

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
            // A `UInt64` is stored as a signed `long`, the type the transform
            // takes it in, which holds no value above the signed range.
            if columns[source_column].col_type == planner::types::Type::UInt64 {
                return Err(unsupported(
                    "its source column is a UBIGINT, whose values above the signed range Iceberg has no value for".to_string(),
                ));
            }
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
            unwritten_trailing_columns: self.derived.len(),
        }
    }

    /// The key columns that are columns of the table, in spec order: the
    /// keys a file's rows carry.
    pub fn list_column_keys(&self) -> Vec<String> {
        self.key_columns
            .iter()
            .filter(|key| self.derived.iter().all(|derived| &derived.name != *key))
            .cloned()
            .collect()
    }

    /// The key columns computed per row, which no file's rows carry, in the
    /// order they are appended.
    pub fn list_derived_keys(&self) -> Vec<String> {
        self.derived.iter().map(|key| key.name.clone()).collect()
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
        assert_eq!(keys.partitioning().unwritten_trailing_columns, 2);
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
