//! Iceberg partition metadata as statistics: a partition field is a transform
//! of a table column, and its values bound that transform in the manifests
//! and data files written under the field's spec.

use crate::pruning::{Bounds, Statistic, Transform};
use arrow_schema::DataType;
use iceberg::spec::{Datum, Literal, ManifestFile, PartitionSpec, PrimitiveType, SchemaRef};
use iceberg::{Error, ErrorKind, Result};
use planner::types::physical_arrow_type;

use super::LiveFile;
use super::values::build_array;
use crate::columns::to_pivot_type;

/// One field of a partition spec.
pub(super) struct PartitionField {
    /// The table column the field is derived from.
    pub column_idx: usize,
    transform: Transform,
    /// Where a partition tuple of the field's spec holds its value.
    position: usize,
    /// The type of the transformed values of the column as it is typed today.
    /// Values written under a narrower column type promote into it.
    value_type: PrimitiveType,
    data_type: DataType,
}

/// The fields of `spec` whose values can prune: a transform Pivot evaluates, of
/// a column the current `schema` still has.
pub(super) fn fields(schema: &SchemaRef, spec: &PartitionSpec) -> Vec<PartitionField> {
    spec.fields()
        .iter()
        .enumerate()
        .filter_map(|(position, field)| {
            let transform = partition_transform(&field.transform)?;
            let columns = schema.as_struct().fields();
            let column_idx = columns
                .iter()
                .position(|column| column.id == field.source_id)?;
            let value_type = field
                .transform
                .result_type(&columns[column_idx].field_type)
                .ok()?;
            Some(PartitionField {
                column_idx,
                transform,
                position,
                value_type: value_type.as_primitive_type()?.clone(),
                data_type: physical_arrow_type(&to_pivot_type(&value_type)?),
            })
        })
        .collect()
}

impl PartitionField {
    /// The range of this field's values in each manifest, from the summary the
    /// manifest-list entry keeps per partition field. `spec_manifests` has one
    /// slot per manifest of the list, filled for those written under the
    /// field's spec; the others, and a manifest without summaries, are unknown.
    pub fn manifest_statistic(
        &self,
        spec_manifests: &[Option<&ManifestFile>],
    ) -> Result<Statistic> {
        let mut lower = Vec::with_capacity(spec_manifests.len());
        let mut upper = Vec::with_capacity(spec_manifests.len());
        let mut all_null = Vec::with_capacity(spec_manifests.len());
        for manifest in spec_manifests {
            // This field's summary in the manifest.
            let summary = match manifest.and_then(|manifest| manifest.partitions.as_ref()) {
                Some(partition_fields) => {
                    Some(partition_fields.get(self.position).ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "manifest summary does not match its partition spec",
                        )
                    })?)
                }
                None => None,
            };
            lower
                .push(self.decode_bound(summary.and_then(|summary| summary.lower_bound.as_ref()))?);
            upper
                .push(self.decode_bound(summary.and_then(|summary| summary.upper_bound.as_ref()))?);
            // A summary without bounds has no non-null value to bound.
            all_null.push(summary.map(|summary| {
                summary.contains_null
                    && summary.lower_bound.is_none()
                    && summary.upper_bound.is_none()
            }));
        }
        Ok(Statistic {
            column: self.column_idx.into(),
            transform: self.transform,
            bounds: Bounds {
                lower: build_array(
                    &self.data_type,
                    lower.iter().map(|bound| bound.as_ref().map(Datum::literal)),
                ),
                upper: build_array(
                    &self.data_type,
                    upper.iter().map(|bound| bound.as_ref().map(Datum::literal)),
                ),
                all_null: all_null.into_iter().collect(),
            },
        })
    }

    fn decode_bound(&self, bytes: Option<&iceberg::spec::ByteBuf>) -> Result<Option<Datum>> {
        bytes
            .map(|bytes| Datum::try_from_bytes(bytes, self.value_type.clone()))
            .transpose()
    }

    /// This field's value in each data file. Every row of a file shares it, so
    /// it is both bounds. `spec_files` has one slot per file, filled for those
    /// written under the field's spec; the others are unknown.
    pub fn file_statistic(&self, spec_files: &[Option<&LiveFile>]) -> Result<Statistic> {
        let mut values: Vec<Option<&Option<Literal>>> = Vec::with_capacity(spec_files.len());
        for file in spec_files {
            let value = match file {
                Some(file) => {
                    let partition = file.data.partition().fields();
                    Some(partition.get(self.position).ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "file partition does not match its spec",
                        )
                    })?)
                }
                None => None,
            };
            values.push(value);
        }
        let bounds = build_array(
            &self.data_type,
            values.iter().map(|value| match value {
                Some(Some(Literal::Primitive(value))) => Some(value),
                _ => None,
            }),
        );
        Ok(Statistic {
            column: self.column_idx.into(),
            transform: self.transform,
            bounds: Bounds {
                lower: bounds.clone(),
                upper: bounds,
                // A NULL partition value means the column is NULL in every row.
                all_null: values
                    .iter()
                    .map(|value| Some((*value)?.is_none()))
                    .collect(),
            },
        })
    }
}

fn partition_transform(transform: &iceberg::spec::Transform) -> Option<Transform> {
    use iceberg::spec::Transform as Iceberg;
    Some(match transform {
        Iceberg::Identity => Transform::Identity,
        Iceberg::Bucket(count) => Transform::Bucket(*count),
        Iceberg::Truncate(width) => Transform::Truncate(*width),
        Iceberg::Year => Transform::Year,
        Iceberg::Month => Transform::Month,
        Iceberg::Day => Transform::Day,
        Iceberg::Hour => Transform::Hour,
        Iceberg::Void | Iceberg::Unknown => return None,
    })
}

#[cfg(test)]
mod tests;
