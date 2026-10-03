//! Iceberg partition metadata as statistics: a partition field is a transform
//! of a table column, and its values bound that transform in a manifest or a
//! data file.

use ::pruning::{Bounds, Statistic, Transform};
use arrow_schema::DataType;
use iceberg::spec::{
    Datum, FieldSummary, Literal, ManifestFile, PartitionSpec, PrimitiveType, SchemaRef,
};
use iceberg::{Error, ErrorKind, Result};
use planner::types::physical_arrow_type;
use std::collections::BTreeMap;

use super::LiveFile;
use crate::columns::to_pivot_type;
use crate::values::build_array;

/// One field of a partition spec.
pub(super) struct PartitionField {
    /// The table column the field is derived from.
    pub column_idx: usize,
    pub transform: Transform,
    /// Where a partition tuple of the field's spec holds its value.
    position: usize,
    /// The type of the transformed values of the column as it is typed today.
    /// Values written under a narrower column type promote into it.
    value_type: PrimitiveType,
    data_type: DataType,
}

/// The fields of `spec` whose values can prune: a transform Pivot evaluates, of
/// a column the current schema still has.
fn fields(schema: &SchemaRef, spec: &PartitionSpec) -> Vec<PartitionField> {
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

/// The prunable fields of the partition specs a table's objects were written
/// under, by spec id. A table that changed its partitioning has several, and
/// each manifest and data file follows the one it was written under.
#[derive(Default)]
pub(super) struct PartitionSpecs {
    fields_by_spec: BTreeMap<i32, Vec<PartitionField>>,
}

impl PartitionSpecs {
    pub fn insert(&mut self, schema: &SchemaRef, spec: &PartitionSpec) {
        self.fields_by_spec
            .entry(spec.spec_id())
            .or_insert_with(|| fields(schema, spec));
    }

    /// One field for each distinct transform of a column among the specs.
    pub fn distinct_fields(&self) -> Vec<&PartitionField> {
        let mut distinct: Vec<&PartitionField> = Vec::new();
        for field in self.fields_by_spec.values().flatten() {
            if !distinct.iter().any(|other| other.is_same_as(field)) {
                distinct.push(field);
            }
        }
        distinct
    }

    /// Where a partition tuple written under `spec_id` holds the same
    /// transform of the same column as `field`, if that spec has it.
    fn position(&self, spec_id: i32, field: &PartitionField) -> Option<usize> {
        let same = self.fields_by_spec[&spec_id]
            .iter()
            .find(|other| other.is_same_as(field))?;
        Some(same.position)
    }
}

impl PartitionField {
    fn is_same_as(&self, other: &Self) -> bool {
        self.column_idx == other.column_idx && self.transform == other.transform
    }

    fn statistic(&self, bounds: Bounds) -> Statistic {
        Statistic {
            column: self.column_idx.into(),
            transform: self.transform,
            bounds,
        }
    }

    /// The range of this field's values in each manifest, from the summary its
    /// manifest-list entry keeps per partition field. A manifest whose spec
    /// lacks the field, or that has no summaries, is unknown.
    pub fn manifest_statistic(
        &self,
        specs: &PartitionSpecs,
        manifests: &[ManifestFile],
    ) -> Result<Statistic> {
        let mut summaries: Vec<Option<&FieldSummary>> = Vec::with_capacity(manifests.len());
        for manifest in manifests {
            let position = specs.position(manifest.partition_spec_id, self);
            let summary = match (position, &manifest.partitions) {
                (Some(position), Some(summaries)) => {
                    Some(summaries.get(position).ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "manifest summary does not match its partition spec",
                        )
                    })?)
                }
                _ => None,
            };
            summaries.push(summary);
        }
        let mut lower = Vec::with_capacity(summaries.len());
        let mut upper = Vec::with_capacity(summaries.len());
        for summary in &summaries {
            lower
                .push(self.decode_bound(summary.and_then(|summary| summary.lower_bound.as_ref()))?);
            upper
                .push(self.decode_bound(summary.and_then(|summary| summary.upper_bound.as_ref()))?);
        }
        let literals = |bounds: &[Option<Datum>]| {
            build_array(
                &self.data_type,
                bounds
                    .iter()
                    .map(|bound| bound.as_ref().map(Datum::literal)),
            )
        };
        Ok(self.statistic(Bounds {
            lower: literals(&lower),
            upper: literals(&upper),
            // A summary without bounds has no non-null value to bound.
            all_null: summaries
                .iter()
                .map(|summary| {
                    let summary = (*summary)?;
                    Some(
                        summary.contains_null
                            && summary.lower_bound.is_none()
                            && summary.upper_bound.is_none(),
                    )
                })
                .collect(),
        }))
    }

    fn decode_bound(&self, bytes: Option<&iceberg::spec::ByteBuf>) -> Result<Option<Datum>> {
        bytes
            .map(|bytes| Datum::try_from_bytes(bytes, self.value_type.clone()))
            .transpose()
    }

    /// This field's value in each data file. Every row of a file shares it, so
    /// it is both bounds. A file whose spec lacks the field is unknown.
    pub fn file_statistic(&self, specs: &PartitionSpecs, files: &[LiveFile]) -> Result<Statistic> {
        let mut values: Vec<Option<&Option<Literal>>> = Vec::with_capacity(files.len());
        for file in files {
            let spec_id = file.manifest.metadata().partition_spec.spec_id();
            let value = match specs.position(spec_id, self) {
                Some(position) => Some(file.data.partition().fields().get(position).ok_or_else(
                    || {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "file partition does not match its spec",
                        )
                    },
                )?),
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
        Ok(self.statistic(Bounds {
            lower: bounds.clone(),
            upper: bounds,
            // A NULL partition value means the column is NULL in every row.
            all_null: values
                .iter()
                .map(|value| Some((*value)?.is_none()))
                .collect(),
        }))
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
