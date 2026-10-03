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

/// One statistic for each distinct transform of a column among the partition
/// fields of `rows`. A row is bounded by the field of its own spec, wherever
/// that spec holds it, and is unknown when its spec has no such field.
pub(super) fn statistics<'a, Row>(
    rows: &'a [Row],
    fields_of: impl Fn(&'a Row) -> &'a Vec<PartitionField>,
    bounds_of: impl Fn(&PartitionField, &[Option<(&'a PartitionField, &'a Row)>]) -> Result<Bounds>,
) -> Result<Vec<Statistic>> {
    let mut statistics: Vec<Statistic> = Vec::new();
    for field in rows.iter().flat_map(&fields_of) {
        let is_described = statistics.iter().any(|statistic| {
            statistic.column.column_idx == field.column_idx
                && statistic.transform == field.transform
        });
        if is_described {
            continue;
        }
        let entries: Vec<_> = rows
            .iter()
            .map(|row| {
                let same = fields_of(row).iter().find(|other| {
                    other.column_idx == field.column_idx && other.transform == field.transform
                })?;
                Some((same, row))
            })
            .collect();
        statistics.push(Statistic {
            column: field.column_idx.into(),
            transform: field.transform,
            bounds: bounds_of(field, &entries)?,
        });
    }
    Ok(statistics)
}

impl PartitionField {
    /// The range of this field's values in each manifest, from the summary its
    /// manifest-list entry keeps per partition field.
    pub fn manifest_bounds(
        &self,
        manifests: &[Option<(&PartitionField, &ManifestFile)>],
    ) -> Result<Bounds> {
        let summaries = manifests
            .iter()
            .map(|entry| {
                let Some((field, manifest)) = entry else {
                    return Ok(None);
                };
                let Some(summaries) = &manifest.partitions else {
                    return Ok(None);
                };
                summaries.get(field.position).map(Some).ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "manifest summary does not match its partition spec",
                    )
                })
            })
            .collect::<Result<Vec<Option<&FieldSummary>>>>()?;
        // A summary's bounds skip NaNs, which Pivot orders outside the finite
        // range, so they bound floating values only when it reports none.
        let is_floating = matches!(
            self.value_type,
            PrimitiveType::Float | PrimitiveType::Double
        );
        let nan_free = |summary: &FieldSummary| !is_floating || summary.contains_nan == Some(false);
        let decode = |bound: fn(&FieldSummary) -> Option<&iceberg::spec::ByteBuf>| {
            let bounds = summaries
                .iter()
                .map(|summary| {
                    summary
                        .filter(|summary| nan_free(summary))
                        .and_then(bound)
                        .map(|bytes| Datum::try_from_bytes(bytes, self.value_type.clone()))
                        .transpose()
                })
                .collect::<Result<Vec<Option<Datum>>>>()?;
            Ok::<_, Error>(build_array(
                &self.data_type,
                bounds
                    .iter()
                    .map(|bound| bound.as_ref().map(Datum::literal)),
            ))
        };
        Ok(Bounds {
            lower: decode(|summary| summary.lower_bound.as_ref())?,
            upper: decode(|summary| summary.upper_bound.as_ref())?,
            all_null: summaries
                .iter()
                .map(|summary| {
                    let summary = (*summary)?;
                    Some(
                        summary.contains_null
                            && summary.lower_bound.is_none()
                            && summary.upper_bound.is_none()
                            && nan_free(summary),
                    )
                })
                .collect(),
        })
    }

    /// This field's value in each data file. Every row of a file shares it, so
    /// it is both bounds.
    pub fn file_bounds(&self, files: &[Option<(&PartitionField, &LiveFile)>]) -> Result<Bounds> {
        let values = files
            .iter()
            .map(|entry| {
                let Some((field, file)) = entry else {
                    return Ok(None);
                };
                let value = file.data.partition().fields().get(field.position);
                value.map(Some).ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "file partition does not match its spec",
                    )
                })
            })
            .collect::<Result<Vec<Option<&Option<Literal>>>>>()?;
        let bounds = build_array(
            &self.data_type,
            values.iter().map(|value| match value {
                Some(Some(Literal::Primitive(value))) => Some(value),
                _ => None,
            }),
        );
        Ok(Bounds {
            lower: bounds.clone(),
            upper: bounds,
            // A NULL partition value means the column is NULL in every row.
            all_null: values
                .iter()
                .map(|value| Some((*value)?.is_none()))
                .collect(),
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
