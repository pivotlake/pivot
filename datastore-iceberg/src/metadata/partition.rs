//! Decode Iceberg partition metadata into shared expressions and Arrow bounds.
//! Predicate projection belongs to the pruning crate.

use ::pruning::{ColumnBounds, PartitionExpression, PartitionTransform};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use iceberg::spec::{
    DataFile, Datum, ManifestFile, PartitionSpec, PrimitiveLiteral, PrimitiveType, SchemaRef,
};
use iceberg::{Error, ErrorKind, Result};
use planner::types::physical_arrow_type;

use crate::columns::to_pivot_type;
use crate::values::build_array;

pub(super) struct PartitionColumn {
    pub expression: PartitionExpression,
    position: usize,
    primitive: PrimitiveType,
    data_type: DataType,
}

pub(super) fn columns(
    current: &SchemaRef,
    stored: &SchemaRef,
    spec: &PartitionSpec,
) -> Result<Vec<PartitionColumn>> {
    let mut columns = Vec::new();
    for (position, field) in spec.fields().iter().enumerate() {
        let Some(transform) = partition_transform(&field.transform) else {
            continue;
        };
        let Some(source_column) = current
            .as_struct()
            .fields()
            .iter()
            .position(|source| source.id == field.source_id)
        else {
            continue;
        };
        let Some(source) = stored.field_by_id(field.source_id) else {
            continue;
        };
        // Unknown transforms cannot supply usable bounds. They must not make
        // an otherwise readable snapshot fail during eager metadata indexing.
        let Ok(result_type) = field.transform.result_type(&source.field_type) else {
            continue;
        };
        let Some(pivot_type) = to_pivot_type(&result_type) else {
            continue;
        };
        let Some(primitive) = result_type.as_primitive_type() else {
            continue;
        };
        let data_type = physical_arrow_type(&pivot_type);
        let Some(source_type) = to_pivot_type(&source.field_type) else {
            continue;
        };
        columns.push(PartitionColumn {
            expression: PartitionExpression {
                source: source_column.into(),
                source_type: physical_arrow_type(&source_type),
                transform,
            },
            position,
            primitive: primitive.clone(),
            data_type,
        });
    }
    Ok(columns)
}

impl PartitionColumn {
    /// Each row carries its own spec's binding, since the same expression may
    /// occur at different positions. Missing bindings are inapplicable bounds.
    pub fn manifest_bounds<'a>(
        &self,
        manifests: impl Iterator<Item = Option<(&'a Self, &'a ManifestFile)>>,
    ) -> Result<ColumnBounds> {
        let manifests: Vec<_> = manifests.collect();
        let validity = applicability(&manifests);
        let summaries = manifests
            .iter()
            .map(|entry| {
                let Some((column, manifest)) = entry else {
                    return Ok(None);
                };
                manifest
                    .partitions
                    .as_ref()
                    .map(|fields| {
                        fields.get(column.position).ok_or_else(|| {
                            Error::new(
                                ErrorKind::DataInvalid,
                                "manifest summary does not match its partition spec",
                            )
                        })
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let decode = |upper| {
            summaries
                .iter()
                .map(|summary| {
                    let bytes = summary.and_then(|summary| {
                        if upper {
                            summary.upper_bound.as_ref()
                        } else {
                            summary.lower_bound.as_ref()
                        }
                    });
                    bytes
                        .map(|bytes| Datum::try_from_bytes(bytes, self.primitive.clone()))
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()
        };
        let lower = decode(false)?;
        let upper = decode(true)?;
        let floating = matches!(self.primitive, PrimitiveType::Float | PrimitiveType::Double);
        Ok(ColumnBounds {
            validity,
            lower: Some(build_array(
                &self.data_type,
                lower.iter().map(|datum| datum.as_ref().map(Datum::literal)),
            )),
            upper: Some(build_array(
                &self.data_type,
                upper.iter().map(|datum| datum.as_ref().map(Datum::literal)),
            )),
            all_null: Some(
                summaries
                    .iter()
                    .map(|summary| {
                        summary.is_some_and(|summary| {
                            summary.contains_null
                                && summary.lower_bound.is_none()
                                && summary.upper_bound.is_none()
                                && (!floating || summary.contains_nan == Some(false))
                        })
                    })
                    .collect(),
            ),
            nan_free: Some(
                summaries
                    .iter()
                    .map(|summary| {
                        summary.is_some_and(|summary| summary.contains_nan == Some(false))
                    })
                    .collect(),
            ),
        })
    }

    pub fn file_bounds<'a>(
        &self,
        files: impl Iterator<Item = Option<(&'a Self, &'a DataFile)>>,
    ) -> Result<ColumnBounds> {
        let files: Vec<_> = files.collect();
        let validity = applicability(&files);
        let values = files
            .iter()
            .map(|entry| {
                let Some((column, file)) = entry else {
                    return Ok(None);
                };
                file.partition()
                    .fields()
                    .get(column.position)
                    .map(Some)
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "file partition does not match its spec",
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let array = build_array(
            &self.data_type,
            values.iter().map(|value| {
                value.and_then(|value| value.as_ref()).and_then(|literal| {
                    if let iceberg::spec::Literal::Primitive(value) = literal {
                        Some(value)
                    } else {
                        None
                    }
                })
            }),
        );
        Ok(ColumnBounds {
            validity,
            lower: Some(array.clone()),
            upper: Some(array),
            all_null: Some(
                values
                    .iter()
                    .map(|value| value.is_some_and(|value| value.is_none()))
                    .collect(),
            ),
            nan_free: Some(
                values
                    .iter()
                    .map(|value| match value.and_then(|value| value.as_ref()) {
                        Some(iceberg::spec::Literal::Primitive(PrimitiveLiteral::Float(value))) => {
                            !value.is_nan()
                        }
                        Some(iceberg::spec::Literal::Primitive(PrimitiveLiteral::Double(
                            value,
                        ))) => !value.is_nan(),
                        _ => true,
                    })
                    .collect(),
            ),
        })
    }
}

fn applicability<T>(rows: &[Option<T>]) -> Option<NullBuffer> {
    let validity = NullBuffer::from(rows.iter().map(Option::is_some).collect::<Vec<_>>());
    (validity.null_count() != 0).then_some(validity)
}

/// One representative per expression, even if several specs contain it or
/// assign it different partition-field IDs and positions.
pub(super) fn distinct<'a>(
    groups: impl Iterator<Item = &'a Vec<PartitionColumn>>,
) -> Vec<&'a PartitionColumn> {
    let mut result: Vec<&PartitionColumn> = Vec::new();
    for column in groups.flatten() {
        if !result
            .iter()
            .any(|other| other.expression == column.expression)
        {
            result.push(column);
        }
    }
    result
}

fn partition_transform(transform: &iceberg::spec::Transform) -> Option<PartitionTransform> {
    use iceberg::spec::Transform;
    Some(match transform {
        Transform::Identity => PartitionTransform::Identity,
        Transform::Bucket(count) => PartitionTransform::Bucket(*count),
        Transform::Truncate(width) => PartitionTransform::Truncate(*width),
        Transform::Year => PartitionTransform::Year,
        Transform::Month => PartitionTransform::Month,
        Transform::Day => PartitionTransform::Day,
        Transform::Hour => PartitionTransform::Hour,
        Transform::Void | Transform::Unknown => return None,
    })
}

#[cfg(test)]
mod tests;
