//! Partition projection and Arrow statistics for manifest summaries and file tuples.

use arrow_schema::DataType;
use iceberg::expr::{
    BinaryExpression, Bind, BoundPredicate, Predicate, PredicateOperator, Reference,
};
use iceberg::spec::{
    DataFile, Datum, ManifestFile, PartitionSpec, PrimitiveLiteral, PrimitiveType, SchemaRef,
};
use iceberg::{Error, ErrorKind, Result};
use parquet_engine::pruning::{ColumnBounds, PruningPredicate, StatisticsBatch};
use planner::expression::CompareType;
use planner::types::physical_arrow_type;

use super::ColumnPredicate;
use crate::columns::to_pivot_type;
use crate::values::{build_array, build_scalar};

struct PartitionColumn {
    position: usize,
    primitive: PrimitiveType,
    data_type: DataType,
}

pub(super) struct PartitionFilter {
    columns: Vec<PartitionColumn>,
    pub(super) predicate: PruningPredicate,
}

impl PartitionFilter {
    pub(super) fn project(
        predicates: &[ColumnPredicate],
        schema: &SchemaRef,
        spec: &PartitionSpec,
    ) -> Result<PartitionFilter> {
        let mut columns = Vec::new();
        let mut projected_predicates = Vec::new();
        for predicate in predicates {
            let Some(source) = schema.field_by_id(predicate.field_id) else {
                continue;
            };
            // A narrowing float conversion can round the constant or fold a
            // comparison outside the numeric range without accounting for NaNs.
            // Keep the original, wider comparison in the file-metrics evaluator.
            if source.field_type.is_floating_type()
                && source.field_type.as_primitive_type() != Some(predicate.datum.data_type())
            {
                continue;
            }
            let bound = Predicate::Binary(BinaryExpression::new(
                predicate_operator(predicate.compare),
                Reference::new(&source.name),
                predicate.datum.clone(),
            ))
            .bind(schema.clone(), true)?;
            if matches!(bound, BoundPredicate::AlwaysFalse) {
                return Ok(PartitionFilter {
                    columns: Vec::new(),
                    predicate: PruningPredicate::Always(false),
                });
            }
            for (position, field) in spec
                .fields()
                .iter()
                .enumerate()
                .filter(|(_, field)| field.source_id == source.id)
            {
                let Some(projected) = field.transform.project(&field.name, &bound)? else {
                    continue;
                };
                let result_type = field.transform.result_type(&source.field_type)?;
                let Some(pivot_type) = to_pivot_type(&result_type) else {
                    continue;
                };
                let data_type = physical_arrow_type(&pivot_type);
                let primitive = result_type
                    .as_primitive_type()
                    .expect("a supported partition type is primitive")
                    .clone();
                projected_predicates.push(project_predicate(projected, columns.len(), &data_type));
                columns.push(PartitionColumn {
                    position,
                    primitive,
                    data_type,
                });
            }
        }
        Ok(PartitionFilter {
            columns,
            predicate: PruningPredicate::And(projected_predicates),
        })
    }

    pub(super) fn manifest_statistics(
        &self,
        manifests: &[&ManifestFile],
    ) -> Result<StatisticsBatch> {
        let columns = self
            .columns
            .iter()
            .map(|column| {
                let summaries = manifests
                    .iter()
                    .map(|manifest| {
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
                                .map(|bytes| Datum::try_from_bytes(bytes, column.primitive.clone()))
                                .transpose()
                        })
                        .collect::<Result<Vec<_>>>()
                };
                let lower = decode(false)?;
                let upper = decode(true)?;
                let floating = matches!(
                    column.primitive,
                    PrimitiveType::Float | PrimitiveType::Double
                );
                Ok(ColumnBounds {
                    lower: Some(build_array(
                        &column.data_type,
                        lower.iter().map(|datum| datum.as_ref().map(Datum::literal)),
                    )),
                    upper: Some(build_array(
                        &column.data_type,
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
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(StatisticsBatch {
            len: manifests.len(),
            columns,
        })
    }

    pub(super) fn file_statistics(&self, files: &[&DataFile]) -> Result<StatisticsBatch> {
        let columns = self
            .columns
            .iter()
            .map(|column| {
                let values = files
                    .iter()
                    .map(|file| {
                        file.partition()
                            .fields()
                            .get(column.position)
                            .ok_or_else(|| {
                                Error::new(
                                    ErrorKind::DataInvalid,
                                    "file partition does not match its spec",
                                )
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let array = build_array(
                    &column.data_type,
                    values.iter().map(|value| {
                        value.as_ref().and_then(|literal| {
                            if let iceberg::spec::Literal::Primitive(value) = literal {
                                Some(value)
                            } else {
                                None
                            }
                        })
                    }),
                );
                Ok(ColumnBounds {
                    lower: Some(array.clone()),
                    upper: Some(array),
                    all_null: Some(values.iter().map(|value| value.is_none()).collect()),
                    nan_free: Some(
                        values
                            .iter()
                            .map(|value| match value {
                                Some(iceberg::spec::Literal::Primitive(
                                    PrimitiveLiteral::Float(value),
                                )) => !value.is_nan(),
                                Some(iceberg::spec::Literal::Primitive(
                                    PrimitiveLiteral::Double(value),
                                )) => !value.is_nan(),
                                _ => true,
                            })
                            .collect(),
                    ),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(StatisticsBatch {
            len: files.len(),
            columns,
        })
    }
}

fn project_predicate(
    predicate: Predicate,
    column: usize,
    data_type: &DataType,
) -> PruningPredicate {
    match predicate {
        Predicate::AlwaysTrue => PruningPredicate::Always(true),
        Predicate::AlwaysFalse => PruningPredicate::Always(false),
        Predicate::Binary(expression) => {
            let compare = match expression.op() {
                PredicateOperator::Eq => CompareType::Equal,
                PredicateOperator::NotEq => CompareType::NotEqual,
                PredicateOperator::LessThan => CompareType::Less,
                PredicateOperator::LessThanOrEq => CompareType::LessEqual,
                PredicateOperator::GreaterThan => CompareType::Greater,
                PredicateOperator::GreaterThanOrEq => CompareType::GreaterEqual,
                _ => return PruningPredicate::Always(true),
            };
            match build_scalar(data_type, expression.literal().literal()) {
                Some(value) => PruningPredicate::Compare {
                    column,
                    compare,
                    value,
                },
                None => PruningPredicate::Always(true),
            }
        }
        Predicate::Set(expression) if expression.op() == PredicateOperator::In => {
            PruningPredicate::Or(
                expression
                    .literals()
                    .iter()
                    .map(|literal| match build_scalar(data_type, literal.literal()) {
                        Some(value) => PruningPredicate::Compare {
                            column,
                            compare: CompareType::Equal,
                            value,
                        },
                        None => PruningPredicate::Always(true),
                    })
                    .collect(),
            )
        }
        // Unsupported projections cannot prove that a partition is irrelevant.
        _ => PruningPredicate::Always(true),
    }
}

fn predicate_operator(compare: CompareType) -> PredicateOperator {
    match compare {
        CompareType::Equal => PredicateOperator::Eq,
        CompareType::NotEqual => PredicateOperator::NotEq,
        CompareType::Less => PredicateOperator::LessThan,
        CompareType::LessEqual => PredicateOperator::LessThanOrEq,
        CompareType::Greater => PredicateOperator::GreaterThan,
        CompareType::GreaterEqual => PredicateOperator::GreaterThanOrEq,
    }
}
