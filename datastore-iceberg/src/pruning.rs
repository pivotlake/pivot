//! Iceberg metadata adapters for the engine's Arrow statistics evaluator.
//! Field IDs survive schema evolution; partition transforms remain Iceberg's
//! public API. Object selection preserves manifest and entry order.

use std::collections::HashMap;

use arrow_array::BooleanArray;
use arrow_schema::{ArrowError, DataType};
use iceberg::spec::{
    DataFile, Datum, Manifest, ManifestFile, ManifestMetadata, SchemaRef, TableMetadata,
};
use iceberg::{Error, ErrorKind, Result};
use parquet_engine::PushedPredicate;
use parquet_engine::pruning::{ColumnBounds, PruningPredicate, StatisticsBatch};
use planner::expression::CompareType;
use planner::types::physical_arrow_type;

use crate::columns::to_pivot_type;
use crate::values::{build_array, to_datum};

mod partition;
use partition::PartitionFilter;

struct StatisticsColumn {
    field_id: i32,
    data_type: DataType,
}

struct ColumnPredicate {
    field_id: i32,
    compare: CompareType,
    datum: Datum,
}

/// A conjunction addressed by stable field IDs. Its constants use Pivot's
/// comparison types; statistics from older schemas are promoted to those types.
pub(crate) struct PruningFilter {
    columns: Vec<StatisticsColumn>,
    predicates: Vec<ColumnPredicate>,
    row_filter: PruningPredicate,
}

impl PruningFilter {
    pub(crate) fn new(schema: &SchemaRef, pushed: &[PushedPredicate]) -> Self {
        let mut columns = Vec::new();
        let mut by_id = HashMap::new();
        let mut predicates = Vec::new();
        let mut comparisons = Vec::new();
        for predicate in pushed {
            let field = &schema.as_struct().fields()[predicate.column_idx];
            let (Some(pivot_type), Some(datum)) =
                (to_pivot_type(&field.field_type), to_datum(&predicate.value))
            else {
                continue;
            };
            let column = *by_id.entry(field.id).or_insert_with(|| {
                let index = columns.len();
                columns.push(StatisticsColumn {
                    field_id: field.id,
                    data_type: physical_arrow_type(&pivot_type),
                });
                index
            });
            comparisons.push(PruningPredicate::Compare {
                column,
                compare: predicate.compare_type,
                value: predicate.value.clone(),
            });
            predicates.push(ColumnPredicate {
                field_id: field.id,
                compare: predicate.compare_type,
                datum,
            });
        }
        Self {
            columns,
            predicates,
            row_filter: PruningPredicate::And(comparisons),
        }
    }

    pub(crate) fn select_manifests<'a>(
        &self,
        metadata: &TableMetadata,
        manifests: &'a [ManifestFile],
    ) -> Result<Vec<&'a ManifestFile>> {
        let mut by_spec: HashMap<i32, Vec<usize>> = HashMap::new();
        for (index, manifest) in manifests.iter().enumerate() {
            by_spec
                .entry(manifest.partition_spec_id)
                .or_default()
                .push(index);
        }
        let mut keep = vec![true; manifests.len()];
        for (spec_id, indices) in by_spec {
            let spec = metadata.partition_spec_by_id(spec_id).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("unknown partition spec {spec_id}"),
                )
            })?;
            let projected =
                PartitionFilter::project(&self.predicates, metadata.current_schema(), spec)?;
            let group: Vec<_> = indices.iter().map(|&index| &manifests[index]).collect();
            let statistics = projected.manifest_statistics(&group)?;
            let mask = evaluate(&statistics, &projected.predicate)?;
            for (row, index) in indices.into_iter().enumerate() {
                keep[index] = mask.value(row);
            }
        }
        Ok(manifests
            .iter()
            .zip(keep)
            .filter_map(|(manifest, keep)| keep.then_some(manifest))
            .collect())
    }

    pub(crate) fn select_files<'a>(&self, manifests: &'a [Manifest]) -> Result<Vec<&'a DataFile>> {
        let mut selected = Vec::new();
        for manifest in manifests {
            let files: Vec<_> = manifest
                .entries()
                .iter()
                .filter(|entry| entry.is_alive())
                .map(|entry| entry.data_file())
                .filter(|file| file.record_count() > 0)
                .collect();
            let mask = self.select_file_mask(manifest.metadata(), &files)?;
            selected.extend(
                files
                    .into_iter()
                    .enumerate()
                    .filter_map(|(row, file)| mask.value(row).then_some(file)),
            );
        }
        Ok(selected)
    }

    fn select_file_mask(
        &self,
        metadata: &ManifestMetadata,
        files: &[&DataFile],
    ) -> Result<BooleanArray> {
        let columns = self
            .columns
            .iter()
            .map(|column| {
                let id = column.field_id;
                let absent = metadata.schema.field_by_id(id).is_none();
                ColumnBounds {
                    lower: Some(build_array(
                        &column.data_type,
                        files
                            .iter()
                            .map(|file| file.lower_bounds().get(&id).map(Datum::literal)),
                    )),
                    upper: Some(build_array(
                        &column.data_type,
                        files
                            .iter()
                            .map(|file| file.upper_bounds().get(&id).map(Datum::literal)),
                    )),
                    // Initial defaults are rejected on table load, so an absent field reads as NULL.
                    all_null: Some(
                        files
                            .iter()
                            .map(|file| {
                                absent
                                    || file.null_value_counts().get(&id)
                                        == Some(&file.record_count())
                            })
                            .collect(),
                    ),
                    nan_free: Some(
                        files
                            .iter()
                            .map(|file| file.nan_value_counts().get(&id) == Some(&0))
                            .collect(),
                    ),
                }
            })
            .collect();
        let metrics = evaluate(
            &StatisticsBatch {
                len: files.len(),
                columns,
            },
            &self.row_filter,
        )?;
        let projected =
            PartitionFilter::project(&self.predicates, &metadata.schema, &metadata.partition_spec)?;
        let partitions = evaluate(&projected.file_statistics(files)?, &projected.predicate)?;
        Ok((0..files.len())
            .map(|row| metrics.value(row) && partitions.value(row))
            .collect())
    }
}

fn evaluate(statistics: &StatisticsBatch, predicate: &PruningPredicate) -> Result<BooleanArray> {
    statistics
        .may_match(predicate)
        .map_err(|source: ArrowError| {
            Error::new(ErrorKind::DataInvalid, "could not evaluate metadata bounds")
                .with_source(source)
        })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use arrow_array::{ArrayRef, Int64Array, Scalar};
    use iceberg::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, FormatVersion, ManifestContentType,
        NestedField, PartitionSpec, PrimitiveType, Schema, Type,
    };
    use planner::expression::{Compare, Expression, Ref, TableFilter};
    use planner::types::Type as PivotType;

    fn schema(id: i32, name: &str, primitive: PrimitiveType) -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_fields(vec![Arc::new(NestedField::optional(
                    id,
                    name,
                    Type::Primitive(primitive),
                ))])
                .build()
                .unwrap(),
        )
    }

    fn prune(
        current: &SchemaRef,
        stored: SchemaRef,
        compare: CompareType,
        constant: i64,
    ) -> impl Fn(&DataFile) -> bool {
        let predicates = PushedPredicate::from_filter(TableFilter::Expression(Box::new(
            Expression::Compare(Compare {
                left: Box::new(Expression::Ref(Ref {
                    column_idx: 0,
                    return_type: PivotType::Int64,
                    name: None,
                })),
                right: Box::new(Expression::Constant(Scalar::new(
                    Arc::new(Int64Array::from(vec![constant])) as ArrayRef,
                ))),
                compare_type: compare,
                return_type: PivotType::Boolean,
            }),
        )));
        let metadata = ManifestMetadata {
            schema: stored.clone(),
            schema_id: stored.schema_id(),
            partition_spec: PartitionSpec::builder(stored).build().unwrap(),
            format_version: FormatVersion::V2,
            content: ManifestContentType::Data,
        };
        let filter = PruningFilter::new(current, &predicates);
        move |file| {
            filter
                .select_file_mask(&metadata, &[file])
                .unwrap()
                .value(0)
        }
    }

    fn file(value: Option<Datum>, null_count: Option<u64>) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("file.parquet".into())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(1)
            .record_count(1)
            .value_counts(HashMap::from([(1, 1)]))
            .null_value_counts(null_count.map(|count| (1, count)).into_iter().collect())
            .lower_bounds(value.clone().map(|value| (1, value)).into_iter().collect())
            .upper_bounds(value.map(|value| (1, value)).into_iter().collect())
            .build()
            .unwrap()
    }

    #[test]
    fn renamed_and_promoted_fields_compare_old_metrics_in_the_current_type() {
        let current = schema(1, "renamed", PrimitiveType::Long);
        let stored = schema(1, "original", PrimitiveType::Int);
        let matching = file(Some(Datum::int(5)), Some(0));
        let other = file(Some(Datum::int(3)), Some(0));
        let pruner = prune(&current, stored, CompareType::Greater, 4);

        let kept = pruner(&matching);
        let excluded = pruner(&other);

        assert!(kept);
        assert!(!excluded);
    }

    #[test]
    fn constants_outside_an_older_types_range_prune_safely() {
        let current = schema(1, "id", PrimitiveType::Long);
        let stored = schema(1, "id", PrimitiveType::Int);
        let file = file(Some(Datum::int(5)), Some(0));
        let greater = prune(
            &current,
            stored.clone(),
            CompareType::Greater,
            i64::from(i32::MAX) + 1,
        );
        let less = prune(&current, stored, CompareType::Less, i64::from(i32::MAX) + 1);

        let above = greater(&file);
        let below = less(&file);

        assert!(!above);
        assert!(below);
    }

    #[test]
    fn missing_metrics_and_null_values_have_different_meanings() {
        let schema = schema(1, "id", PrimitiveType::Long);
        let pruner = prune(&schema, schema.clone(), CompareType::Equal, 5);
        let unknown = file(None, None);
        let all_null = file(None, Some(1));

        let unknown_matches = pruner(&unknown);
        let null_matches = pruner(&all_null);

        assert!(unknown_matches);
        assert!(!null_matches);
    }

    #[test]
    fn promoted_float_partition_predicates_do_not_round_the_query_constant() {
        let current = schema(1, "value", PrimitiveType::Double);
        let stored = schema(1, "value", PrimitiveType::Float);
        let partition_spec = PartitionSpec::builder(stored.clone())
            .add_partition_field(
                "value",
                "value_partition",
                iceberg::spec::Transform::Identity,
            )
            .unwrap()
            .build()
            .unwrap();
        let metadata = ManifestMetadata {
            schema: stored.clone(),
            schema_id: stored.schema_id(),
            partition_spec,
            format_version: FormatVersion::V2,
            content: ManifestContentType::Data,
        };
        let predicates = PushedPredicate::from_filter(TableFilter::Expression(Box::new(
            Expression::Compare(Compare {
                left: Box::new(Expression::Ref(Ref {
                    column_idx: 0,
                    return_type: PivotType::Float64,
                    name: None,
                })),
                right: Box::new(Expression::Constant(Scalar::new(Arc::new(
                    arrow_array::Float64Array::from(vec![1.00000005]),
                )
                    as ArrayRef))),
                compare_type: CompareType::NotEqual,
                return_type: PivotType::Boolean,
            }),
        )));
        let file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("file.parquet".into())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(1)
            .record_count(1)
            .partition(iceberg::spec::Struct::from_iter([Some(
                iceberg::spec::Literal::Primitive(Datum::float(1.0_f32).literal().clone()),
            )]))
            .lower_bounds(HashMap::from([(1, Datum::float(1.0_f32))]))
            .upper_bounds(HashMap::from([(1, Datum::float(1.0_f32))]))
            .nan_value_counts(HashMap::from([(1, 0)]))
            .build()
            .unwrap();

        let keep = PruningFilter::new(&current, &predicates)
            .select_file_mask(&metadata, &[&file])
            .unwrap();

        assert!(keep.value(0));
    }

    #[test]
    fn a_column_added_after_a_manifest_was_written_is_null() {
        let current = schema(2, "added", PrimitiveType::Long);
        let stored = schema(1, "original", PrimitiveType::Long);
        let pruner = prune(&current, stored, CompareType::Equal, 5);
        let file = file(Some(Datum::long(5)), Some(0));

        let matches = pruner(&file);

        assert!(!matches);
    }
}
