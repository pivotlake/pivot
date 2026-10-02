//! The file set's immutable Arrow bounds, rebuilt with each catalog snapshot.
//! Delta entries remain the persistence/compaction representation; queries read
//! these arrays directly instead of converting their per-file values.

use ::pruning::{
    ColumnBounds, ColumnStatistics, PartitionExpression, PartitionStatistics, PartitionTransform,
    StatisticsBatch,
};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef, BooleanArray, new_null_array};
use arrow_schema::DataType;
use planner::catalog::Column;
use planner::types::physical_arrow_type;
use std::collections::BTreeMap;

use crate::manifest::DeltaFileEntry;

pub(super) fn file_statistics<'a>(
    columns: &[Column],
    partition_by: &[String],
    files: impl IntoIterator<Item = &'a DeltaFileEntry>,
) -> StatisticsBatch {
    let files: Vec<_> = files.into_iter().collect();
    let stats: Vec<_> = files.iter().map(|file| file.stats.as_deref()).collect();
    let mut bounds = BTreeMap::new();
    let mut partitions = Vec::new();
    for (index, column) in columns.iter().enumerate() {
        // VARIANT paths are pruned only with per-file Parquet row-group metadata.
        if matches!(column.col_type, planner::types::Type::Variant) {
            continue;
        }
        let data_type = physical_arrow_type(&column.col_type);
        bounds.insert(
            index,
            ColumnStatistics::from(ColumnBounds {
                validity: None,
                lower: Some(collect_bounds(
                    &data_type,
                    stats
                        .iter()
                        .map(|stats| stats.and_then(|stats| stats.min_values.get(&column.name))),
                )),
                upper: Some(collect_bounds(
                    &data_type,
                    stats
                        .iter()
                        .map(|stats| stats.and_then(|stats| stats.max_values.get(&column.name))),
                )),
                all_null: Some(
                    stats
                        .iter()
                        .map(|stats| {
                            stats.and_then(|stats| {
                                Some(*stats.null_counts.get(&column.name)? == stats.num_records?)
                            })
                        })
                        .collect(),
                ),
                // Delta's min/max do not establish that floats contain no NaNs.
                nan_free: None,
            }),
        );
        if partition_by.contains(&column.name) {
            let values: Vec<_> = files
                .iter()
                .map(|file| {
                    file.partition
                        .as_ref()?
                        .get(&column.name)
                        .map(|value| value.clone().into_inner())
                })
                .collect();
            let array = collect_bounds(&data_type, values.iter().map(Option::as_ref));
            let nan_free = (0..array.len())
                .map(|row| match array.data_type() {
                    DataType::Float32 => !array.as_primitive::<Float32Type>().value(row).is_nan(),
                    DataType::Float64 => !array.as_primitive::<Float64Type>().value(row).is_nan(),
                    _ => true,
                })
                .collect::<BooleanArray>();
            partitions.push(PartitionStatistics {
                expression: PartitionExpression {
                    source: index.into(),
                    source_type: data_type.clone(),
                    transform: PartitionTransform::Identity,
                },
                bounds: ColumnBounds {
                    validity: None,
                    lower: Some(array.clone()),
                    upper: Some(array),
                    // Missing partition metadata is unknown, not an exact NULL.
                    all_null: Some(
                        values
                            .iter()
                            .map(|value| value.as_ref().map(|array| array.is_null(0)))
                            .collect(),
                    ),
                    nan_free: Some(nan_free),
                },
            });
        }
    }
    StatisticsBatch::new(files.len(), bounds, partitions)
        .expect("file statistics have one slot per file")
}

fn collect_bounds<'a>(
    data_type: &DataType,
    values: impl Iterator<Item = Option<&'a ArrayRef>>,
) -> ArrayRef {
    let null = new_null_array(data_type, 1);
    let values: Vec<_> = values
        .map(|value| {
            value
                .filter(|value| value.data_type() == data_type && value.len() == 1)
                .unwrap_or(&null)
                .as_ref()
        })
        .collect();
    if values.is_empty() {
        return new_null_array(data_type, 0);
    }
    arrow_select::concat::concat(&values)
        .unwrap_or_else(|_| new_null_array(data_type, values.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::DeltaFileEntry;
    use ::pruning::ColumnPredicate;
    use arrow_array::{Int64Array, Scalar};
    use object_storage::{FileRef, ObjectPath};
    use parquet_engine::FileStats;
    use planner::expression::{Compare, CompareType, Expression, Ref, TableFilter};
    use planner::types::Type;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn predicate(value: i64) -> Vec<ColumnPredicate> {
        TableFilter::Expression(Box::new(Expression::Compare(Compare {
            left: Box::new(Expression::Ref(Ref {
                column_idx: 0,
                return_type: Type::Int64,
                name: None,
            })),
            right: Box::new(Expression::Constant(Scalar::new(
                Arc::new(Int64Array::from(vec![value])) as ArrayRef,
            ))),
            compare_type: CompareType::Equal,
            return_type: Type::Boolean,
        })))
        .pruning_predicates()
    }

    #[test]
    fn exact_null_partitions_and_missing_partitions_remain_distinct() {
        let columns = [Column {
            name: "part".into(),
            col_type: Type::Int64,
        }];
        let file = |partition| DeltaFileEntry {
            file: FileRef {
                path: ObjectPath::new("unused"),
                size: 0,
            },
            partition,
            stats: None,
        };
        let partition = |value| {
            Some(HashMap::from([(
                "part".into(),
                Scalar::new(Arc::new(Int64Array::from(vec![value])) as ArrayRef),
            )]))
        };
        let files = [
            file(partition(Some(1))),
            file(partition(None)),
            file(None),
            file(Some(HashMap::new())),
        ];
        let statistics = file_statistics(&columns, &["part".into()], &files);
        assert_eq!(
            statistics.prune(&predicate(1)).unwrap(),
            BooleanArray::from(vec![true, false, true, true])
        );
        assert_eq!(
            statistics.prune(&predicate(2)).unwrap(),
            BooleanArray::from(vec![false, false, true, true])
        );
        assert_eq!(
            statistics.prune(&predicate(1)).unwrap(),
            BooleanArray::from(vec![true, false, true, true])
        );
        let exact = &statistics.partition_stats()[0].bounds;
        assert!(Arc::ptr_eq(
            exact.lower.as_ref().unwrap(),
            exact.upper.as_ref().unwrap()
        ));
    }

    #[test]
    fn rebuilding_file_order_does_not_mutate_an_older_statistics_snapshot() {
        let columns = [Column {
            name: "id".into(),
            col_type: Type::Int64,
        }];
        let file = |value: i64| {
            let bound = Arc::new(Int64Array::from(vec![value])) as ArrayRef;
            DeltaFileEntry {
                file: FileRef {
                    path: ObjectPath::new(value.to_string()),
                    size: 0,
                },
                partition: None,
                stats: Some(Arc::new(FileStats {
                    num_records: Some(1),
                    min_values: HashMap::from([("id".into(), bound.clone())]),
                    max_values: HashMap::from([("id".into(), bound)]),
                    null_counts: HashMap::from([("id".into(), 0)]),
                })),
            }
        };
        let old = file_statistics(&columns, &[], &[file(1), file(2)]);
        let new = file_statistics(&columns, &[], &[file(2), file(3)]);
        assert_eq!(
            old.prune(&predicate(1)).unwrap(),
            BooleanArray::from(vec![true, false])
        );
        assert_eq!(
            new.prune(&predicate(1)).unwrap(),
            BooleanArray::from(vec![false, false])
        );
        assert_eq!(
            new.prune(&predicate(2)).unwrap(),
            BooleanArray::from(vec![true, false])
        );
        assert_eq!(
            old.prune(&predicate(2)).unwrap(),
            BooleanArray::from(vec![false, true])
        );
    }

    #[test]
    fn file_bounds_omit_variant_even_when_the_log_has_statistics() {
        let columns = [
            Column {
                name: "doc".into(),
                col_type: Type::Variant,
            },
            Column {
                name: "id".into(),
                col_type: Type::Int64,
            },
        ];
        let bound = Arc::new(Int64Array::from(vec![10])) as ArrayRef;
        let file = DeltaFileEntry {
            file: FileRef {
                path: ObjectPath::new("unused"),
                size: 0,
            },
            partition: None,
            stats: Some(Arc::new(FileStats {
                num_records: Some(1),
                min_values: HashMap::from([("id".into(), bound.clone())]),
                max_values: HashMap::from([("id".into(), bound)]),
                null_counts: HashMap::from([("doc".into(), 1), ("id".into(), 0)]),
            })),
        };
        let statistics = file_statistics(&columns, &[], &[file]);
        assert_eq!(statistics.column_stats().len(), 1);
        let mut predicates = predicate(20);
        // Neither the whole VARIANT nor a path uses file-level null proofs.
        assert!(statistics.prune(&predicates).unwrap().value(0));
        predicates[0].path = vec!["price".into()];
        predicates[0].as_type = Some(DataType::Int64);
        assert!(statistics.prune(&predicates).unwrap().value(0));
        predicates[0].column_idx = 1;
        predicates[0].path.clear();
        predicates[0].as_type = None;
        // Skipping a VARIANT column preserves the other columns' logical indexes.
        assert!(!statistics.prune(&predicates).unwrap().value(0));
    }
}
