//! What the Delta log says about each data file, as the statistics that prune
//! whole files before their row groups are looked at.

use ::pruning::{Bounds, Predicate, Statistic, Statistics, Transform};
use arrow_array::{Array, ArrayRef, new_null_array};
use arrow_schema::DataType;
use planner::catalog::Column;
use planner::types::{Type, physical_arrow_type};

use crate::manifest::DeltaFileEntry;

/// The statistics of `files` for the columns `predicates` read, one slot per
/// file: each column's recorded min/max and null count, and the exact value of
/// a partition column. A VARIANT column has neither; its fields are pruned per
/// row group.
pub(super) fn file_statistics<'a>(
    columns: &[Column],
    partition_by: &[String],
    files: impl IntoIterator<Item = &'a DeltaFileEntry>,
    predicates: &[Predicate],
) -> Statistics {
    let files: Vec<_> = files.into_iter().collect();
    let mut statistics = Vec::new();
    for column_path in Predicate::columns(predicates) {
        let column = &columns[column_path.column_idx];
        if !column_path.path.is_empty() || column.col_type == Type::Variant {
            continue;
        }
        let data_type = physical_arrow_type(&column.col_type);
        let mut push = |bounds| {
            statistics.push(Statistic {
                column: column_path.clone(),
                transform: Transform::Identity,
                bounds,
            })
        };
        push(column_bounds(&files, &column.name, &data_type));
        if partition_by.contains(&column.name) {
            push(partition_bounds(&files, &column.name, &data_type));
        }
    }
    Statistics::new(files.len(), statistics)
}

fn column_bounds(files: &[&DeltaFileEntry], column: &str, data_type: &DataType) -> Bounds {
    let stats = || files.iter().map(|file| file.stats.as_deref());
    Bounds {
        lower: collect_values(
            data_type,
            stats().map(|stats| stats?.min_values.get(column)),
        ),
        upper: collect_values(
            data_type,
            stats().map(|stats| stats?.max_values.get(column)),
        ),
        all_null: stats()
            .map(|stats| {
                let stats = stats?;
                Some(*stats.null_counts.get(column)? == stats.num_records?)
            })
            .collect(),
    }
}

/// Every row of a file holds the file's partition value, so it is both bounds.
fn partition_bounds(files: &[&DeltaFileEntry], column: &str, data_type: &DataType) -> Bounds {
    let values: Vec<Option<ArrayRef>> = files
        .iter()
        .map(|file| Some(file.partition.as_ref()?.get(column)?.clone().into_inner()))
        .collect();
    let bounds = collect_values(data_type, values.iter().map(Option::as_ref));
    Bounds {
        lower: bounds.clone(),
        upper: bounds,
        // A file without a recorded value is unknown, not all NULL.
        all_null: values
            .iter()
            .map(|value| value.as_ref().map(|value| value.is_null(0)))
            .collect(),
    }
}

/// One slot per file: its value, or null where the file records none of
/// `data_type`.
fn collect_values<'a>(
    data_type: &DataType,
    values: impl Iterator<Item = Option<&'a ArrayRef>>,
) -> ArrayRef {
    let unknown = new_null_array(data_type, 1);
    let values: Vec<&dyn Array> = values
        .map(|value| {
            value
                .filter(|value| value.data_type() == data_type && value.len() == 1)
                .unwrap_or(&unknown)
                .as_ref()
        })
        .collect();
    if values.is_empty() {
        return new_null_array(data_type, 0);
    }
    arrow_select::concat::concat(&values).expect("arrays of one type concatenate")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::pruning::Comparison;
    use arrow_array::{BooleanArray, Int64Array, Scalar};
    use object_storage::{FileRef, ObjectPath};
    use parquet_engine::FileStats;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn int(value: Option<i64>) -> ArrayRef {
        Arc::new(Int64Array::from(vec![value]))
    }

    fn column(name: &str, col_type: Type) -> Column {
        Column {
            name: name.into(),
            col_type,
        }
    }

    fn equals(column_idx: usize, value: i64) -> Vec<Predicate> {
        vec![Predicate {
            column: column_idx.into(),
            comparison: Comparison::Equal,
            value: Scalar::new(int(Some(value))),
        }]
    }

    fn file(
        partition: Option<HashMap<String, Scalar<ArrayRef>>>,
        stats: Option<FileStats>,
    ) -> DeltaFileEntry {
        DeltaFileEntry {
            file: FileRef {
                path: ObjectPath::new("unused"),
                size: 0,
            },
            partition,
            stats: stats.map(Arc::new),
        }
    }

    /// The statistics of a one-row-per-value file of column `id`.
    fn id_stats(min: Option<i64>, max: Option<i64>, nulls: i64, rows: i64) -> FileStats {
        let bound = |value: Option<i64>| -> HashMap<String, ArrayRef> {
            value
                .map(|value| ("id".to_string(), int(Some(value))))
                .into_iter()
                .collect()
        };
        FileStats {
            num_records: Some(rows),
            min_values: bound(min),
            max_values: bound(max),
            null_counts: HashMap::from([("id".into(), nulls)]),
        }
    }

    #[test]
    fn a_file_is_pruned_by_its_column_range_or_when_the_column_is_all_null() {
        let columns = [column("id", Type::Int64)];
        let files = [
            file(None, Some(id_stats(Some(1), Some(5), 0, 5))),
            file(None, Some(id_stats(Some(6), Some(9), 0, 4))),
            file(None, Some(id_stats(None, None, 3, 3))),
            file(None, None),
        ];
        let predicates = equals(0, 7);

        let keep = file_statistics(&columns, &[], &files, &predicates).prune(&predicates);

        assert_eq!(keep, BooleanArray::from(vec![false, true, false, true]));
    }

    #[test]
    fn a_null_partition_matches_nothing_and_a_missing_one_is_unknown() {
        let columns = [column("part", Type::Int64)];
        let partition = |value| {
            Some(HashMap::from([(
                "part".to_string(),
                Scalar::new(int(value)),
            )]))
        };
        let files = [
            file(partition(Some(1)), None),
            file(partition(Some(2)), None),
            file(partition(None), None),
            file(None, None),
            file(Some(HashMap::new()), None),
        ];
        let predicates = equals(0, 1);

        let keep =
            file_statistics(&columns, &["part".into()], &files, &predicates).prune(&predicates);

        assert_eq!(
            keep,
            BooleanArray::from(vec![true, false, false, true, true])
        );
    }

    #[test]
    fn variant_columns_and_fields_are_left_to_the_row_groups() {
        let columns = [column("doc", Type::Variant), column("id", Type::Int64)];
        let mut stats = id_stats(Some(10), Some(10), 0, 1);
        stats.null_counts.insert("doc".into(), 1);
        let files = [file(None, Some(stats))];
        let mut on_field = equals(0, 20);
        on_field[0].column.path = vec!["price".into()];

        let by_column =
            file_statistics(&columns, &[], &files, &equals(0, 20)).prune(&equals(0, 20));
        let by_field = file_statistics(&columns, &[], &files, &on_field).prune(&on_field);
        let by_id = file_statistics(&columns, &[], &files, &equals(1, 20)).prune(&equals(1, 20));

        assert!(by_column.value(0));
        assert!(by_field.value(0));
        assert!(!by_id.value(0));
    }
}
