//! Parquet adapters for static pruning and decoder predicates. Logical SQL
//! filters are translated in the planner; bounds evaluation lives in `pruning`.

use super::types::leaves::{
    first_leaf, leaf_fields, variant_shredded_leaves, variant_shredded_paths,
    variant_statistics_are_null,
};
use super::types::metadata::{DecodedLeafStatistics, RowGroupMetadata};
use crate::{FileRowGroups, ScanEqualityPredicate};
use arrow_array::Datum;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Fields};
use pruning::{ColumnBounds, ColumnPredicate, ColumnStatistics, Comparison, VariantPathStatistics};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Group footer bounds by logical table column. VARIANT paths share the typed
/// leaf arrays and carry per-group coverage of their unshredded fallbacks.
pub(crate) fn statistics_columns(
    fields: &Fields,
    leaves: &[Option<DecodedLeafStatistics>],
    row_counts: &[i64],
) -> BTreeMap<usize, ColumnStatistics> {
    let bounds = |leaf: &DecodedLeafStatistics| ColumnBounds {
        lower: leaf.min.clone(),
        upper: leaf.max.clone(),
        all_null: Some(
            leaf.null_counts
                .iter()
                .zip(row_counts)
                .map(|(nulls, rows)| nulls.map(|nulls| nulls == *rows))
                .collect(),
        ),
        ..Default::default()
    };
    let mut columns = BTreeMap::new();
    let physical = leaf_fields(fields);
    for (column_idx, field) in fields.iter().enumerate() {
        if !crate::is_variant_field(field) {
            if let Some(Some(leaf)) = leaves.get(first_leaf(fields, column_idx)) {
                columns.insert(column_idx, ColumnStatistics::from(bounds(leaf)));
            }
            continue;
        }
        let mut paths = BTreeMap::new();
        for path in variant_shredded_paths(field) {
            let Some(resolved) = variant_shredded_leaves(fields, column_idx, &path) else {
                continue;
            };
            let Some(leaf) = &leaves[resolved.typed_leaf] else {
                continue;
            };
            let data_type = physical[resolved.typed_leaf].data_type();
            let validity = NullBuffer::from(
                row_counts
                    .iter()
                    .enumerate()
                    .map(|(row, &num_rows)| {
                        resolved
                            .value_leaves
                            .iter()
                            .enumerate()
                            .all(|(level, &leaf)| {
                                leaves[leaf].as_ref().is_some_and(|statistics| {
                                    variant_statistics_are_null(
                                        &statistics.row(row),
                                        num_rows,
                                        data_type,
                                        level + 1 == resolved.value_leaves.len(),
                                    )
                                })
                            })
                    })
                    .collect::<Vec<_>>(),
            );
            let mut bounds = bounds(leaf);
            bounds.validity = (validity.null_count() != 0).then_some(validity);
            paths.insert(
                path,
                VariantPathStatistics {
                    data_type: data_type.clone(),
                    bounds,
                },
            );
        }
        if !paths.is_empty() {
            columns.insert(column_idx, ColumnStatistics::Variant(paths));
        }
    }
    columns
}

/// The equality predicates among `predicates`, in the shape a scan applies
/// per row group (dictionary pruning and batch pre-filtering). NaN constants
/// stay with SQL because decoder dictionary lookup uses native float equality.
pub fn equality_predicates(predicates: &[ColumnPredicate]) -> Vec<ScanEqualityPredicate> {
    predicates
        .iter()
        .filter(|predicate| matches!(predicate.compare_type, Comparison::Equal))
        // An empty decoder path denotes a plain column, not a typed VARIANT root.
        .filter(|predicate| !predicate.path.is_empty() || predicate.as_type.is_none())
        .filter(|predicate| {
            let (value, _) = predicate.value.get();
            match value.data_type() {
                DataType::Float32 => !value.as_primitive::<Float32Type>().value(0).is_nan(),
                DataType::Float64 => !value.as_primitive::<Float64Type>().value(0).is_nan(),
                _ => true,
            }
        })
        .map(|predicate| ScanEqualityPredicate {
            column_idx: predicate.column_idx,
            path: predicate.path.clone(),
            value: predicate.value.clone(),
        })
        .collect()
}

impl FileRowGroups {
    /// Select this file's row groups using its own schema and bounds. The
    /// captured metadata stays unchanged and surviving groups keep their order.
    pub fn prune(&self, predicates: &[ColumnPredicate]) -> Vec<Arc<RowGroupMetadata>> {
        let row_groups = self.row_groups();
        if predicates.is_empty() {
            return row_groups.to_vec();
        }
        let Some(first) = row_groups.first() else {
            return Vec::new();
        };
        let Ok(keep) = first.statistics.bounds.prune(predicates) else {
            // Unusable statistics cannot prove that any group is empty.
            return row_groups.to_vec();
        };
        row_groups
            .iter()
            .filter(|group| keep.value(group.file_row_group_idx))
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests;
