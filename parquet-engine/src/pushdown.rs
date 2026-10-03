//! Static filter pushdown for Parquet-backed table bindings.
//!
//! The planner translates a SQL filter into [`Predicate`]s once; a binding
//! reuses them for partition, file, row-group, and decoder pruning. This module
//! describes a file's footer statistics to the `pruning` crate, which decides
//! what they exclude.

use std::borrow::Borrow;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{ArrayRef, BooleanArray, Datum, new_null_array};
use arrow_schema::DataType;
use pruning::{Bounds, ColumnPath, Comparison, Predicate, Statistic, Statistics, Transform};

use super::types::leaves::{
    first_leaf, leaf_fields, variant_shredded_leaves, variant_value_leaf_is_semantically_null,
};
use super::types::metadata::RowGroupMetadata;
use super::types::table::ParquetTable;
use crate::ScanEqualityPredicate;

/// The equality predicates among `predicates`, in the shape a scan applies
/// per row group (dictionary pruning and batch pre-filtering). NaN constants
/// stay with SQL because decoder dictionary lookup uses native float equality.
pub fn equality_predicates(predicates: &[Predicate]) -> Vec<ScanEqualityPredicate> {
    predicates
        .iter()
        .filter(|predicate| predicate.comparison == Comparison::Equal)
        .filter(|predicate| {
            let (value, _) = predicate.value.get();
            match value.data_type() {
                DataType::Float32 => !value.as_primitive::<Float32Type>().value(0).is_nan(),
                DataType::Float64 => !value.as_primitive::<Float64Type>().value(0).is_nan(),
                _ => true,
            }
        })
        .map(|predicate| ScanEqualityPredicate {
            column_idx: predicate.column.column_idx,
            path: predicate.column.path.clone(),
            value: predicate.value.clone(),
        })
        .collect()
}

/// Clone a table and retain only the row groups which the recorded predicates
/// do not eliminate. The clone preserves [`ParquetTable`]'s captured schema
/// even when every row group is removed.
pub fn prune_parquet(parquet: &ParquetTable, predicates: &[Predicate]) -> ParquetTable {
    let mut parquet = parquet.clone();
    *parquet.row_groups_mut() = prune_row_groups(parquet.row_groups(), predicates);
    parquet
}

/// The row groups that may hold a row satisfying every predicate, in order.
pub fn prune_row_groups(
    row_groups: &[Arc<RowGroupMetadata>],
    predicates: &[Predicate],
) -> Vec<Arc<RowGroupMetadata>> {
    if predicates.is_empty() {
        return row_groups.to_vec();
    }
    // A file's row groups share one set of statistics, so each run of
    // consecutive row groups of one file is pruned at once.
    row_groups
        .chunk_by(|previous, next| {
            Arc::ptr_eq(&previous.statistics, &next.statistics)
                && next.file_row_group_idx == previous.file_row_group_idx + 1
        })
        .flat_map(|run| row_group_statistics(run, predicates).select(run, predicates))
        .cloned()
        .collect()
}

/// What a file's footer says about the columns `predicates` read, one slot for
/// each of `row_groups`: consecutive row groups of that file.
pub(crate) fn row_group_statistics<G: Borrow<RowGroupMetadata>>(
    row_groups: &[G],
    predicates: &[Predicate],
) -> Statistics {
    let mut statistics: Vec<Statistic> = Predicate::columns(predicates)
        .into_iter()
        .filter_map(|column| {
            Some(Statistic {
                column: column.clone(),
                transform: Transform::Identity,
                bounds: column_bounds(row_groups, column)?,
            })
        })
        .collect();
    statistics.extend(
        predicates
            .iter()
            .filter_map(|predicate| null_variant_statistic(row_groups, predicate)),
    );
    Statistics::new(row_groups.len(), statistics)
}

/// Which row groups hold only NULLs in the VARIANT column `predicate` reads. A
/// NULL VARIANT is NULL at every path and under every cast, whether or not the
/// file shreds the path, so the statistic takes the type of the constant.
fn null_variant_statistic<G: Borrow<RowGroupMetadata>>(
    row_groups: &[G],
    predicate: &Predicate,
) -> Option<Statistic> {
    let first = row_groups.first()?.borrow();
    let fields = first.schema.fields();
    let field = fields.get(predicate.column.column_idx)?;
    if !crate::is_variant_field(field) {
        return None;
    }
    // Every VARIANT value has metadata, its first leaf, so the leaf is null
    // exactly where the VARIANT is.
    let metadata = first_leaf(fields, predicate.column.column_idx);
    let statistics = first.statistics.get(metadata)?.as_ref()?;
    let unknown = new_null_array(predicate.value.get().0.data_type(), row_groups.len());
    Some(Statistic {
        column: predicate.column.clone(),
        transform: Transform::Identity,
        bounds: Bounds {
            lower: unknown.clone(),
            upper: unknown,
            all_null: row_groups
                .iter()
                .map(|row_group| {
                    let row_group = row_group.borrow();
                    let null_count = statistics.null_counts[row_group.file_row_group_idx];
                    Some(null_count == Some(row_group.num_rows))
                })
                .collect(),
        },
    })
}

/// The bounds of `column` in each row group, or `None` when the file records
/// no statistics for it. A plain column is bounded by its own leaf, and a
/// VARIANT field by the typed leaf the file shreds it into.
fn column_bounds<G: Borrow<RowGroupMetadata>>(
    row_groups: &[G],
    column: &ColumnPath,
) -> Option<Bounds> {
    let fields = row_groups.first()?.borrow().schema.fields();
    let field = fields.get(column.column_idx)?;
    if !crate::is_variant_field(field) {
        if !column.path.is_empty() || field.data_type().is_nested() {
            return None;
        }
        let leaf = first_leaf(fields, column.column_idx);
        return leaf_bounds(row_groups, leaf, field.data_type(), |_| true);
    }
    let leaves = variant_shredded_leaves(fields, column.column_idx, &column.path)?;
    let typed = leaf_fields(fields)[leaves.typed_leaf].data_type().clone();
    let terminal = leaves.value_leaves.len().saturating_sub(1);
    // The typed leaf describes the field only in a row group whose untyped
    // fallbacks hold nothing a cast to the leaf's type would read.
    leaf_bounds(row_groups, leaves.typed_leaf, &typed, |row_group| {
        leaves
            .value_leaves
            .iter()
            .enumerate()
            .all(|(level, &leaf)| {
                variant_value_leaf_is_semantically_null(row_group, leaf, &typed, level == terminal)
            })
    })
}

/// The bounds that the statistics of a leaf of `data_type` give each row group
/// that `is_described` by them, and unknown bounds for the others.
fn leaf_bounds<G: Borrow<RowGroupMetadata>>(
    row_groups: &[G],
    leaf: usize,
    data_type: &DataType,
    is_described: impl Fn(&RowGroupMetadata) -> bool,
) -> Option<Bounds> {
    let first = row_groups.first()?.borrow();
    let statistics = first.statistics.get(leaf)?.as_ref()?;
    let undescribed: BooleanArray = row_groups
        .iter()
        .map(|row_group| Some(!is_described(row_group.borrow())))
        .collect();
    let bound = |bounds: &Option<ArrayRef>| match bounds {
        // One array holds the bound of every row group in the file.
        Some(bounds) => {
            let bounds = bounds.slice(first.file_row_group_idx, row_groups.len());
            if undescribed.has_true() {
                arrow_select::nullif::nullif(&bounds, &undescribed)
                    .expect("one bound per row group")
            } else {
                bounds
            }
        }
        None => new_null_array(data_type, row_groups.len()),
    };
    Some(Bounds {
        lower: bound(&statistics.min),
        upper: bound(&statistics.max),
        all_null: row_groups
            .iter()
            .zip(undescribed.values())
            .map(|(row_group, undescribed)| {
                let row_group = row_group.borrow();
                let null_count = statistics.null_counts[row_group.file_row_group_idx];
                Some(!undescribed && null_count == Some(row_group.num_rows))
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests;
