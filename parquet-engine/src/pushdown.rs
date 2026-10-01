//! Static filter pushdown for Parquet-backed table bindings.
//!
//! Planner filters are translated into [`PushedPredicate`]s once, then reused
//! for partition, file, row-group, and decoder pruning. [`prune_parquet`] uses
//! scalar statistics; [`prune_file_row_groups`] evaluates per-file arrays with
//! the conservative bounds semantics in [`crate::pruning`].

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, Scalar, TimestampMicrosecondArray};
use arrow_schema::{DataType, Fields, TimeUnit};

use planner::expression::{CompareType, Expression, Function, JsonPath, TableFilter};
use planner::types::{Type, UTC_TIMEZONE, physical_arrow_type};

use super::pruning::ColumnBounds;
use super::row_group_stats::row_group_eliminated;
use super::types::leaves::{
    first_leaf, leaf_fields, variant_shredded_leaves, variant_value_leaf_is_semantically_null,
};
use super::types::metadata::RowGroupMetadata;
use super::types::table::ParquetTable;
use crate::ScanEqualityPredicate;

/// A single-column constant comparison offered by DuckDB during filter
/// pushdown. Parquet-backed bindings record it and apply it when they compile,
/// while retaining the upstream SQL filter for correctness.
#[derive(Clone, Debug)]
pub struct PushedPredicate {
    /// The top-level column the comparison reads. For a variant path this is
    /// the variant column; the predicate prunes against the path's shredded
    /// leaf.
    pub column_idx: usize,
    /// The path inside the variant column, empty for a plain column comparison.
    pub path: JsonPath,
    /// The SQL cast's physical output type for a variant path. Pruning against
    /// the raw typed leaf is sound only when this is a semantic identity.
    as_type: Option<arrow_schema::DataType>,
    pub compare_type: CompareType,
    pub value: Scalar<ArrayRef>,
}

impl PushedPredicate {
    /// Recognize the predicate shapes whose Parquet statistics can safely
    /// eliminate row groups. Unsupported shapes are left entirely upstream and
    /// yield no predicates.
    pub fn from_filter(filter: TableFilter) -> Vec<Self> {
        let TableFilter::Expression(expr) = filter else {
            return Vec::new();
        };
        match expr.as_ref() {
            Expression::Compare(compare) => {
                // DuckDB normally canonicalizes the column to the left. Keep
                // constant-left comparisons upstream instead of risking an
                // incorrect direction during metadata pruning.
                let Expression::Constant(constant) = compare.right.as_ref() else {
                    return Vec::new();
                };
                Self::from_bound(&compare.left, compare.compare_type, constant)
                    .into_iter()
                    .collect()
            }
            // DuckDB's filter combiner folds a lower and an upper bound on one
            // column into a single BETWEEN before offering it for pushdown, so
            // a two-sided range always arrives as one expression. Each bound
            // prunes on its own.
            Expression::Between(between) => {
                let (Expression::Constant(lower), Expression::Constant(upper)) =
                    (between.lower.as_ref(), between.upper.as_ref())
                else {
                    return Vec::new();
                };
                let lower_compare = if between.lower_inclusive {
                    CompareType::GreaterEqual
                } else {
                    CompareType::Greater
                };
                let upper_compare = if between.upper_inclusive {
                    CompareType::LessEqual
                } else {
                    CompareType::Less
                };
                [
                    Self::from_bound(&between.input, lower_compare, lower),
                    Self::from_bound(&between.input, upper_compare, upper),
                ]
                .into_iter()
                .flatten()
                .collect()
            }
            _ => Vec::new(),
        }
    }

    /// A single `column <compare_type> constant` bound, when the column side
    /// is one whose statistics can prune.
    fn from_bound(
        column_expr: &Expression,
        compare_type: CompareType,
        constant: &Scalar<ArrayRef>,
    ) -> Option<Self> {
        // DuckDB casts a TIMESTAMP column to TIMESTAMPTZ for a mixed-type
        // comparison. In UTC the cast preserves the microsecond value, so
        // retag the constant as TIMESTAMP for statistics pruning.
        let (column, value) = match column_expr {
            Expression::Cast(cast)
                if cast.target == Type::TimestampTz
                    && matches!(
                        cast.source(),
                        Expression::Ref(reference) if reference.return_type == Type::Timestamp
                    ) =>
            {
                (
                    prunable_column_and_json_path(cast.source())?,
                    as_timestamp_constant(constant)?,
                )
            }
            column => (prunable_column_and_json_path(column)?, constant.clone()),
        };
        Some(Self {
            column_idx: column.column_idx,
            path: column.path,
            as_type: column.as_type,
            compare_type,
            value,
        })
    }

    /// The column-chunk index this predicate's statistics live on in `rg`: the
    /// column's own leaf for a plain predicate, or the shredded typed leaf for
    /// a variant path. `None` means pruning is not sound for this row group.
    fn leaf_for_row_group(&self, rg: &RowGroupMetadata) -> Option<usize> {
        let fields = rg.schema.fields();
        if self.path.is_empty() {
            return Some(first_leaf(fields, self.column_idx));
        }
        let leaves = variant_shredded_leaves(fields, self.column_idx, &self.path)?;
        let target = self
            .as_type
            .as_ref()
            .expect("a variant path predicate has a cast target");
        if leaf_fields(fields)[leaves.typed_leaf].data_type() != target {
            return None;
        }
        let terminal = leaves.value_leaves.len().saturating_sub(1);
        leaves
            .value_leaves
            .iter()
            .enumerate()
            .all(|(level, &leaf)| {
                variant_value_leaf_is_semantically_null(rg, leaf, target, level == terminal)
            })
            .then_some(leaves.typed_leaf)
    }

    /// Resolve the statistics leaf and any VARIANT fallback leaves in this
    /// file's schema. Fallbacks must be semantically null in a row group before
    /// the typed leaf's bounds can exclude it.
    fn resolve_leaf(&self, fields: &Fields) -> Option<(usize, Vec<usize>)> {
        if self.path.is_empty() {
            return Some((first_leaf(fields, self.column_idx), Vec::new()));
        }
        let leaves = variant_shredded_leaves(fields, self.column_idx, &self.path)?;
        let target = self
            .as_type
            .as_ref()
            .expect("a variant path predicate has a cast target");
        if leaf_fields(fields)[leaves.typed_leaf].data_type() != target {
            return None;
        }
        Some((leaves.typed_leaf, leaves.value_leaves))
    }
}

/// The equality predicates among `predicates`, in the shape a scan applies
/// per row group (dictionary pruning and batch pre-filtering).
pub fn equality_predicates(predicates: &[PushedPredicate]) -> Vec<ScanEqualityPredicate> {
    predicates
        .iter()
        .filter(|predicate| matches!(predicate.compare_type, CompareType::Equal))
        .map(|predicate| ScanEqualityPredicate {
            column_idx: predicate.column_idx,
            path: predicate.path.clone(),
            value: predicate.value.clone(),
        })
        .collect()
}

/// Clone a table and retain only the row groups which the recorded predicates
/// do not eliminate. The clone preserves [`ParquetTable`]'s captured schema
/// even when every row group is removed.
pub fn prune_parquet(parquet: &ParquetTable, predicates: &[PushedPredicate]) -> ParquetTable {
    let mut parquet = parquet.clone();
    parquet.row_groups_mut().retain(|rg| {
        !predicates.iter().any(|predicate| {
            predicate
                .leaf_for_row_group(rg.as_ref())
                .is_some_and(|leaf| {
                    row_group_eliminated(
                        rg.as_ref(),
                        leaf,
                        predicate.compare_type,
                        &predicate.value,
                    )
                    .unwrap_or(false)
                })
        })
    });
    parquet
}

/// Select row groups from one file, preserving their order. The groups must
/// share a schema and file statistics; their file-local indexes may have gaps.
/// Call this while metadata is still grouped by file, before assembling the
/// flat scan view used by scanning and materialization.
pub fn prune_file_row_groups(
    row_groups: &[Arc<RowGroupMetadata>],
    predicates: &[PushedPredicate],
) -> Vec<Arc<RowGroupMetadata>> {
    if predicates.is_empty() {
        return row_groups.to_vec();
    }
    let Some(first) = row_groups.first() else {
        return Vec::new();
    };
    let mut keep = vec![true; row_groups.len()];
    for predicate in predicates {
        let Some((leaf, fallback_leaves)) = predicate.resolve_leaf(first.schema.fields()) else {
            continue;
        };
        let Some(statistics) = first.statistics.get(leaf).and_then(Option::as_ref) else {
            continue;
        };
        let len = statistics.null_counts.len();
        let bounds = ColumnBounds {
            lower: statistics.min.clone(),
            upper: statistics.max.clone(),
            nan_free: Some(BooleanArray::from(vec![statistics.nan_free; len])),
            ..Default::default()
        };
        // Unusable statistics cannot prove that a group is empty.
        let Ok(mask) = bounds.may_match(len, predicate.compare_type, &predicate.value) else {
            continue;
        };
        for (keep, row_group) in keep.iter_mut().zip(row_groups) {
            if !*keep {
                continue;
            }
            let fallback_is_null = fallback_leaves.iter().enumerate().all(|(level, &leaf)| {
                variant_value_leaf_is_semantically_null(
                    row_group,
                    leaf,
                    predicate
                        .as_type
                        .as_ref()
                        .expect("a variant path predicate has a cast target"),
                    level + 1 == fallback_leaves.len(),
                )
            });
            let index = row_group.file_row_group_idx;
            if fallback_is_null {
                *keep &=
                    mask.value(index) && statistics.null_counts[index] != Some(row_group.num_rows);
            }
        }
    }
    row_groups
        .iter()
        .zip(keep)
        .filter(|(_, keep)| *keep)
        .map(|(group, _)| group.clone())
        .collect()
}

/// A top-level column and the optional variant path whose statistics can prune.
struct PrunableColumn {
    column_idx: usize,
    path: JsonPath,
    as_type: Option<DataType>,
}

/// Returns the column and optional variant path that can use row-group stats.
fn prunable_column_and_json_path(expr: &Expression) -> Option<PrunableColumn> {
    match expr {
        Expression::Ref(reference) => Some(PrunableColumn {
            column_idx: reference.column_idx,
            path: Vec::new(),
            as_type: None,
        }),
        Expression::Function(Function::VariantGet(read)) if read.as_type.is_some() => {
            match read.input.as_ref() {
                Expression::Ref(reference) => Some(PrunableColumn {
                    column_idx: reference.column_idx,
                    path: read.path.clone(),
                    as_type: read.as_type.as_ref().map(physical_arrow_type),
                }),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Remove the UTC timezone annotation from a one-value TIMESTAMPTZ constant
/// without touching its value buffer. This mirrors DuckDB's UTC
/// TIMESTAMP-to-TIMESTAMPTZ cast in the other direction for statistics only.
fn as_timestamp_constant(value: &Scalar<ArrayRef>) -> Option<Scalar<ArrayRef>> {
    let array = value.clone().into_inner();
    let DataType::Timestamp(TimeUnit::Microsecond, Some(timezone)) = array.data_type() else {
        return None;
    };
    if timezone.as_ref() != UTC_TIMEZONE {
        return None;
    }
    let timestamp = array
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()?
        .clone()
        .with_timezone_opt(None::<Arc<str>>);
    Some(Scalar::new(Arc::new(timestamp) as ArrayRef))
}

#[cfg(test)]
mod tests;
