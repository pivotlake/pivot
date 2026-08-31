//! Static filter pushdown for Parquet-backed table bindings.
//!
//! Planner filters are translated into [`PushedPredicate`]s once, then reused
//! for partition, file, row-group, and decoder pruning. The row-group decision
//! itself remains in [`super::row_group_stats`] so static and dynamic pruning
//! share the same min/max semantics.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Scalar, TimestampMicrosecondArray};
use arrow_schema::{DataType, TimeUnit};

use planner::expression::{Compare, CompareType, Expression, Function, JsonPath, TableFilter};
use planner::types::{Type, UTC_TIMEZONE, physical_arrow_type};

use super::reading::{ConstantMatch, ScanConstantPredicate};
use super::row_group_stats::row_group_eliminated;
use super::types::leaves::{
    first_leaf, leaf_fields, variant_shredded_leaves, variant_value_leaf_is_semantically_null,
};
use super::types::metadata::RowGroupMetadata;
use super::types::table::ParquetTable;

/// What a pushed predicate applies between its column and its constant.
#[derive(Clone, Copy, Debug)]
pub enum PredicateOperation {
    /// An ordered comparison. Its min/max statistics can eliminate row groups,
    /// and an equality additionally prunes by dictionary at scan time.
    Compare(CompareType),
    /// A substring-flavored string match (`contains`/`prefix`/`suffix`,
    /// DuckDB's rewrites of the single-literal `LIKE` patterns). Statistics
    /// cannot answer it; only dictionary pruning applies. Never
    /// [`ConstantMatch::Equals`]: an equality arrives as
    /// `Compare(CompareType::Equal)`.
    Match(ConstantMatch),
}

/// A single-column constant predicate offered by DuckDB during filter
/// pushdown. Parquet-backed bindings record it and apply it when they compile,
/// while retaining the upstream SQL filter for correctness.
#[derive(Clone, Debug)]
pub struct PushedPredicate {
    /// The top-level column the predicate reads. For a variant path this is
    /// the variant column; the predicate prunes against the path's shredded
    /// leaf.
    pub column_idx: usize,
    /// The path inside the variant column, empty for a plain column comparison.
    pub path: JsonPath,
    /// The SQL cast's physical output type for a variant path. Pruning against
    /// the raw typed leaf is sound only when this is a semantic identity.
    as_type: Option<arrow_schema::DataType>,
    pub operation: PredicateOperation,
    pub value: Scalar<ArrayRef>,
}

impl PushedPredicate {
    /// Recognize the predicate shapes the scan can prune by: constant
    /// comparisons (whose Parquet statistics can safely eliminate row groups)
    /// and substring-flavored string matches (which prune by dictionary).
    /// Unsupported shapes are left entirely upstream.
    pub fn from_filter(filter: TableFilter) -> Option<Self> {
        let TableFilter::Expression(expr) = filter else {
            return None;
        };
        match expr.as_ref() {
            Expression::Compare(compare) => Self::from_compare(compare),
            Expression::Function(function) => Self::from_string_match(function),
            _ => None,
        }
    }

    fn from_compare(compare: &Compare) -> Option<Self> {
        let (column_expr, constant) = match (compare.left.as_ref(), compare.right.as_ref()) {
            (column, Expression::Constant(value)) => (column, value),
            // DuckDB normally canonicalizes the column to the left. Keep
            // constant-left comparisons upstream instead of risking an
            // incorrect direction during metadata pruning.
            _ => return None,
        };
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
            operation: PredicateOperation::Compare(compare.compare_type),
            value,
        })
    }

    /// Recognize `contains(column, 'x')` / `prefix(column, 'x')` /
    /// `suffix(column, 'x')` on a plain string column. Variant paths stay
    /// upstream: a substring match reaches the scan only through these string
    /// functions, whose haystack is a direct column reference.
    fn from_string_match(function: &Function) -> Option<Self> {
        let (haystack, needle, match_type) = match function {
            Function::Contains(contains) => (
                &contains.haystack,
                &contains.needle,
                ConstantMatch::Contains,
            ),
            Function::Prefix(prefix) => {
                (&prefix.haystack, &prefix.prefix, ConstantMatch::StartsWith)
            }
            Function::Suffix(suffix) => (&suffix.haystack, &suffix.suffix, ConstantMatch::EndsWith),
            _ => return None,
        };
        let Expression::Ref(reference) = haystack.as_ref() else {
            return None;
        };
        if reference.return_type != Type::Utf8 {
            return None;
        }
        let Expression::Constant(value) = needle.as_ref() else {
            return None;
        };
        Some(Self {
            column_idx: reference.column_idx,
            path: Vec::new(),
            as_type: None,
            operation: PredicateOperation::Match(match_type),
            value: value.clone(),
        })
    }

    /// This predicate as a decoder-side [`ScanConstantPredicate`] (dictionary
    /// pruning, and batch filtering for an equality), or `None` for a
    /// comparison a dictionary cannot answer.
    pub fn scan_predicate(&self) -> Option<ScanConstantPredicate> {
        let match_type = match self.operation {
            PredicateOperation::Compare(CompareType::Equal) => ConstantMatch::Equals,
            PredicateOperation::Match(match_type) => match_type,
            PredicateOperation::Compare(_) => return None,
        };
        Some(ScanConstantPredicate {
            column_idx: self.column_idx,
            path: self.path.clone(),
            value: self.value.clone(),
            match_type,
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
}

/// Clone a table and retain only the row groups which the recorded predicates
/// do not eliminate. The clone preserves [`ParquetTable`]'s captured schema
/// even when every row group is removed.
pub fn prune_parquet(parquet: &ParquetTable, predicates: &[PushedPredicate]) -> ParquetTable {
    let mut parquet = parquet.clone();
    parquet.row_groups_mut().retain(|rg| {
        !predicates.iter().any(|predicate| {
            // Only a comparison has min/max semantics; a substring match
            // prunes by dictionary at scan time instead.
            let PredicateOperation::Compare(compare_type) = predicate.operation else {
                return false;
            };
            predicate
                .leaf_for_row_group(rg.as_ref())
                .is_some_and(|leaf| {
                    row_group_eliminated(rg.as_ref(), leaf, compare_type, &predicate.value)
                        .unwrap_or(false)
                })
        })
    });
    parquet
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
