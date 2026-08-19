//! The single-column constant comparisons a binding collects from DuckDB, and
//! the row-group pruning they drive.
//!
//! Recorded during binding (the `BoundTable` trait gives no channel from
//! `pushdown_filter` to `compile`), applied at compile time once the row-group
//! metadata exists: min/max stats drop whole row groups, and equality
//! additionally prunes by dictionary contents inside the decoder. The upstream
//! `Filter` always runs, so all of this is a pure optimization.
//!
//! Shared by every binding that scans Parquet — a catalog table and a location
//! read straight from storage alike — so both prune a row group on exactly the
//! same evidence.

use arrow_array::{Array, ArrayRef, Scalar};
use planner::catalog::{Result as CatalogResult, TableRevision};
use planner::expression::{CompareType, Expression, Function, JsonPath, TableFilter};

use crate::parquet::types::leaves::{
    first_leaf, leaf_fields, variant_shredded_leaves, variant_value_leaf_is_semantically_null,
};
use crate::parquet::types::metadata::RowGroupMetadata;
use crate::parquet::{ParquetTable, ScanEqualityPredicate, row_group_eliminated};

/// A single-column constant comparison (`col <cmp> const`) pushed down by
/// DuckDB during binding.
#[derive(Clone, Debug)]
pub(crate) struct PushedPredicate {
    /// The top-level column the comparison reads. For a variant path this is
    /// the variant column; the predicate prunes against the path's shredded
    /// leaf.
    pub(crate) column_idx: usize,
    /// The path inside the variant column (`CAST(col->'a'->'b' AS T) <cmp>
    /// const`), empty for a plain column comparison.
    pub(crate) path: JsonPath,
    /// The SQL cast's physical output type for a variant path. Pruning against
    /// the raw typed leaf is sound only when this is a semantic identity.
    pub(crate) as_type: Option<arrow_schema::DataType>,
    pub(crate) compare_type: CompareType,
    pub(crate) value: Scalar<ArrayRef>,
}

impl PushedPredicate {
    /// The column-chunk index this predicate's statistics live on in `rg`: the
    /// column's own leaf for a plain predicate, or the shredded typed leaf for
    /// a variant path. `None` means pruning isn't sound for this row group
    /// (always safe): the path isn't shredded in this file, or some rows may
    /// hold the path's value in an untyped `value` leaf along the path, where
    /// the typed leaf's statistics can't see them. Stats-based skipping is
    /// allowed only when every such fallback is proven to contribute SQL NULL
    /// for this cast.
    pub(crate) fn get_leaf_for_row_group(&self, rg: &RowGroupMetadata) -> Option<usize> {
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

/// Returns the column and optional variant path that can use row-group stats.
///
/// Plain columns use an empty path. Typed variant reads use the corresponding
/// shredded leaf. Untyped variant reads cannot be compared and are ignored.
fn get_prunable_column_and_json_path(
    expr: &Expression,
) -> Option<(usize, JsonPath, Option<arrow_schema::DataType>)> {
    match expr {
        Expression::Ref(r) => Some((r.column_idx, Vec::new(), None)),
        Expression::Function(Function::VariantGet(read)) if read.as_type.is_some() => {
            match read.input.as_ref() {
                Expression::Ref(r) => Some((
                    r.column_idx,
                    read.path.clone(),
                    read.as_type
                        .as_ref()
                        .map(planner::types::physical_arrow_type),
                )),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Record `filter` on `predicates` if it is a comparison this layer can prune
/// with. Always answers `Ok(false)`: the upstream `Filter` is kept, so what is
/// recorded here can only make the scan read less, never change its rows.
pub(crate) fn record_pushed_filter(
    filter: TableFilter,
    predicates: &mut Vec<PushedPredicate>,
) -> CatalogResult<bool> {
    let TableFilter::Expression(expr) = filter else {
        return Ok(false);
    };
    let Expression::Compare(compare) = expr.as_ref() else {
        return Ok(false);
    };
    let ((column_idx, path, as_type), constant) =
        match (compare.left.as_ref(), compare.right.as_ref()) {
            (column, Expression::Constant(k)) | (Expression::Constant(k), column) => {
                let Some(prunable) = get_prunable_column_and_json_path(column) else {
                    return Ok(false);
                };
                (prunable, k)
            }
            _ => return Ok(false),
        };

    predicates.push(PushedPredicate {
        column_idx,
        path,
        as_type,
        compare_type: compare.compare_type,
        value: constant.clone(),
    });

    Ok(false)
}

/// Clone `parquet`'s row groups and keep only those that survive `predicates` —
/// i.e. what the scan actually reads. A min/max stat that proves no row in a
/// group can match drops it; a stats-comparison error means "can't prune"
/// (kept) — never wrong, just unoptimized. No footer I/O.
pub(crate) fn prune_row_groups(
    parquet: &ParquetTable,
    predicates: &[PushedPredicate],
) -> ParquetTable {
    let mut parquet = parquet.clone();
    parquet.row_groups_mut().retain(|rg| {
        !predicates.iter().any(|p| {
            p.get_leaf_for_row_group(rg.as_ref()).is_some_and(|leaf| {
                row_group_eliminated(rg.as_ref(), leaf, p.compare_type, &p.value).unwrap_or(false)
            })
        })
    });
    parquet
}

/// The equality predicates among `predicates`, as the decoder's dictionary
/// pruning takes them. A variant path reaches its shredded typed leaf only in
/// the files that shred it, so each row group resolves the path itself.
pub(crate) fn equality_predicates(predicates: &[PushedPredicate]) -> Vec<ScanEqualityPredicate> {
    predicates
        .iter()
        .filter(|p| matches!(p.compare_type, CompareType::Equal))
        .map(|p| ScanEqualityPredicate {
            column_idx: p.column_idx,
            path: p.path.clone(),
            value: p.value.clone(),
        })
        .collect()
}

/// The bounds of column `column` across `parquet`'s row groups, or `None` when
/// any of them fails to carry both. Feeds the planner's min/max peephole, so it
/// is only ever asked of an unfiltered view of a table.
pub(crate) fn column_min_max(
    parquet: &ParquetTable,
    column: usize,
) -> Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)> {
    let row_groups = parquet.row_groups();
    if row_groups.is_empty() {
        return None;
    }
    let mut min: Option<Scalar<ArrayRef>> = None;
    let mut max: Option<Scalar<ArrayRef>> = None;
    for rg in row_groups {
        let stats = rg.column_statistics(column)?;
        let (rg_min, rg_max) = (stats.min.as_ref()?, stats.max.as_ref()?);
        min = Some(match min {
            Some(m) if scalar_lt(&m, rg_min) => m,
            _ => rg_min.clone(),
        });
        max = Some(match max {
            Some(m) if scalar_lt(rg_max, &m) => m,
            _ => rg_max.clone(),
        });
    }
    Some((min?, max?))
}

/// `a < b` over two single-value scalars of the same physical type. A null,
/// type mismatch, or kernel error reads as `false`. In [`column_min_max`] every
/// comparison is between two same-typed integer/temporal stat bounds, where the
/// kernel never errors and the bounds are non-null. (Mirrors the stricter,
/// private `scalar_lt` in `parquet::reading::fetching`; worth consolidating.)
fn scalar_lt(a: &Scalar<ArrayRef>, b: &Scalar<ArrayRef>) -> bool {
    arrow_ord::cmp::lt(a, b).is_ok_and(|r| r.len() == 1 && r.is_valid(0) && r.value(0))
}

/// The revision a binding reports for a set of files read straight from
/// storage: the location it was bound from, at no catalog version. Such a
/// binding is never plan-cached, so this only ever names the scan.
pub(crate) fn location_revision(location: &str) -> TableRevision {
    TableRevision {
        identity: location.to_string(),
        version: 0,
    }
}
