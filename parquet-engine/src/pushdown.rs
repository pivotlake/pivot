//! Static filter pushdown for Parquet-backed table bindings.
//!
//! Planner filters are translated into [`PushedPredicate`]s once, then reused
//! for partition, file, row-group, and decoder pruning. The row-group decision
//! itself remains in [`super::row_group_stats`] so static and dynamic pruning
//! share the same min/max semantics. A scan also evaluates the filters it can
//! as [`ScanCondition`]s, to drop rows before extracting their variant paths.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Scalar, TimestampMicrosecondArray};
use arrow_schema::{DataType, TimeUnit};
use dispatch::{Projection, VariantExtract};

use planner::expression::{CompareType, Expression, Function, JsonPath, Ref};
use planner::types::{Type, UTC_TIMEZONE, physical_arrow_type};

use super::row_group_stats::row_group_eliminated;
use super::types::leaves::{
    first_leaf, leaf_fields, variant_shredded_leaves, variant_value_leaf_is_semantically_null,
};
use super::types::metadata::RowGroupMetadata;
use super::types::table::ParquetTable;
use crate::{ScanCondition, ScanEqualityPredicate, ScanPushdown};

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
    pub fn from_filter(filter: &Expression) -> Vec<Self> {
        match filter {
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
}

/// The equality predicates among `predicates`, in the shape a scan applies
/// per row group (dictionary pruning and batch pre-filtering).
fn equality_predicates(predicates: &[PushedPredicate]) -> Vec<ScanEqualityPredicate> {
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

impl ScanPushdown {
    /// What a scan of `projection` applies of the `filters` pushed down into
    /// its table.
    pub fn new(filters: &[Expression], projection: &Projection) -> Self {
        let predicates: Vec<PushedPredicate> = filters
            .iter()
            .flat_map(PushedPredicate::from_filter)
            .collect();
        Self {
            eq_predicates: Arc::new(equality_predicates(&predicates)),
            conditions: Arc::new(
                filters
                    .iter()
                    .filter_map(|filter| create_scan_condition(filter, projection))
                    .collect(),
            ),
        }
    }
}

/// `filter` as a condition over the output columns of a scan of `projection`,
/// when it is one such a scan can decide: comparisons, ranges, `IN` lists and
/// NULL checks of columns the scan emits, and boolean combinations of those.
///
/// Evaluating one of these cannot fail, so the scan deciding it on rows the
/// query's `Filter` would never have reached changes no outcome. Every other
/// filter is left to that `Filter` alone.
fn create_scan_condition(filter: &Expression, projection: &Projection) -> Option<ScanCondition> {
    if filter.result_type().ok()? != Type::Boolean {
        return None;
    }
    let mut condition = filter.clone();
    let mut output_columns = Vec::new();
    if !rebind_to_scan_outputs(&mut condition, projection, &mut output_columns) {
        return None;
    }
    // A condition that does not compile is the `Filter`'s to report.
    let evaluator = condition.compile().ok()?;
    Some(ScanCondition::new(evaluator, output_columns))
}

/// Rewrites every column `condition` reads, by its index in the table, into
/// the output column of `projection` holding the same read, collecting those
/// into `output_columns`. A variant path read becomes a plain reference to
/// the output that extracts that path as that type.
///
/// Returns whether the whole condition was rewritten. It is not when it has
/// a shape a scan does not decide, or reads something the scan does not emit.
fn rebind_to_scan_outputs(
    condition: &mut Expression,
    projection: &Projection,
    output_columns: &mut Vec<usize>,
) -> bool {
    let mut rebind =
        |operand: &mut Expression| rebind_to_scan_outputs(operand, projection, output_columns);
    match condition {
        Expression::Constant(_) => true,
        Expression::Ref(column) => {
            let Some(output_idx) = find_output_column(projection, column.column_idx, None) else {
                return false;
            };
            column.column_idx = output_idx;
            output_columns.push(output_idx);
            true
        }
        Expression::Function(Function::VariantGet(read)) => {
            let Expression::Ref(column) = read.input.as_ref() else {
                return false;
            };
            let extract = VariantExtract {
                path: read.path.clone(),
                as_type: read.as_type.as_ref().map(physical_arrow_type),
            };
            let Some(output_idx) =
                find_output_column(projection, column.column_idx, Some(&extract))
            else {
                return false;
            };
            let return_type = read.result_type();
            *condition = Expression::Ref(Ref {
                column_idx: output_idx,
                return_type,
                name: None,
            });
            output_columns.push(output_idx);
            true
        }
        Expression::Compare(compare) => rebind(&mut compare.left) && rebind(&mut compare.right),
        Expression::Between(between) => {
            rebind(&mut between.input) && rebind(&mut between.lower) && rebind(&mut between.upper)
        }
        Expression::InList(list) => rebind(&mut list.input) && list.values.iter_mut().all(rebind),
        Expression::Conjunction(conjunction) => conjunction.children.iter_mut().all(rebind),
        Expression::Not(not) => rebind(&mut not.input),
        Expression::IsNull(is_null) => rebind(&mut is_null.input),
        _ => false,
    }
}

/// The output column of `projection` that reads table column `column_idx`:
/// whole when `extract` is `None`, else through exactly that extract.
fn find_output_column(
    projection: &Projection,
    column_idx: usize,
    extract: Option<&VariantExtract>,
) -> Option<usize> {
    (0..projection.column_indices.len()).find(|&output_idx| {
        projection.column_indices[output_idx] == column_idx
            && projection.extract_at(output_idx) == extract
    })
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
