//! Static filter pushdown for Parquet-backed table bindings.
//!
//! Planner filters are translated into [`PushedPredicate`]s once, then reused
//! for partition, file, row-group, and decoder pruning. The row-group decision
//! itself remains in [`super::row_group_stats`] so static and dynamic pruning
//! share the same min/max semantics.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Datum, Scalar, TimestampMicrosecondArray};
use arrow_schema::{DataType, TimeUnit};

use planner::expression::{CompareType, Expression, Function, JsonPath, TableFilter};
use planner::types::{Type, UTC_TIMEZONE, physical_arrow_type};

use super::row_group_stats::row_group_eliminated;
use super::types::leaves::{
    first_leaf, leaf_fields, leaf_range, variant_shredded_leaves,
    variant_value_leaf_is_semantically_null,
};
use super::types::metadata::RowGroupMetadata;
use super::types::table::ParquetTable;

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
    /// eliminate row groups: a single-column constant comparison, or a
    /// BETWEEN, which contributes one comparison per constant bound.
    /// Unsupported shapes are left entirely upstream.
    pub fn from_filter(filter: TableFilter) -> Vec<Self> {
        let TableFilter::Expression(expr) = filter else {
            return Vec::new();
        };
        match expr.as_ref() {
            Expression::Compare(compare) => match (compare.left.as_ref(), compare.right.as_ref()) {
                (column, Expression::Constant(value)) => {
                    Self::from_comparison(column, compare.compare_type, value)
                        .into_iter()
                        .collect()
                }
                // DuckDB normally canonicalizes the column to the left. Keep
                // constant-left comparisons upstream instead of risking an
                // incorrect direction during metadata pruning.
                _ => Vec::new(),
            },
            Expression::Between(between) => {
                let lower_compare = match between.lower_inclusive {
                    true => CompareType::GreaterEqual,
                    false => CompareType::Greater,
                };
                let upper_compare = match between.upper_inclusive {
                    true => CompareType::LessEqual,
                    false => CompareType::Less,
                };
                [
                    (between.lower.as_ref(), lower_compare),
                    (between.upper.as_ref(), upper_compare),
                ]
                .into_iter()
                .filter_map(|(bound, compare_type)| match bound {
                    Expression::Constant(value) => {
                        Self::from_comparison(between.input.as_ref(), compare_type, value)
                    }
                    _ => None,
                })
                .collect()
            }
            _ => Vec::new(),
        }
    }

    /// One `column <op> constant` comparison, when the column side's
    /// statistics can prune.
    fn from_comparison(
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

/// Decode payload columns only when they outweigh the gate columns by this
/// factor, so the two-wave scan's overhead (the gate columns decode once for
/// the gate and once for the output) stays a small fraction of the saving.
const MIN_PAYLOAD_TO_GATE_BYTE_RATIO: i64 = 4;

/// Whether `PIVOT_SCAN_GATE=off` disables predicate-gated scans, an escape
/// hatch for isolating the gate's effect on a live server. Read once per
/// process.
fn scan_gate_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var_os("PIVOT_SCAN_GATE").is_some_and(|v| v == "off"))
}

/// Decide whether a scan should late-materialize behind its pushed
/// predicates, and plan the gate if so.
///
/// The gate fires only when it is cheap and the deferral is worth it:
/// every gate column must be a plain fixed-width primitive the projection
/// reads whole (so evaluating the comparison is a vectorized kernel over a
/// cheap decode), and the remaining projected columns must carry at least
/// [`MIN_PAYLOAD_TO_GATE_BYTE_RATIO`] times the gate columns' compressed
/// bytes. Whether a surviving row group's request actually masks its rows is
/// decided per row group at run time from the observed selectivity.
pub fn plan_gated_scan(
    parquet: &ParquetTable,
    predicates: &[PushedPredicate],
    projection: &crate::types::projection::Projection,
) -> Option<crate::GatedScanPlan> {
    if scan_gate_disabled() {
        return None;
    }
    let row_groups = parquet.row_groups();
    if row_groups.is_empty() {
        return None;
    }

    let mut gate_columns: Vec<usize> = Vec::new();
    let mut eligible: Vec<&PushedPredicate> = Vec::new();
    for predicate in predicates {
        if !predicate.path.is_empty() || predicate.as_type.is_some() {
            continue;
        }
        // The predicate column must be projected whole somewhere, so the gate
        // scan's decoded column is the raw column the comparison reads.
        let projected_plain =
            projection.indices().iter().enumerate().any(|(pos, &col)| {
                col == predicate.column_idx && projection.extract_at(pos).is_none()
            });
        if !projected_plain {
            continue;
        }
        let value_type = predicate.value.get().0.data_type();
        let cheap_everywhere = row_groups.iter().all(|rg| {
            let fields = rg.schema.fields();
            let field = &fields[predicate.column_idx];
            field.data_type().is_primitive() && field.data_type() == value_type
        });
        if !cheap_everywhere {
            continue;
        }
        eligible.push(predicate);
        if !gate_columns.contains(&predicate.column_idx) {
            gate_columns.push(predicate.column_idx);
        }
    }
    if eligible.is_empty() {
        return None;
    }

    let mut projected_columns: Vec<usize> = projection.indices().to_vec();
    projected_columns.sort_unstable();
    projected_columns.dedup();
    let (mut gate_bytes, mut payload_bytes) = (0i64, 0i64);
    for rg in row_groups {
        let fields = rg.schema.fields();
        for &column in &projected_columns {
            let bytes: i64 = leaf_range(fields, column)
                .map(|leaf| rg.columns[leaf].total_compressed_size)
                .sum();
            if gate_columns.contains(&column) {
                gate_bytes += bytes;
            } else {
                payload_bytes += bytes;
            }
        }
    }
    if payload_bytes < gate_bytes.checked_mul(MIN_PAYLOAD_TO_GATE_BYTE_RATIO)? {
        return None;
    }

    gate_columns.sort_unstable();
    let predicates = eligible
        .into_iter()
        .map(|predicate| crate::GatePredicate {
            batch_column_idx: gate_columns
                .iter()
                .position(|&col| col == predicate.column_idx)
                .expect("every gate predicate's column is a gate column"),
            compare_type: predicate.compare_type,
            value: predicate.value.clone(),
        })
        .collect();
    Some(crate::GatedScanPlan {
        gate_projection: crate::types::projection::Projection::columns(gate_columns),
        predicates,
    })
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
