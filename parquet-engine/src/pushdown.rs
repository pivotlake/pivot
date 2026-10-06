//! Static filter pushdown for Parquet-backed table bindings.
//!
//! Planner filters are translated into [`PushedPredicate`]s once, then reused
//! for partition, file, row-group, and decoder pruning. The row-group decision
//! itself remains in [`super::row_group_stats`] so static and dynamic pruning
//! share the same min/max semantics.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Scalar, TimestampMicrosecondArray};
use arrow_schema::{DataType, TimeUnit};

use planner::expression::{CompareType, ConjunctionOp, Expression, Function, JsonPath};
use planner::types::{Type, UTC_TIMEZONE, physical_arrow_type};

use super::row_group_stats::row_group_eliminated;
use super::types::leaves::{
    first_leaf, leaf_fields, variant_shredded_leaves, variant_value_leaf_is_semantically_null,
};
use super::types::metadata::RowGroupMetadata;
use super::types::table::ParquetTable;
use crate::ScanEqualityPredicate;

/// Each value of an `IN` list (or of an `OR` of equalities) is compared with
/// every row group's bounds, so a longer list is left to the filter above the
/// scan.
const MAX_IN_LIST_VALUES: usize = 200;

/// A single-column comparison with constants offered by DuckDB during filter
/// pushdown: `column <compare_type> value` for any one of `values`. A plain
/// comparison has one value; an `IN` list, or an `OR` of equalities on one
/// column, has one per listed value. Parquet-backed bindings record it and
/// apply it when they compile, while retaining the upstream SQL filter for
/// correctness.
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
    /// The constants compared with, never empty: a row group is eliminated
    /// only when every one of them eliminates it.
    pub values: Vec<Scalar<ArrayRef>>,
}

impl PushedPredicate {
    /// Recognize the predicate shapes whose Parquet statistics can safely
    /// eliminate row groups. Unsupported shapes are left entirely upstream and
    /// yield no predicates.
    pub fn from_filter(filter: &Expression) -> Vec<Self> {
        match filter {
            // A row is in the list when it equals one of its values, so the
            // list prunes as one equality with every value.
            Expression::InList(list) => Self::from_equalities(list.values.iter().map(|value| {
                let Expression::Constant(constant) = value else {
                    return None;
                };
                Self::from_bound(&list.input, CompareType::Equal, constant)
            }))
            .into_iter()
            .collect(),
            // DuckDB spells a short `IN` list as an `OR` of equalities on the
            // column, which prunes exactly like the list.
            Expression::Conjunction(conjunction) if matches!(conjunction.op, ConjunctionOp::Or) => {
                Self::from_equalities(conjunction.children.iter().map(|child| {
                    match Self::from_filter(child).as_slice() {
                        [predicate] if matches!(predicate.compare_type, CompareType::Equal) => {
                            Some(predicate.clone())
                        }
                        _ => None,
                    }
                }))
                .into_iter()
                .collect()
            }
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

    /// The disjunction of `equalities` as one predicate, when every one of
    /// them prunes and they all compare the same column (or variant path): a
    /// row group none of their values can be in is eliminated. A disjunct that
    /// cannot prune, one on another column, or more values than
    /// [`MAX_IN_LIST_VALUES`] leaves the whole disjunction to the filter above
    /// the scan.
    fn from_equalities(equalities: impl Iterator<Item = Option<Self>>) -> Option<Self> {
        let mut merged: Option<Self> = None;
        for equality in equalities {
            let equality = equality?;
            let Some(merged) = merged.as_mut() else {
                merged = Some(equality);
                continue;
            };
            if merged.column_idx != equality.column_idx
                || merged.path != equality.path
                || merged.as_type != equality.as_type
            {
                return None;
            }
            merged.values.extend(equality.values);
        }
        merged.filter(|merged| merged.values.len() <= MAX_IN_LIST_VALUES)
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
            values: vec![value],
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
/// per row group (dictionary pruning and batch pre-filtering). A scan looks
/// one constant up per column, so an equality with several values prunes by
/// statistics alone.
pub fn equality_predicates(predicates: &[PushedPredicate]) -> Vec<ScanEqualityPredicate> {
    predicates
        .iter()
        .filter(|predicate| matches!(predicate.compare_type, CompareType::Equal))
        .filter_map(|predicate| match predicate.values.as_slice() {
            [value] => Some(ScanEqualityPredicate {
                column_idx: predicate.column_idx,
                path: predicate.path.clone(),
                value: value.clone(),
            }),
            _ => None,
        })
        .collect()
}

/// Clone a table and retain only the row groups which the recorded predicates
/// do not eliminate: a predicate eliminates a row group when each of its
/// values does. The clone preserves [`ParquetTable`]'s captured schema even
/// when every row group is removed.
pub fn prune_parquet(parquet: &ParquetTable, predicates: &[PushedPredicate]) -> ParquetTable {
    let mut parquet = parquet.clone();
    parquet.row_groups_mut().retain(|rg| {
        !predicates.iter().any(|predicate| {
            predicate
                .leaf_for_row_group(rg.as_ref())
                .is_some_and(|leaf| {
                    predicate.values.iter().all(|value| {
                        row_group_eliminated(rg.as_ref(), leaf, predicate.compare_type, value)
                            .unwrap_or(false)
                    })
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

#[cfg(test)]
mod tests {
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use arrow_array::{Datum as _, Int32Array};
    use planner::expression::{Compare, Conjunction, InList, Ref};

    use super::*;

    fn column(column_idx: usize) -> Expression {
        Expression::Ref(Ref {
            column_idx,
            return_type: Type::Int32,
            name: None,
        })
    }

    fn constant(value: i32) -> Expression {
        Expression::Constant(Scalar::new(
            Arc::new(Int32Array::from(vec![value])) as ArrayRef
        ))
    }

    fn equals(column_idx: usize, value: i32) -> Expression {
        Expression::Compare(Compare {
            left: Box::new(column(column_idx)),
            right: Box::new(constant(value)),
            compare_type: CompareType::Equal,
            return_type: Type::Boolean,
        })
    }

    fn in_list(column_idx: usize, values: impl Iterator<Item = i32>) -> Expression {
        Expression::InList(InList {
            input: Box::new(column(column_idx)),
            values: values.map(constant).collect(),
        })
    }

    fn any_of(children: Vec<Expression>) -> Expression {
        Expression::Conjunction(Conjunction {
            op: ConjunctionOp::Or,
            children,
        })
    }

    fn int_values(predicate: &PushedPredicate) -> Vec<i32> {
        predicate
            .values
            .iter()
            .map(|value| value.get().0.as_primitive::<Int32Type>().value(0))
            .collect()
    }

    #[test]
    fn an_in_list_is_one_equality_with_every_listed_value() {
        let predicates = PushedPredicate::from_filter(&in_list(2, [7, 8, 9].into_iter()));

        let [predicate] = predicates.as_slice() else {
            panic!("one predicate, got {predicates:?}");
        };
        assert_eq!(predicate.column_idx, 2);
        assert!(matches!(predicate.compare_type, CompareType::Equal));
        assert_eq!(int_values(predicate), [7, 8, 9]);
    }

    #[test]
    fn an_or_of_equalities_on_one_column_merges_their_values() {
        let filter = any_of(vec![equals(1, 4), in_list(1, [5, 6].into_iter())]);

        let predicates = PushedPredicate::from_filter(&filter);

        assert_eq!(int_values(&predicates[0]), [4, 5, 6]);
        assert_eq!(predicates.len(), 1);
    }

    #[test]
    fn an_or_across_columns_or_with_a_range_prunes_nothing() {
        let across_columns = any_of(vec![equals(0, 1), equals(1, 1)]);
        let with_range = any_of(vec![
            equals(0, 1),
            Expression::Compare(Compare {
                left: Box::new(column(0)),
                right: Box::new(constant(100)),
                compare_type: CompareType::Greater,
                return_type: Type::Boolean,
            }),
        ]);

        assert!(PushedPredicate::from_filter(&across_columns).is_empty());
        assert!(PushedPredicate::from_filter(&with_range).is_empty());
    }

    #[test]
    fn a_list_longer_than_the_limit_prunes_nothing() {
        let at_limit = in_list(0, 0..MAX_IN_LIST_VALUES as i32);
        let over_limit = in_list(0, 0..MAX_IN_LIST_VALUES as i32 + 1);

        assert_eq!(PushedPredicate::from_filter(&at_limit).len(), 1);
        assert!(PushedPredicate::from_filter(&over_limit).is_empty());
    }

    #[test]
    fn a_list_with_several_values_is_not_a_scan_equality() {
        let single = PushedPredicate::from_filter(&equals(0, 1));
        let several = PushedPredicate::from_filter(&in_list(0, [1, 2].into_iter()));

        assert_eq!(equality_predicates(&single).len(), 1);
        assert!(equality_predicates(&several).is_empty());
    }
}
