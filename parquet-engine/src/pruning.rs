//! Conservative predicate evaluation over columnar metadata, one slot per
//! manifest, file, or row group. A false result proves that no row can match;
//! absent statistics always retain the object. Bounds need not be exact extrema.

use std::sync::Arc;

use arrow_arith::boolean::{and, and_kleene, or, or_kleene};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef, BooleanArray, Datum, Scalar};
use arrow_ord::cmp;
use arrow_schema::{ArrowError, DataType};
use planner::catalog::DynamicScanPredicate;
use planner::expression::CompareType;

use crate::{PushedPredicate, RowGroupFilter, RowGroupMetadata, ScanEqualityPredicate};

/// Equality predicates that the decoder can safely apply to externally written
/// files. Its dictionary lookup uses native float equality, which cannot find
/// NaNs; leave those comparisons to the SQL filter above the scan.
pub fn equality_predicates(predicates: &[PushedPredicate]) -> Vec<ScanEqualityPredicate> {
    crate::equality_predicates(predicates)
        .into_iter()
        .filter(|predicate| {
            let (value, _) = predicate.value.get();
            match value.data_type() {
                DataType::Float32 => !value.as_primitive::<Float32Type>().value(0).is_nan(),
                DataType::Float64 => !value.as_primitive::<Float64Type>().value(0).is_nan(),
                _ => true,
            }
        })
        .collect()
}

/// Build a dynamic row-group filter using conservative bounds, including
/// file-level NaN proofs when supplied by a table format.
pub fn row_group_filter_from(predicates: Vec<DynamicScanPredicate>) -> Option<RowGroupFilter> {
    if predicates.is_empty() {
        return None;
    }
    Some(Arc::new(move |row_group: &RowGroupMetadata| -> bool {
        for pred in &predicates {
            let Some(constant) = pred.slot.boundary() else {
                continue;
            };
            // Dynamic predicates only target top-level columns. DuckDB does
            // not push them down for JSON paths.
            let leaf = crate::types::leaves::first_leaf(row_group.schema.fields(), pred.column_idx);
            if let Ok(true) = row_group_eliminated(row_group, leaf, pred.compare_type, &constant) {
                return false;
            }
        }
        true
    }))
}

/// Returns whether min/max statistics prove that the row group cannot match.
///
/// `leaf` is a column-chunk index in this row group's schema. Missing or
/// incompatible statistics cannot eliminate the row group.
pub fn row_group_eliminated(
    row_group: &RowGroupMetadata,
    leaf: usize,
    compare_type: CompareType,
    constant: &Scalar<ArrayRef>,
) -> Result<bool, ArrowError> {
    let Some(stats) = row_group.leaf_statistics(leaf) else {
        return Ok(false);
    };
    let bounds = ColumnBounds {
        lower: stats.min().map(Scalar::into_inner),
        upper: stats.max().map(Scalar::into_inner),
        all_null: Some(BooleanArray::from(vec![
            stats.null_count == Some(row_group.num_rows),
        ])),
        nan_free: Some(BooleanArray::from(vec![stats.nan_free])),
    };
    Ok(!bounds.may_match(1, compare_type, constant)?.value(0))
}

/// Bounds for one column across a batch of objects. Null array entries mean
/// unknown statistics, not SQL NULL values. Each present array has the batch's
/// length. Floating-point bounds alone do not describe the presence of NaNs.
#[derive(Default)]
pub struct ColumnBounds {
    pub lower: Option<ArrayRef>,
    pub upper: Option<ArrayRef>,
    /// True proves that every value is SQL NULL (or the object is empty).
    pub all_null: Option<BooleanArray>,
    /// True proves that the object contains no NaNs. Absent/null entries leave
    /// NaN presence unknown; bounds can then exclude only comparisons that no
    /// NaN can satisfy, such as equality with a non-NaN constant.
    pub nan_free: Option<BooleanArray>,
}

/// A predicate over statistics columns, after format-specific projection.
/// NOT must be pushed into comparisons before construction: negating a
/// "may match" result does not prove that the opposite predicate cannot match.
pub enum PruningPredicate {
    Always(bool),
    Compare {
        column: usize,
        compare: CompareType,
        value: Scalar<ArrayRef>,
    },
    And(Vec<Self>),
    Or(Vec<Self>),
}

/// A batch of statistics with stable positions identifying the input objects.
pub struct StatisticsBatch {
    pub len: usize,
    pub columns: Vec<ColumnBounds>,
}

impl StatisticsBatch {
    /// Return a non-null mask: false excludes an object, true keeps it.
    pub fn may_match(&self, predicate: &PruningPredicate) -> Result<BooleanArray, ArrowError> {
        match predicate {
            PruningPredicate::Always(value) => Ok(BooleanArray::from(vec![*value; self.len])),
            PruningPredicate::Compare {
                column,
                compare,
                value,
            } => {
                let bounds = self.columns.get(*column).ok_or_else(|| {
                    ArrowError::InvalidArgumentError(format!("missing statistics column {column}"))
                })?;
                bounds.may_match(self.len, *compare, value)
            }
            PruningPredicate::And(predicates) | PruningPredicate::Or(predicates) => {
                let conjunction = matches!(predicate, PruningPredicate::And(_));
                let mut result = BooleanArray::from(vec![conjunction; self.len]);
                for child in predicates {
                    let next = self.may_match(child)?;
                    result = if conjunction {
                        and(&result, &next)?
                    } else {
                        or(&result, &next)?
                    };
                }
                Ok(result)
            }
        }
    }
}

impl ColumnBounds {
    /// Evaluate one comparison for every object. One-sided bounds can prove
    /// exclusions too; unknown and incompatible bounds contribute no proof.
    pub fn may_match(
        &self,
        len: usize,
        compare: CompareType,
        constant: &Scalar<ArrayRef>,
    ) -> Result<BooleanArray, ArrowError> {
        for array in [
            self.lower.as_deref(),
            self.upper.as_deref(),
            self.all_null.as_ref().map(|array| array as &dyn Array),
            self.nan_free.as_ref().map(|array| array as &dyn Array),
        ]
        .into_iter()
        .flatten()
        {
            if array.len() != len {
                return Err(ArrowError::InvalidArgumentError(
                    "statistics arrays have different lengths".into(),
                ));
            }
        }
        let (value, _) = constant.get();
        if value.is_null(0) {
            return Ok(BooleanArray::from(vec![false; len]));
        }
        let test = |bound: &Option<ArrayRef>,
                    kernel: fn(&dyn Datum, &dyn Datum) -> Result<BooleanArray, ArrowError>|
         -> Result<BooleanArray, ArrowError> {
            match bound {
                Some(bound) if bound.data_type() == value.data_type() => {
                    let result = kernel(bound, constant)?;
                    // A NaN bound supplies no ordering information, even
                    // when separate counts prove the data has no NaNs.
                    Ok(match bound.data_type() {
                        DataType::Float32 => result
                            .iter()
                            .zip(bound.as_primitive::<Float32Type>().iter())
                            .map(|(excluded, value)| {
                                value.filter(|value| !value.is_nan()).and(excluded)
                            })
                            .collect(),
                        DataType::Float64 => result
                            .iter()
                            .zip(bound.as_primitive::<Float64Type>().iter())
                            .map(|(excluded, value)| {
                                value.filter(|value| !value.is_nan()).and(excluded)
                            })
                            .collect(),
                        _ => result,
                    })
                }
                _ => Ok(BooleanArray::new_null(len)),
            }
        };
        let excluded = match compare {
            CompareType::Equal => {
                or_kleene(&test(&self.lower, cmp::gt)?, &test(&self.upper, cmp::lt)?)?
            }
            CompareType::NotEqual => {
                and_kleene(&test(&self.lower, cmp::eq)?, &test(&self.upper, cmp::eq)?)?
            }
            CompareType::Less => test(&self.lower, cmp::gt_eq)?,
            CompareType::LessEqual => test(&self.lower, cmp::gt)?,
            CompareType::Greater => test(&self.upper, cmp::lt_eq)?,
            CompareType::GreaterEqual => test(&self.upper, cmp::lt)?,
        };
        let constant_is_nan = match value.data_type() {
            DataType::Float32 => Some(value.as_primitive::<Float32Type>().value(0).is_nan()),
            DataType::Float64 => Some(value.as_primitive::<Float64Type>().value(0).is_nan()),
            _ => None,
        };
        // Arrow's total ordering puts negative NaNs below finite values and
        // positive NaNs above them. Either sign can satisfy a range comparison;
        // neither can equal a non-NaN constant.
        let needs_nan_proof =
            constant_is_nan.is_some_and(|is_nan| is_nan || !matches!(compare, CompareType::Equal));
        let proven = |mask: &BooleanArray, index| mask.is_valid(index) && mask.value(index);
        Ok((0..len)
            .map(|index| {
                let all_null = self
                    .all_null
                    .as_ref()
                    .is_some_and(|mask| proven(mask, index));
                let usable_bounds = !needs_nan_proof
                    || self
                        .nan_free
                        .as_ref()
                        .is_some_and(|mask| proven(mask, index));
                !all_null && !(usable_bounds && proven(&excluded, index))
            })
            .collect())
    }
}

#[cfg(test)]
mod tests;
